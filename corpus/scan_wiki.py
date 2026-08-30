"""Scan Wikipedia parquet shards for concept mentions and accumulate Karkada-style windowed
co-occurrence counts (L=16, f(d)=L+1-d, symmetric), unigram counts, total word count, and
sequential-interval statistics. One output JSON per shard. Run with multiprocessing.

P_ij = (1/Z) sum_nu delta_{C[nu],i} sum_{d=1}^{L} f(d) (delta_{C[nu+d],j} + delta_{C[nu-d],j})
Z = sum over all token positions and offsets of f(d) (both directions) = 2 * N_words * sum_d f(d)
(up to boundary effects) — we accumulate Z exactly per document.
Positions are word indices (whitespace tokenization); a multiword mention (e.g. "C major") is
located at its first word.
"""
import sys, os, json, re, bisect, time
from collections import defaultdict
import pyarrow.parquet as pq
from corpus.concepts import (KEY_RE, MONTH_RE, WEEKDAY_RE, NOTE_RE, COF_PHRASES, key_name,
                             is_sentence_initial)

L = 16
FW = [0] + [L + 1 - d for d in range(1, L + 1)]  # f(d)
SUM_F = sum(FW)
TOKEN_RE = re.compile(r"\S+")

PREFILTER = ("major", "minor", "January", "February", "March", "April", "May", "June", "July", "August",
             "September", "October", "November", "December", "Monday", "Tuesday", "Wednesday", "Thursday",
             "Friday", "Saturday", "Sunday", "note ", "pitch ", "tonic ")


class Tally:
    def __init__(self):
        self.uni = defaultdict(int)           # concept -> count
        self.uni_sentinit = defaultdict(int)  # excluded sentence-initial key matches
        self.co = defaultdict(float)          # (a,b) -> weighted count (both orders accumulated)
        self.co_d = defaultdict(int)          # (a,b,d) -> raw count for d=1..L (a before b)
        self.doc_co = defaultdict(int)        # (a,b) -> number of docs containing both (a<=b)
        self.doc_uni = defaultdict(int)       # concept -> docs containing it
        self.nwords = 0
        self.Z = 0.0
        self.ndocs = 0
        self.adjacent = defaultdict(int)      # (a,b) for consecutive key mentions within 40 words
        self.cof_docs = 0

    def to_json(self):
        return {"uni": dict(self.uni), "uni_sentinit": dict(self.uni_sentinit),
                "co": {f"{a}|{b}": v for (a, b), v in self.co.items()},
                "co_d": {f"{a}|{b}|{d}": v for (a, b, d), v in self.co_d.items()},
                "doc_co": {f"{a}|{b}": v for (a, b), v in self.doc_co.items()},
                "doc_uni": dict(self.doc_uni), "nwords": self.nwords, "Z": self.Z, "ndocs": self.ndocs,
                "adjacent": {f"{a}|{b}": v for (a, b), v in self.adjacent.items()}, "cof_docs": self.cof_docs}


def find_mentions(text):
    """Return list of (char_offset, concept_string) for all families."""
    out = []
    sentinit = []
    for m in KEY_RE.finditer(text):
        name = "KEY:" + key_name(m) + ":" + m.group("mode")
        if is_sentence_initial(text, m.start()):
            sentinit.append(name)
            continue
        out.append((m.start(), name))
    for m in MONTH_RE.finditer(text):
        out.append((m.start(), "MONTH:" + m.group("month")))
    for m in WEEKDAY_RE.finditer(text):
        out.append((m.start(), "WDAY:" + m.group("weekday")))
    for m in NOTE_RE.finditer(text):
        out.append((m.start("letter"), "NOTE:" + key_name(m)))
    out.sort()
    return out, sentinit


def process_doc(text, tallies):
    """tallies: dict name -> Tally; 'all' always updated; 'nocof' updated iff doc lacks COF phrases;
    'cof' updated iff doc has them."""
    nw = text.count(" ") + text.count("\n") + 1  # cheap approx of len(text.split()) for Z/N; exact below for matched docs
    tl = text.lower()
    has_cof = any(p in tl for p in COF_PHRASES)
    groups = [tallies["all"], tallies["cof"] if has_cof else tallies["nocof"]]
    if not any(p in text for p in PREFILTER):
        for T in groups:
            T.nwords += nw; T.Z += 2.0 * nw * SUM_F; T.ndocs += 1
            if has_cof: T.cof_docs += 1
        return
    mentions, sentinit = find_mentions(text)
    toks = [m.start() for m in TOKEN_RE.finditer(text)]
    nw = len(toks)
    for T in groups:
        T.nwords += nw; T.Z += 2.0 * nw * SUM_F; T.ndocs += 1
        if has_cof: T.cof_docs += 1
        for s in sentinit: T.uni_sentinit[s] += 1
    if not mentions:
        return
    pos = [bisect.bisect_right(toks, off) - 1 for off, _ in mentions]
    names = [n for _, n in mentions]
    present = set(names)
    for T in groups:
        for n in names: T.uni[n] += 1
        for n in present: T.doc_uni[n] += 1
        pl = sorted(present)
        for i in range(len(pl)):
            for j in range(i, len(pl)):
                T.doc_co[(pl[i], pl[j])] += 1
    n = len(names)
    last_key_i = None
    for i in range(n):
        a = names[i]
        if a.startswith("KEY:"):
            if last_key_i is not None and pos[i] - pos[last_key_i] <= 40:
                for T in groups: T.adjacent[(names[last_key_i], a)] += 1
            last_key_i = i
        j = i + 1
        while j < n and pos[j] - pos[i] <= L:
            d = pos[j] - pos[i]
            b = names[j]
            if d >= 1:
                w = FW[d]
                for T in groups:
                    T.co[(a, b)] += w; T.co[(b, a)] += w
                    T.co_d[(a, b, d)] += 1
            j += 1


def scan_shard(path, outdir):
    t0 = time.time()
    tallies = {k: Tally() for k in ("all", "nocof", "cof")}
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=2000, columns=["text"]):
        for text in batch.column("text").to_pylist():
            process_doc(text, tallies)
    base = os.path.basename(path).replace(".parquet", "")
    with open(os.path.join(outdir, base + ".json"), "w") as f:
        json.dump({k: T.to_json() for k, T in tallies.items()}, f)
    return base, tallies["all"].ndocs, tallies["all"].nwords, time.time() - t0


if __name__ == "__main__":
    import multiprocessing as mp
    indir, outdir = sys.argv[1], sys.argv[2]
    nproc = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    os.makedirs(outdir, exist_ok=True)
    files = sorted(os.path.join(indir, f) for f in os.listdir(indir) if f.endswith(".parquet"))
    files = [f for f in files if not os.path.exists(os.path.join(outdir, os.path.basename(f).replace(".parquet", ".json")))]
    print(f"{len(files)} shards to scan with {nproc} procs", flush=True)
    with mp.Pool(nproc) as pool:
        results = [pool.apply_async(scan_shard, (f, outdir)) for f in files]
        for r in results:
            base, nd, nwords, dt = r.get()
            print(f"{base}: {nd} docs, {nwords/1e6:.1f}M words, {dt:.0f}s", flush=True)
