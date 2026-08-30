"""Phase V extraction families over a corpus of documents (parquet 'text' column, or Dolma json.gz 'text' field):
A: directional window N=40 (target j within 40 words after source i);
B: cue-conditioned directional window (modulation / chord / signature / enharmonic / relation cues in the intervening text);
C: high-precision relational patterns; D: document-conditioned directional (j occurs anywhere later in the doc after first i).
Only major keys, 15 spellings (symbol, word and unicode forms mapped to the symbol spelling). Outputs npz of count matrices
+ source counts. Usage: python -m phase5.scan_conditional <input dir or file list> <out.npz> <nproc> [fmt: parquet|jsonl.gz]"""
import sys, os, re, gzip, json, numpy as np
sys.path.insert(0, ".")
from corpus.concepts import KEY_RE, key_name, is_sentence_initial
from phase2.keys15 import KEYS15
KI = {k: i for i, k in enumerate(KEYS15)}; n = 15
CUES = {"modulation": ["modulat", "transpos", "moves to", "shifts to", "changes to", "then to", "moving to", "returns to", "back to"],
        "chord": ["chord", "cadence", "progression", "tonic", "dominant", "subdominant", "harmon"],
        "signature": ["sharp", "flat", "key signature", "accidental"],
        "enharmonic": ["enharmonic", "equivalent", "same as", "identical"],
        "relation": ["relative", "parallel", "related", "neighbo", "close to", "distant"]}
PAT_C = [re.compile(r"from\s+\{X\}\s+major\s+(?:to|into)\s+\{Y\}\s+major"), re.compile(r"\{X\}\s+major\s+(?:to|→|-)\s+\{Y\}\s+major"),
         re.compile(r"\{X\}\s+major(?:\W+\w+){0,11}?\W+modulat\w*\s+(?:to|into)\s+\{Y\}\s+major"), re.compile(r"\{X\}\s+major\s+and\s+\{Y\}\s+major")]
NAME = r"(?:[A-G](?:#|♯|-sharp| sharp|b|♭|-flat| flat)?)"
FAMS = ["A_win40", "B_modulation", "B_chord", "B_signature", "B_enharmonic", "B_relation", "B_any", "C_patterns", "D_doc"]
def keyname_norm(s):
    s = s.replace("♯", "#").replace("♭", "b").replace("-sharp", "#").replace(" sharp", "#").replace("-flat", "b").replace(" flat", "b"); return s
def iter_docs(path):
    if path.endswith(".parquet"):
        import pyarrow.parquet as pq
        for batch in pq.ParquetFile(path).iter_batches(batch_size=2000, columns=["text"]):
            for t in batch.column("text").to_pylist(): yield t
    elif path.endswith(".zst") or path.endswith(".zstd"):
        import zstandard as zstd, io
        with open(path, "rb") as fh:
            with zstd.ZstdDecompressor().stream_reader(fh) as r:
                for line in io.TextIOWrapper(r, encoding="utf-8", errors="ignore"):
                    try: yield json.loads(line)["text"]
                    except Exception: continue
    else:
        with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
            for line in f:
                try: yield json.loads(line)["text"]
                except Exception: continue
PERDOC = os.environ.get("PF_PERDOC", "") != ""
def scan(path):
    N = {f: np.zeros((n, n)) for f in FAMS}; uni = np.zeros(n); ndocs = 0; nwords = 0; per = []
    for text in iter_docs(path):
        if " major" not in text: continue
        ms = [(m.start(), m.end(), key_name(m)) for m in KEY_RE.finditer(text) if m.group("mode") == "major" and not is_sentence_initial(text, m.start())]
        ms = [(s, e, k) for s, e, k in ms if k in KI]
        if not ms: continue
        ndocs += 1
        if PERDOC: dA, dD, du = {}, {}, np.zeros(n); A0 = N["A_win40"].copy(); D0 = N["D_doc"].copy()
        for s, e, k in ms: uni[KI[k]] += 1
        seen_first = {}
        for s, e, k in ms:
            if k not in seen_first: seen_first[k] = s
        for k, s in seen_first.items():
            for k2, s2 in seen_first.items():
                if k2 != k and s2 > s: N["D_doc"][KI[k], KI[k2]] += 1
        for i in range(len(ms)):
            for j in range(i + 1, len(ms)):
                seg = text[ms[i][1]:ms[j][0]]; nw = len(seg.split())
                if nw > 40: break
                a, b = KI[ms[i][2]], KI[ms[j][2]]; N["A_win40"][a, b] += 1; low = seg.lower(); hit = False
                for c, cues in CUES.items():
                    if any(q in low for q in cues): N[f"B_{c}"][a, b] += 1; hit = True
                if hit: N["B_any"][a, b] += 1
        # C: pattern matches on the raw text with normalized names
        for m in re.finditer(r"(?<![A-Za-z0-9\-])(" + NAME + r")\s+major", text):
            pass
        for pat in PAT_C:
            rx = re.compile(pat.pattern.replace(r"\{X\}", "(?P<X>" + NAME + ")").replace(r"\{Y\}", "(?P<Y>" + NAME + ")"))
            for m in rx.finditer(text):
                x, y = keyname_norm(m.group("X")), keyname_norm(m.group("Y"))
                if x in KI and y in KI and x != y: N["C_patterns"][KI[x], KI[y]] += 1
        if PERDOC:
            dA = N["A_win40"] - A0; dD = N["D_doc"] - D0; du = np.zeros(n)
            for s, e, k in ms: du[KI[k]] += 1
            per.append((dA, dD, du))
    return N, uni, ndocs, per
if __name__ == "__main__":
    import multiprocessing as mp
    src, out, nproc = sys.argv[1], sys.argv[2], int(sys.argv[3])
    files = sorted(os.path.join(src, f) for f in os.listdir(src) if f.endswith((".parquet", ".json.gz", ".jsonl.gz", ".zst", ".zstd"))) if os.path.isdir(src) else [l.strip() for l in open(src) if l.strip()]
    with mp.Pool(min(nproc, len(files))) as pool: res = pool.map(scan, files)
    N = {f: sum(r[0][f] for r in res) for f in FAMS}; uni = sum(r[1] for r in res); nd = sum(r[2] for r in res)
    np.savez(out, uni=uni, ndocs=nd, **N); print("saved", out, "key docs", nd, "source counts", uni.astype(int).tolist(), {f: int(N[f].sum()) for f in FAMS})
    if PERDOC:
        per = [x for r in res for x in r[3]]; A_doc, A_i, A_j, A_c, D_doc, D_i, D_j, D_c = [], [], [], [], [], [], [], []
        for d, (dA, dD, du) in enumerate(per):
            ii, jj = np.nonzero(dA); A_doc += [d] * len(ii); A_i += ii.tolist(); A_j += jj.tolist(); A_c += dA[ii, jj].tolist()
            ii, jj = np.nonzero(dD); D_doc += [d] * len(ii); D_i += ii.tolist(); D_j += jj.tolist(); D_c += dD[ii, jj].tolist()
        np.savez(os.environ["PF_PERDOC"], uni_docs=np.array([x[2] for x in per]), A_win40_doc=np.array(A_doc), A_win40_i=np.array(A_i), A_win40_j=np.array(A_j), A_win40_c=np.array(A_c), D_doc_doc=np.array(D_doc), D_doc_i=np.array(D_i), D_doc_j=np.array(D_j), D_doc_c=np.array(D_c))
        print("saved per-document counts", os.environ["PF_PERDOC"], len(per), "docs")
