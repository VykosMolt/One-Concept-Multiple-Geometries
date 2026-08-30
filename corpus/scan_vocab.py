"""Second corpus pass: documents containing >=1 key mention. Build a vocabulary (keys as merged tokens
+ top-V words in those docs) and accumulate the full V×V Karkada-weighted co-occurrence matrix (L=16,
f(d)=L+1-d, symmetric) so that the theory's own embedding prediction W = Phi sqrt|Lambda| can be computed
with helper words (Karkada Fig. 4). Output: npz with counts matrix, unigrams, vocab, N, Z.
Usage: python -m corpus.scan_vocab <parquet dir> <out.npz> <V> <nproc>"""
import sys, os, re, json, time, numpy as np
from collections import Counter
import pyarrow.parquet as pq
from corpus.concepts import KEY_RE, key_name, is_sentence_initial

L = 16; FW = np.array([0] + [L + 1 - d for d in range(1, L + 1)], float)
WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*|\d+|[^\sA-Za-z\d]")


def key_docs(path):
    """yield token lists for docs with key mentions, keys replaced by single tokens KEY_<spelling>_<mode>."""
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=2000, columns=["text"]):
        for text in batch.column("text").to_pylist():
            if " major" not in text and " minor" not in text: continue
            ms = [m for m in KEY_RE.finditer(text) if not is_sentence_initial(text, m.start())]
            if not ms: continue
            out = []; last = 0
            for m in ms:
                out.extend(w.lower() for w in WORD_RE.findall(text[last:m.start()]))
                out.append(f"KEY_{key_name(m)}_{m.group('mode')}"); last = m.end()
            out.extend(w.lower() for w in WORD_RE.findall(text[last:]))
            yield out


def pass1(path):
    c = Counter(); nd = 0
    for toks in key_docs(path): c.update(toks); nd += 1
    return c, nd


def pass2(args):
    path, vocab_index, V = args
    Cm = np.zeros((V, V), np.float64); uni = np.zeros(V); N = 0
    for toks in key_docs(path):
        ids = np.array([vocab_index.get(t, -1) for t in toks]); N += len(ids)
        valid = ids >= 0
        np.add.at(uni, ids[valid], 1)
        for d in range(1, L + 1):
            a, b = ids[:-d], ids[d:]; m = (a >= 0) & (b >= 0)
            if m.any():
                np.add.at(Cm, (a[m], b[m]), FW[d]); np.add.at(Cm, (b[m], a[m]), FW[d])
    return Cm, uni, N


if __name__ == "__main__":
    import multiprocessing as mp
    indir, out, V, nproc = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    files = sorted(os.path.join(indir, f) for f in os.listdir(indir) if f.endswith(".parquet"))
    t0 = time.time()
    with mp.Pool(nproc) as pool:
        res = pool.map(pass1, files)
    total = Counter(); nd = 0
    for c, n in res: total.update(c); nd += n
    keys = sorted(t for t in total if t.startswith("KEY_"))
    words = [w for w, _ in total.most_common() if not w.startswith("KEY_")][:V - len(keys)]
    vocab = keys + words; vi = {w: i for i, w in enumerate(vocab)}
    print(f"pass1: {nd} key docs, {len(total)} types, vocab {len(vocab)} ({len(keys)} keys), {time.time()-t0:.0f}s", flush=True)
    with mp.Pool(nproc) as pool:
        res = pool.map(pass2, [(f, vi, len(vocab)) for f in files])
    Cm = sum(r[0] for r in res); uni = sum(r[1] for r in res); N = sum(r[2] for r in res)
    Z = 2.0 * N * FW.sum()
    np.savez_compressed(out, C=Cm, uni=uni, vocab=np.array(vocab), N=N, Z=Z, ndocs=nd)
    print(f"pass2 done: N={N} words in key docs; saved {out}; {time.time()-t0:.0f}s")
