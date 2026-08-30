"""Track 6: task-aligned corpus statistics. Over key-containing Wikipedia documents, for every ordered pair of major-key
mentions (x then y) within 24 words, classify the intervening text by cue class:
  modulation {modulat, transpos, moves to, shifts to, changes to, then to},
  chord      {chord, cadence, progression, tonic, dominant, subdominant, harmon},
  signature  {sharp, flat, key signature, accidental},
  enharmonic {enharmonic, equivalent, same as}, other.
Build conditional count matrices N_c[x, y] over the 15 spellings and report, per class, RSA/partials with circle vs line
(same controls as elsewhere) and the enharmonic collapse index, with Poisson bootstrap. Usage: python -m phase2.corpus_conditional"""
import sys, os, re, json, numpy as np
sys.path.insert(0, ".")
import pyarrow.parquet as pq
from corpus.concepts import KEY_RE, key_name, is_sentence_initial
from phase2.keys15 import KEYS15, GLYPH, ENH_PAIRS, candidate_geometries, n
CUES = {"modulation": ["modulat", "transpos", "moves to", "shifts to", "changes to", "then to", "moving to"],
        "chord": ["chord", "cadence", "progression", "tonic", "dominant", "subdominant", "harmon"],
        "signature": ["sharp", "flat", "key signature", "accidental"],
        "enharmonic": ["enharmonic", "equivalent", "same as"]}
KI = {k: i for i, k in enumerate(KEYS15)}
def scan(path):
    N = {c: np.zeros((n, n)) for c in list(CUES) + ["other", "all"]}; uni = np.zeros(n)
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=2000, columns=["text"]):
        for text in batch.column("text").to_pylist():
            if " major" not in text: continue
            ms = [(m.start(), m.end(), key_name(m)) for m in KEY_RE.finditer(text) if m.group("mode") == "major" and not is_sentence_initial(text, m.start()) and key_name(m) in KI]
            for s, e, k in ms: uni[KI[k]] += 1
            for i in range(len(ms)):
                for j in range(i + 1, len(ms)):
                    seg = text[ms[i][1]:ms[j][0]]
                    if len(seg.split()) > 24: break
                    low = seg.lower(); a, b = KI[ms[i][2]], KI[ms[j][2]]
                    hit = False
                    for c, cues in CUES.items():
                        if any(q in low for q in cues): N[c][a, b] += 1; hit = True
                    if not hit: N["other"][a, b] += 1
                    N["all"][a, b] += 1
    return N, uni
if __name__ == "__main__":
    import multiprocessing as mp
    files = sorted(os.path.join("data/wiki", f) for f in os.listdir("data/wiki") if f.endswith(".parquet"))
    with mp.Pool(20) as pool: res = pool.map(scan, files)
    N = {c: sum(r[0][c] for r in res) for c in res[0][0]}; uni = sum(r[1] for r in res)
    os.makedirs("results/phase2/corpus", exist_ok=True)
    np.savez("results/phase2/corpus/conditional15.npz", uni=uni, **N)
    from scipy.stats import spearmanr, rankdata
    iu = np.triu_indices(n, 1); NP = len(iu[0]); G = candidate_geometries(logfreq=np.log(uni + 1)); R = {k: rankdata(v[iu]) for k, v in G.items()}
    CTRL = ["glyph_class", "edit_distance", "same_letter", "alphabet", "commonness"]
    def partial(dvec, target, controls):
        t = rankdata(dvec); X = np.column_stack([np.ones(NP)] + [R[c] for c in controls]); g = R[target]
        rt = t - X @ np.linalg.lstsq(X, t, rcond=None)[0]; rg = g - X @ np.linalg.lstsq(X, g, rcond=None)[0]; return float(rt @ rg / np.sqrt((rt @ rt) * (rg @ rg)))
    pk = np.array([[i, j] for i, j in zip(*iu)]); idx = [np.where((pk[:, 0] == i) & (pk[:, 1] == j))[0][0] for i, j in ENH_PAIRS]
    rng = np.random.default_rng(0)
    print(f"{'class':11s} {'pairs':>7s} | ECI  circle|line line|circle | RSA circle line | top directed intervals (y-x on line, excl 0)")
    for c in ["all", "other", "modulation", "chord", "signature", "enharmonic"]:
        M = N[c]; tot = M.sum()
        if tot < 50: print(f"{c:11s} {tot:7.0f} | too sparse"); continue
        Ssym = M + M.T; # symmetric conditional association: log((count+0.5)/(expected)) with expected from row/col totals
        with np.errstate(divide="ignore"):
            E = np.outer(Ssym.sum(1), Ssym.sum(0)) / Ssym.sum(); A = np.log((Ssym + 0.5) / (E + 0.5))
        d = -A[iu]; ranks = rankdata(d) / NP
        cl = partial(d, "circle_fifths", CTRL + ["line_fifths"]); lc = partial(d, "line_fifths", CTRL + ["circle_fifths"])
        # bootstrap
        bs = []
        for _ in range(200):
            Mb = rng.poisson(Ssym); Mb = np.triu(Mb) + np.triu(Mb, 1).T
            Eb = np.outer(Mb.sum(1), Mb.sum(0)) / max(Mb.sum(), 1); Ab = np.log((Mb + 0.5) / (Eb + 0.5)); db = -Ab[iu]
            bs.append((partial(db, "circle_fifths", CTRL + ["line_fifths"]), partial(db, "line_fifths", CTRL + ["circle_fifths"]), (rankdata(db) / NP)[idx].mean()))
        bs = np.array(bs)
        # directed interval histogram on the line (signed s difference), excluding self
        S = np.arange(-7, 8); hist = {}
        for i in range(n):
            for j in range(n):
                if i != j: hist[S[j] - S[i]] = hist.get(S[j] - S[i], 0) + M[i, j]
        top = sorted(hist.items(), key=lambda t: -t[1])[:5]
        print(f"{c:11s} {tot:7.0f} | {ranks[idx].mean():.2f} {cl:+.2f}±{bs[:,0].std():.2f} {lc:+.2f}±{bs[:,1].std():.2f} | {spearmanr(d, G['circle_fifths'][iu]).correlation:+.2f} {spearmanr(d, G['line_fifths'][iu]).correlation:+.2f} | " + " ".join(f"{k:+d}:{int(v)}" for k, v in top))
