"""§12: three corpus operators on the same corpus, side by side, over the 15 spellings:
  SYM  = symmetric PMI (Karkada window, Phase I tallies; Phase II corpus15);
  HELP = helper-word factorization W = Φ√|Λ| of the V=3000 PMI over key documents (Phase I keydocs_V3000.npz), key rows;
  COND = directional conditional rows (Phase V family A_win40 / B_any / D_doc).
For each: geometry (ECI, circle|line, line|circle with orthographic controls; and the *raw* line/circle RSA, spelled view),
and how well its 15×15 (dis)similarity predicts (a) key-name span-mean geometry, (b) prompt-final geometry, (c) behaviour
(Phase II E_modulation), via Spearman over pairs / ordered pairs. Usage: python -m phase5.operators"""
import json, numpy as np, sys
sys.path.insert(0, ".")
from phase2.keys15 import KEYS15, candidate_geometries, ENH_PAIRS, n
from phase2.contexts import FAMILIES
from pf.fourier import center
from scipy.stats import spearmanr, rankdata
iu = np.triu_indices(n, 1); NP = len(iu[0]); pk = np.array([[i, j] for i, j in zip(*iu)]); eidx = [np.where((pk[:, 0] == i) & (pk[:, 1] == j))[0][0] for i, j in ENH_PAIRS]
corpus = json.load(open("results/corpus_merged/wiki_full.json"))["all"]["uni"]; lf = np.log(np.array([corpus.get(f"KEY:{k}:major", 0) + 1 for k in KEYS15], float))
G = candidate_geometries(logfreq=lf); R = {k: rankdata(v[iu]) for k, v in G.items()}; CTRL = ["glyph_class", "edit_distance", "same_letter", "alphabet", "commonness"]
def partial(d, t, c):
    tt = rankdata(d); X = np.column_stack([np.ones(NP)] + [R[x] for x in c]); g = R[t]; rt = tt - X @ np.linalg.lstsq(X, tt, rcond=None)[0]; rg = g - X @ np.linalg.lstsq(X, g, rcond=None)[0]
    return float(rt @ rg / np.sqrt((rt @ rt) * (rg @ rg))) if rt @ rt > 1e-12 else float("nan")
def geom(D, name):
    d = D[iu]; e = (rankdata(d) / NP)[eidx].mean()
    print(f"  {name:44s} ECI {e:.2f} | raw RSA circle {spearmanr(d, G['circle_fifths'][iu]).correlation:+.2f} line {spearmanr(d, G['line_fifths'][iu]).correlation:+.2f} | controlled circle|line {partial(d, 'circle_fifths', CTRL + ['line_fifths']):+.2f} line|circle {partial(d, 'line_fifths', CTRL + ['circle_fifths']):+.2f}")
    return d
ops = {}
# SYM: Phase II 15-key PMI
P15 = json.load(open("results/phase2/corpus/pmi15.json")); Msym = np.array(P15["M"]); ops["SYM (PMI, symmetric window)"] = -Msym
# HELP: theory embedding rows for the 15 spellings
z = np.load("results/corpus_merged/keydocs_V3000.npz", allow_pickle=True); C, uni, vocab, N, Z = z["C"], z["uni"], list(z["vocab"]), float(z["N"]), float(z["Z"]); vi = {w: i for i, w in enumerate(vocab)}
with np.errstate(divide="ignore", invalid="ignore"):
    M = np.log((C / Z) / np.outer(uni / N, uni / N)); M = np.where(np.isfinite(M), M, np.nan); M = np.where(np.isnan(M), np.nanmin(M) - 1, M)
M = (M + M.T) / 2; w, Phi = np.linalg.eigh(M); o = np.argsort(-np.abs(w))[:300]; W = Phi[:, o] * np.sqrt(np.abs(w[o]))
idx = [vi.get(f"KEY_{k}_major") for k in KEYS15]; missing = [k for k, i in zip(KEYS15, idx) if i is None]
if not missing:
    Hk = W[idx]; Hc = Hk - Hk.mean(0); ops["HELP (helper-word factorization)"] = np.linalg.norm(Hc[:, None] - Hc[None], axis=-1)
else: print("helper factorization missing keys:", missing)
# COND: directional conditional rows -> symmetrized row divergence (JS over 15 spelled targets) and also the raw log-conditional
Zc = np.load("results/phase5/cond_wikipedia.npz")
def js(p, q):
    m = (p + q) / 2; f = lambda a, b: np.nansum(np.where(a > 0, a * np.log(a / b), 0)); return 0.5 * f(p, m) + 0.5 * f(q, m)
for ex in ("A_win40", "B_any", "D_doc"):
    Mx = Zc[ex]; rows = (Mx + 0.5) / (Mx + 0.5).sum(1, keepdims=True); ops[f"COND rows JS ({ex})"] = np.array([[js(rows[i], rows[j]) for j in range(n)] for i in range(n)])
# matched operators built from the SAME 40-word ordered counts N: symmetric counts S = N + N^T
N40 = Zc["A_win40"].astype(float); S40 = N40 + N40.T
rows = (S40 + 0.5) / (S40 + 0.5).sum(1, keepdims=True); ops["MATCHED sym-conditional JS (N+N^T rows)"] = np.array([[js(rows[i], rows[j]) for j in range(n)] for i in range(n)])
with np.errstate(divide="ignore"):
    tot = S40.sum(); pmi40 = np.log(((S40 + 0.5) / tot) / np.outer(S40.sum(1) / tot, S40.sum(0) / tot))
ops["MATCHED PMI (N+N^T), distance = -PMI"] = -pmi40
rowsT = (N40.T + 0.5) / (N40.T + 0.5).sum(1, keepdims=True); ops["MATCHED reverse conditional JS (N^T rows)"] = np.array([[js(rowsT[i], rowsT[j]) for j in range(n)] for i in range(n)])
print("Corpus operators — geometry over 15 spellings (spelled tonal view = raw RSA; neutralized/controlled = partials):")
dvecs = {name: geom(D, name) for name, D in ops.items()}
# targets: model geometries and behaviour
print("\nWhat each operator predicts (Spearman over 105 pairs of distances; behaviour uses symmetrized −log q):")
print(f"{'operator':28s} | " + " ".join(f"{t:>28s}" for t in ["keyname span-mean (7B L24)", "prompt-final (7B L18)", "behaviour E (7B)", "behaviour E (1B)", "keyname span-mean (Qwen L27)", "behaviour E (Qwen)", "behaviour E (Gemma)"]))
targets = {}
for tag, layer, key in (("olmo2_7b", 24, "keyname span-mean (7B L24)"), ("qwen25_3b", 27, "keyname span-mean (Qwen L27)")):
    Zh = np.load(f"results/phase2/hidden/{tag}_symbol_v2.npz"); H = np.stack([Zh[f"E_modulation__t{ti}__mean"] for ti in range(4)], 0).mean(0)[layer]; Hc = center(H); targets[key] = np.linalg.norm(Hc[:, None] - Hc[None], axis=-1)[iu]
Zh = np.load("results/phase2/hidden/olmo2_7b_symbol.npz"); H = np.stack([Zh[f"E_modulation__t{ti}__final"] for ti in range(4)], 0).mean(0)[18]; Hc = center(H); targets["prompt-final (7B L18)"] = np.linalg.norm(Hc[:, None] - Hc[None], axis=-1)[iu]
for tag, key in (("olmo2_7b", "behaviour E (7B)"), ("olmo2_1b", "behaviour E (1B)"), ("qwen25_3b", "behaviour E (Qwen)"), ("gemma2_2b", "behaviour E (Gemma)")):
    J = json.load(open(f"results/phase2/behavior/{tag}.json")); L = np.mean([np.array(J[f"E_modulation__t{ti}"]["total"]) for ti in range(4)], 0); Sm = -(L + L.T) / 2; targets[key] = Sm[iu]
for name, d in dvecs.items():
    print(f"{name:28s} | " + " ".join(f"{spearmanr(d, targets[t]).correlation:+28.2f}" for t in ["keyname span-mean (7B L24)", "prompt-final (7B L18)", "behaviour E (7B)", "behaviour E (1B)", "keyname span-mean (Qwen L27)", "behaviour E (Qwen)", "behaviour E (Gemma)"]))
