"""Directed corpus statistic from co_d tallies: D[x,y] = log P(y mentioned within d<=K words AFTER x) / P(y),
compared with model predictive matrices L[x,y] over 132 ordered pairs; plus circle-vs-line on D.
Usage: python scripts/directed_corpus.py <merged.json> <K> <tags comma>"""
import sys, json, numpy as np
sys.path.insert(0, '.')
from pf.families import PC_CANON_MAJOR
from scipy.stats import spearmanr, rankdata
T = json.load(open(sys.argv[1]))["all"]; K = int(sys.argv[2]); tags = sys.argv[3].split(",")
labs = [f"KEY:{k}:major" for k in PC_CANON_MAJOR]; n = 12
uni = np.array([T["uni"].get(l, 0) for l in labs], float); N = T["nwords"]
Cd = np.zeros((n, n))
for key, v in T["co_d"].items():
    a, b, d = key.rsplit("|", 2)
    if a in labs and b in labs and int(d) <= K: Cd[labs.index(a), labs.index(b)] += v
# P(y after x within K) / P(y): counts of (x then y) / (count of x * K) divided by unigram rate of y
with np.errstate(divide="ignore"):
    D = np.log((Cd / (uni[:, None] * K)) / (uni[None] / N))
D = np.where(np.isfinite(D), D, np.nanmin(np.where(np.isfinite(D), D, np.nan)) - 1)
off = ~np.eye(n, dtype=bool)
x = np.arange(12); fp = (7 * x) % 12; signed = np.where(fp <= 6, fp, fp - 12)
circ = np.minimum((fp[:, None] - fp[None]) % 12, (fp[None] - fp[:, None]) % 12).astype(float); line = np.abs(signed[:, None] - signed[None]).astype(float)
BLACK = np.array([0,1,0,1,0,0,1,0,1,0,1,0]); blk = (BLACK[:, None] == BLACK[None]).astype(float); com = np.log(uni)[:, None] + np.log(uni)[None]
def partial(S, target, controls):
    t = rankdata(S[off]); g = rankdata(target[off]); X = np.column_stack([np.ones(off.sum())] + [rankdata(c[off]) for c in controls])
    rt = t - X @ np.linalg.lstsq(X, t, rcond=None)[0]; rg = g - X @ np.linalg.lstsq(X, g, rcond=None)[0]
    if np.linalg.norm(rt) < 1e-9 * max(1.0, np.linalg.norm(t)) or np.linalg.norm(rg) < 1e-9 * max(1.0, np.linalg.norm(g)): return float("nan")
    return float(np.corrcoef(rt, rg)[0, 1])
asym = float(np.abs(D - D.T)[off].mean() / np.abs(D - D.mean())[off].mean())
print(f"directed corpus statistic, window K={K}: total ordered pair count {Cd[off].sum():.0f}; asymmetry index {asym:.2f}")
print(f"  RSA(D, -circle)={spearmanr(D[off], -circ[off]).correlation:+.2f}  RSA(D, -line)={spearmanr(D[off], -line[off]).correlation:+.2f} | partial (ctrl blk+common): circle|line={partial(D, -circ, [blk, com, line]):+.2f}  line|circle={partial(D, -line, [blk, com, circ]):+.2f}")
# interval profile: mean D by directed interval (y - x mod 12)
iv = (x[None] - x[:, None]) % 12
prof = [float(D[iv == k].mean()) for k in range(1, 12)]
print("  mean D by directed interval +1..+11:", np.round(prof, 2), " (fifth up=+7, fourth up=+5, semitone=+1)")
for t in tags:
    P = json.load(open(f"results/predictive/{t}.json"))
    for c in ("modulates_to", "then_key", "next_song"):
        L = np.array(P[c]["L"])
        print(f"  {t:10s} {c:13s}: RSA(L, D) over 132 ordered pairs = {spearmanr(L[off], D[off]).correlation:+.2f}; partial(L, D | blk, common) = {partial(L, D, [blk, com]):+.2f}; partial(L, D | blk, common, circle, line) = {partial(L, D, [blk, com, circ, line]):+.2f}")
