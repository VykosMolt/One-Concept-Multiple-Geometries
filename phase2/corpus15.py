"""15-key corpus PMI from the Phase-I tallies (Karkada window), with Poisson bootstrap; ECI, circle|line, line|circle,
RSA with candidate geometries; nulls (free / glyph-preserving relabeling)."""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase2.keys15 import KEYS15, GLYPH, ENH_PAIRS, candidate_geometries, n
from scipy.stats import spearmanr, rankdata
T = json.load(open("results/corpus_merged/wiki_full.json"))["all"]; N = T["nwords"]; Z = T["Z"]
labs = [f"KEY:{k}:major" for k in KEYS15]
uni = np.array([T["uni"].get(l, 0) for l in labs], float); C = np.array([[T["co"].get(f"{a}|{b}", 0.0) for b in labs] for a in labs])
iu = np.triu_indices(n, 1); NP = len(iu[0])
G = candidate_geometries(logfreq=np.log(uni + 1)); R = {k: rankdata(v[iu]) for k, v in G.items()}
CTRL = ["glyph_class", "edit_distance", "same_letter", "commonness"]
def pmi(c, u):
    with np.errstate(divide="ignore", invalid="ignore"):
        m = np.log((c / Z) / np.outer(u / N, u / N)); m = np.where(np.isfinite(m), m, np.nan)
    return np.where(np.isnan(m), np.nanmin(m) - 1.0, m)
def partial(dvec, target, controls):
    t = rankdata(dvec); X = np.column_stack([np.ones(NP)] + [R[c] for c in controls]); g = R[target]
    rt = t - X @ np.linalg.lstsq(X, t, rcond=None)[0]; rg = g - X @ np.linalg.lstsq(X, g, rcond=None)[0]; return float(rt @ rg / np.sqrt((rt @ rt) * (rg @ rg)))
def stats(M):
    d = -M[iu]  # distance = -PMI
    ranks = rankdata(d) / NP; pk = np.array([[i, j] for i, j in zip(*iu)])
    idx = [np.where((pk[:, 0] == i) & (pk[:, 1] == j))[0][0] for i, j in ENH_PAIRS]
    return {"eci": float(ranks[idx].mean()), "cl": partial(d, "circle_fifths", CTRL + ["line_fifths"]), "lc": partial(d, "line_fifths", CTRL + ["circle_fifths"]),
            "rsa_circle": float(spearmanr(d, G["circle_fifths"][iu]).correlation), "rsa_line": float(spearmanr(d, G["line_fifths"][iu]).correlation),
            "rsa_glyph": float(spearmanr(d, G["glyph_class"][iu]).correlation), "circle_partial": partial(d, "circle_fifths", CTRL), "line_partial": partial(d, "line_fifths", CTRL)}
M = pmi(C, uni); s0 = stats(M)
print("15-key corpus PMI (Wikipedia). min key count", int(uni.min()), "; enharmonic pair weighted counts:", [(KEYS15[i], KEYS15[j], int(C[i, j])) for i, j in ENH_PAIRS])
print("  PMI of enharmonic pairs:", [round(M[i, j], 2) for i, j in ENH_PAIRS], " median off-diag PMI:", round(float(np.median(M[iu])), 2))
print(f"  observed: ECI {s0['eci']:.2f}  circle|line {s0['cl']:+.2f}  line|circle {s0['lc']:+.2f}  RSA circle {s0['rsa_circle']:+.2f} line {s0['rsa_line']:+.2f} glyph {s0['rsa_glyph']:+.2f}  partial circle {s0['circle_partial']:+.2f} line {s0['line_partial']:+.2f}")
rng = np.random.default_rng(0); B = 500; boots = []
for _ in range(B):
    u = rng.poisson(uni); c = rng.poisson(C); c = np.triu(c) + np.triu(c, 1).T; boots.append(stats(pmi(c, u)))
for k in ("eci", "cl", "lc", "circle_partial", "line_partial"):
    v = np.array([b[k] for b in boots]); print(f"  bootstrap {k:15s}: {v.mean():+.2f} ± {v.std():.2f}   (Poisson on f-weighted counts: understates true uncertainty)")
print(f"  P(circle|line > line|circle) = {np.mean([b['cl'] > b['lc'] for b in boots]):.3f}")
# relabeling nulls
flats = np.where(GLYPH == -1)[0]; nats = np.where(GLYPH == 0)[0]; sharps = np.where(GLYPH == 1)[0]
def perm_glyph():
    p = np.arange(n); p[flats] = flats[rng.permutation(len(flats))]; p[nats] = nats[rng.permutation(len(nats))]; p[sharps] = sharps[rng.permutation(len(sharps))]; return p
null_free = [stats(M[np.ix_(p, p)]) for p in (rng.permutation(n) for _ in range(1000))]; null_g = [stats(M[np.ix_(p, p)]) for p in (perm_glyph() for _ in range(1000))]
for k, better in (("eci", "low"), ("cl", "high"), ("lc", "high"), ("circle_partial", "high"), ("line_partial", "high")):
    f = np.array([b[k] for b in null_free]); g = np.array([b[k] for b in null_g])
    pf = np.mean(f <= s0[k]) if better == "low" else np.mean(f >= s0[k]); pg = np.mean(g <= s0[k]) if better == "low" else np.mean(g >= s0[k])
    print(f"  null p for {k:15s}: free {pf:.3f}  glyph-preserving {pg:.3f}")
# excluding the three rare spellings' enharmonic pairs? sensitivity: drop Cb (38 mentions)
keep = [i for i in range(n) if KEYS15[i] != "Cb"]
os.makedirs("results/phase2/corpus", exist_ok=True); json.dump({"M": M.tolist(), "uni": uni.tolist(), "C": C.tolist(), "stats": s0}, open("results/phase2/corpus/pmi15.json", "w"))
