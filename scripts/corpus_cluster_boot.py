"""Shard-cluster bootstrap for the Phase-I corpus statistics (replaces the Poisson-on-counts bootstrap, which ignores
document clustering). Clusters = the 41 Wikipedia parquet shards (documents are assigned to shards independently of
content, so shards are exchangeable clusters that preserve within-document dependence). Resamples shards with
replacement, re-aggregates unigram/pair tallies, rebuilds the 12-key PMI (canonical and enharmonic-merged families) and
recomputes: partial fifths (controls: black-key block, commonness, letter identity, alphabet distance; as in
scripts/partial_nulls.py), circle|line and line|circle (controls: block, commonness, the other geometry; as in
scripts/corpus_seam.py), and the 15-spelling ECI. Usage: python scripts/corpus_cluster_boot.py [B]"""
import sys, os, json, glob, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pf.families import PC_CANON_MAJOR
from phase2.keys15 import KEYS15, ENH_PAIRS
from scipy.stats import rankdata
B = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
files = sorted(glob.glob("results/corpus_wiki/train-*.json")); shards = [json.load(open(f))["all"] for f in files]; K = len(shards)
ENH = {0: ["C", "B#"], 1: ["C#", "Db"], 2: ["D"], 3: ["D#", "Eb"], 4: ["E", "Fb"], 5: ["F", "E#"], 6: ["F#", "Gb"], 7: ["G"], 8: ["G#", "Ab"], 9: ["A"], 10: ["A#", "Bb"], 11: ["B", "Cb"]}
FAMS = {"canonical": [[f"KEY:{s}:major"] for s in PC_CANON_MAJOR], "merged": [[f"KEY:{s}:major" for s in ENH[i]] for i in range(12)], "spelled15": [[f"KEY:{k}:major"] for k in KEYS15]}
def per_shard(labels):
    U = np.zeros((K, len(labels))); C = np.zeros((K, len(labels), len(labels)))
    for s, sh in enumerate(shards):
        for i, li in enumerate(labels):
            U[s, i] = sum(sh["uni"].get(l, 0) for l in li)
            for j, lj in enumerate(labels): C[s, i, j] = sum(sh["co"].get(f"{a}|{b}", 0.0) for a in li for b in lj)
    return U, C
PS = {f: per_shard(l) for f, l in FAMS.items()}; NW = np.array([sh["nwords"] for sh in shards]); ZZ = np.array([sh["Z"] for sh in shards])
def pmi(U, C, N, Z):
    with np.errstate(divide="ignore", invalid="ignore"):
        m = np.log((C / Z) / np.outer(U / N, U / N)); m = np.where(np.isfinite(m), m, np.nan); return np.where(np.isnan(m), np.nanmin(m) - 1, m)
x = np.arange(12); fp = (7 * x) % 12; signed = np.where(fp <= 6, fp, fp - 12)
circ = np.minimum((fp[:, None] - fp[None]) % 12, (fp[None] - fp[:, None]) % 12).astype(float); line = np.abs(signed[:, None] - signed[None]).astype(float)
BLACK = np.array([0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0]); blk = (BLACK[:, None] == BLACK[None]).astype(float)
letters = np.array([ord(n[0]) - 65 for n in PC_CANON_MAJOR]); sl = (letters[:, None] == letters[None]).astype(float); al = np.abs(letters[:, None] - letters[None]).astype(float)
iu = np.triu_indices(12, 1)
def partial(Sm, target, controls):
    t = rankdata(Sm[iu]); g = rankdata(target[iu]); X = np.column_stack([np.ones(len(t))] + [rankdata(c[iu]) for c in controls])
    rt = t - X @ np.linalg.lstsq(X, t, rcond=None)[0]; rg = g - X @ np.linalg.lstsq(X, g, rcond=None)[0]; return float(np.corrcoef(rt, rg)[0, 1])
iu15 = np.triu_indices(15, 1); pk = np.array([[i, j] for i, j in zip(*iu15)]); eidx = [np.where((pk[:, 0] == i) & (pk[:, 1] == j))[0][0] for i, j in ENH_PAIRS]
def stats(w):
    N = w @ NW; Z = w @ ZZ; out = {}
    for fam in ("canonical", "merged"):
        U, C = PS[fam]; u = w @ U; c = np.tensordot(w, C, 1); M = pmi(u, c, N, Z); com = np.log(u)[:, None] + np.log(u)[None]
        out[f"{fam}_fifths"] = partial(M, -circ, [blk, com, sl, al]); out[f"{fam}_cl"] = partial(M, -circ, [blk, com, line]); out[f"{fam}_lc"] = partial(M, -line, [blk, com, circ])
    U, C = PS["spelled15"]; u = w @ U; c = np.tensordot(w, C, 1); M = pmi(u, c, N, Z); d = -M[iu15]; out["eci15"] = float((rankdata(d) / len(d))[eidx].mean())
    return out
obs = stats(np.ones(K)); rng = np.random.default_rng(0); boots = []
for _ in range(B):
    w = np.bincount(rng.integers(0, K, K), minlength=K).astype(float); boots.append(stats(w))
keys = list(obs); arr = {k: np.array([b[k] for b in boots]) for k in keys}
print(f"Shard-cluster bootstrap ({K} shards resampled with replacement, B = {B}); point estimate, bootstrap SD, 2.5/97.5 percentiles:")
for k in keys: print(f"  {k:18s} {obs[k]:+.3f}  sd {arr[k].std():.3f}  [{np.percentile(arr[k], 2.5):+.3f}, {np.percentile(arr[k], 97.5):+.3f}]")
pw = np.mean(arr["merged_cl"] > arr["merged_lc"]); pc = np.mean(arr["canonical_cl"] > arr["canonical_lc"])
print(f"  P(circle|line > line|circle): merged {pw:.3f}, canonical {pc:.3f}")
json.dump({"obs": obs, "boot_sd": {k: float(arr[k].std()) for k in keys}, "ci": {k: [float(np.percentile(arr[k], 2.5)), float(np.percentile(arr[k], 97.5))] for k in keys}, "B": B, "K": K}, open("results/corpus/wiki/cluster_boot.json", "w"), indent=1)
