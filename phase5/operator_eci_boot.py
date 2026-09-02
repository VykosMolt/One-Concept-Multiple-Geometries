"""Uncertainty and null checks for the conditional-row ECI of §6 (review round 5, finding M2/M4).
(1) Document-cluster bootstrap (9,168 key documents, B resamples) of the ECI of the 40-word directional conditional,
    the symmetrized conditional, the reverse conditional and the same-count PMI, all rebuilt from per-document counts;
(2) a marginals-only null: count matrices drawn from the independence model with the observed row/column totals and
    total mass (no cell structure), scored with the same operators;
(3) B's row subsampled to Cb's event count. Output: results/phase5/operator_eci_boot_v4.txt (+ .json).
Usage: python -m phase5.operator_eci_boot [B]"""
import json, sys, numpy as np
sys.path.insert(0, ".")
from phase2.keys15 import KEYS15, ENH_PAIRS, n
from scipy.stats import rankdata
B = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
rng = np.random.default_rng(20260830)
iu = np.triu_indices(n, 1); NP = len(iu[0]); pk = np.array([[i, j] for i, j in zip(*iu)])
eidx = [np.where((pk[:, 0] == i) & (pk[:, 1] == j))[0][0] for i, j in ENH_PAIRS]
def js(p, q):
    m = (p + q) / 2; f = lambda a, b: np.sum(np.where(a > 0, a * np.log(np.where(a > 0, a, 1) / b), 0)); return 0.5 * f(p, m) + 0.5 * f(q, m)
def jsmat(M):
    rows = (M + 0.5) / (M + 0.5).sum(1, keepdims=True); return np.array([[js(rows[i], rows[j]) for j in range(n)] for i in range(n)])
def eci(D): return float((rankdata(D[iu]) / NP)[eidx].mean())
def operators(N):
    S = N + N.T
    with np.errstate(divide="ignore"):
        tot = S.sum(); pmi = np.log(((S + 0.5) / tot) / np.outer(S.sum(1) / tot, S.sum(0) / tot))
    return {"directional": eci(jsmat(N)), "symmetrized": eci(jsmat(S)), "reverse": eci(jsmat(N.T)), "pmi_same_counts": eci(-pmi)}
Z = np.load("results/phase5/cond_wikipedia_perdoc.npz"); doc, ii, jj, cc = Z["A_win40_doc"], Z["A_win40_i"], Z["A_win40_j"], Z["A_win40_c"]
ndocs = int(Z["uni_docs"].shape[0])
def build(weights):
    N = np.zeros((n, n)); np.add.at(N, (ii, jj), cc * weights[doc]); return N
N0 = build(np.ones(ndocs)); obs = operators(N0)
full = np.load("results/phase5/cond_wikipedia.npz")["A_win40"]; assert np.allclose(N0, full), "per-document counts do not rebuild the full matrix"
boot = {k: [] for k in obs}
for _ in range(B):
    w = np.bincount(rng.integers(0, ndocs, ndocs), minlength=ndocs).astype(float); o = operators(build(w))
    for k in obs: boot[k].append(o[k])
null = {k: [] for k in obs}; T = N0.sum(); r = N0.sum(1) / T; c = N0.sum(0) / T; p = np.outer(r, c).ravel()
for _ in range(B):
    Nn = rng.multinomial(int(round(T)), p / p.sum()).reshape(n, n).astype(float); o = operators(Nn)
    for k in obs: null[k].append(o[k])
ki = {k: i for i, k in enumerate(KEYS15)}; cb, b = ki["Cb"], ki["B"]; ncb = N0[cb].sum()
sub = []
for _ in range(B):
    Ns = N0.copy(); row = N0[b]; Ns[b] = rng.multinomial(int(round(ncb)), row / row.sum()); sub.append(operators(Ns)["directional"])
out = {"B": B, "observed": obs, "docboot_q025_q975": {k: [float(np.quantile(v, .025)), float(np.quantile(v, .975))] for k, v in boot.items()},
       "docboot_sd": {k: float(np.std(v, ddof=1)) for k, v in boot.items()}, "marginals_null_mean_sd": {k: [float(np.mean(v)), float(np.std(v, ddof=1))] for k, v in null.items()},
       "marginals_null_q025_q975": {k: [float(np.quantile(v, .025)), float(np.quantile(v, .975))] for k, v in null.items()},
       "B_row_subsampled_to_Cb_events": {"Cb_events": float(ncb), "B_events": float(N0[b].sum()), "directional_eci_mean_sd": [float(np.mean(sub)), float(np.std(sub, ddof=1))]},
       "row_totals": {k: float(N0[i].sum()) for k, i in ki.items()}, "seed": 20260830}
json.dump(out, open("results/phase5/operator_eci_boot_v4.json", "w"), indent=1)
with open("results/phase5/operator_eci_boot_v4.txt", "w") as f:
    f.write(f"Conditional-row ECI (40-word counts), B={B}, seed 20260830. ECI: mean percentile rank of the three enharmonic pairs (0 closest, 0.5 null).\n")
    for k in obs: f.write(f"{k:16s} observed {obs[k]:.3f}  doc-cluster 95% CI [{out['docboot_q025_q975'][k][0]:.3f}, {out['docboot_q025_q975'][k][1]:.3f}]  marginals-only null mean {out['marginals_null_mean_sd'][k][0]:.3f} sd {out['marginals_null_mean_sd'][k][1]:.3f} (95% [{out['marginals_null_q025_q975'][k][0]:.3f}, {out['marginals_null_q025_q975'][k][1]:.3f}])\n")
    f.write(f"B row subsampled to Cb's {ncb:.0f} events (from {N0[b].sum():.0f}): directional ECI {np.mean(sub):.3f} sd {np.std(sub, ddof=1):.3f}\n")
    f.write("row totals: " + ", ".join(f"{k} {v:.0f}" for k, v in out["row_totals"].items()) + "\n")
print(open("results/phase5/operator_eci_boot_v4.txt").read())
