"""Fit competing geometries to 15-key hidden-state distance matrices, per (family, position, layer), with nulls.
Statistics per cell:
  - RSA (Spearman over 105 pairs) with each candidate geometry;
  - enharmonic collapse index ECI = mean percentile rank of the 3 enharmonic-pair distances among all 105 (0 = closest;
    null 0.5), and ratio d_enh / mean d(line-adjacent pairs);
  - partial Spearman circle|line and line|circle, both controlling for orthography (glyph_class, edit_distance,
    same_letter, tokcount) and commonness;
  - nested leave-one-key-out CV rank regression: R2_cv for {ortho}, {ortho+circle}, {ortho+line}, {ortho+both};
  - nulls: free relabeling and glyph-class-preserving relabeling (500 each), applied identically across layers so that a
    max-over-layers p-value can be reported per (family, position).
Usage: python phase2/geometry_fit.py <tag> [spelling] [nperm]"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase2.keys15 import KEYS15, S, PC, GLYPH, ENH_PAIRS, candidate_geometries, n
from phase2.contexts import FAMILIES
from pf.fourier import center
from scipy.stats import spearmanr, rankdata
tag = sys.argv[1]; spelling = sys.argv[2] if len(sys.argv) > 2 else "symbol"; NPERM = int(sys.argv[3]) if len(sys.argv) > 3 else 500
POSITIONS = sys.argv[4].split(",") if len(sys.argv) > 4 else ["last", "final"]
Z = np.load(f"results/phase2/hidden/{tag}_{spelling}.npz"); tok = json.load(open(f"results/phase2/hidden/{tag}_{spelling}_tokens.json"))
corpus = json.load(open("results/corpus_merged/wiki_full.json"))["all"]["uni"]
logfreq = np.log(np.array([corpus.get(f"KEY:{k}:major", 0) + 1 for k in KEYS15], float))
tokcounts = np.array(tok[list(tok)[0]]["n_span"], float)
G = candidate_geometries(tokcounts=tokcounts, logfreq=logfreq)
iu = np.triu_indices(n, 1); NP = len(iu[0])
ORTHO = ["glyph_class", "edit_distance", "same_letter", "alphabet", "tokcount"]; CTRL = ORTHO + ["commonness"]
line_adj = [(i, i + 1) for i in range(n - 1)]
from phase2.keys15 import LETTER
ENH_SET = set(ENH_PAIRS)
CTRL_PAIRS = [(i, j) for i, j in zip(*iu) if abs(LETTER[i] - LETTER[j]) == 1 and GLYPH[i] != GLYPH[j] and (i, j) not in ENH_SET]  # letter-adjacent, cross-glyph, non-enharmonic
rng = np.random.default_rng(0)
flats = np.where(GLYPH == -1)[0]; nats = np.where(GLYPH == 0)[0]; sharps = np.where(GLYPH == 1)[0]
def perm_free(): return rng.permutation(n)
def perm_glyph():
    p = np.arange(n); p[flats] = flats[rng.permutation(len(flats))]; p[nats] = nats[rng.permutation(len(nats))]; p[sharps] = sharps[rng.permutation(len(sharps))]; return p
PERMS = {"free": [perm_free() for _ in range(NPERM)], "glyph": [perm_glyph() for _ in range(NPERM)]}
R = {k: rankdata(v[iu]) for k, v in G.items()}
def partial(dvec, target, controls):
    t = rankdata(dvec); X = np.column_stack([np.ones(NP)] + [R[c] for c in controls]); g = R[target]
    rt = t - X @ np.linalg.lstsq(X, t, rcond=None)[0]; rg = g - X @ np.linalg.lstsq(X, g, rcond=None)[0]
    if rt @ rt < 1e-12 or rg @ rg < 1e-12: return float("nan")
    return float(rt @ rg / np.sqrt((rt @ rt) * (rg @ rg)))
pair_keys = np.array([[i, j] for i, j in zip(*iu)])
def cv_r2(dvec, preds):
    """leave-one-key-out CV R^2 of rank regression of pair distances on candidate ranks."""
    t = rankdata(dvec); X = np.column_stack([np.ones(NP)] + [R[c] for c in preds]); sse = 0.0; sst = 0.0
    for k in range(n):
        test = (pair_keys[:, 0] == k) | (pair_keys[:, 1] == k); train = ~test
        b = np.linalg.lstsq(X[train], t[train], rcond=None)[0]; pred = X[test] @ b
        sse += float(((t[test] - pred) ** 2).sum()); sst += float(((t[test] - t[train].mean()) ** 2).sum())
    return 1 - sse / sst
def stats_for(D):
    d = D[iu]; out = {}
    out["rsa"] = {k: float(spearmanr(d, v[iu]).correlation) for k, v in G.items()}
    ranks = rankdata(d) / NP
    idx = [np.where((pair_keys[:, 0] == i) & (pair_keys[:, 1] == j))[0][0] for i, j in ENH_PAIRS]
    out["eci"] = float(np.mean(ranks[idx]))
    cidx = [np.where((pair_keys[:, 0] == i) & (pair_keys[:, 1] == j))[0][0] for i, j in CTRL_PAIRS]
    out["eci_ctrl"] = float(np.mean(ranks[cidx]))          # same statistic for letter-adjacent cross-glyph non-enharmonic pairs
    out["eci_minus_ctrl"] = out["eci"] - out["eci_ctrl"]
    out["enh_over_adjacent"] = float(np.mean(D[[i for i, j in ENH_PAIRS], [j for i, j in ENH_PAIRS]]) / np.mean(D[[i for i, j in line_adj], [j for i, j in line_adj]]))
    out["circle_given_line"] = partial(d, "circle_fifths", CTRL + ["line_fifths"]); out["line_given_circle"] = partial(d, "line_fifths", CTRL + ["circle_fifths"])
    out["circle_partial"] = partial(d, "circle_fifths", CTRL); out["line_partial"] = partial(d, "line_fifths", CTRL)
    out["cv_ortho"] = cv_r2(d, CTRL); out["cv_circle"] = cv_r2(d, CTRL + ["circle_fifths"]); out["cv_line"] = cv_r2(d, CTRL + ["line_fifths"]); out["cv_both"] = cv_r2(d, CTRL + ["circle_fifths", "line_fifths"])
    return out
def dist(H):
    Hc = center(H); return np.linalg.norm(Hc[:, None] - Hc[None], axis=-1)
results = {}
for fam in FAMILIES:
    for pos in POSITIONS:
        Hs = np.stack([Z[f"{fam}__t{ti}__{pos}"] for ti in range(len(FAMILIES[fam]))], 0)  # (T, L+1, 15, d)
        Havg = Hs.mean(0); Lp1 = Havg.shape[0]
        per_layer = []; Ds = []
        for l in range(Lp1):
            D = dist(Havg[l]); Ds.append(D)
            if np.allclose(D, 0): per_layer.append(None); continue
            per_layer.append(stats_for(D))
        # nulls for circle|line, line|circle, ECI: same permutation across layers -> max-over-layers
        keys = ["circle_given_line", "line_given_circle", "circle_partial", "line_partial"]
        nulls = {kind: {k: np.zeros((NPERM, Lp1)) for k in keys + ["eci"]} for kind in PERMS}
        for kind, perms in PERMS.items():
            for pi, p in enumerate(perms):
                for l in range(Lp1):
                    if per_layer[l] is None: nulls[kind]["eci"][pi, l] = 0.5; continue
                    Dp = Ds[l][np.ix_(p, p)]; d = Dp[iu]
                    nulls[kind]["circle_given_line"][pi, l] = partial(d, "circle_fifths", CTRL + ["line_fifths"]); nulls[kind]["line_given_circle"][pi, l] = partial(d, "line_fifths", CTRL + ["circle_fifths"])
                    nulls[kind]["circle_partial"][pi, l] = partial(d, "circle_fifths", CTRL); nulls[kind]["line_partial"][pi, l] = partial(d, "line_fifths", CTRL)
                    ranks = rankdata(d) / NP; idx = [np.where((pair_keys[:, 0] == i) & (pair_keys[:, 1] == j))[0][0] for i, j in ENH_PAIRS]; nulls[kind]["eci"][pi, l] = np.mean(ranks[idx])
        summ = {}
        for k in keys:
            obs = np.array([pl[k] if pl else np.nan for pl in per_layer]); ob = np.nan_to_num(obs, nan=-9)
            b = int(np.argmax(ob)); summ[k] = {"best": float(ob[b]), "best_layer": b, "final": float(obs[-1]),
                                               "p_max_free": float(np.mean(np.nan_to_num(nulls["free"][k], nan=-9).max(1) >= ob[b])),
                                               "p_max_glyph": float(np.mean(np.nan_to_num(nulls["glyph"][k], nan=-9).max(1) >= ob[b])),
                                               "p_final_free": float(np.mean(np.nan_to_num(nulls["free"][k][:, -1], nan=-9) >= obs[-1]))}
        eci = np.array([pl["eci"] if pl else np.nan for pl in per_layer]); e = np.nan_to_num(eci, nan=9); b = int(np.argmin(e))
        summ["eci"] = {"best": float(e[b]), "best_layer": b, "final": float(eci[-1]), "p_min_free": float(np.mean(nulls["free"]["eci"].min(1) <= e[b])),
                       "p_min_glyph": float(np.mean(nulls["glyph"]["eci"].min(1) <= e[b])), "p_final_free": float(np.mean(nulls["free"]["eci"][:, -1] <= eci[-1]))}
        results[f"{fam}__{pos}"] = {"per_layer": per_layer, "summary": summ}
        pl = [x for x in per_layer if x]; bl = summ["circle_given_line"]["best_layer"]; st = per_layer[bl]
        ecb = summ['eci']['best_layer']; ec = per_layer[ecb]['eci_ctrl'] if per_layer[ecb] else float('nan')
        print(f"{tag:10s} {fam:13s} [{pos:5s}] ECI best {summ['eci']['best']:.2f} @L{ecb} (ctrl pairs {ec:.2f}; p_min free {summ['eci']['p_min_free']:.3f} glyph {summ['eci']['p_min_glyph']:.3f}) final {summ['eci']['final']:.2f} | "
              f"circle|line best {summ['circle_given_line']['best']:+.2f} @L{bl} (p_max {summ['circle_given_line']['p_max_free']:.3f}/{summ['circle_given_line']['p_max_glyph']:.3f}) final {summ['circle_given_line']['final']:+.2f} | "
              f"line|circle best {summ['line_given_circle']['best']:+.2f} @L{summ['line_given_circle']['best_layer']} (p_max {summ['line_given_circle']['p_max_free']:.3f}/{summ['line_given_circle']['p_max_glyph']:.3f}) final {summ['line_given_circle']['final']:+.2f} | "
              f"CV R2 @L{bl}: ortho {st['cv_ortho']:.2f} +circle {st['cv_circle']:.2f} +line {st['cv_line']:.2f} +both {st['cv_both']:.2f}", flush=True)
os.makedirs("results/phase2/geometry", exist_ok=True)
json.dump(results, open(f"results/phase2/geometry/{tag}_{spelling}{'_' + '_'.join(POSITIONS) if POSITIONS != ['last', 'final'] else ''}.json", "w"))
