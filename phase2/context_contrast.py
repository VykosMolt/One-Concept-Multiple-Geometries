"""H4 test without layer/family cherry-picking: for each model and position, compute per-template circle|line and
line|circle (and ECI) at EVERY layer, then the family contrast 'relational (B,C,D,E) minus non-relational (A,F)' averaged
over a fixed layer band (middle half of the network, chosen a priori) — and its null by randomly regrouping the 24
templates into 6 pseudo-families of 4 (1000 shuffles). Also reports the contrast per family vs the rest.
Usage: python phase2/context_contrast.py <tag> [spelling]"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase2.keys15 import KEYS15, GLYPH, ENH_PAIRS, candidate_geometries, n
from phase2.contexts import FAMILIES
from pf.fourier import center
from scipy.stats import rankdata
tag = sys.argv[1]; spelling = sys.argv[2] if len(sys.argv) > 2 else "symbol"; POSITIONS = sys.argv[3].split(",") if len(sys.argv) > 3 else ["last", "final"]
Z = np.load(f"results/phase2/hidden/{tag}_{spelling}.npz"); tok = json.load(open(f"results/phase2/hidden/{tag}_{spelling}_tokens.json"))
corpus = json.load(open("results/corpus_merged/wiki_full.json"))["all"]["uni"]; logfreq = np.log(np.array([corpus.get(f"KEY:{k}:major", 0) + 1 for k in KEYS15], float))
G = candidate_geometries(tokcounts=np.array(tok[list(tok)[0]]["n_span"], float), logfreq=logfreq); iu = np.triu_indices(n, 1); NP = len(iu[0]); R = {k: rankdata(v[iu]) for k, v in G.items()}
CTRL = ["glyph_class", "edit_distance", "same_letter", "alphabet", "tokcount", "commonness"]
Xc = {t: np.column_stack([np.ones(NP)] + [R[c] for c in CTRL + [o]]) for t, o in (("circle_fifths", "line_fifths"), ("line_fifths", "circle_fifths"))}
Qc = {t: np.linalg.qr(X)[0] for t, X in Xc.items()}; rt = {t: R[t] - Qc[t] @ (Qc[t].T @ R[t]) for t in Xc}
def partial(dvec, t):
    g = rankdata(dvec); rg = g - Qc[t] @ (Qc[t].T @ g); return float(rg @ rt[t] / np.sqrt((rg @ rg) * (rt[t] @ rt[t]))) if rg @ rg > 1e-12 else np.nan
pk = np.array([[i, j] for i, j in zip(*iu)]); eidx = [np.where((pk[:, 0] == i) & (pk[:, 1] == j))[0][0] for i, j in ENH_PAIRS]
fams = list(FAMILIES); tpl_keys = [(f, ti) for f in fams for ti in range(4)]
out = {}
for pos in POSITIONS:
    Lp1 = Z[f"A_spelling__t0__{pos}"].shape[0]; band = range(Lp1 // 4, 3 * Lp1 // 4 + 1)   # a-priori middle half
    stat = {k: np.full((len(tpl_keys), Lp1), np.nan) for k in ("cl", "lc", "eci")}
    for a, (f, ti) in enumerate(tpl_keys):
        H = Z[f"{f}__t{ti}__{pos}"]
        for l in range(Lp1):
            Hc = center(H[l]); D = np.linalg.norm(Hc[:, None] - Hc[None], axis=-1); d = D[iu]
            if np.allclose(d, 0): continue
            stat["cl"][a, l] = partial(d, "circle_fifths"); stat["lc"][a, l] = partial(d, "line_fifths"); stat["eci"][a, l] = (rankdata(d) / NP)[eidx].mean()
    fam_of = np.array([fams.index(f) for f, _ in tpl_keys]); rel = np.isin(fam_of, [1, 2, 3, 4])
    rng = np.random.default_rng(0); res = {}
    for k in ("cl", "lc", "eci"):
        M = np.nanmean(stat[k][:, list(band)], axis=1)   # per-template band mean
        obs = M[rel].mean() - M[~rel].mean()
        null = np.array([M[p][rel].mean() - M[p][~rel].mean() for p in (rng.permutation(len(M)) for _ in range(2000))])
        pv = float(np.mean(null >= obs)) if k != "eci" else float(np.mean(null <= obs))
        per_fam = {f: float(M[fam_of == i].mean()) for i, f in enumerate(fams)}
        per_fam_p = {}
        for i, f in enumerate(fams):
            o = M[fam_of == i].mean() - M[fam_of != i].mean(); nl = np.array([M[p][fam_of == i].mean() - M[p][fam_of != i].mean() for p in (rng.permutation(len(M)) for _ in range(1000))])
            per_fam_p[f] = float(np.mean(nl >= o)) if k != "eci" else float(np.mean(nl <= o))
        res[k] = {"relational_minus_other": float(obs), "p": pv, "per_family_band_mean": per_fam, "per_family_vs_rest_p": per_fam_p}
        print(f"{tag} [{pos}] {k:3s} band L{min(band)}-{max(band)}: relational(B,C,D,E) − (A,F) = {obs:+.3f} (template-shuffle p = {pv:.3f}) | per family: " + " ".join(f"{f[:1]}={per_fam[f]:+.2f}(p{per_fam_p[f]:.2f})" for f in fams), flush=True)
    out[pos] = res
os.makedirs("results/phase2/contrast", exist_ok=True); json.dump(out, open(f"results/phase2/contrast/{tag}_{spelling}{'_' + '_'.join(POSITIONS) if POSITIONS != ['last', 'final'] else ''}.json", "w"))
