"""Checkpoint trajectory: for each OLMo-2-1B checkpoint's behaviour (families C/D/E), report line|circle, circle|line,
twin ECI / mean twin-target asymmetry, and the Wikipedia fingerprint statistics (residual r, ΔCV, twin-diff cosines) via
phase5.fingerprint machinery. Usage: python -m phase5.ckpt_analysis"""
import json, os, sys, subprocess, numpy as np
sys.path.insert(0, ".")
from phase2.keys15 import KEYS15, candidate_geometries, ENH_PAIRS, n
from scipy.stats import rankdata, spearmanr
revs = ["stage1-step300-tokens1B", "stage1-step10000-tokens21B", "stage1-step23100-tokens49B", "stage1-step50000-tokens105B", "stage1-step140000-tokens294B", "stage1-step480000-tokens1007B", "stage1-step950000-tokens1993B", "stage1-step1907359-tokens4001B", "stage2-ingredient3-step23852-tokens51B"]
corpus = json.load(open("results/corpus_merged/wiki_full.json"))["all"]["uni"]; lf = np.log(np.array([corpus.get(f"KEY:{k}:major", 0) + 1 for k in KEYS15], float))
G = candidate_geometries(logfreq=lf); iu = np.triu_indices(n, 1); NP = len(iu[0]); R = {k: rankdata(v[iu]) for k, v in G.items()}; CTRL = ["glyph_class", "edit_distance", "same_letter", "alphabet", "commonness"]
pk = np.array([[i, j] for i, j in zip(*iu)]); eidx = [np.where((pk[:, 0] == i) & (pk[:, 1] == j))[0][0] for i, j in ENH_PAIRS]
def partial(d, t, c):
    tt = rankdata(d); X = np.column_stack([np.ones(NP)] + [R[x] for x in c]); g = R[t]; rt = tt - X @ np.linalg.lstsq(X, tt, rcond=None)[0]; rg = g - X @ np.linalg.lstsq(X, g, rcond=None)[0]
    return float(rt @ rg / np.sqrt((rt @ rt) * (rg @ rg))) if rt @ rt > 1e-12 else float("nan")
tags = [f"olmo2_1b_{r}" for r in revs] + ["olmo2_1b"]
print(f"{'checkpoint':40s} fam | line|circle circle|line | twin ECI  twin tgt asym | top-1 intervals")
for t in tags:
    f = f"results/phase2/behavior/{t}.json"
    if not os.path.exists(f): continue
    J = json.load(open(f))
    for fam in ("C_harmonic", "D_chord", "E_modulation"):
        Ls = [np.array(J[k]["total"]) for k in J if k.startswith(fam)]
        if not Ls: continue
        L = np.mean(Ls, 0); Sm = -(L + L.T) / 2; d = Sm[iu]; e = (rankdata(d) / NP)[eidx].mean(); asym = np.mean([abs(L[i, a] - L[i, b]) for a, b in ENH_PAIRS for i in range(n)])
        from collections import Counter
        top = np.argmax(np.where(np.eye(n, dtype=bool), -1e9, L), 1); S = np.arange(-7, 8); iv = Counter(int(S[j] - S[i]) for i, j in enumerate(top)).most_common(3)
        print(f"{t[9:]:40s} {fam[:1]}   | {partial(d, 'line_fifths', CTRL + ['circle_fifths']):+.2f}       {partial(d, 'circle_fifths', CTRL + ['line_fifths']):+.2f}      | {e:.2f}      {asym:.2f}         | " + " ".join(f"{k:+d}:{v}" for k, v in iv))
