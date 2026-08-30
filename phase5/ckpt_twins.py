"""Enharmonic-twin difference vectors across OLMo-2-1B checkpoints: raw and line-controlled centred cosine between the
model's ΔQ (log q(·|Cb) − log q(·|B), etc.; 13 non-alias targets) and the Wikipedia ΔC (A_win40, D_doc); target-permutation
null (2000); plus stability of ΔQ itself (cosine with the final model's ΔQ). Usage: python -m phase5.ckpt_twins"""
import json, os, sys, numpy as np
sys.path.insert(0, ".")
from phase2.keys15 import KEYS15, S, GLYPH, ENH_PAIRS, n
from scipy.stats import spearmanr
revs = ["stage1-step300-tokens1B", "stage1-step10000-tokens21B", "stage1-step23100-tokens49B", "stage1-step50000-tokens105B", "stage1-step140000-tokens294B", "stage1-step480000-tokens1007B", "stage1-step950000-tokens1993B", "stage1-step1907359-tokens4001B", "stage2-ingredient3-step23852-tokens51B", "stage2-ingredient1-step23852-tokens51B", "stage2-ingredient2-step23852-tokens51B", "main"]
Zc = np.load("results/phase5/cond_wikipedia.npz")
def logC_of(M): R = (M + 0.5) / (M + 0.5).sum(1, keepdims=True); return np.log(R)
C = {ex: logC_of(Zc[ex].astype(float)) for ex in ("A_win40", "D_doc")}
def load(r, fam):
    t = "olmo2_1b" if r == "main" else f"olmo2_1b_{r}"; Jb = json.load(open(f"results/phase2/behavior/{t}.json"))
    L = np.mean([np.array(Jb[k]["total"]) for k in Jb if k.startswith(fam)], 0); return L - np.logaddexp.reduce(L, axis=1, keepdims=True)
rng = np.random.default_rng(0)
def ctrl(dq, dc, keep, a, b, nperm=2000):
    Xd = np.column_stack([np.ones(len(keep)), S[keep], np.abs(S[keep] - S[a]) - np.abs(S[keep] - S[b]), (GLYPH[keep] == -1).astype(float), (GLYPH[keep] == 1).astype(float)])
    rq = dq - Xd @ np.linalg.lstsq(Xd, dq, rcond=None)[0]; rc = dc - Xd @ np.linalg.lstsq(Xd, dc, rcond=None)[0]
    dq0 = dq - dq.mean(); dc0 = dc - dc.mean(); cos = float(dq0 @ dc0 / (np.linalg.norm(dq0) * np.linalg.norm(dc0) + 1e-12)); cosr = float(rq @ rc / (np.linalg.norm(rq) * np.linalg.norm(rc) + 1e-12))
    nul = []
    for _ in range(nperm):
        p = rng.permutation(len(keep)); dcp = dc[p]; rcp = dcp - Xd @ np.linalg.lstsq(Xd, dcp, rcond=None)[0]; nul.append(float(rq @ rcp / (np.linalg.norm(rq) * np.linalg.norm(rcp) + 1e-12)))
    return cos, cosr, float(np.mean(np.array(nul) >= cosr))
out = []
for fam in ("E_modulation", "C_harmonic"):
    Lf = load("main", fam)
    print(f"== {fam}: per checkpoint, per twin pair: |ΔQ| (mean abs, nats) ; cos(ΔQ, ΔQ_final) ; line-controlled cos with Wikipedia ΔC A_win40 (p) / D_doc (p)  [raw cos in brackets]")
    for r in revs:
        f = f"results/phase2/behavior/{'olmo2_1b' if r == 'main' else 'olmo2_1b_' + r}.json"
        if not os.path.exists(f): continue
        L = load(r, fam); cells = []; rec = {"rev": r, "fam": fam, "pairs": {}}
        for a, b in ENH_PAIRS:
            keep = np.array([j for j in range(n) if j not in (a, b)]); dq = L[a, keep] - L[b, keep]; dqf = Lf[a, keep] - Lf[b, keep]
            stab = float(((dq - dq.mean()) @ (dqf - dqf.mean())) / (np.linalg.norm(dq - dq.mean()) * np.linalg.norm(dqf - dqf.mean()) + 1e-12))
            res = {ex: ctrl(dq, C[ex][a, keep] - C[ex][b, keep], keep, a, b) for ex in C}
            rec["pairs"][f"{KEYS15[a]}|{KEYS15[b]}"] = {"mag": float(np.abs(dq).mean()), "stab": stab, **{ex: list(v) for ex, v in res.items()}}
            cells.append(f"{KEYS15[a]}|{KEYS15[b]}: {np.abs(dq).mean():.2f} ; {stab:+.2f} ; A {res['A_win40'][1]:+.2f} ({res['A_win40'][2]:.2f}) D {res['D_doc'][1]:+.2f} ({res['D_doc'][2]:.2f}) [{res['A_win40'][0]:+.2f}/{res['D_doc'][0]:+.2f}]")
        lab = r.replace("stage1-step", "s1 ").replace("stage2-ingredient3-step", "s2i3 ").replace("stage2-ingredient1-step", "s2i1 ").replace("stage2-ingredient2-step", "s2i2 ").replace("-tokens", " ") if r != "main" else "released (other run)"
        print(f"{lab:22s} | " + " || ".join(cells)); out.append(rec)
json.dump(out, open("results/phase5/ckpt_twins.json", "w"), indent=1)
