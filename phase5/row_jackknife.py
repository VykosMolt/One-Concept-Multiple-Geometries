"""Leave-one-source-row-out decomposition of the flagship held-out gain and percentile-vs-pivotal document-cluster
intervals (review round 5, findings B1/M1). Reads the stored per-row KL arrays; no recomputation.
Output: results/phase5/row_jackknife_v4.{txt,json}. Usage: python -m phase5.row_jackknife"""
import json, sys, numpy as np
sys.path.insert(0, ".")
from phase2.keys15 import KEYS15
M = ["olmo2_1b", "gemma2_2b", "qwen25_3b", "olmo2_7b"]; R = "results/phase5/fingerprint/"
out = {}; lines = ["Row jackknife of dKL (target-aggregated modulation view; mean over 15 held-out source rows) and bootstrap intervals",
                   "cell | dKL | largest row (share) | dKL without Cb | min/max over single-row drops | percentile 95% CI | pivotal 95% CI (2*dKL - q)"]
for lab, fn, bfn in (("base", "wikipedia_v4_neutral.json", "wikipedia_v4_neutral_docboot.json"), ("rich", "wikipedia_v4_neutral_rich.json", "wikipedia_v4_neutral_rich_docboot.json"), ("rich+tp", "wikipedia_v4_neutral_rich_tp.json", None)):
    d = json.load(open(R + fn)); b = json.load(open(R + bfn)) if bfn else None
    for ex in ("A_win40", "D_doc"):
        for m in M:
            c = d[f"{m}|E_modulation|{ex}"]; g = np.array(c["loo"]["kl_rows"]["theory"]) - np.array(c["loo"]["kl_rows"]["both"])
            full = float(g.mean()); jk = [(g.sum() - g[i]) / (len(g) - 1) for i in range(len(g))]; imax = int(np.argmax(g)); cb = KEYS15.index("Cb")
            rec = {"dkl": full, "row_dkl": {k: float(v) for k, v in zip(KEYS15, g)}, "largest_row": KEYS15[imax], "largest_share": float(g[imax] / g.sum()) if g.sum() else None,
                   "dkl_without_Cb": float(jk[cb]), "jackknife_min": float(min(jk)), "jackknife_max": float(max(jk))}
            ci = ""
            if b:
                db = b[f"{m}|E_modulation|{ex}"]["docboot"]; lo, hi = db["q025"], db["q975"]; h = b[f"{m}|E_modulation|{ex}"]["dkl"]
                rec.update({"percentile_ci": [lo, hi], "pivotal_ci": [2 * h - hi, 2 * h - lo], "docboot_B": db["B"]})
                ci = f" | [{lo:+.4f},{hi:+.4f}] | [{2*h-hi:+.4f},{2*h-lo:+.4f}]"
            out[f"{lab}|{m}|{ex}"] = rec
            lines.append(f"{lab:7s} {ex:7s} {m:10s} | {full:+.5f} | {KEYS15[imax]} ({100*rec['largest_share']:+.0f}%) | {jk[cb]:+.5f} | {min(jk):+.5f}/{max(jk):+.5f}{ci}")
json.dump(out, open("results/phase5/row_jackknife_v4.json", "w"), indent=1); open("results/phase5/row_jackknife_v4.txt", "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
