"""Ridge-penalty sensitivity of the flagship held-out gain (review round 5, finding M1).
Recomputes the target-aggregated modulation cells (base and rich theory, window and document conditionals, four
models) at lambda in {0.01, 0.1, 1, 10, 100} with the same fold-local pipeline as phase5.fingerprint (no nulls), and
records the eigenvalue range of X'X for the standardized theory+corpus training design in every fold.
Output: results/phase5/ridge_lambda_sensitivity_v4.{txt,json}. Usage: python -m phase5.ridge_lambda_sensitivity"""
import json, sys, numpy as np
sys.path.insert(0, ".")
import phase5.fingerprint as fp
LAMS = (0.01, 0.1, 1.0, 10.0, 100.0)
MODELS = ["olmo2_1b", "gemma2_2b", "qwen25_3b", "olmo2_7b"]
zc, frequencies = fp._load_corpus("results/phase5/cond_wikipedia.npz", ["A_win40", "D_doc"])
eig_log = {}
_orig = fp.ridge_fit_predict
def make_patched(lam, key):
    def patched(Xtr, ytr, Xte, lam_=lam):
        if Xtr.shape[1] == eig_log.get("_p", -1):
            w = np.linalg.eigvalsh(Xtr.T @ Xtr); eig_log.setdefault(key, []).append((float(w.min()), float(w.max())))
        return _orig(Xtr, ytr, Xte, lam=lam_)
    return patched
out = {}
lines = ["lambda-sensitivity of the corrected held-out gain (target-aggregated modulation view; dKL nats/row, then % of theory KL0)",
         "cell               | " + " | ".join(f"lam={l:g}" for l in LAMS) + " | eig(X'X) both-design min..max at lam=1 (over folds)"]
for rich in (False, True):
    for ex in ("A_win40", "D_doc"):
        for m in MODELS:
            q = fp._merge_behavior(fp._load_behavior(m, ["E_modulation"], (0, 1, 2, 3))["E_modulation"]); tok = fp._load_tokens(m)
            row = []; key = f"{m}|E_modulation|{ex}|{'rich' if rich else 'base'}"
            for lam in LAMS:
                cfg = fp.CellConfig("wikipedia_v4", m, "E_modulation", ex, "aggregated", rich, False, (0, 1, 2, 3), 0, 0, 0)
                eig_log["_p"] = -1
                fp.ridge_fit_predict = make_patched(lam, key if lam == 1.0 else "_skip")
                rec, _ = fp.compute_cell(cfg, q, np.asarray(zc[ex], float), frequencies, tok)
                fp.ridge_fit_predict = _orig
                if lam == 1.0:
                    # record eigenvalues of the theory+corpus design: rerun the fits capturing the widest design width
                    p_both = max(len(f) for f in rec["loo_feature_names_by_fold"]) + 1
                    eig_log["_p"] = p_both; eig_log.pop(key, None)
                    fp.ridge_fit_predict = make_patched(lam, key); fp.compute_cell(cfg, q, np.asarray(zc[ex], float), frequencies, tok); fp.ridge_fit_predict = _orig
                row.append({"lam": lam, "dkl": rec["dkl"], "kl0": rec["kl"]["theory"], "pct": 100 * rec["dkl"] / rec["kl"]["theory"], "r2gain": rec["r2gain"]})
            e = eig_log.get(key, []); emin = min(x[0] for x in e) if e else float("nan"); emax = max(x[1] for x in e) if e else float("nan")
            out[key] = {"lams": row, "eig_min": emin, "eig_max": emax, "n_fits_recorded": len(e)}
            lines.append(f"{key:50s} | " + " | ".join(f"{r['dkl']:+.4f} ({r['pct']:+.1f}%)" for r in row) + f" | {emin:.3g}..{emax:.3g} (n={len(e)})")
            print(lines[-1], flush=True)
json.dump(out, open("results/phase5/ridge_lambda_sensitivity_v4.json", "w"), indent=1)
open("results/phase5/ridge_lambda_sensitivity_v4.txt", "w").write("\n".join(lines) + "\n")
