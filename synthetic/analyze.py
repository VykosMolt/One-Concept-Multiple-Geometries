"""Aggregate Phase-III runs: per condition mean±sd over seeds of final behaviour/hidden statistics; paired ALIGNED−PERMUTED
differences per seed with sign tests; trajectories; figures. Usage: python -m synthetic.analyze [tag]"""
import sys, os, json, glob, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
tag = sys.argv[1] if len(sys.argv) > 1 else "main"
runs = {}
for d in sorted(glob.glob(f"results/phase3/runs/{tag}/*_s*")):
    if not os.path.exists(f"{d}/trajectory.json"): continue
    cfg = json.load(open(f"{d}/config.json")); T = json.load(open(f"{d}/trajectory.json")); runs[(cfg["law"], cfg["code"], cfg["seed"])] = (cfg, T)
laws = ["circle", "line"]; codes = ["aligned", "permuted"]; seeds = sorted({k[2] for k in runs})
BKEYS = ["kl", "rsa_oracle", "circle_given_line", "line_given_circle", "code_given_both", "rsa_code", "twin_source_js", "twin_target_asym", "eci"]
HKEYS = ["circle_given_line", "line_given_circle", "code_given_both", "rsa_code", "eci"]
def final(T): return T["trajectory"][-1]
out = {"final": {}, "paired": {}}
print("=== FINAL CHECKPOINT (mean ± sd over seeds) ===")
print(f"{'condition':18s} | " + " ".join(f"{k[:12]:>12s}" for k in BKEYS) + " | hidden(last layer): c|l l|c code eci | hidden(mid): c|l l|c code eci")
for law in laws:
    for code in codes:
        ks = [k for k in runs if k[0] == law and k[1] == code]
        if not ks: continue
        B = {k: [final(runs[x][1])["behaviour"][k] for x in ks] for k in BKEYS}
        HL = {k: [final(runs[x][1])["hidden"][-1][k] for x in ks] for k in HKEYS}; nl = len(final(runs[ks[0]][1])["hidden"]); HM = {k: [final(runs[x][1])["hidden"][nl // 2][k] for x in ks] for k in HKEYS}
        out["final"][f"{law}_{code}"] = {"n": len(ks), "behaviour": {k: (float(np.mean(v)), float(np.std(v))) for k, v in B.items()}, "hidden_last": {k: (float(np.mean(v)), float(np.std(v))) for k, v in HL.items()}, "hidden_mid": {k: (float(np.mean(v)), float(np.std(v))) for k, v in HM.items()}}
        print(f"{law}×{code:9s} n={len(ks)} | " + " ".join(f"{np.mean(B[k]):+7.3f}±{np.std(B[k]):.3f}" for k in BKEYS) + " | " + " ".join(f"{np.mean(HL[k]):+.2f}" for k in ["circle_given_line", "line_given_circle", "rsa_code", "eci"]) + " | " + " ".join(f"{np.mean(HM[k]):+.2f}" for k in ["circle_given_line", "line_given_circle", "rsa_code", "eci"]))
print("\n=== PAIRED ALIGNED − PERMUTED per seed (final checkpoint) ===")
for law in laws:
    for key, where in [(k, "behaviour") for k in ["line_given_circle", "circle_given_line", "rsa_code", "twin_target_asym", "twin_source_js", "kl", "eci"]] + [(k, "hidden_last") for k in ["line_given_circle", "circle_given_line", "rsa_code", "eci"]] + [(k, "hidden_mid") for k in ["line_given_circle", "circle_given_line", "rsa_code", "eci"]]:
        diffs = []
        for s in seeds:
            a, p = runs.get((law, "aligned", s)), runs.get((law, "permuted", s))
            if a is None or p is None: continue
            fa, fp = final(a[1]), final(p[1])
            if where == "behaviour": va, vp = fa["behaviour"][key], fp["behaviour"][key]
            elif where == "hidden_last": va, vp = fa["hidden"][-1][key], fp["hidden"][-1][key]
            else: nl = len(fa["hidden"]); va, vp = fa["hidden"][nl // 2][key], fp["hidden"][nl // 2][key]
            diffs.append(va - vp)
        if not diffs: continue
        d = np.array(diffs); npos = int((d > 0).sum()); n = len(d)
        from scipy.stats import binomtest, ttest_1samp
        pt = float(ttest_1samp(d, 0).pvalue) if n > 1 and d.std() > 0 else float("nan")
        out["paired"][f"{law}|{where}|{key}"] = {"diffs": d.tolist(), "mean": float(d.mean()), "sign_p": float(binomtest(npos, n, 0.5).pvalue), "t_p": pt}
        print(f"{law:6s} {where:11s} {key:18s}: mean diff {d.mean():+.3f} (sd {d.std():.3f}); {npos}/{n} positive; sign p={binomtest(npos, n, 0.5).pvalue:.3f}; t p={pt:.3f}; per-seed " + " ".join(f"{x:+.3f}" for x in d))
# trajectories figure
fig, axes = plt.subplots(2, 4, figsize=(18, 7))
for c, (key, where, title) in enumerate([("kl", "behaviour", "KL(q‖oracle)"), ("line_given_circle", "behaviour", "behaviour line|circle"), ("twin_target_asym", "behaviour", "twin-target asymmetry |log q(m)-log q(m')|"), ("circle_given_line", "behaviour", "behaviour circle|line")]):
    ax = axes[0][c]
    for law, ls in (("circle", "-"), ("line", "--")):
        for code, col in (("aligned", "C3"), ("permuted", "C0")):
            for s in seeds:
                r = runs.get((law, code, s));
                if r is None: continue
                tr = r[1]["trajectory"]; ax.plot([t["step"] for t in tr], [t[where][key] for t in tr], ls, color=col, alpha=0.5, label=f"{law}×{code}" if s == seeds[0] else None)
    ax.set_title(title, fontsize=9); ax.set_xscale("symlog"); ax.set_xlabel("step")
    if key == "kl": ax.set_yscale("log")
    if c == 0: ax.legend(fontsize=7)
for c, (key, title) in enumerate([("line_given_circle", "Q-hidden (last layer) line|circle"), ("circle_given_line", "Q-hidden (last layer) circle|line"), ("rsa_code", "Q-hidden (last layer) RSA with code Hamming"), ("eci", "Q-hidden (last layer) twin-source collapse ECI")]):
    ax = axes[1][c]
    for law, ls in (("circle", "-"), ("line", "--")):
        for code, col in (("aligned", "C3"), ("permuted", "C0")):
            for s in seeds:
                r = runs.get((law, code, s))
                if r is None: continue
                tr = r[1]["trajectory"]; ax.plot([t["step"] for t in tr], [t["hidden"][-1][key] for t in tr], ls, color=col, alpha=0.5)
    ax.set_title(title, fontsize=9); ax.set_xscale("symlog"); ax.set_xlabel("step")
fig.suptitle("Phase III trajectories: solid = CIRCLE law, dashed = LINE law; red = LINE_ALIGNED code, blue = PERMUTED code; one line per seed", fontsize=10)
fig.tight_layout(); os.makedirs("figures/phase3", exist_ok=True); fig.savefig(f"figures/phase3/trajectories_{tag}.png", dpi=120)
# layerwise hidden geometry at final checkpoint
fig, axes = plt.subplots(1, 4, figsize=(18, 3.6))
for c, key in enumerate(["circle_given_line", "line_given_circle", "rsa_code", "eci"]):
    ax = axes[c]
    for law, ls in (("circle", "-"), ("line", "--")):
        for code, col in (("aligned", "C3"), ("permuted", "C0")):
            vals = np.array([[h[key] for h in final(runs[k][1])["hidden"]] for k in runs if k[0] == law and k[1] == code])
            if len(vals) == 0: continue
            ax.plot(range(vals.shape[1]), vals.mean(0), ls, color=col, marker="o", ms=3, label=f"{law}×{code}"); ax.fill_between(range(vals.shape[1]), vals.mean(0) - vals.std(0), vals.mean(0) + vals.std(0), color=col, alpha=0.1)
    ax.set_title(f"Q-hidden {key} by layer (final ckpt, mean±sd over seeds)", fontsize=8); ax.set_xlabel("layer")
    if c == 0: ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(f"figures/phase3/layers_{tag}.png", dpi=120)
json.dump(out, open(f"results/phase3/summary_{tag}.json", "w"), indent=1); print("figures written")
