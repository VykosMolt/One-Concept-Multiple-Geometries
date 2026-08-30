"""Summary figure: partial fifths (ctrl block+commonness+letter+alphabet) for corpus PMI, theory embedding, and each model
(best layer and final layer), plus few-shot dominant/subdominant/fifth accuracy. Usage: python scripts/fig_summary.py tag1,tag2,..."""
import sys, os, json, re, numpy as np
sys.path.insert(0, '.')
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
tags = sys.argv[1].split(",")
rows = [("corpus PMI\n(12×12)", 0.62, None), ("theory embedding\n(V=3000, d=300)", 0.52, None)]
def parse_partial(path):
    vals = []
    for line in open(path):
        m = re.match(r"\s+(\d+)\s+fifths=([+-][\d.]+)", line)
        if m: vals.append((int(m.group(1)), float(m.group(2))))
    return vals
for t in tags:
    f = f"results/multictx/{t}_decompose_major_last.txt"
    if not os.path.exists(f):
        # olmo/gemma were run interactively; regenerate quickly if missing
        os.system(f".venv/bin/python scripts/decompose_rsa.py {t} results/corpus/wiki/report.json major last 2>&1 | grep -E '^##|^layer|^ +[0-9]+ ' > {f}")
    v = parse_partial(f)
    if not v: continue
    best = max(v, key=lambda x: x[1]); final = v[-1]
    acc = None
    fs = f"results/behavior/{t}_fewshot.json"
    if os.path.exists(fs):
        J = json.load(open(fs)); acc = np.mean([J[k]["acc"] for k in ("dominant(+7)", "subdominant(+5)", "fifth_up(+7)")])
    pred = None
    fp = f"results/predictive/{t}.json"
    if os.path.exists(fp): pred = json.load(open(fp))["modulates_to"]["partial_fifths"]
    rows.append((f"{t}\nbest L{best[0]} / final", best[1], (final[1], acc, pred)))
fig, ax = plt.subplots(figsize=(1.6 * len(rows) + 3, 4))
xs = np.arange(len(rows))
ax.bar(xs, [r[1] for r in rows], color=["C2", "C2"] + ["C0"] * (len(rows) - 2), label="token-geometry partial fifths (best layer for models)")
for i, r in enumerate(rows):
    if r[2] is not None:
        ax.plot([i], [r[2][0]], "kv", label="token geometry, final layer" if i == 2 else None)
        if r[2][2] is not None: ax.plot([i], [r[2][2]], "D", color="C1", ms=8, label="predictive P(next key), 'modulates to'" if i == 2 else None)
        if r[2][1] is not None: ax.text(i, max(r[1], r[2][0], r[2][2] or 0) + 0.03, f"few-shot\nfifth-rel acc {r[2][1]:.2f}", ha="center", fontsize=7)
ax.axhline(0, color="k", lw=0.8); ax.set_xticks(xs); ax.set_xticklabels([r[0] for r in rows], fontsize=8)
ax.set_ylabel("partial Spearman(Gram, −fifths distance)\nctrl: black block, commonness, letter, alphabet"); ax.set_ylim(-0.3, 0.9); ax.legend(fontsize=8)
ax.set_title("Corpus statistic vs theory embedding vs LLM key-name geometry (major keys)")
fig.tight_layout(); fig.savefig("figures/summary/fig4_partial_fifths_summary.png", dpi=130)
print("ok", rows)
