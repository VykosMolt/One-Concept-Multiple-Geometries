"""Cross-corpus specificity. For each corpus fingerprint JSON (spelled or neutral), tabulate residual r, ΔKL, ΔR² per model ×
family × extraction, then for each model and corpus compare per-row held-out ΔKL (theory − both) against the
size-matched Wikipedia baseline with a paired Wilcoxon over the 15 source rows, and report the difference-in-differences
(OLMo models vs non-OLMo models). Usage: python -m phase5.crosscorpus_compare <suffix: '' or _neutral> corpus1 corpus2 ..."""
import json, sys, numpy as np
from scipy.stats import wilcoxon
suf = sys.argv[1] if sys.argv[1] != "-" else ""; corpora = sys.argv[2:]
models = ["olmo2_1b", "olmo2_7b", "gemma2_2b", "qwen25_3b"]; fams = ["C_harmonic", "D_chord", "E_modulation"]; exs = ["A_win40", "B_any", "D_doc"]
J = {c: json.load(open(f"results/phase5/fingerprint/{c}{suf}.json")) for c in corpora}
base = {c: (f"wikipedia_thin_{c}" if f"wikipedia_thin_{c}" in corpora else "wikipedia_v3") for c in corpora}
print(f"view: {'neutral' if 'neutral' in suf else 'spelled'}; corpora: {corpora}")
print(f"{'model':10s} {'family':13s} {'extract':8s} | " + " | ".join(f"{c[:18]:>24s}" for c in corpora))
print(" " * 35 + "| " + " | ".join(f"{'r     ΔKL(p)   ΔR²':>24s}" for c in corpora))
for m in models:
    for fam in fams:
        for ex in exs:
            cells = []
            for c in corpora:
                v = J[c].get(f"{m}|{fam}|{ex}")
                cells.append(f"{v['resid_r']:+.2f} {v['dkl']:+.4f}({v['dkl_p']:.2f}) {v['r2gain']:+.2f}" if v and "dkl" in v else f"{'n/a':>24s}")
            print(f"{m:10s} {fam:13s} {ex:8s} | " + " | ".join(f"{s:>24s}" for s in cells))
print("\nPaired over the 15 source rows: per-row ΔKL(corpus) − ΔKL(size-matched Wikipedia); mean (Wilcoxon p)")
for c in corpora:
    if c.startswith("wikipedia"): continue
    b = base[c]
    if b not in J: continue
    print(f"== {c} vs {b}")
    did = {}
    for fam in fams:
        for ex in exs:
            row = []
            for m in models:
                v = J[c].get(f"{m}|{fam}|{ex}"); w = J[b].get(f"{m}|{fam}|{ex}")
                if not v or not w or "kl_rows" not in v["loo"] or "kl_rows" not in w["loo"]: row.append(f"{m}: n/a"); continue
                d = (np.array(v["loo"]["kl_rows"]["theory"]) - np.array(v["loo"]["kl_rows"]["both"])) - (np.array(w["loo"]["kl_rows"]["theory"]) - np.array(w["loo"]["kl_rows"]["both"]))
                try: p = wilcoxon(d).pvalue
                except ValueError: p = float("nan")
                did[(fam, ex, m)] = d; row.append(f"{m}: {d.mean():+.4f} ({p:.2f})")
            print(f"  {fam:13s} {ex:8s} | " + "  ".join(row))
    # difference-in-differences: OLMo mean − non-OLMo mean of the per-row differences
    for fam in fams:
        for ex in exs:
            o = [did[(fam, ex, m)] for m in ("olmo2_1b", "olmo2_7b") if (fam, ex, m) in did]; n = [did[(fam, ex, m)] for m in ("gemma2_2b", "qwen25_3b") if (fam, ex, m) in did]
            if o and n:
                dd = np.mean(o, 0) - np.mean(n, 0)
                try: p = wilcoxon(dd).pvalue
                except ValueError: p = float("nan")
                print(f"  DiD {fam:13s} {ex:8s}: OLMo − others = {dd.mean():+.4f} nats/row (Wilcoxon over rows p {p:.2f})")
