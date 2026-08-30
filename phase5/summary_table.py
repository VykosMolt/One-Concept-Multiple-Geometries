"""Compact central-number table from results/phase5/fingerprint/*.json. Usage: python -m phase5.summary_table <json names...>
Prints, per model × family × extraction: residual r (p), ΔR² (p), ΔKL (p) and theory KL."""
import json, sys
names = sys.argv[1:] or ["wikipedia_v3_neutral"]
for nm in names:
    J = json.load(open(f"results/phase5/fingerprint/{nm}.json"))
    print(f"== {nm}")
    print(f"{'model':10s} {'family':13s} {'extract':8s} | resid r (p)     | ΔR² (p)         | ΔKL (p)  [theory KL] | pairs")
    for k, v in J.items():
        m, fam, ex = k.split("|")
        r2 = v.get("r2gain"); dkl = v.get("dkl")
        s = f"{m:10s} {fam:13s} {ex:8s} | {v['resid_r']:+.2f} ({v['resid_p']:.3f})   |"
        s += f" {r2:+.3f} ({v['r2gain_p']:.3f})  | {dkl:+.4f} ({v['dkl_p']:.3f}) [{v['kl']['theory']:.3f}]" if r2 is not None else " n/a"
        print(s + f" | {v['pairs']}")
