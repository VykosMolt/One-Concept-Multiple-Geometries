"""Is the released OLMo-2-0425-1B a soup of the three stage-2 ingredient endpoints? Compares main to each ingredient and to
their mean over a sample of tensors: relative L2 distance ||A−B||/||A||. Usage: python -m phase5.soup_check"""
import glob, numpy as np, torch
from safetensors import safe_open
dirs = {"main": "models/OLMo-2-0425-1B", **{f"ing{i}": f"models/olmo2_1b_ckpts/stage2-ingredient{i}-step23852-tokens51B" for i in (1, 2, 3)}}
def loader(d):
    idx = {}
    for f in sorted(glob.glob(d + "/*.safetensors")):
        with safe_open(f, "pt") as h:
            for k in h.keys(): idx[k] = f
    return idx
L = {n: loader(d) for n, d in dirs.items()}; keys = sorted(L["main"])[::9][:25] + ["model.embed_tokens.weight", "lm_head.weight"]
def get(n, k):
    with safe_open(L[n][k], "pt") as h: return h.get_tensor(k).float()
rel = lambda a, b: float((a - b).norm() / a.norm())
acc = {c: [] for c in ("ing1", "ing2", "ing3", "mean3", "ing1-ing2", "ing1-ing3")}
for k in keys:
    m = get("main", k); t = {i: get(i, k) for i in ("ing1", "ing2", "ing3")}; mean3 = (t["ing1"] + t["ing2"] + t["ing3"]) / 3
    for i in ("ing1", "ing2", "ing3"): acc[i].append(rel(m, t[i]))
    acc["mean3"].append(rel(m, mean3)); acc["ing1-ing2"].append(rel(t["ing1"], t["ing2"])); acc["ing1-ing3"].append(rel(t["ing1"], t["ing3"]))
for c, v in acc.items(): print(f"relative distance {c:10s}: median {np.median(v):.4f}  max {np.max(v):.4f}  (n={len(v)} tensors)")
