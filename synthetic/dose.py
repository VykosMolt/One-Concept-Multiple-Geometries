"""Dose-response (run only if the binary aligned/permuted contrast is non-null): CIRCLE law, permutations with
Spearman(code distance, |n-m|) ≈ 0, .25, .5, .75, 1.0; seeds 0-2. Usage: python -m synthetic.dose"""
import subprocess, json, numpy as np, sys
from synthetic.codes import assignment_with_alignment
targets = [0.0, 0.25, 0.5, 0.75, 1.0]; plan = []
for t in targets:
    for s in range(3):
        if t == 1.0: p, rho = np.arange(15), 1.0
        else: p, rho = assignment_with_alignment(t, seed=100 * s + int(t * 100))
        plan.append({"target": t, "seed": s, "rho": float(rho), "perm": [int(x) for x in p]})
json.dump(plan, open("results/phase3/dose_plan.json", "w"), indent=1)
import os
for item in plan:
    tag = f"dose_r{int(item['target']*100):03d}"
    if os.path.exists(f"results/phase3/runs/{tag}/circle_custom_s{item['seed']}/trajectory.json"): continue
    np.save(f"results/phase3/dose_perm_{tag}_s{item['seed']}.npy", np.array(item["perm"]))
    subprocess.run([sys.executable, "-m", "synthetic.train", "circle", "custom", str(item["seed"]), "6000", tag, f"results/phase3/dose_perm_{tag}_s{item['seed']}.npy"], check=True)
