"""Rare-twin follow-up: excess twin-target asymmetry (model − oracle) and KL, aligned vs permuted, over training."""
import json, numpy as np, sys
sys.path.insert(0, '.')
from synthetic.laws import build, TWIN_PAIRS, N_STATES
P = build(2.0)["circle_rare"]; oracle_asym = float(np.mean([abs(np.log(P[n, a]) - np.log(P[n, b])) for a, b in TWIN_PAIRS for n in range(N_STATES)]))
print(f"oracle twin-target asymmetry (mean |log P(m)-log P(m')|) = {oracle_asym:.3f}")
steps = [100, 200, 500, 1000, 2000, 3000, 6000]
print(f"{'step':>5s} | excess asym aligned  permuted  Δ (per seed) | KL aligned permuted | hidden mid l|c aligned permuted Δ")
for st in steps:
    ea, ep, ka, kp, ha, hp = [], [], [], [], [], []
    for s in range(3):
        for code, E, K, Hh in (("aligned", ea, ka, ha), ("permuted", ep, kp, hp)):
            T = {t["step"]: t for t in json.load(open(f"results/phase3/runs/rare/circle_rare_{code}_s{s}/trajectory.json"))["trajectory"]}
            E.append(T[st]["behaviour"]["twin_target_asym"] - oracle_asym); K.append(T[st]["behaviour"]["kl"]); Hh.append(T[st]["hidden"][2]["line_given_circle"])
    d = np.array(ea) - np.array(ep); dh = np.array(ha) - np.array(hp)
    print(f"{st:5d} | {np.mean(ea):+.3f} {np.mean(ep):+.3f} {d.mean():+.3f} ({' '.join(f'{x:+.2f}' for x in d)}) | {np.mean(ka):.4f} {np.mean(kp):.4f} | {np.mean(ha):+.2f} {np.mean(hp):+.2f} {dh.mean():+.2f} ({int((dh>0).sum())}/3)")
