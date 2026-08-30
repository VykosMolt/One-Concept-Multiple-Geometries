"""Interval statistics of consecutive key mentions (<=40 words apart) in the corpus.
Usage: python scripts/adjacency.py <merged.json> [group]"""
import sys, json, numpy as np
from collections import defaultdict
sys.path.insert(0, '.')
from corpus.concepts import SPELLINGS

path = sys.argv[1]; group = sys.argv[2] if len(sys.argv) > 2 else "all"
T = json.load(open(path))[group]
adj = T["adjacent"]
hist = {("major", "major"): np.zeros(12), ("minor", "minor"): np.zeros(12), ("major", "minor"): np.zeros(12), ("minor", "major"): np.zeros(12)}
same_spelling_same_key = 0
for k, v in adj.items():
    a, b = k.split("|")
    _, sa, ma = a.split(":"); _, sb, mb = b.split(":")
    iv = (SPELLINGS[sb] - SPELLINGS[sa]) % 12
    hist[(ma, mb)][iv] += v
names = ["0", "+1", "+2", "+3", "+4", "+5", "+6", "+7", "+8", "+9", "+10", "+11"]
print(f"group={group}: consecutive key-mention pairs (<=40 words). Interval = second - first (semitones mod 12)")
for key, h in hist.items():
    tot = h.sum()
    if tot == 0: continue
    p = h / tot
    print(f"  {key[0]:5s}->{key[1]:5s} n={int(tot):6d}  " + " ".join(f"{n}:{x:.3f}" for n, x in zip(names, p)))
    # generator summary (excluding interval 0)
    nz = h.copy(); nz[0] = 0; nz = nz / nz.sum()
    print(f"      excluding 0: semitone(±1)={nz[1]+nz[11]:.3f}  fifth(±5/7)={nz[5]+nz[7]:.3f}  whole-tone(±2)={nz[2]+nz[10]:.3f} "
          f"minor3(±3)={nz[3]+nz[9]:.3f} major3(±4)={nz[4]+nz[8]:.3f} tritone(6)={nz[6]:.3f}   [uniform=0.182 per pair, 0.091 tritone]")
