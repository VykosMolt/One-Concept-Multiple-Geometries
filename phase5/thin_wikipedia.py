"""Poisson-thin the Wikipedia conditional count matrices to the pair mass of another corpus (per extraction family) so that
cross-corpus comparisons are made at equal sample size. Usage: python -m phase5.thin_wikipedia <target npz> <out npz> [seed]"""
import sys, numpy as np
tgt = np.load(sys.argv[1]); wiki = np.load("results/phase5/cond_wikipedia.npz"); seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
rng = np.random.default_rng(seed); out = {}
for k in wiki.files:
    W = wiki[k].astype(float)
    if W.ndim == 0: out[k] = W; continue
    if k in tgt.files and W.sum() > 0 and tgt[k].sum() < W.sum():
        f = float(tgt[k].sum()) / W.sum(); out[k] = rng.binomial(W.astype(int), f).astype(float); print(f"{k}: wiki {int(W.sum())} → {int(out[k].sum())} (target {int(tgt[k].sum())})")
    else: out[k] = W
np.savez(sys.argv[2], **out)
