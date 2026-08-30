import numpy as np, sys
sys.path.insert(0, '.')
from synthetic.phase4 import source_probs, merge_targets, js, metrics, RARE, COMMON, UNIQUE_CTRL, LABELS_OF_CLASS
from synthetic.laws import build, Z, TWIN_PAIRS, N_STATES
from synthetic.codes import CODEWORDS, hamming, assignment
P = build(2.0)["circle"]
for r in (0.5, 0.1, 0.01, 0.001):
    p = source_probs("alias", r)
    cm = np.array([p[LABELS_OF_CLASS[z]].sum() for z in range(12)]); assert np.allclose(cm, 1 / 12), "class mass must be constant"
    for a, b in TWIN_PAIRS: assert np.isclose(p[a] / (p[a] + p[b]), r), "rare alias share must be r"
    for a, b in TWIN_PAIRS: assert np.allclose(P[a], P[b]), "twin oracle rows identical for every r"
pu = source_probs("unique", 0.01); cmu = np.array([pu[LABELS_OF_CLASS[z]].sum() for z in range(12)]); assert np.isclose(cmu[Z[UNIQUE_CTRL]], 0.01 / 12) and np.isclose(pu.sum(), 1)
for a, b in TWIN_PAIRS: assert np.isclose(pu[a], pu[b]), "aliases balanced in the unique control"
lz = merge_targets(np.log(P)); assert np.allclose(np.exp(lz).sum(1), 1)
for a, b in TWIN_PAIRS: assert js(lz[a], lz[b]) < 1e-12 and js(np.log(P[a]), np.log(P[b])) < 1e-12, "oracle twin JS is zero"
m = metrics(np.log(P), P, hamming(CODEWORDS[assignment("aligned")])); assert m["kl_global"] < 1e-9 and m["twin_jsz"] < 1e-12
# a model that mixes up the rare alias (uniform row) has positive latent JS and positive rare KL but zero common KL
lq = np.log(P).copy(); lq[RARE] = np.log(np.full(N_STATES, 1 / N_STATES)); m2 = metrics(lq, P, hamming(CODEWORDS)); assert m2["twin_jsz"] > 0.05 and m2["kl_rare"] > 0.3 and m2["kl_common"] < 1e-9
# merging target aliases: class mass equals the sum of alias columns
assert np.allclose(np.exp(merge_targets(lq))[:, 5], np.exp(lq)[:, 0] + np.exp(lq)[:, 12])
print("ALL PHASE-4 TESTS PASSED")
