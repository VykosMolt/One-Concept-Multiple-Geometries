"""Unit tests for the Phase-III synthetic experiment. Run: .venv/bin/python -m tests.test_synthetic_phase3"""
import numpy as np, torch
from synthetic.laws import build, circle_law, line_law, row_entropy, STATES, Z, MULT, TWIN_PAIRS, N_STATES
from synthetic.codes import CODEWORDS, hamming, assignment, geometry_report, SEM_LINE, iu, L
from synthetic.data import make_dataset, VOCAB, SEQ_LEN, Q_TOK, SYM0
from synthetic.model import TinyGPT
from synthetic.measure import behaviour, behaviour_stats, hidden_geometry
from scipy.stats import spearmanr
O = build(2.0); Pc, Pl = O["circle"], O["line"]
# 1 circle periodicity: rows depend only on z(n); shifting all classes by k is a symmetry
for a, b in TWIN_PAIRS: assert np.allclose(Pc[a], Pc[b]), "twin sources must have identical rows"
Pz = np.array([[Pc[n, Z == z].sum() for z in range(12)] for n in range(N_STATES)])
for n in range(N_STATES): assert np.allclose(Pz[n], np.roll(Pz[7], Z[n] - Z[7])), "class-marginal must be a rotation of the n=0 row"
# 2 duplicate-state equality
for a, b in TWIN_PAIRS: assert np.allclose(Pc[:, a], Pc[:, b]), "twin targets must receive equal mass"
assert np.allclose(Pc.sum(1), 1) and np.allclose(Pl.sum(1), 1)
# no open-line coordinate in CIRCLE: the ratio of mass to the two spellings of a class is exactly 1 for every source
for a, b in TWIN_PAIRS: assert np.allclose(Pc[:, a] / Pc[:, b], 1.0)
# 3 line kernel: monotone in |m-n|, twin sources differ
for n in range(N_STATES):
    d = np.abs(STATES - STATES[n]); order = np.argsort(d); assert np.all(np.diff(Pl[n, order]) <= 1e-12)
assert not np.allclose(Pl[0], Pl[12])
# 4 entropy matching
assert abs(row_entropy(Pc).mean() - row_entropy(Pl).mean()) < 0.02
# 5 equal codeword lengths, fixed alphabet
assert CODEWORDS.shape == (15, L) and CODEWORDS.max() <= 2 and len(set(map(tuple, CODEWORDS))) == 15
# 6 aligned code geometry: Hamming = min(|i-j|, L)
Hm = hamming(CODEWORDS); assert np.array_equal(Hm, np.minimum(np.abs(np.arange(15)[:, None] - np.arange(15)[None]), L))
assert geometry_report(assignment("aligned"))["rho_hamming_line"] > 0.95
# 7 permutation preservation: same multiset of codewords, same symbol counts, lengths; low alignment
for k in range(5):
    p = assignment("permuted", k); assert sorted(p.tolist()) == list(range(15))
    g = geometry_report(p); assert g["symbol_counts"] == geometry_report(assignment("aligned"))["symbol_counts"] and g["lengths"] == [8] and abs(g["rho_hamming_line"]) < 0.15
# 8 candidate-sequence scorer: q sums to 1; equals direct per-token computation for an untrained model
torch.manual_seed(0); m = TinyGPT(VOCAB, SEQ_LEN); idx = assignment("aligned"); logq = behaviour(m, idx, "cpu")
assert np.allclose(np.exp(logq).sum(1), 1)
with torch.no_grad():
    X = np.zeros((1, SEQ_LEN), np.int64); X[0, 0] = 3; X[0, 1] = Q_TOK; X[0, 2:] = SYM0 + CODEWORDS[idx[5]]
    lp = torch.log_softmax(m(torch.tensor(X)).float(), -1); s = sum(float(lp[0, 1 + p, X[0, 2 + p]]) for p in range(L))
    X2 = np.zeros((15, SEQ_LEN), np.int64); X2[:, 0] = 3; X2[:, 1] = Q_TOK; X2[:, 2:] = SYM0 + CODEWORDS[idx]
    lp2 = torch.log_softmax(m(torch.tensor(X2)).float(), -1); allS = np.array([sum(float(lp2[j, 1 + p, X2[j, 2 + p]]) for p in range(L)) for j in range(15)])
assert np.isclose(logq[3, 5], s - np.logaddexp.reduce(allS))
# 9 semantic remapping: behaviour columns index target states m (not codeword indices); permuting the code must not change which
#   state a column refers to — check that dataset targets map through idx into the codeword actually emitted
Xd, src, tgt = make_dataset(Pc, assignment("permuted", 1), 500, 0); assert np.array_equal(Xd[:, 2:] - SYM0, CODEWORDS[assignment("permuted", 1)[tgt]])
# 10 oracle KL: a model that reproduces the oracle exactly has KL 0 and rsa_oracle 1
bs = behaviour_stats(np.log(Pc), Pc, Hm); assert abs(bs["kl"]) < 1e-9 and bs["rsa_oracle"] > 0.999 and bs["twin_source_js"] < 1e-12 and bs["twin_target_asym"] < 1e-12
bl = behaviour_stats(np.log(Pl), Pl, Hm); assert bl["line_given_circle"] > bl["circle_given_line"] and bs["circle_given_line"] > bs["line_given_circle"]
# 11 seed pairing: same seed -> same sources and uniform variates across laws
Xa, sa, _ = make_dataset(Pc, idx, 1000, 7); Xb, sb, _ = make_dataset(Pl, idx, 1000, 7); assert np.array_equal(sa, sb)
print("ALL PHASE-3 TESTS PASSED; oracle geometry: circle law c|l=%.2f l|c=%.2f ; line law c|l=%.2f l|c=%.2f" % (bs["circle_given_line"], bs["line_given_circle"], bl["circle_given_line"], bl["line_given_circle"]))
