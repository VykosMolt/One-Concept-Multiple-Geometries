"""Dataset generation: <S_n> <Q> code(m)_1..8. Vocabulary: 15 source tokens (0..14), Q = 15, symbols 16..18, PAD=19."""
import numpy as np
from synthetic.laws import N_STATES
from synthetic.codes import CODEWORDS, L
Q_TOK = 15; SYM0 = 16; VOCAB = 20; SEQ_LEN = 2 + L


def make_dataset(P: np.ndarray, idx: np.ndarray, n_examples: int, seed: int):
    """Seed-paired sampling: the same seed yields the same source sequence and the same uniform variates for m,
    so datasets for different laws/codes differ only through P (inverse-CDF sampling) and idx."""
    rng = np.random.default_rng(seed)
    src = rng.integers(0, N_STATES, n_examples); u = rng.random(n_examples)
    cdf = np.cumsum(P, axis=1); tgt = np.array([np.searchsorted(cdf[s], uu) for s, uu in zip(src, u)]); tgt = np.minimum(tgt, N_STATES - 1)
    X = np.zeros((n_examples, SEQ_LEN), np.int64); X[:, 0] = src; X[:, 1] = Q_TOK; X[:, 2:] = SYM0 + CODEWORDS[idx[tgt]]
    return X, src, tgt
