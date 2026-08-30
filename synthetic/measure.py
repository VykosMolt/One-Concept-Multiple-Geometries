"""Behavioural and hidden-state measurements against oracle / circle / line / code geometries."""
import numpy as np, torch
from scipy.stats import spearmanr, rankdata
from synthetic.laws import N_STATES, STATES, TWIN_PAIRS
from synthetic.codes import CODEWORDS, SEM_LINE, SEM_CIRC, hamming, iu
from synthetic.data import Q_TOK, SYM0, L
NP = len(iu[0])
R_LINE = rankdata(SEM_LINE[iu]); R_CIRC = rankdata(SEM_CIRC[iu])
pk = np.array([[i, j] for i, j in zip(*iu)]); TWIN_IDX = [np.where((pk[:, 0] == a) & (pk[:, 1] == b))[0][0] for a, b in TWIN_PAIRS]


def partial(dvec, target_r, control_rs):
    t = rankdata(dvec); X = np.column_stack([np.ones(NP)] + list(control_rs)); rt = t - X @ np.linalg.lstsq(X, t, rcond=None)[0]; rg = target_r - X @ np.linalg.lstsq(X, target_r, rcond=None)[0]
    return float(rt @ rg / np.sqrt((rt @ rt) * (rg @ rg))) if rt @ rt > 1e-12 and rg @ rg > 1e-12 else float("nan")


def geometry_stats(D, code_hamming):
    """D: 15x15 distance matrix (symmetric). Returns RSA/partials/ECI with the code Hamming matrix as control."""
    d = D[iu]; rc = rankdata(code_hamming[iu])
    ranks = rankdata(d) / NP
    return {"rsa_circle": float(spearmanr(d, SEM_CIRC[iu]).correlation), "rsa_line": float(spearmanr(d, SEM_LINE[iu]).correlation), "rsa_code": float(spearmanr(d, code_hamming[iu]).correlation),
            "circle_given_line": partial(d, R_CIRC, [R_LINE, rc]), "line_given_circle": partial(d, R_LINE, [R_CIRC, rc]), "code_given_both": partial(d, rc, [R_LINE, R_CIRC]),
            "eci": float(ranks[TWIN_IDX].mean())}


@torch.no_grad()
def behaviour(model, idx, device):
    """q(m|n) over the 15 codewords by teacher forcing; returns log q (15x15, rows = source n, cols = target state m)."""
    model.eval(); S = np.zeros((N_STATES, N_STATES))
    for n in range(N_STATES):
        X = np.zeros((N_STATES, 2 + L), np.int64); X[:, 0] = n; X[:, 1] = Q_TOK; X[:, 2:] = SYM0 + CODEWORDS[idx]   # candidate m = codeword idx[m]
        x = torch.tensor(X, device=device); logits = model(x); lp = torch.log_softmax(logits.float(), -1)
        for m in range(N_STATES):
            S[n, m] = sum(float(lp[m, 1 + p, x[m, 2 + p]]) for p in range(L))
    logq = S - np.logaddexp.reduce(S, axis=1, keepdims=True)
    return logq


def behaviour_stats(logq, P, code_hamming):
    q = np.exp(logq)
    with np.errstate(divide="ignore"):
        kl = float(np.mean([(P[n] * (np.log(P[n]) - logq[n])).sum() for n in range(N_STATES)]))
    off = ~np.eye(N_STATES, dtype=bool)
    rsa_oracle = float(spearmanr(-logq[off], -np.log(P[off])).correlation)
    D = -(logq + logq.T) / 2; np.fill_diagonal(D, 0.0)
    g = geometry_stats(D, code_hamming)
    js = []
    for a, b in TWIN_PAIRS:
        m = (q[a] + q[b]) / 2; js.append(0.5 * (q[a] * (logq[a] - np.log(m))).sum() + 0.5 * (q[b] * (logq[b] - np.log(m))).sum())
    tgt_asym = float(np.mean([abs(logq[n, a] - logq[n, b]) for a, b in TWIN_PAIRS for n in range(N_STATES)]))
    return {"kl": kl, "rsa_oracle": rsa_oracle, "twin_source_js": float(np.mean(js)), "twin_target_asym": tgt_asym, **g}


@torch.no_grad()
def hidden_geometry(model, code_hamming, device):
    """Residual at the <Q> position for the 15 sources, every layer (0 = embeddings)."""
    model.eval(); X = np.zeros((N_STATES, 2), np.int64); X[:, 0] = np.arange(N_STATES); X[:, 1] = Q_TOK
    _, hs = model(torch.tensor(X, device=device), return_hidden=True)
    out = []
    for h in hs:
        H = h[:, 1].float().cpu().numpy(); Hc = H - H.mean(0); D = np.linalg.norm(Hc[:, None] - Hc[None], axis=-1)
        out.append(geometry_stats(D, code_hamming))
    return out


def relabel_null(D_or_logq, P, code_hamming, kind, n=1000, seed=0):
    """Null distribution of the key statistics under free relabeling of the 15 states (applied to the matrix rows/cols;
    the code_hamming control is kept fixed in the analysis frame)."""
    rng = np.random.default_rng(seed); res = []
    for _ in range(n):
        p = rng.permutation(N_STATES)
        if kind == "behaviour":
            lq = D_or_logq[np.ix_(p, p)]; D = -(lq + lq.T) / 2; np.fill_diagonal(D, 0.0); res.append(geometry_stats(D, code_hamming))
        else:
            res.append(geometry_stats(D_or_logq[np.ix_(p, p)], code_hamming))
    return res
