"""Oracle transition laws over 15 observed states n in {-7..7} with periodic latent identity z(n) = n mod 12."""
import numpy as np
from scipy.optimize import brentq
N_STATES = 15
STATES = np.arange(-7, 8)
Z = STATES % 12
MULT = np.array([int(np.sum(Z == Z[i])) for i in range(N_STATES)])   # 2 for duplicated classes, else 1
TWIN_PAIRS = [(0, 12), (1, 13), (2, 14)]                               # (-7,+5), (-6,+6), (-5,+7)


def circle_law(beta: float, alt_share: float = 0.5) -> np.ndarray:
    """P(m|n) = P_z(z(m)|z(n)) · w(m), with P_z(z'|z) ∝ exp(beta cos(2π(z'−z)/12)) over the 12 classes and, for duplicated
    classes, the class mass split between the primary spelling (indices 12,13,14 = +5,+6,+7) and the alternative spelling
    (0,1,2 = −7,−6,−5) as (1 − alt_share, alt_share); alt_share = 0.5 is the symmetric law, alt_share < 0.5 the data-sparse
    'rare alternative spelling' variant (still periodic at the class level)."""
    P = np.zeros((N_STATES, N_STATES))
    for i in range(N_STATES):
        w = np.exp(beta * np.cos(2 * np.pi * (np.arange(12) - Z[i]) / 12)); pz = w / w.sum()
        for j in range(N_STATES):
            share = 1.0 if MULT[j] == 1 else (alt_share if j < 3 else 1.0 - alt_share)
            P[i, j] = pz[Z[j]] * share
    return P


def line_law(tau: float) -> np.ndarray:
    """P(m|n) ∝ exp(−|m−n|/tau) over the 15 observed labels (open line)."""
    D = np.abs(STATES[:, None] - STATES[None])
    W = np.exp(-D / tau); return W / W.sum(1, keepdims=True)


def row_entropy(P: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore"):
        return -(P * np.where(P > 0, np.log(P), 0)).sum(1)


def match_tau(beta: float, lo=0.2, hi=20.0) -> float:
    """tau such that mean row entropy of LINE equals that of CIRCLE(beta)."""
    target = row_entropy(circle_law(beta)).mean()
    return brentq(lambda t: row_entropy(line_law(t)).mean() - target, lo, hi)


def build(beta: float = 2.0):
    tau = match_tau(beta)
    return {"beta": beta, "tau": tau, "circle": circle_law(beta), "line": line_law(tau), "circle_rare": circle_law(beta, alt_share=0.1)}


if __name__ == "__main__":
    o = build(2.0)
    print("beta", o["beta"], "tau", round(o["tau"], 4))
    for k in ("circle", "line"):
        P = o[k]; print(k, "mean row entropy", round(row_entropy(P).mean(), 4), "row sums ok", np.allclose(P.sum(1), 1))
    P = o["circle"]; print("twin source rows equal:", all(np.allclose(P[a], P[b]) for a, b in TWIN_PAIRS), " twin target mass equal:", all(np.allclose(P[:, a], P[:, b]) for a, b in TWIN_PAIRS))
    print("class-marginal from source n=0:", np.round(np.array([P[7, Z == z].sum() for z in range(12)]), 4))
    print("line twin source rows differ:", not np.allclose(o["line"][0], o["line"][12]))
    np.savez("results/phase3/oracles.npz", **o, states=STATES, z=Z, mult=MULT)
