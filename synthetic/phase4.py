"""Phase IV: sparse source-alias equivalence. Online-sampled CIRCLE process with source-alias rarity r (or an equally rare
unique-state control), aligned/permuted output codes, checkpoint metrics including latent-class twin JS and exposure counts.
Usage: python -m synthetic.phase4 <manip: alias|unique> <r> <code: aligned|permuted> <seed> <steps> <tag>"""
import sys, os, json, time, numpy as np, torch, torch.nn.functional as F
from synthetic.laws import build, N_STATES, STATES, Z, TWIN_PAIRS
from synthetic.codes import CODEWORDS, assignment, hamming
from synthetic.data import VOCAB, SEQ_LEN, Q_TOK, SYM0
from synthetic.model import TinyGPT
from synthetic.measure import behaviour, geometry_stats, partial, R_LINE, R_CIRC
from synthetic.codes import iu, L
RARE = [0, 1, 2]; COMMON = [12, 13, 14]        # rare aliases -7,-6,-5 ; common +5,+6,+7 (twin pairs (0,12),(1,13),(2,14))
UNIQUE_CTRL = 7                                  # state 0 (index 7) used as the equally-rare unique control
CLASS_OF = Z; LABELS_OF_CLASS = {z: [i for i in range(N_STATES) if Z[i] == z] for z in range(12)}


def source_probs(manip, r):
    """Probability of each of the 15 source labels per example. alias: classes uniform (1/12), duplicated classes split
    (1-r, r) between common and rare alias. unique: aliases balanced, class of label 0 gets mass r/12, rest uniform."""
    p = np.zeros(N_STATES)
    if manip == "alias":
        for z in range(12):
            labs = LABELS_OF_CLASS[z]
            if len(labs) == 1: p[labs[0]] = 1 / 12
            else:
                rare = [i for i in labs if i in RARE][0]; com = [i for i in labs if i in COMMON][0]; p[rare] = r / 12; p[com] = (1 - r) / 12
    elif manip == "unique":
        cm = np.full(12, (1 - r / 12) / 11); cm[Z[UNIQUE_CTRL]] = r / 12
        for z in range(12):
            labs = LABELS_OF_CLASS[z]
            for i in labs: p[i] = cm[z] / len(labs)
    else: raise ValueError(manip)
    assert np.isclose(p.sum(), 1); return p


def sample_batch(rng, P, idx, psrc, B):
    src = rng.choice(N_STATES, size=B, p=psrc); u = rng.random(B); cdf = np.cumsum(P, 1)
    tgt = np.minimum(np.array([np.searchsorted(cdf[s], x) for s, x in zip(src, u)]), N_STATES - 1)
    X = np.zeros((B, SEQ_LEN), np.int64); X[:, 0] = src; X[:, 1] = Q_TOK; X[:, 2:] = SYM0 + CODEWORDS[idx[tgt]]
    return X, src


def merge_targets(logq):
    """15-way log q(m|n) -> 12-way log q_z(z'|n) by log-sum-exp over target aliases."""
    out = np.full((N_STATES, 12), -np.inf)
    for z in range(12):
        cols = LABELS_OF_CLASS[z]; out[:, z] = np.logaddexp.reduce(logq[:, cols], axis=1)
    return out


def js(lp, lq):
    p, q = np.exp(lp), np.exp(lq); m = (p + q) / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(0.5 * np.nansum(p * (lp - np.log(m))) + 0.5 * np.nansum(q * (lq - np.log(m))))


def metrics(logq, P, H):
    lz = merge_targets(logq); Pz = merge_targets(np.log(P))
    with np.errstate(divide="ignore"):
        rowkl = np.array([(P[n] * (np.log(P[n]) - logq[n])).sum() for n in range(N_STATES)])
    twin_js15 = [js(logq[a], logq[b]) for a, b in TWIN_PAIRS]; twin_jsz = [js(lz[a], lz[b]) for a, b in TWIN_PAIRS]
    D = -(logq + logq.T) / 2; np.fill_diagonal(D, 0.0); g = geometry_stats(D, H)
    return {"kl_global": float(rowkl.mean()), "kl_common": float(rowkl[COMMON].mean()), "kl_rare": float(rowkl[RARE].mean()), "kl_unique_ctrl": float(rowkl[UNIQUE_CTRL]),
            "kl_others": float(np.delete(rowkl, RARE + COMMON + [UNIQUE_CTRL]).mean()), "twin_js15": float(np.mean(twin_js15)), "twin_jsz": float(np.mean(twin_jsz)), "twin_jsz_each": twin_jsz,
            "beh_circle_given_line": g["circle_given_line"], "beh_line_given_circle": g["line_given_circle"]}


@torch.no_grad()
def hidden_metrics(model, H, device):
    X = np.zeros((N_STATES, 2), np.int64); X[:, 0] = np.arange(N_STATES); X[:, 1] = Q_TOK
    _, hs = model(torch.tensor(X, device=device), return_hidden=True); out = []
    for h in hs:
        Hh = h[:, 1].float().cpu().numpy(); Hc = Hh - Hh.mean(0); D = np.linalg.norm(Hc[:, None] - Hc[None], axis=-1)
        nontwin = np.mean([D[i, j] for i, j in zip(*iu) if (i, j) not in TWIN_PAIRS])
        cos = [float(Hc[a] @ Hc[b] / (np.linalg.norm(Hc[a]) * np.linalg.norm(Hc[b]) + 1e-9)) for a, b in TWIN_PAIRS]
        g = geometry_stats(D, H); out.append({"twin_dist_rel": float(np.mean([D[a, b] for a, b in TWIN_PAIRS]) / nontwin), "twin_cos": float(np.mean(cos)), "eci": g["eci"], "circle_given_line": g["circle_given_line"], "line_given_circle": g["line_given_circle"], "rsa_code": g["rsa_code"],
                    "unique_ctrl_dist_rel": float(np.mean([D[UNIQUE_CTRL, j] for j in range(N_STATES) if j != UNIQUE_CTRL]) / nontwin)})
    return out


if __name__ == "__main__":
    manip, r, code, seed, steps, tag = sys.argv[1], float(sys.argv[2]), sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    O = build(2.0); P = O["circle"]; idx = assignment(code, seed); H = hamming(CODEWORDS[idx]); psrc = source_probs(manip, r)
    rng = np.random.default_rng(1000 + seed); torch.manual_seed(seed)
    model = TinyGPT(VOCAB, SEQ_LEN).to(device); opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 200) * (0.5 * (1 + np.cos(np.pi * min(s, steps) / steps)) * 0.9 + 0.1))  # cosine to 10%, not 0, so long runs keep learning
    outdir = f"results/phase4/runs/{tag}/{manip}_r{r:g}_{code}_s{seed}"; os.makedirs(outdir, exist_ok=True)
    json.dump({"manip": manip, "r": r, "code": code, "seed": seed, "steps": steps, "idx": idx.tolist(), "psrc": psrc.tolist()}, open(f"{outdir}/config.json", "w"))
    B = 256; ck = sorted(set([int(x) for x in np.geomspace(25, steps, 28)] + [0, steps])); traj = []; exposure = 0; exposure_unique = 0; t0 = time.time()
    def loss_on(X):
        logits = model(X[:, :-1]); return F.cross_entropy(logits[:, 1:].reshape(-1, VOCAB), X[:, 2:].reshape(-1))
    def measure(step):
        model.eval(); logq = behaviour(model, idx, device); m = metrics(logq, P, H); hm = hidden_metrics(model, H, device)
        traj.append({"step": step, "exposure_rare": int(exposure), "exposure_unique_ctrl": int(exposure_unique), "time": time.time() - t0, **m, "hidden": hm}); np.save(f"{outdir}/logq_{step}.npy", logq)
        print(f"[{tag} {manip} r={r:g} {code} s{seed}] step {step:6d} expo {exposure:7d} | KL global {m['kl_global']:.4f} common {m['kl_common']:.4f} rare {m['kl_rare']:.4f} uniq {m['kl_unique_ctrl']:.4f} | twin JSz {m['twin_jsz']:.4f} JS15 {m['twin_js15']:.4f} | Q twin dist_rel {hm[-1]['twin_dist_rel']:.2f} cos {hm[-1]['twin_cos']:.2f} eci {hm[-1]['eci']:.2f} l|c {hm[-1]['line_given_circle']:+.2f}", flush=True)
        model.train()
    measure(0); model.train()
    for step in range(1, steps + 1):
        X, src = sample_batch(rng, P, idx, psrc, B); exposure += int(np.isin(src, RARE).sum()); exposure_unique += int((src == UNIQUE_CTRL).sum())
        loss = loss_on(torch.tensor(X, device=device)); opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        if step in ck: measure(step)
    json.dump({"trajectory": traj}, open(f"{outdir}/trajectory.json", "w")); torch.save(model.state_dict(), f"{outdir}/final.pt"); print("done", outdir)
