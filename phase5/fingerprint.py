"""Phase V central tests. For each model × context family × corpus × extraction family:
 (1) residual correspondence: fit log Q and log C on nuisance/theory features (ridge, ordered off-diagonal pairs);
     Spearman(C_resid, Q_resid) with a joint key-relabeling null;
 (2) leave-one-source-row-out prediction of Q: theory-only / corpus-only / theory+corpus; scored by within-row Spearman
     (ΔCV), softmax-KL(actual‖predicted) (ΔKL) and within-row R² gain; relabeling null (corpus keys permuted jointly),
     optional Poisson bootstrap over corpus counts and optional document-cluster bootstrap;
 (3) twin difference vectors ΔQ vs ΔC (13 non-alias targets): centered cosine, line-controlled cosine, target-permutation null.
All Monte-Carlo p-values use the finite-sample estimator p = (b + 1) / (B + 1), b = #{null ≥ observed}.
Usage: python -m phase5.fingerprint <corpus npz> <corpus name> [models] [families] [extractions] [flags]
Flags: --neutral (merge target columns into 12 classes) --rich (quadratic/harmonic theory terms) --targetprior (training-row
target fixed effect in the theory model) --templates=0,1,2 (which templates to average) --nperm-res=N --nperm-loo=N
--nboot=N --docboot=<perdoc npz>:<B> --jobs=N --tag=<suffix for output json>"""
import sys, os, json, numpy as np
sys.path.insert(0, ".")
from phase2.keys15 import KEYS15, S, GLYPH, LETTER, ENH_PAIRS, levenshtein, n
from scipy.stats import spearmanr, rankdata
args = [a for a in sys.argv[1:] if not a.startswith("--")]; flags = [a for a in sys.argv[1:] if a.startswith("--")]
def flag(name, default=None):
    for f in flags:
        if f == f"--{name}": return True
        if f.startswith(f"--{name}="): return f.split("=", 1)[1]
    return default
corpus_npz, corpus_name = args[0], args[1]
models = args[2].split(",") if len(args) > 2 else ["olmo2_1b", "gemma2_2b", "qwen25_3b", "olmo2_7b"]
fams = args[3].split(",") if len(args) > 3 else ["C_harmonic", "D_chord", "E_modulation"]
EXTR = args[4].split(",") if len(args) > 4 else ["A_win40", "B_any", "D_doc"]
NEUTRAL = bool(flag("neutral", False)); RICH = bool(flag("rich", False)); TARGETPRIOR = bool(flag("targetprior", False))
TEMPLATES = [int(x) for x in str(flag("templates", "0,1,2,3")).split(",")]
NPERM_RES = int(flag("nperm-res", 5000)); NPERM_LOO = int(flag("nperm-loo", 5000)); NBOOT = int(flag("nboot", 0)); JOBS = int(flag("jobs", 1))
DOCBOOT = flag("docboot", None); TAG = flag("tag", "") or ""
Zc = dict(np.load(corpus_npz)); uni = Zc["uni"].astype(float)   # materialized: a lazy NpzFile cannot be shared by forked workers
pc = (7 * S) % 12
if NEUTRAL:
    off = np.ones((n, 12), dtype=bool)
    for i in range(n): off[i, pc[i]] = False
    I, J = np.where(off); NCOL = 12
    CLASS_S = np.array([S[np.where(pc == z)[0]].mean() for z in range(12)])
else:
    off = ~np.eye(n, dtype=bool); I, J = np.where(off); NCOL = n
def merge_cols(logM):
    out = np.full((n, 12), -np.inf)
    for z in range(12):
        cols = np.where(pc == z)[0]; out[:, z] = np.logaddexp.reduce(logM[:, cols], axis=1)
    return out
def feats(tokcount, uni_):
    if NEUTRAL:
        circ = np.minimum((pc[I] - J) % 12, (J - pc[I]) % 12); line = np.abs(S[I] - CLASS_S[J]); signed = CLASS_S[J] - S[I]
        clsfreq = np.array([np.log(uni_[np.where(pc == z)[0]].sum() + 1) for z in range(12)])
        F = np.column_stack([circ, line, signed, (GLYPH[I] != 0).astype(float), np.log(uni_[I] + 1), clsfreq[J], tokcount[I]])
    else:
        circ = np.minimum((pc[I] - pc[J]) % 12, (pc[J] - pc[I]) % 12); line = np.abs(S[I] - S[J]); signed = S[J] - S[I]
        F = np.column_stack([circ, line, signed, (GLYPH[I] == GLYPH[J]).astype(float), (GLYPH[I] != 0).astype(float), (GLYPH[J] != 0).astype(float), (LETTER[I] == LETTER[J]).astype(float),
                             [levenshtein(KEYS15[i], KEYS15[j]) for i, j in zip(I, J)], np.log(uni_[I] + 1), np.log(uni_[J] + 1), tokcount[I], tokcount[J]])
    if RICH: F = np.column_stack([F, circ ** 2, line ** 2, signed ** 2, circ * line, np.cos(2 * np.pi * circ / 12), np.cos(4 * np.pi * circ / 12), np.cos(6 * np.pi * circ / 12)])
    F = (F - F.mean(0)) / (F.std(0) + 1e-9); return F
def ridge_fit_predict(Xtr, ytr, Xte, lam=1.0):
    Xtr1 = np.column_stack([np.ones(len(Xtr)), Xtr]); Xte1 = np.column_stack([np.ones(len(Xte)), Xte]); A = Xtr1.T @ Xtr1 + lam * np.diag([0] + [1] * Xtr.shape[1])
    b = np.linalg.solve(A, Xtr1.T @ ytr); return Xte1 @ b
def logC_of(M):
    R = (M + 0.5) / (M + 0.5).sum(1, keepdims=True); L = np.log(R); return merge_cols(L) if NEUTRAL else L
def loo(logQ, F, logC):
    """leave-one-source-row-out; mean within-row Spearman for theory / corpus / both, pooled within-row R² gain, softmax-KL."""
    res = {"theory": [], "corpus": [], "both": []}; sse = {"theory": 0.0, "corpus": 0.0, "both": 0.0}; kl = {"theory": [], "corpus": [], "both": []}
    qv = logQ[off]; cv = logC[off]
    for i in range(n):
        te = I == i; tr = ~te; y = qv[te]; py = np.exp(y - np.logaddexp.reduce(y))
        Fx = F
        if TARGETPRIOR:   # training-row target fixed effect: mean log q of each target over the 14 training rows (held-out row never used)
            tp = np.array([qv[tr][J[tr] == j].mean() if np.any(J[tr] == j) else 0.0 for j in range(NCOL)]); tpf = tp[J]
            Fx = np.column_stack([F, (tpf - tpf[tr].mean()) / (tpf[tr].std() + 1e-9)])
        for name, X in (("theory", Fx), ("corpus", cv[:, None]), ("both", np.column_stack([Fx, cv]))):
            pred = ridge_fit_predict(X[tr], qv[tr], X[te]); res[name].append(spearmanr(pred, y).correlation)
            sse[name] += float((((pred - pred.mean()) - (y - y.mean())) ** 2).sum()); pp = pred - np.logaddexp.reduce(pred); kl[name].append(float((py * (np.log(py + 1e-12) - pp)).sum()))
    out = {k: float(np.mean(v)) for k, v in res.items()}
    out["r2gain"] = float(1 - sse["both"] / sse["theory"]); out["kl"] = {k: float(np.mean(v)) for k, v in kl.items()}; out["dkl"] = out["kl"]["theory"] - out["kl"]["both"]
    out["kl_rows"] = {k: [float(x) for x in v] for k, v in kl.items()}
    return out
def perm_keys(M, p): return M[np.ix_(p, p)]
def mc_p(null, obs, ge=True):
    null = np.asarray(null); b = int((null >= obs).sum() if ge else (null <= obs).sum()); return float((b + 1) / (len(null) + 1))
PERDOC = None
if DOCBOOT:
    path, nb = DOCBOOT.split(":"); PERDOC = dict(np.load(path)); NDOCBOOT = int(nb)
def cell(task):
    m, fam, ex, logQ, tok = task
    rng = np.random.default_rng(abs(hash((m, fam, ex, corpus_name))) % (2 ** 32))
    F = feats(tok, uni); M = Zc[ex].astype(float)
    if M.sum() < 100: return f"{m}|{fam}|{ex}", None, f"{m:10s} {fam:13s} {ex:13s} | too sparse ({int(M.sum())} pairs)"
    logC = logC_of(M)
    qr = logQ[off] - ridge_fit_predict(F, logQ[off], F); cr = logC[off] - ridge_fit_predict(F, logC[off], F); r_res = spearmanr(qr, cr).correlation
    null = []
    for _ in range(NPERM_RES):
        p = rng.permutation(n); Cp = logC_of(perm_keys(M, p)); crp = Cp[off] - ridge_fit_predict(F, Cp[off], F); null.append(spearmanr(qr, crp).correlation)
    p_res = mc_p(null, r_res)
    sc = loo(logQ, F, logC); dcv = sc["both"] - sc["theory"]
    nullcv, nullr2, nullkl = [], [], []
    for _ in range(NPERM_LOO):
        p = rng.permutation(n); s2 = loo(logQ, F, logC_of(perm_keys(M, p))); nullcv.append(s2["both"] - s2["theory"]); nullr2.append(s2["r2gain"]); nullkl.append(s2["dkl"])
    p_cv = mc_p(nullcv, dcv); p_r2 = mc_p(nullr2, sc["r2gain"]); p_kl = mc_p(nullkl, sc["dkl"])
    boot = [loo(logQ, F, logC_of(rng.poisson(M)))["dkl"] for _ in range(NBOOT)]
    docboot = None
    if PERDOC is not None and ex in ("A_win40", "D_doc"):
        d_id, ii, jj, cc = PERDOC[f"{ex}_doc"], PERDOC[f"{ex}_i"], PERDOC[f"{ex}_j"], PERDOC[f"{ex}_c"].astype(float); U = PERDOC["uni_docs"].astype(float); ND = U.shape[0]
        vals = []
        for _ in range(NDOCBOOT):
            w = rng.multinomial(ND, np.ones(ND) / ND).astype(float); Mb = np.zeros((n, n)); np.add.at(Mb, (ii, jj), cc * w[d_id]); ub = w @ U
            Fb = feats(tok, ub); vals.append(loo(logQ, Fb, logC_of(Mb))["dkl"])
        vals = np.array(vals); docboot = {"sd": float(vals.std()), "q025": float(np.percentile(vals, 2.5)), "q975": float(np.percentile(vals, 97.5)), "B": NDOCBOOT}
    tw = []
    for a, b in (ENH_PAIRS if not NEUTRAL else []):
        keep = np.array([j for j in range(n) if j not in (a, b)]); dq = logQ[a, keep] - logQ[b, keep]; dc = logC[a, keep] - logC[b, keep]
        Xd = np.column_stack([np.ones(len(keep)), S[keep], np.abs(S[keep] - S[a]) - np.abs(S[keep] - S[b]), (GLYPH[keep] == -1).astype(float), (GLYPH[keep] == 1).astype(float)])
        rq = dq - Xd @ np.linalg.lstsq(Xd, dq, rcond=None)[0]; rc = dc - Xd @ np.linalg.lstsq(Xd, dc, rcond=None)[0]
        dq0 = dq - dq.mean(); dc0 = dc - dc.mean(); cos = float(dq0 @ dc0 / (np.linalg.norm(dq0) * np.linalg.norm(dc0) + 1e-12)); cosr = float(rq @ rc / (np.linalg.norm(rq) * np.linalg.norm(rc) + 1e-12))
        nul = []
        for p in (rng.permutation(len(keep)) for _ in range(NPERM_RES)):
            dcp = dc[p]; rcp = dcp - Xd @ np.linalg.lstsq(Xd, dcp, rcond=None)[0]; nul.append(float(rq @ rcp / (np.linalg.norm(rq) * np.linalg.norm(rcp) + 1e-12)))
        tw.append((cos, cosr, mc_p(nul, cosr), float(spearmanr(dq, dc).correlation)))
    rec = {"resid_r": float(r_res), "resid_p": p_res, "loo": sc, "dcv": float(dcv), "dcv_p": p_cv, "dcv_boot_sd": (float(np.std(boot)) if boot else None), "r2gain": sc["r2gain"], "r2gain_p": p_r2,
           "kl": sc["kl"], "dkl": sc["dkl"], "dkl_p": p_kl, "twins": tw, "pairs": int(M.sum()), "nperm_res": NPERM_RES, "nperm_loo": NPERM_LOO, "docboot": docboot, "targetprior": TARGETPRIOR, "templates": TEMPLATES}
    line = f"{m:10s} {fam:13s} {ex:13s} | {r_res:+.2f} ({p_res:.4f}) | {sc['theory']:+.2f} {sc['corpus']:+.2f} {sc['both']:+.2f}  Δ {dcv:+.3f} ({p_cv:.4f}) ΔR² {sc['r2gain']:+.3f} ({p_r2:.4f}) ΔKL {sc['dkl']:+.4f} ({p_kl:.4f}) [KL th {sc['kl']['theory']:.3f}]"
    if docboot: line += f" docboot sd {docboot['sd']:.4f} [{docboot['q025']:+.4f}, {docboot['q975']:+.4f}]"
    line += " | " + " ".join(f"{c:+.2f}/{cr:+.2f}({p:.3f})" for c, cr, p, _ in tw)
    return f"{m}|{fam}|{ex}", rec, line
if __name__ == "__main__":
    tasks = []
    for m in models:
        src = f"results/phase2/behavior/{m}.json"
        if not os.path.exists(src): print(m, "no behaviour file"); continue
        Jb = json.load(open(src))
        tokf = f"results/phase2/hidden/{m if not m.startswith('olmo') else 'olmo2_1b'}_symbol_tokens.json"
        tok = np.array(json.load(open(tokf))["A_spelling__t0"]["n_span"], float) if os.path.exists(tokf) else np.ones(n)
        for fam in fams:
            Ls = [np.array(Jb[f"{fam}__t{t}"]["total"]) for t in TEMPLATES if f"{fam}__t{t}" in Jb]
            if not Ls: continue
            logQ = np.mean(Ls, 0); logQ = logQ - np.logaddexp.reduce(logQ, axis=1, keepdims=True)
            if NEUTRAL: logQ = merge_cols(logQ)
            for ex in EXTR: tasks.append((m, fam, ex, logQ, tok))
    print(f"corpus={corpus_name}; source counts {uni.astype(int).tolist()}; neutral={NEUTRAL} rich={RICH} targetprior={TARGETPRIOR} templates={TEMPLATES} nperm_res={NPERM_RES} nperm_loo={NPERM_LOO} nboot={NBOOT} docboot={DOCBOOT}")
    print(f"{'model':10s} {'family':13s} {'extract':13s} | resid r (p) | LOO Spearman: theory corpus both  ΔCV (p) ΔR² (p) ΔKL (p) [theory KL] | twins raw/line-controlled (p): Cb|B Gb|F# Db|C#")
    if JOBS > 1:
        import multiprocessing as mp
        with mp.Pool(JOBS) as pool: results = pool.map(cell, tasks)
    else: results = [cell(t) for t in tasks]
    out = {}
    for key, rec, line in results:
        print(line, flush=True)
        if rec is not None: out[key] = rec
    os.makedirs("results/phase5/fingerprint", exist_ok=True)
    name = f"{corpus_name}{'_neutral' if NEUTRAL else ''}{'_rich' if RICH else ''}{'_tp' if TARGETPRIOR else ''}{TAG}"
    json.dump(out, open(f"results/phase5/fingerprint/{name}.json", "w"), indent=1); print("saved", name)
