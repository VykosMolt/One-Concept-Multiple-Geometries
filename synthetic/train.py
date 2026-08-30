"""Train one condition/seed with checkpoint measurements.
Usage: python -m synthetic.train <law: circle|line> <code: aligned|permuted> <seed> [steps] [tag]"""
import sys, os, json, time, numpy as np, torch, torch.nn.functional as F
from synthetic.laws import build, N_STATES
from synthetic.codes import CODEWORDS, assignment, hamming, geometry_report
from synthetic.data import make_dataset, VOCAB, SEQ_LEN
from synthetic.model import TinyGPT
from synthetic.measure import behaviour, behaviour_stats, hidden_geometry, relabel_null
law, code, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]); steps = int(sys.argv[4]) if len(sys.argv) > 4 else 6000; tag = sys.argv[5] if len(sys.argv) > 5 else "main"   # law in {circle, line, circle_rare}
perm_seed = seed
custom_perm = None
if len(sys.argv) > 6:
    if code == "custom": custom_perm = np.load(sys.argv[6])
    else: perm_seed = int(sys.argv[6])
device = "cuda" if torch.cuda.is_available() else "cpu"
O = build(2.0); P = O[law]; idx = custom_perm if custom_perm is not None else assignment(code, perm_seed); H = hamming(CODEWORDS[idx])
Xtr, _, _ = make_dataset(P, idx, 300_000, 1000 + seed); Xva, _, _ = make_dataset(P, idx, 20_000, 2000 + seed)
torch.manual_seed(seed); np.random.seed(seed)
model = TinyGPT(VOCAB, SEQ_LEN).to(device); opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 200) * 0.5 * (1 + np.cos(np.pi * min(s, steps) / steps)))
Xtr_t = torch.tensor(Xtr, device=device); Xva_t = torch.tensor(Xva, device=device); B = 256
outdir = f"results/phase3/runs/{tag}/{law}_{code}_s{seed}"; os.makedirs(outdir, exist_ok=True)
json.dump({"law": law, "code": code, "seed": seed, "perm_seed": perm_seed, "idx": idx.tolist(), "code_geometry": geometry_report(idx), "beta": O["beta"], "tau": O["tau"], "steps": steps}, open(f"{outdir}/config.json", "w"))
def loss_on(X):
    logits = model(X[:, :-1]); return F.cross_entropy(logits[:, 1:].reshape(-1, VOCAB), X[:, 2:].reshape(-1))  # predict code positions only
traj = []; g = torch.Generator(device=device); g.manual_seed(seed); t0 = time.time()
CK = sorted(set([0, 50, 100, 150, 200, 300, 400, 500, 750, 1000, 1500, 2000, 3000, 4000, 5000, steps]))
def measure(step):
    model.eval()
    with torch.no_grad():
        va = float(np.mean([loss_on(Xva_t[i:i + 2000]).item() for i in range(0, len(Xva_t), 2000)]))
    logq = behaviour(model, idx, device); bs = behaviour_stats(logq, P, H); hg = hidden_geometry(model, H, device)
    traj.append({"step": step, "val_loss": va, "behaviour": bs, "hidden": hg, "time": time.time() - t0}); np.save(f"{outdir}/logq_{step}.npy", logq)
    print(f"[{law}/{code}/s{seed}] step {step:5d} val {va:.4f} KL {bs['kl']:.4f} rsa_oracle {bs['rsa_oracle']:+.2f} c|l {bs['circle_given_line']:+.2f} l|c {bs['line_given_circle']:+.2f} twinJS {bs['twin_source_js']:.3f} tgtAsym {bs['twin_target_asym']:.3f} | Q-hidden last layer: c|l {hg[-1]['circle_given_line']:+.2f} l|c {hg[-1]['line_given_circle']:+.2f} code {hg[-1]['rsa_code']:+.2f} eci {hg[-1]['eci']:.2f}", flush=True)
    model.train()
measure(0)
model.train()
for step in range(1, steps + 1):
    bi = torch.randint(0, len(Xtr_t), (B,), device=device, generator=g); loss = loss_on(Xtr_t[bi])
    opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
    if step in CK: measure(step)
# final nulls
logq = behaviour(model, idx, device); D = -(logq + logq.T) / 2; np.fill_diagonal(D, 0.0)
nb = relabel_null(logq, P, H, "behaviour", n=1000, seed=seed)
X = np.zeros((N_STATES, 2), np.int64); X[:, 0] = np.arange(N_STATES); X[:, 1] = 15
with torch.no_grad(): _, hs = model(torch.tensor(X, device=device), return_hidden=True)
nh = {}
for l in (len(hs) // 2, len(hs) - 1):
    Hh = hs[l][:, 1].float().cpu().numpy(); Hc = Hh - Hh.mean(0); Dh = np.linalg.norm(Hc[:, None] - Hc[None], axis=-1); nh[str(l)] = relabel_null(Dh, P, H, "hidden", n=1000, seed=seed)
def summ(null, key): return {"mean": float(np.mean([r[key] for r in null])), "sd": float(np.std([r[key] for r in null])), "q95": float(np.quantile([r[key] for r in null], 0.95)), "q05": float(np.quantile([r[key] for r in null], 0.05))}
json.dump({"trajectory": traj, "null_behaviour": {k: summ(nb, k) for k in ("circle_given_line", "line_given_circle", "code_given_both", "eci")},
           "null_hidden": {l: {k: summ(v, k) for k in ("circle_given_line", "line_given_circle", "code_given_both", "eci")} for l, v in nh.items()}}, open(f"{outdir}/trajectory.json", "w"))
torch.save(model.state_dict(), f"{outdir}/final.pt"); print("done", outdir)
