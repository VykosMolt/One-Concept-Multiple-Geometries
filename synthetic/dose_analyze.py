"""Dose-response analysis: hidden line|circle shift, PC2-quadratic signature, early-training behaviour vs alignment ρ."""
import json, glob, os, numpy as np, torch, sys
sys.path.insert(0, '.')
from synthetic.model import TinyGPT
from synthetic.data import VOCAB, SEQ_LEN
from synthetic.laws import STATES
from synthetic.codes import SEM_LINE, iu
from synthetic.measure import partial, R_LINE, R_CIRC
from scipy.stats import spearmanr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plan = json.load(open("results/phase3/dose_plan.json")); rows = []
for item in plan:
    tag = f"dose_r{int(item['target']*100):03d}"; d = f"results/phase3/runs/{tag}/circle_custom_s{item['seed']}"
    if not os.path.exists(f"{d}/trajectory.json"): continue
    T = json.load(open(f"{d}/trajectory.json"))["trajectory"]; fin = T[-1]; st = {t["step"]: t for t in T}
    m = TinyGPT(VOCAB, SEQ_LEN); m.load_state_dict(torch.load(f"{d}/final.pt", map_location="cpu")); m.eval()
    X = np.zeros((15, 2), np.int64); X[:, 0] = np.arange(15); X[:, 1] = 15
    with torch.no_grad(): _, hs = m(torch.tensor(X), return_hidden=True)
    H = hs[2][:, 1].numpy(); Hc = H - H.mean(0); U, S, Vt = np.linalg.svd(Hc, full_matrices=False); pc = Hc @ Vt[:2].T
    lq100 = np.load(f"{d}/logq_100.npy"); D = -(lq100 + lq100.T) / 2; np.fill_diagonal(D, 0)
    rows.append({"rho": item["rho"], "seed": item["seed"], "hid_mid_lc": fin["hidden"][2]["line_given_circle"], "hid_last_lc": fin["hidden"][-1]["line_given_circle"], "hid_mid_code": fin["hidden"][2]["rsa_code"],
                 "pc2_quad": abs(spearmanr(pc[:, 1], (STATES - STATES.mean()) ** 2).correlation), "pc1_lin": abs(spearmanr(pc[:, 0], STATES).correlation),
                 "kl100": st[100]["behaviour"]["kl"], "beh_lc100": partial(D[iu], R_LINE, [R_CIRC]), "tgt_asym500": st[500]["behaviour"]["twin_target_asym"], "kl_final": fin["behaviour"]["kl"]})
rows.sort(key=lambda r: r["rho"])
print(f"{'rho':>6s} seed | final KL | hidden mid l|c  last l|c  rsa_code | PC1 lin  PC2 quad | KL@100  beh l|c@100 (uncontrolled)  twin asym@500")
for r in rows: print(f"{r['rho']:+.2f} {r['seed']}    | {r['kl_final']:.4f} | {r['hid_mid_lc']:+.2f} {r['hid_last_lc']:+.2f} {r['hid_mid_code']:+.2f} | {r['pc1_lin']:.2f} {r['pc2_quad']:.2f} | {r['kl100']:.3f} {r['beh_lc100']:+.2f} {r['tgt_asym500']:.2f}")
rho = np.array([r["rho"] for r in rows])
for k in ("hid_mid_lc", "hid_last_lc", "pc2_quad", "kl100", "beh_lc100", "tgt_asym500"):
    v = np.array([r[k] for r in rows]); print(f"Spearman(alignment rho, {k}) = {spearmanr(rho, v).correlation:+.2f} (n={len(v)})")
fig, axes = plt.subplots(1, 4, figsize=(16, 3.4))
for ax, k, t in zip(axes, ("hid_mid_lc", "pc2_quad", "kl100", "beh_lc100"), ("<Q> hidden line|circle (mid layer, final)", "PC2 quadratic-in-n |ρ| (mid layer, final)", "KL to oracle at step 100", "behaviour line|circle at step 100")):
    ax.scatter(rho, [r[k] for r in rows]); ax.set_xlabel("code alignment ρ(Hamming, |n−m|)"); ax.set_title(t, fontsize=9)
fig.suptitle("Dose-response, CIRCLE law: output-code alignment vs representational / early-behavioural line", fontsize=9); fig.tight_layout(); fig.savefig("figures/phase3/dose_response.png", dpi=120); json.dump(rows, open("results/phase3/dose_summary.json", "w"), indent=1)
