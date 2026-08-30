import numpy as np, torch, sys
sys.path.insert(0, '.')
from synthetic.model import TinyGPT
from synthetic.data import VOCAB, SEQ_LEN
from synthetic.laws import STATES
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
for ax, (law, code) in zip(axes, [("circle", "aligned"), ("circle", "permuted"), ("line", "aligned"), ("line", "permuted")]):
    m = TinyGPT(VOCAB, SEQ_LEN); m.load_state_dict(torch.load(f"results/phase3/runs/main/{law}_{code}_s0/final.pt", map_location="cpu")); m.eval()
    X = np.zeros((15, 2), np.int64); X[:, 0] = np.arange(15); X[:, 1] = 15
    with torch.no_grad(): _, hs = m(torch.tensor(X), return_hidden=True)
    H = hs[2][:, 1].numpy(); Hc = H - H.mean(0); U, S, Vt = np.linalg.svd(Hc, full_matrices=False); pc = Hc @ Vt[:2].T
    ax.plot(pc[:, 0], pc[:, 1], "-", color="0.7"); sc = ax.scatter(pc[:, 0], pc[:, 1], c=STATES, cmap="coolwarm", s=50)
    for i, n in enumerate(STATES): ax.annotate(f"{n:+d}", (pc[i, 0], pc[i, 1]), fontsize=7)
    ax.set_title(f"{law}×{code} — <Q> hidden layer 2, PC1/PC2 (seed 0)", fontsize=9)
fig.suptitle("Twins (−7,+5), (−6,+6), (−5,+7) coincide under CIRCLE law; LINE×ALIGNED is a horseshoe along n; LINE×PERMUTED is a distorted open curve", fontsize=9)
fig.tight_layout(); fig.savefig("figures/phase3/pca_Q.png", dpi=120); print("ok")
