"""Residual-stream geometry at the PREDICTING position (final token of a context that is about to name the next key),
per layer: Gram RSA vs corpus PMI / fifths / chromatic / block / commonness; partial fifths (full controls);
circle|line and line|circle; P1/P5 raw and black-projected; k=7 isotropy.
Usage: python scripts/predict_position.py <model_path> <tag> [dtype] [device_map]"""
import sys, os, json, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pf.extract import Extractor
from pf.families import PC_CANON_MAJOR
from pf.fourier import center, mode_energies, paired_vector, project_out, mode_isotropy
from scipy.stats import spearmanr, rankdata
model_path, tag = sys.argv[1], sys.argv[2]
dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[sys.argv[3] if len(sys.argv) > 3 else "fp32"]
dm = sys.argv[4] if len(sys.argv) > 4 else None
if dm:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    ex = Extractor.__new__(Extractor); ex.tok = AutoTokenizer.from_pretrained(model_path)
    ex.model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype, device_map=dm).eval(); ex.device = next(ex.model.parameters()).device
else:
    ex = Extractor(model_path, dtype=dtype)
CTX = {"modulates_to": "The piece modulates from {x} major to", "then_key": "The first movement is in {x} major and the second movement is in",
       "related_key": "A key closely related to {x} major is", "next_song": "The first song on the album is in {x} major. The second song is in",
       "ctrl_nonpredicting": "The piece modulates from {x} major very often in the"}
rep = json.load(open("results/corpus/wiki/report.json"))["all"]["major_canon@pmi"]; Cp = np.array(rep["M"]); uni = np.array(rep["uni"])
x = np.arange(12); fp = (7 * x) % 12; signed = np.where(fp <= 6, fp, fp - 12)
fd = np.minimum((fp[:, None] - fp[None]) % 12, (fp[None] - fp[:, None]) % 12).astype(float); line = np.abs(signed[:, None] - signed[None]).astype(float)
cd = np.minimum((x[:, None] - x[None]) % 12, (x[None] - x[:, None]) % 12).astype(float)
BLACK = np.array([0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0], float); blk = (BLACK[:, None] == BLACK[None]).astype(float); lf = np.log(uni); common = lf[:, None] + lf[None]
letters = np.array([ord(n[0]) - 65 for n in PC_CANON_MAJOR]); same_letter = (letters[:, None] == letters[None]).astype(float); alpha = np.abs(letters[:, None] - letters[None]).astype(float)
iu = np.triu_indices(12, 1)
def partial(T, target, controls):
    t = rankdata(T[iu]); g = rankdata(target[iu]); X = np.column_stack([np.ones(66)] + [rankdata(c[iu]) for c in controls])
    rt = t - X @ np.linalg.lstsq(X, t, rcond=None)[0]; rg = g - X @ np.linalg.lstsq(X, g, rcond=None)[0]
    if np.linalg.norm(rt) < 1e-9 * max(1.0, np.linalg.norm(t)) or np.linalg.norm(rg) < 1e-9 * max(1.0, np.linalg.norm(g)): return float("nan")
    return float(np.corrcoef(rt, rg)[0, 1])
ctrl = [blk, common, same_letter, alpha]
out = {}; saved = {}
for cname, tpl in CTX.items():
    H, meta = ex.extract(tpl, PC_CANON_MAJOR, positions=("last", "final"))
    Hs = H["final"]  # (L+1, 12, d)
    saved[cname] = {"last": H["last"].astype(np.float32), "final": H["final"].astype(np.float32)}
    rows = []
    print(f"## {tag} predicting position, context '{tpl}'")
    print("layer | RSA PMI  -fifths  block  common | partial fifths  chrom | circle|line  line|circle | P1 P5 raw | P1 P5 blackproj | iso7")
    for l in range(Hs.shape[0]):
        Hl = Hs[l]; G = center(Hl) @ center(Hl).T
        v = paired_vector(mode_energies(Hl)); p = v / v.sum(); vb = paired_vector(mode_energies(project_out(Hl, BLACK))); pb = vb / vb.sum()
        r = {"layer": l, "rsa_pmi": spearmanr(G[iu], Cp[iu]).correlation, "rsa_fifths": spearmanr(G[iu], -fd[iu]).correlation, "rsa_block": spearmanr(G[iu], blk[iu]).correlation,
             "rsa_common": spearmanr(G[iu], common[iu]).correlation, "partial_fifths": partial(G, -fd, ctrl), "partial_chrom": partial(G, -cd, ctrl),
             "circle_given_line": partial(G, -fd, [blk, common, line]), "line_given_circle": partial(G, -line, [blk, common, fd]),
             "P1": p[0], "P5": p[4], "P1b": pb[0], "P5b": pb[4], "iso7": mode_isotropy(Hl, 7)[2]}
        rows.append({k: float(v) for k, v in r.items()})
        if l % 2 == 0 or l == Hs.shape[0] - 1:
            print(f"{l:5d} | {r['rsa_pmi']:+.2f}  {r['rsa_fifths']:+.2f}  {r['rsa_block']:+.2f}  {r['rsa_common']:+.2f} | {r['partial_fifths']:+.2f}  {r['partial_chrom']:+.2f} | {r['circle_given_line']:+.2f}  {r['line_given_circle']:+.2f} | {r['P1']:.2f} {r['P5']:.2f} | {r['P1b']:.2f} {r['P5b']:.2f} | {r['iso7']:.2f}", flush=True)
    out[cname] = rows
os.makedirs("results/predict_position", exist_ok=True); json.dump(out, open(f"results/predict_position/{tag}.json", "w"))
np.savez_compressed(f"results/predict_position/{tag}_H.npz", **{f"{c}__{p}": v for c, d in saved.items() for p, v in d.items()})
