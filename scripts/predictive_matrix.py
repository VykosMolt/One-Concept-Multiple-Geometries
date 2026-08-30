"""Model predictive key-transition matrix: L[x,y] = log P(" {y} <mode>" | context(x)) for several contexts;
compare with corpus PMI, fifths/chromatic distance, black block, commonness (RSA + partials); and Fourier profile
of the (symmetrized, centered) matrix. Usage: python scripts/predictive_matrix.py <model_path> <tag> [dtype] [device_map]"""
import sys, os, json, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import AutoTokenizer, AutoModelForCausalLM
from pf.families import PC_CANON_MAJOR, PC_CANON_MINOR
from pf.fourier import circulant_projection, kernel_dft, paired_vector, predicted_energy_from_M
from scipy.stats import spearmanr, rankdata
model_path, tag = sys.argv[1], sys.argv[2]
dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[sys.argv[3] if len(sys.argv) > 3 else "fp32"]
dm = sys.argv[4] if len(sys.argv) > 4 else "cuda"
tok = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype, device_map=dm).eval(); dev = next(model.parameters()).device
CTX = {
  "modulates_to": "The piece modulates from {x} major to",
  "then_key": "The first movement is in {x} major and the second movement is in",
  "related_key": "A key closely related to {x} major is",
  "transposed_to": "The song, originally in {x} major, was transposed to",
  "next_song": "The first song on the album is in {x} major. The second song is in",
  "chord_after": "In the key of {x} major, the tonic chord is usually followed by the chord of",
}
rep = json.load(open("results/corpus/wiki/report.json"))["all"]["major_canon@pmi"]; Cpmi = np.array(rep["M"]); uni = np.array(rep["uni"])
x = np.arange(12); fd = np.minimum((7*(x[:,None]-x[None]))%12, (7*(x[None]-x[:,None]))%12); cd = np.minimum((x[:,None]-x[None])%12, (x[None]-x[:,None])%12)
fp_ = (7 * x) % 12; signed_ = np.where(fp_ <= 6, fp_, fp_ - 12); line = np.abs(signed_[:, None] - signed_[None]).astype(float)
BLACK = np.array([0,1,0,1,0,0,1,0,1,0,1,0]); blk = (BLACK[:,None]==BLACK[None]).astype(float); lf = np.log(uni); common = lf[:,None]+lf[None]
letters = np.array([ord(n[0])-65 for n in PC_CANON_MAJOR]); same_letter = (letters[:,None]==letters[None]).astype(float); alpha = np.abs(letters[:,None]-letters[None]).astype(float)
iu = np.triu_indices(12, 1); offd = ~np.eye(12, dtype=bool)
def partial(T, target, controls):
    t = rankdata(T[iu]); g = rankdata(target[iu]); X = np.column_stack([np.ones(66)] + [rankdata(c[iu]) for c in controls])
    rt = t - X @ np.linalg.lstsq(X, t, rcond=None)[0]; rg = g - X @ np.linalg.lstsq(X, g, rcond=None)[0]
    if np.linalg.norm(rt) < 1e-9 * max(1.0, np.linalg.norm(t)) or np.linalg.norm(rg) < 1e-9 * max(1.0, np.linalg.norm(g)): return float("nan")
    return float(np.corrcoef(rt, rg)[0, 1])
def seq_logprob(prefix_ids, cont_ids):
    ids = torch.tensor([prefix_ids + cont_ids], device=dev)
    with torch.no_grad(): lp = torch.log_softmax(model(ids).logits[0].float(), -1)
    return float(sum(lp[len(prefix_ids)-1+i, t] for i, t in enumerate(cont_ids)))
out = {}
print(f"## {tag} predictive matrices (major keys). RSA over 66 off-diagonal pairs of symmetrized L; partial ctrl = block+commonness+letter+alpha")
print(f"{'context':14s} | diag-self | RSA: PMI  -fifths -chrom  block  common | partial: fifths chrom PMI | Fourier profile of sym-centered L (P1..P5,E6) | top-1 target hist (interval)")
for cname, tpl in CTX.items():
    ALT = {"Db": "C#", "Eb": "D#", "F#": "Gb", "Ab": "G#", "Bb": "A#"}
    L = np.zeros((12, 12)); Lm = np.zeros((12, 12))
    for i, xk in enumerate(PC_CANON_MAJOR):
        pre = tok(tpl.replace("{x}", xk))["input_ids"]
        for j, yk in enumerate(PC_CANON_MAJOR):
            L[i, j] = seq_logprob(pre, tok(" " + yk + " major", add_special_tokens=False)["input_ids"])
            Lm[i, j] = np.logaddexp(L[i, j], seq_logprob(pre, tok(" " + ALT[yk] + " major", add_special_tokens=False)["input_ids"])) if yk in ALT else L[i, j]
    L = L - np.logaddexp.reduce(L, axis=1, keepdims=True)   # normalize over the 12 candidates per row
    Lm = Lm - np.logaddexp.reduce(Lm, axis=1, keepdims=True)  # enharmonic-merged targets (both spellings summed)
    S = (L + L.T) / 2
    self_share = float(np.mean(np.exp(np.diag(L))))
    r = {"PMI": spearmanr(S[iu], Cpmi[iu]).correlation, "fifths": spearmanr(S[iu], -fd[iu]).correlation, "chrom": spearmanr(S[iu], -cd[iu]).correlation,
         "block": spearmanr(S[iu], blk[iu]).correlation, "common": spearmanr(S[iu], common[iu]).correlation}
    ctrl = [blk, common, same_letter, alpha]
    pf_ = partial(S, -fd, ctrl); pc_ = partial(S, -cd, ctrl); pp_ = partial(S, Cpmi, ctrl)
    # Fourier profile: treat off-diagonal symmetric matrix as a similarity; energy via circulant projection of centered S
    Sc = S.copy(); np.fill_diagonal(Sc, np.nan); Sc = np.where(np.isnan(Sc), np.nanmean(Sc, axis=1, keepdims=True), Sc); Sc = (Sc + Sc.T)/2
    kap, _, cf = circulant_projection(Sc - Sc.mean()); lam = np.abs(kernel_dft(kap).real); lam[0] = 0; prof = paired_vector(lam)/paired_vector(lam).sum()
    top = np.argmax(L - np.where(np.eye(12, dtype=bool), 1e9, 0), axis=1); hist = np.bincount((top - x) % 12, minlength=12)
    Sm = (Lm + Lm.T) / 2
    print(f"   [enharmonic-merged targets] partial fifths={partial(Sm, -fd, ctrl):+.2f}  circle|line={partial(Sm, -fd, [blk, common, line]):+.2f}  line|circle={partial(Sm, -line, [blk, common, fd]):+.2f}   (canonical: {partial(S, -fd, [blk, common, line]):+.2f} / {partial(S, -line, [blk, common, fd]):+.2f})", flush=True)
    out[cname] = {"L": L.tolist(), "L_merged": Lm.tolist(), "rsa": {k: float(v) for k, v in r.items()}, "partial_fifths": pf_, "partial_chrom": pc_, "partial_pmi": pp_, "profile": prof.tolist(), "self_share": self_share, "top_interval_hist": hist.tolist()}
    print(f"{cname:14s} | {self_share:.2f}      | {r['PMI']:+.2f} {r['fifths']:+.2f} {r['chrom']:+.2f} {r['block']:+.2f} {r['common']:+.2f} | {pf_:+.2f} {pc_:+.2f} {pp_:+.2f} | {np.round(prof,2)} | {hist.tolist()}", flush=True)
os.makedirs("results/predictive", exist_ok=True); json.dump(out, open(f"results/predictive/{tag}.json", "w"))
