"""Multi-context averaged extraction + Gram RSA vs corpus PMI.
Usage: python scripts/multicontext.py <model_path> <tag> [major|minor|both] [dtype]"""
import sys, os, json, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pf.extract import Extractor
from pf.families import PC_CANON_MAJOR, PC_CANON_MINOR
from pf.contexts import MAJOR_CONTEXTS, MINOR_CONTEXTS

model_path, tag = sys.argv[1], sys.argv[2]
which = sys.argv[3] if len(sys.argv) > 3 else "both"
dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[sys.argv[4] if len(sys.argv) > 4 else "fp32"]
device_map = sys.argv[5] if len(sys.argv) > 5 else None
outdir = f"results/multictx/{tag}"; os.makedirs(outdir, exist_ok=True)
if device_map:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    ex = Extractor.__new__(Extractor)
    ex.tok = AutoTokenizer.from_pretrained(model_path)
    ex.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, device_map=device_map).eval()
    ex.device = next(ex.model.parameters()).device; ex.n_layers = ex.model.config.num_hidden_layers
else:
    ex = Extractor(model_path, dtype=dtype)
fams = {"major": (PC_CANON_MAJOR, MAJOR_CONTEXTS), "minor": (PC_CANON_MINOR, MINOR_CONTEXTS)}
for fam, (concepts, ctxs) in fams.items():
    if which != "both" and which != fam: continue
    allH = {p: [] for p in ("last", "mean", "anchor")}
    tokcheck = []
    for c in ctxs:
        H, meta = ex.extract(c, concepts, positions=("last", "mean", "anchor"))
        for p in allH: allH[p].append(H[p])
        tokcheck.append([ex.describe_tokens(c, x)["anchor_token"] for x in concepts])
    np.savez(f"{outdir}/{fam}.npz", concepts=np.array(concepts), contexts=np.array(ctxs),
             **{p: np.stack(v, 0) for p, v in allH.items()})  # (n_ctx, L+1, 12, d)
    json.dump(tokcheck, open(f"{outdir}/{fam}_anchors.json", "w"))
    bad = [(i, a) for i, a in enumerate(tokcheck) if len(set(a)) != 1]
    print(fam, "saved; contexts with non-identical anchor tokens:", bad, flush=True)
print("done")
