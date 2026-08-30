"""Extract hidden states for all families/templates -> results/hidden/<model>/<family>__t<i>.npz"""
import sys, os, json, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pf.extract import Extractor
from pf.families import FAMILIES

model_path = sys.argv[1]; tag = sys.argv[2]
fams = sys.argv[3].split(",") if len(sys.argv) > 3 else list(FAMILIES)
outdir = f"results/hidden/{tag}"; os.makedirs(outdir, exist_ok=True)
ex = Extractor(model_path)
tokinfo = {}
for fam in fams:
    spec = FAMILIES[fam]
    # static embeddings
    E, meta = ex.input_embeddings(spec["concepts"])
    np.savez(f"{outdir}/{fam}__embed.npz", H=E)
    tokinfo[fam + "__embed"] = meta
    for ti, tpl in enumerate(spec["templates"]):
        H, meta = ex.extract(tpl, spec["concepts"])
        np.savez(f"{outdir}/{fam}__t{ti}.npz", template=tpl, concepts=np.array(spec["concepts"]), **H)
        tokinfo[f"{fam}__t{ti}"] = {"template": tpl, "tokens": [ex.describe_tokens(tpl, x) for x in spec["concepts"]]}
        print(fam, ti, repr(tpl), "n_span:", [m["n_span"] for m in meta], flush=True)
json.dump(tokinfo, open(f"{outdir}/tokenization.json", "w"), indent=1)
print("done")
