"""Extract hidden states for the 15 keys under all context families/templates, key-token ('last') and prompt-final
('final') positions, all layers. Usage: python phase2/extract15.py <model_path> <tag> [dtype] [device_map] [spelling: symbol|word]"""
import sys, os, json, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pf.extract import Extractor
from phase2.keys15 import KEYS15, WORDS
from phase2.contexts import FAMILIES
model_path, tag = sys.argv[1], sys.argv[2]
dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[sys.argv[3] if len(sys.argv) > 3 else "fp32"]
dm = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] != "-" else None
spelling = sys.argv[5] if len(sys.argv) > 5 else "symbol"
names = KEYS15 if spelling == "symbol" else [WORDS[k] for k in KEYS15]
if dm:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    ex = Extractor.__new__(Extractor); ex.tok = AutoTokenizer.from_pretrained(model_path)
    ex.model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype, device_map=dm).eval(); ex.device = next(ex.model.parameters()).device
else:
    ex = Extractor(model_path, dtype=dtype)
out = {}; tokinfo = {}
for fam, tpls in FAMILIES.items():
    for ti, tpl in enumerate(tpls):
        H, meta = ex.extract(tpl, names, positions=("last", "mean", "final"))
        out[f"{fam}__t{ti}__last"] = H["last"].astype(np.float32); out[f"{fam}__t{ti}__final"] = H["final"].astype(np.float32); out[f"{fam}__t{ti}__mean"] = H["mean"].astype(np.float32)
        tokinfo[f"{fam}__t{ti}"] = {"template": tpl, "n_span": [m["n_span"] for m in meta]}
        print(fam, ti, "n_span", [m["n_span"] for m in meta], flush=True)
os.makedirs("results/phase2/hidden", exist_ok=True)
np.savez_compressed(f"results/phase2/hidden/{tag}_{spelling}{'_v2' if '--v2' in sys.argv else ''}.npz", **out)
json.dump(tokinfo, open(f"results/phase2/hidden/{tag}_{spelling}_tokens.json", "w"), indent=1)
print("saved")
