"""Track 8: enharmonic respelling design. 9 neutral tonics with two standard-ish spellings each (C/B#, Db/C#, Eb/D#, E/Fb,
F/E#, Gb/F#, Ab/G#, Bb/A#, B/Cb) plus D, G, A: 21 labels. Extract key-token and final-token residuals under all six
families. Usage: python phase2/extract_respell.py <model_path> <tag> [dtype] [device_map]"""
import sys, os, json, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pf.extract import Extractor
from phase2.contexts import FAMILIES
LABELS = ["C", "B#", "Db", "C#", "Eb", "D#", "E", "Fb", "F", "E#", "Gb", "F#", "Ab", "G#", "Bb", "A#", "B", "Cb", "D", "G", "A"]
PCS = [0, 0, 1, 1, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 10, 10, 11, 11, 2, 7, 9]
model_path, tag = sys.argv[1], sys.argv[2]
dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[sys.argv[3] if len(sys.argv) > 3 else "fp32"]; dm = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] != "-" else None
if dm:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    ex = Extractor.__new__(Extractor); ex.tok = AutoTokenizer.from_pretrained(model_path); ex.model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype, device_map=dm).eval(); ex.device = next(ex.model.parameters()).device
else: ex = Extractor(model_path, dtype=dtype)
out = {}
for fam, tpls in FAMILIES.items():
    for ti, tpl in enumerate(tpls):
        H, meta = ex.extract(tpl, LABELS, positions=("last", "final")); out[f"{fam}__t{ti}__last"] = H["last"].astype(np.float32); out[f"{fam}__t{ti}__final"] = H["final"].astype(np.float32)
os.makedirs("results/phase2/respell", exist_ok=True); np.savez_compressed(f"results/phase2/respell/{tag}.npz", labels=np.array(LABELS), pcs=np.array(PCS), **out); print("saved")
