"""Is the 'black-key block' musical or orthographic? Respell four WHITE keys enharmonically (C->B#, E->Fb, F->E#, B->Cb)
so that 'name has an accidental glyph' (9/12) no longer coincides with 'is a black key' (5/12). Extract last-token
residuals over 12 contexts, average, and compare Gram RSA with the black-key block vs the accidental-glyph block per layer.
Usage: python scripts/decouple_orthography.py <model_path> <tag> [dtype] [device_map]"""
import sys, os, json, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pf.extract import Extractor
from pf.families import PC_CANON_MAJOR
from pf.contexts import MAJOR_CONTEXTS
from pf.fourier import center
from scipy.stats import spearmanr
model_path, tag = sys.argv[1], sys.argv[2]
dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[sys.argv[3] if len(sys.argv) > 3 else "fp32"]
dm = sys.argv[4] if len(sys.argv) > 4 else None
if dm:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    ex = Extractor.__new__(Extractor); ex.tok = AutoTokenizer.from_pretrained(model_path)
    ex.model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype, device_map=dm).eval(); ex.device = next(ex.model.parameters()).device
else:
    ex = Extractor(model_path, dtype=dtype)
DECOUPLED = ["B#", "Db", "D", "Eb", "Fb", "E#", "F#", "G", "Ab", "A", "Bb", "Cb"]   # same pitch classes, semitone order
BLACK = np.array([0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0]); blk = (BLACK[:, None] == BLACK[None]).astype(float)
iu = np.triu_indices(12, 1)
CTX = MAJOR_CONTEXTS[:12]
out = {}
for setname, names in (("canonical", PC_CANON_MAJOR), ("decoupled", DECOUPLED)):
    glyph = np.array([1 if len(n) > 1 else 0 for n in names]); gb = (glyph[:, None] == glyph[None]).astype(float)
    Hs = []
    for c in CTX:
        H, meta = ex.extract(c, names, positions=("last",)); Hs.append(H["last"])
    Hs = np.stack(Hs, 0)  # (nctx, L+1, 12, d)
    rows = []
    for l in range(Hs.shape[1]):
        Havg = Hs[:, l].mean(0); G = center(Havg) @ center(Havg).T
        Gn = G / np.sqrt(np.outer(np.diag(G), np.diag(G)))
        r_black = spearmanr(G[iu], blk[iu]).correlation; r_glyph = spearmanr(G[iu], gb[iu]).correlation
        rows.append({"layer": l, "rsa_black": float(r_black), "rsa_glyph": float(r_glyph)})
    out[setname] = {"names": names, "glyph": glyph.tolist(), "rows": rows}
    L = Hs.shape[1]
    print(f"## {tag} {setname} set {names}: glyph indicator {glyph.tolist()}")
    for l in (2, L // 4, L // 2, 3 * L // 4, L - 3, L - 1):
        r = rows[l]; print(f"   L{l:2d}: RSA(Gram, black-key block) = {r['rsa_black']:+.2f}   RSA(Gram, accidental-glyph block) = {r['rsa_glyph']:+.2f}", flush=True)
os.makedirs("results/decouple", exist_ok=True); json.dump(out, open(f"results/decouple/{tag}.json", "w"))
