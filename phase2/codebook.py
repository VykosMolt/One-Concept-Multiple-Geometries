"""Track 11: arbitrary output codes. In-context codebook maps each of the 15 spelled keys to an arbitrary single-token
label (counterbalanced random codebooks). Step 1: identity check ('The code for X major is' -> code). Step 2: relation
questions answered in codes; 15x15 matrices scored over the 15 code tokens. If the line persists under codes, it is not a
surface-spelling effect; if enharmonic twins' codes are treated as equivalent, the model uses neutral pitch identity.
Usage: python phase2/codebook.py <model_path> <tag> [dtype] [device_map] [n_codebooks]"""
import sys, os, json, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import AutoTokenizer, AutoModelForCausalLM
from phase2.keys15 import KEYS15, WORDS, n
model_path, tag = sys.argv[1], sys.argv[2]
dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[sys.argv[3] if len(sys.argv) > 3 else "fp32"]
dm = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] != "-" else "cuda"; NCB = int(sys.argv[5]) if len(sys.argv) > 5 else 3
tok = AutoTokenizer.from_pretrained(model_path); model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype, device_map=dm).eval(); dev = next(model.parameters()).device
POOL = ["apple", "river", "stone", "cloud", "tiger", "candle", "forest", "silver", "window", "garden", "bridge", "planet", "pepper", "castle", "velvet", "marble", "falcon", "harbor", "meadow", "copper", "lantern", "orchid", "saddle", "thunder"]
POOL = [w for w in POOL if len(tok(" " + w, add_special_tokens=False)["input_ids"]) == 1]
assert len(POOL) >= 15, POOL
rng = np.random.default_rng(0)
TASKS = {"identity": "The code for {x} major is", "dominant": "The dominant key of {x} major, written as a code, is", "modulation": "The piece modulates from {x} major (as a code) to the code",
         "chord": "In the key of {x} major, the tonic chord is usually followed by the chord whose code is", "enharmonic": "The enharmonic equivalent of {x} major, written as a code, is"}
@torch.no_grad()
def logprobs(prefix_ids, cands, chunk=5):
    out = []
    for s0 in range(0, len(cands), chunk):
        cc = cands[s0:s0 + chunk]; seqs = [prefix_ids + c for c in cc]; L = max(len(s) for s in seqs); pad = tok.pad_token_id or 0
        ids = torch.tensor([s + [pad] * (L - len(s)) for s in seqs], device=dev); logits = model(ids).logits
        lp = torch.log_softmax(logits[:, len(prefix_ids) - 1:, :].float(), -1)
        out += [sum(float(lp[b, i, t]) for i, t in enumerate(c)) for b, c in enumerate(cc)]
        del logits, lp
    return np.array(out)
out = {}
for cb in range(NCB):
    codes = list(rng.permutation(POOL)[:15]); cand = [tok(" " + c, add_special_tokens=False)["input_ids"] for c in codes]
    order = list(rng.permutation(n)) if "--random-order" in sys.argv else list(range(n))   # header listing order (line order by default; randomized control)
    header = "Each musical key is assigned a code word.\n" + "\n".join(f"{WORDS[KEYS15[i]]} major = {codes[i]}" for i in order) + "\n"
    res = {"codes": codes, "header_order": [int(i) for i in order]}
    for task, tpl in TASKS.items():
        M = np.zeros((n, n))
        for i, k in enumerate(KEYS15):
            pre = tok(header + tpl.replace("{x}", WORDS[k]))["input_ids"]; M[i] = logprobs(pre, cand)
        M = M - np.logaddexp.reduce(M, axis=1, keepdims=True); res[task] = M.tolist()
        if task == "identity": acc = float(np.mean(np.argmax(M, 1) == np.arange(n))); res["identity_acc"] = acc; print(f"{tag} codebook {cb}: identity accuracy {acc:.2f}", flush=True)
        else:
            top = np.argmax(np.where(np.eye(n, dtype=bool), -1e9, M), 1); print(f"{tag} codebook {cb} {task}: top-1 (non-self) " + " ".join(f"{KEYS15[i]}->{KEYS15[j]}" for i, j in enumerate(top)), flush=True)
    out[f"cb{cb}"] = res
os.makedirs("results/phase2/codebook", exist_ok=True); json.dump(out, open(f"results/phase2/codebook/{tag}{'_random' if '--random-order' in sys.argv else ''}.json", "w"))
