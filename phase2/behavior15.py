"""Sequence-level next-key behaviour over the 15 spellings under the context families where the next token should be a key
(B enharmonic, C harmonic, D chord, E modulation). For each context x (15) and candidate y (15) score the completion
" <y> major" with three pre-registered scorers: (1) total log-prob; (2) length-normalized (mean per token);
(3) enharmonic-merged mass: log-sum-exp over the two spellings of the same neutral pc (defined for the 3 enharmonic pairs;
other keys unchanged) — appropriate for neutral-pitch tasks, destructive for spelling tasks; all three are saved.
Usage: python phase2/behavior15.py <model_path> <tag> [dtype] [device_map] [families comma]"""
import sys, os, json, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import AutoTokenizer, AutoModelForCausalLM
from phase2.keys15 import KEYS15, ENH_PAIRS, PC
from phase2.contexts import FAMILIES
model_path, tag = sys.argv[1], sys.argv[2]
dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[sys.argv[3] if len(sys.argv) > 3 else "fp32"]
dm = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] != "-" else "cuda"
fams = sys.argv[5].split(",") if len(sys.argv) > 5 else ["B_enharmonic", "C_harmonic", "D_chord", "E_modulation"]
tok = AutoTokenizer.from_pretrained(model_path); model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype, device_map=dm).eval(); dev = next(model.parameters()).device
cont = {y: tok(" " + y + " major", add_special_tokens=False)["input_ids"] for y in KEYS15}
@torch.no_grad()
def score_all(prefix_ids):
    """total and mean log-prob of each of the 15 completions after prefix (batched)."""
    seqs = [prefix_ids + cont[y] for y in KEYS15]; L = max(len(s) for s in seqs)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    ids = torch.tensor([s + [pad] * (L - len(s)) for s in seqs], device=dev)
    lp = torch.log_softmax(model(ids).logits.float(), -1)
    tot, mean = [], []
    for b, y in enumerate(KEYS15):
        c = cont[y]; s = sum(float(lp[b, len(prefix_ids) - 1 + i, t]) for i, t in enumerate(c)); tot.append(s); mean.append(s / len(c))
    return np.array(tot), np.array(mean)
out = {}
for fam in fams:
    for ti, tpl in enumerate(FAMILIES[fam]):
        Ltot = np.zeros((15, 15)); Lmean = np.zeros((15, 15))
        for i, x in enumerate(KEYS15):
            pre = tok(tpl.replace("{x}", x))["input_ids"]; t, m = score_all(pre); Ltot[i] = t; Lmean[i] = m
        Ltot_n = Ltot - np.logaddexp.reduce(Ltot, axis=1, keepdims=True)   # normalized over the 15 candidates
        Lmerged = Ltot_n.copy()
        for a, b in ENH_PAIRS:
            m = np.logaddexp(Ltot_n[:, a], Ltot_n[:, b]); Lmerged[:, a] = m; Lmerged[:, b] = m
        out[f"{fam}__t{ti}"] = {"template": tpl, "total": Ltot_n.tolist(), "mean": Lmean.tolist(), "merged": Lmerged.tolist()}
        top = np.argmax(np.where(np.eye(15, dtype=bool), -1e9, Ltot_n), axis=1)
        print(f"{tag} {fam} t{ti}: top-1 (non-self) by total: " + " ".join(f"{KEYS15[i]}->{KEYS15[j]}" for i, j in enumerate(top)), flush=True)
os.makedirs("results/phase2/behavior", exist_ok=True); json.dump(out, open(f"results/phase2/behavior/{tag}.json", "w"))
