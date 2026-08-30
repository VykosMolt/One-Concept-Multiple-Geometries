"""Does the model know key relations? Next-token distributions for relation prompts.
Usage: python scripts/behavior_keys.py <model_path> <tag> [dtype] [device_map]"""
import sys, os, json, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import AutoTokenizer, AutoModelForCausalLM
from pf.families import PC_CANON_MAJOR, PC_CANON_MINOR, PC_ALL_SHARP, PC_ALL_FLAT
model_path, tag = sys.argv[1], sys.argv[2]
dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[sys.argv[3] if len(sys.argv) > 3 else "fp32"]
kw = {"device_map": sys.argv[4]} if len(sys.argv) > 4 else {}
tok = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, **({"device_map": "cuda"} | kw)).eval()
dev = next(model.parameters()).device
ALL = sorted(set(PC_CANON_MAJOR + PC_CANON_MINOR + PC_ALL_SHARP + PC_ALL_FLAT))
first_ids = {n: tok(" " + n, add_special_tokens=False)["input_ids"][0] for n in ALL}  # first token of each spelling
pc_of = {}
for n in ALL:
    base = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}[n[0]]; pc_of[n] = (base + (1 if n.endswith("#") else -1 if n.endswith("b") else 0)) % 12
RELS = {
    "dominant(+7)": ("The dominant key of {x} major is", 7, "major"),
    "subdominant(+5)": ("The subdominant key of {x} major is", 5, "major"),
    "relative_minor(+9)": ("The relative minor of {x} major is", 9, "minor"),
    "relative_major(+3)": ("The relative major of {x} minor is", 3, "major"),
    "semitone_up(+1)": ("Transposed up by one semitone, {x} major becomes", 1, "major"),
    "whole_tone_up(+2)": ("Transposed up by a whole tone, {x} major becomes", 2, "major"),
    "fifth_up(+7)": ("Transposed up by a perfect fifth, {x} major becomes", 7, "major"),
    "parallel_minor(0)": ("The parallel minor of {x} major is", 0, "minor"),
}
out = {}
print(f"## {tag}: P(correct pitch class | prompt), summing over enharmonic spellings of the first token; chance ≈ 1/12")
for rel, (tpl, shift, mode) in RELS.items():
    concepts = PC_CANON_MAJOR if "minor}" not in tpl and "{x} minor" not in tpl else PC_CANON_MINOR
    rows = []
    for x in concepts:
        ids = tok(tpl.replace("{x}", x), return_tensors="pt")["input_ids"].to(dev)
        with torch.no_grad(): logits = model(ids).logits[0, -1].float()
        probs = torch.softmax(logits, -1).cpu().numpy()
        # mass on each pitch class via first-token of any spelling (sharps/flats share first token with naturals -> ambiguous;
        # so we score by *letter-level* first token and count as correct if the target pc's canonical spelling's first token wins)
        target = (pc_of[x] + shift) % 12
        # group candidate spellings by pitch class and take the max over their first tokens
        pc_mass = np.zeros(12)
        for n in ALL:
            pc_mass[pc_of[n]] = max(pc_mass[pc_of[n]], probs[first_ids[n]])
        top_pc = int(np.argmax(pc_mass)); pred_tok = tok.decode([int(np.argmax(probs))])
        rows.append({"x": x, "target_pc": target, "top_pc": top_pc, "correct": top_pc == target, "p_target": float(pc_mass[target]), "top_token": pred_tok})
    acc = np.mean([r["correct"] for r in rows])
    print(f"  {rel:20s} acc={acc:.2f}  " + " ".join(f"{r['x']}->{r['top_token'].strip()}" for r in rows))
    out[rel] = rows
os.makedirs("results/behavior", exist_ok=True); json.dump(out, open(f"results/behavior/{tag}.json", "w"), indent=1)
