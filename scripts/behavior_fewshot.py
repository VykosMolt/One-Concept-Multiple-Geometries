"""Few-shot key-relation knowledge test. For each relation, 3 demonstration lines with keys disjoint from the query,
then the query; score the pitch class of the first generated key token. Usage: python scripts/behavior_fewshot.py <model_path> <tag> [dtype] [device_map]"""
import sys, os, json, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import AutoTokenizer, AutoModelForCausalLM
from pf.families import PC_CANON_MAJOR, PC_CANON_MINOR
model_path, tag = sys.argv[1], sys.argv[2]
dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[sys.argv[3] if len(sys.argv) > 3 else "fp32"]
dm = sys.argv[4] if len(sys.argv) > 4 else "cuda"
tok = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype, device_map=dm).eval()
dev = next(model.parameters()).device
PC = {"C": 0, "Db": 1, "C#": 1, "D": 2, "Eb": 3, "D#": 3, "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "Ab": 8, "G#": 8, "A": 9, "Bb": 10, "A#": 10, "B": 11}
SPELL_MAJ = PC_CANON_MAJOR; SPELL_MIN = PC_CANON_MINOR
def name(pc, mode): return (SPELL_MAJ if mode == "major" else SPELL_MIN)[pc]
RELS = {  # (template with {x} {y}, shift, src mode, tgt mode)
    "dominant(+7)":        ("The dominant of {x} major is {y} major.", 7, "major", "major"),
    "subdominant(+5)":     ("The subdominant of {x} major is {y} major.", 5, "major", "major"),
    "relative_minor(+9)":  ("The relative minor of {x} major is {y} minor.", 9, "major", "minor"),
    "relative_major(+3)":  ("The relative major of {x} minor is {y} major.", 3, "minor", "major"),
    "semitone_up(+1)":     ("{x} major transposed up a semitone is {y} major.", 1, "major", "major"),
    "whole_tone_up(+2)":   ("{x} major transposed up a whole tone is {y} major.", 2, "major", "major"),
    "fifth_up(+7)":        ("{x} major transposed up a perfect fifth is {y} major.", 7, "major", "major"),
    "minor_third_up(+3)":  ("{x} major transposed up a minor third is {y} major.", 3, "major", "major"),
    "tritone(+6)":         ("{x} major transposed up a tritone is {y} major.", 6, "major", "major"),
}
# candidate first tokens for each pitch class (both spellings; the letter token for sharps/flats is ambiguous, so we
# score with the *full* answer: compare log-prob of each of the 12 candidate completions " <name> <mode>." )
def seq_logprob(prefix_ids, cont_ids):
    ids = torch.tensor([prefix_ids + cont_ids], device=dev)
    with torch.no_grad(): lp = torch.log_softmax(model(ids).logits[0].float(), -1)
    return float(sum(lp[len(prefix_ids) - 1 + i, t] for i, t in enumerate(cont_ids)))
out = {}; rng = np.random.default_rng(0)
print(f"## {tag} few-shot (3 demos, disjoint keys): accuracy of argmax over 12 candidate completions; chance 1/12")
for rel, (tpl, shift, sm, tm) in RELS.items():
    rows = []
    for x in range(12):
        demos = [d for d in rng.permutation(12) if d != x and (d + shift) % 12 != x][:3]
        prefix = "\n".join(tpl.replace("{x}", name(d, sm)).replace("{y}", name((d + shift) % 12, tm)) for d in demos)
        q = tpl.replace("{x}", name(x, sm)).split(" {y}")[0]
        prefix_ids = tok(prefix + "\n" + q)["input_ids"]
        scores = []
        for y in range(12):
            cont = " " + name(y, tm) + " " + tm + "."
            scores.append(seq_logprob(prefix_ids, tok(cont, add_special_tokens=False)["input_ids"]))
        pred = int(np.argmax(scores)); target = (x + shift) % 12
        rows.append({"x": name(x, sm), "pred": name(pred, tm), "target": name(target, tm), "correct": pred == target})
    acc = np.mean([r["correct"] for r in rows]); out[rel] = {"acc": float(acc), "rows": rows}
    print(f"  {rel:20s} acc={acc:.2f}  " + " ".join(f"{r['x']}->{r['pred']}{'' if r['correct'] else '(x)'}" for r in rows), flush=True)
os.makedirs("results/behavior", exist_ok=True); json.dump(out, open(f"results/behavior/{tag}_fewshot.json", "w"), indent=1)
