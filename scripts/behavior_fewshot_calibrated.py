"""Length-fair few-shot relation test: score each of the 12 candidate completions by
  log P(cont | prompt) - log P(cont | neutral prefix)   (PMI / 'calibrate-before-use' style),
which removes the unconditional length/frequency bias of unequal-token-length key names.
Usage: python scripts/behavior_fewshot_calibrated.py <model_path> <tag> [dtype] [device_map]"""
import sys, os, json, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import AutoTokenizer, AutoModelForCausalLM
from pf.families import PC_CANON_MAJOR, PC_CANON_MINOR
model_path, tag = sys.argv[1], sys.argv[2]
dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[sys.argv[3] if len(sys.argv) > 3 else "fp32"]
dm = sys.argv[4] if len(sys.argv) > 4 else "cuda"
tok = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype, device_map=dm).eval(); dev = next(model.parameters()).device
def name(pc, mode): return (PC_CANON_MAJOR if mode == "major" else PC_CANON_MINOR)[pc]
RELS = {
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
def seq_logprob(prefix_ids, cont_ids):
    ids = torch.tensor([prefix_ids + cont_ids], device=dev)
    with torch.no_grad(): lp = torch.log_softmax(model(ids).logits[0].float(), -1)
    return float(sum(lp[len(prefix_ids) - 1 + i, t] for i, t in enumerate(cont_ids)))
neutral = {m: tok("The key is")["input_ids"] for m in ("major", "minor")}
cont_ids = {(y, m): tok(" " + name(y, m) + " " + m + ".", add_special_tokens=False)["input_ids"] for y in range(12) for m in ("major", "minor")}
prior = {k: seq_logprob(neutral[k[1]], v) for k, v in cont_ids.items()}
out = {}; rng = np.random.default_rng(0)
print(f"## {tag} few-shot, three scorers: raw sum logP | mean per token | calibrated (minus logP under 'The key is'); 3 demos, disjoint keys; chance 1/12")
for rel, (tpl, shift, sm, tm) in RELS.items():
    rows = []
    for x in range(12):
        demos = [d for d in rng.permutation(12) if d != x and (d + shift) % 12 != x][:3]
        prefix = "\n".join(tpl.replace("{x}", name(d, sm)).replace("{y}", name((d + shift) % 12, tm)) for d in demos)
        q = tpl.replace("{x}", name(x, sm)).split(" {y}")[0]
        prefix_ids = tok(prefix + "\n" + q)["input_ids"]
        raw = [seq_logprob(prefix_ids, cont_ids[(y, tm)]) for y in range(12)]
        cal = [raw[y] - prior[(y, tm)] for y in range(12)]
        meanlp = [raw[y] / len(cont_ids[(y, tm)]) for y in range(12)]
        pred = int(np.argmax(cal)); pred_raw = int(np.argmax(raw)); pred_mean = int(np.argmax(meanlp)); target = (x + shift) % 12
        rows.append({"x": name(x, sm), "pred_cal": name(pred, tm), "pred_raw": name(pred_raw, tm), "pred_mean": name(pred_mean, tm), "target": name(target, tm),
                     "correct_cal": pred == target, "correct_raw": pred_raw == target, "correct_mean": pred_mean == target, "ntok_target": len(cont_ids[(target, tm)])})
    acc_cal = np.mean([r["correct_cal"] for r in rows]); acc_raw = np.mean([r["correct_raw"] for r in rows]); acc_mean = np.mean([r["correct_mean"] for r in rows])
    mn = min(len(v) for v in cont_ids.values())
    multi_raw = [r["correct_raw"] for r in rows if r["ntok_target"] > mn]; multi_mean = [r["correct_mean"] for r in rows if r["ntok_target"] > mn]
    out[rel] = {"acc_cal": float(acc_cal), "acc_raw": float(acc_raw), "acc_mean": float(acc_mean), "rows": rows}
    print(f"  {rel:20s} raw={acc_raw:.2f} mean-per-token={acc_mean:.2f} calibrated={acc_cal:.2f} | multi-token targets (n={len(multi_raw)}): raw {np.mean(multi_raw) if multi_raw else float('nan'):.2f} mean {np.mean(multi_mean) if multi_mean else float('nan'):.2f}  " + " ".join(f"{r['x']}->{r['pred_mean']}{'' if r['correct_mean'] else '(x)'}" for r in rows), flush=True)
json.dump(out, open(f"results/behavior/{tag}_fewshot_calibrated.json", "w"), indent=1)
