import os, sys, time
from huggingface_hub import HfApi, hf_hub_download
api = HfApi(token=os.environ["HF_TOKEN"])
revs = ["stage1-step950000-tokens1993B", "stage1-step1907359-tokens4001B", "stage2-ingredient3-step23852-tokens51B"]
for r in revs:
    d = f"models/olmo2_1b_ckpts/{r}"
    if os.path.exists(f"{d}/.complete"): continue
    files = [f for f in api.list_repo_files("allenai/OLMo-2-0425-1B", revision=r) if f.endswith((".safetensors", ".json", ".txt"))]
    for f in files:
        for attempt in range(20):
            try:
                hf_hub_download("allenai/OLMo-2-0425-1B", f, revision=r, local_dir=d, token=os.environ["HF_TOKEN"]); break
            except Exception as e:
                print("retry", r, f, attempt, type(e).__name__, str(e)[:120], flush=True); time.sleep(30)
    open(f"{d}/.complete", "w").write("ok"); print("DONE", r, flush=True)
print("ALL_CKPTS_DONE", flush=True)
