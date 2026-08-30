"""Merge per-shard JSON tallies into a single tally per group (all / nocof / cof)."""
import sys, os, json
from collections import defaultdict

def merge(files):
    out = {}
    for g in ("all", "nocof", "cof"):
        agg = {"uni": defaultdict(int), "uni_sentinit": defaultdict(int), "co": defaultdict(float),
               "co_d": defaultdict(int), "doc_co": defaultdict(int), "doc_uni": defaultdict(int),
               "adjacent": defaultdict(int), "nwords": 0, "Z": 0.0, "ndocs": 0, "cof_docs": 0}
        for f in files:
            d = json.load(open(f))[g]
            for k in ("uni", "uni_sentinit", "co", "co_d", "doc_co", "doc_uni", "adjacent"):
                for kk, v in d[k].items(): agg[k][kk] += v
            for k in ("nwords", "Z", "ndocs", "cof_docs"): agg[k] += d[k]
        out[g] = {k: (dict(v) if isinstance(v, defaultdict) else v) for k, v in agg.items()}
    return out

if __name__ == "__main__":
    indir, outfile = sys.argv[1], sys.argv[2]
    files = sorted(os.path.join(indir, f) for f in os.listdir(indir) if f.endswith(".json") and f.startswith("train-"))
    m = merge(files)
    m["_meta"] = {"n_shards": len(files), "shards": [os.path.basename(f) for f in files]}
    json.dump(m, open(outfile, "w"))
    print("merged", len(files), "shards ->", outfile, {g: (m[g]["ndocs"], m[g]["nwords"]) for g in ("all","nocof","cof")})
