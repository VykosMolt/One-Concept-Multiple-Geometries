import sys, json, numpy as np
tag = sys.argv[1]; names = sys.argv[2].split(","); pos = sys.argv[3] if len(sys.argv) > 3 else "anchor"
for name in names:
    import os
    if not os.path.exists(f"results/spectra/{tag}/{name}.json"): print("missing", name); continue
    r = json.load(open(f"results/spectra/{tag}/{name}.json"))
    p = pos if pos in r else "embed"
    P = np.array(r[p]["profile"]); Z = np.array(r[p]["z"]); N95 = np.array(r[p]["null95"])
    labels = ["P1", "P2", "P3", "P4", "P5", "E6"] if P.shape[1] == 6 else [f"P{m}" for m in range(1, P.shape[1])] + ["E"]
    print(f"## {name} [{p}] template={r.get('template','')!r}")
    print("layer " + " ".join(f"{l:>7s}" for l in labels) + "   | z: " + " ".join(f"{l:>5s}" for l in labels))
    for l in range(P.shape[0]):
        if np.isnan(P[l]).any(): print(f"{l:5d}  nan"); continue
        print(f"{l:5d} " + " ".join(f"{x:7.3f}" for x in P[l]) + "   | " + " ".join(f"{z:5.1f}" for z in Z[l]))
