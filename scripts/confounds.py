"""Black-key / token-count confound diagnostics for 12-key families.
Usage: python scripts/confounds.py <tag> <family_prefix_list comma> <pos>"""
import sys, os, json, glob, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pf.fourier import mode_energies, paired_vector, project_out, mode_isotropy, rsa_line, permutation_null

tag, fams, pos = sys.argv[1], sys.argv[2].split(","), sys.argv[3]
BLACK = np.array([0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0], float)  # semitone order
WHITE_IDX = [0, 2, 4, 5, 7, 9, 11]                               # C D E F G A B
WHITE_FIFTHS_LINE = [5, 0, 7, 2, 9, 4, 11]                       # F C G D A E B
WHITE_CHROM_LINE = [0, 2, 4, 5, 7, 9, 11]                        # C D E F G A B
out = {}
for f in sorted(glob.glob(f"results/hidden/{tag}/*.npz")):
    name = os.path.basename(f)[:-4]
    if not any(name.startswith(p + "__") for p in fams): continue
    z = np.load(f, allow_pickle=True)
    if name.endswith("__embed"): Hs = z["H"][None]; p = "embed"
    else:
        if pos not in z: continue
        Hs = z[pos]; p = pos
    tokinfo = json.load(open(f"results/hidden/{tag}/tokenization.json"))
    ntok = np.array([t["n_span"] for t in tokinfo[name]["tokens"]], float) if name in tokinfo and "tokens" in tokinfo[name] else np.ones(12)
    rows = []
    print(f"## {name} [{p}]  ntok={ntok.astype(int).tolist()}")
    print("layer  P1raw P5raw | P1 P5 (black proj) | P1 P5 (black+ntok proj) | iso k=1  iso k=7 | RSA white: fifths chrom | P5 z(proj)")
    for l in range(Hs.shape[0]):
        H = Hs[l]
        if np.isnan(H).any(): continue
        v0 = paired_vector(mode_energies(H)); p0 = v0 / v0.sum()
        Hb = project_out(H, BLACK); v1 = paired_vector(mode_energies(Hb)); p1 = v1 / v1.sum()
        Hbt = project_out(Hb, ntok); v2 = paired_vector(mode_energies(Hbt)); p2 = v2 / v2.sum()
        iso1 = mode_isotropy(H, 1)[2]; iso7 = mode_isotropy(H, 7)[2]
        rf = rsa_line(H, WHITE_FIFTHS_LINE); rc = rsa_line(H, WHITE_CHROM_LINE)
        # null for projected P5: permute labels of the *projected* H? No — project after permuting (indicator is label-tied).
        rng = np.random.default_rng(0); nulls = []
        for _ in range(500):
            perm = rng.permutation(12); Hp = H[perm]
            Hpb = project_out(project_out(Hp, BLACK), ntok); vv = paired_vector(mode_energies(Hpb)); nulls.append(vv[4] / vv.sum())
        nulls = np.array(nulls); zp = (p2[4] - nulls.mean()) / nulls.std()
        rows.append({"layer": l, "p_raw": p0.tolist(), "p_black": p1.tolist(), "p_black_ntok": p2.tolist(), "iso1": iso1, "iso7": iso7,
                     "rsa_white_fifths": rf, "rsa_white_chrom": rc, "z_P5_proj": float(zp)})
        print(f"{l:5d}  {p0[0]:.3f} {p0[4]:.3f} |  {p1[0]:.3f} {p1[4]:.3f}      |    {p2[0]:.3f} {p2[4]:.3f}          |  {iso1:.2f}     {iso7:.2f}  |   {rf:+.2f}   {rc:+.2f}   |  {zp:+.1f}")
    out[name] = rows
os.makedirs(f"results/confounds/{tag}", exist_ok=True)
json.dump(out, open(f"results/confounds/{tag}/{'_'.join(fams)}__{pos}.json", "w"))
