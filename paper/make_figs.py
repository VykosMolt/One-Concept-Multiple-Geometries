"""Regenerate every main-text figure of the paper from saved results (no hand-typed numbers), in the shared
Kirin figure style. Run from the repository root: .venv-cpu/bin/python paper/make_figs.py"""
import json, glob, os, re, sys, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, ".")
from paper_style import figure_style as st
st.apply(paper=1)
plt.rcParams.update({"font.size": 8.4, "axes.titlesize": 9.0, "xtick.labelsize": 7.8, "ytick.labelsize": 7.8, "legend.fontsize": 7.4, "lines.linewidth": 1.4, "lines.markersize": 4.5, "axes.titlepad": 5})
OUT = "paper/figures"; os.makedirs(OUT, exist_ok=True)
from phase2.keys15 import KEYS15, S, GLYPH, n as N15
MODELS = ["olmo2_1b", "gemma2_2b", "qwen25_3b", "olmo2_7b"]; MLAB = {"olmo2_1b": "OLMo-2-1B", "gemma2_2b": "Gemma-2-2B", "qwen25_3b": "Qwen2.5-3B", "olmo2_7b": "OLMo-2-7B"}
MCOL = {"olmo2_1b": st.P1, "gemma2_2b": st.POSITIVE, "qwen25_3b": st.VIOLET, "olmo2_7b": st.P1_DARK}
def save(fig, name):
    try: fig.tight_layout()
    except Exception: pass
    st.save(fig, f"{OUT}/{name}.pdf"); plt.close(fig); print("wrote", name)
def nogrid(ax): ax.grid(False)

# ---------------------------------------------------------------- Fig 1: the two tonal spaces
def fig_spaces():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.55), gridspec_kw={"width_ratios": [1, 1.7]})
    ax = axes[0]; ax.set_aspect("equal"); ax.axis("off"); nogrid(ax)
    circ = np.linspace(0, 2 * np.pi, 200); ax.plot(np.cos(circ), np.sin(circ), color=st.RULE, lw=1.2)
    for z in range(12):
        th = np.pi / 2 - 2 * np.pi * z / 12
        idx = [i for i in range(N15) if (7 * S[i]) % 12 == (7 * z) % 12]; names = [KEYS15[i] for i in idx]
        col = st.P2 if len(names) > 1 else st.INK
        ax.plot(np.cos(th), np.sin(th), "o", ms=5.5, color=col, mec="white", mew=0.6)
        ax.text(1.27 * np.cos(th), 1.27 * np.sin(th), "/".join(names), ha="center", va="center", fontsize=7.5, color=col, fontweight="bold" if len(names) > 1 else "normal")
    ax.set_xlim(-1.55, 1.55); ax.set_ylim(-1.6, 1.6)
    ax.text(0, -1.62, "(a) neutral pitch classes: circle of fifths", ha="center", va="top", fontsize=8.4, color=st.INK)
    ax = axes[1]; ax.axis("off"); nogrid(ax)
    ax.plot([-7.6, 7.6], [0, 0], color=st.RULE, lw=1.2)
    for i in range(N15):
        col = st.P2 if KEYS15[i] in ("Cb", "Gb", "Db", "B", "F#", "C#") else st.INK
        ax.plot(S[i], 0, "o", ms=5.5, color=col, mec="white", mew=0.6)
        ax.text(S[i], -0.3, KEYS15[i], ha="center", va="top", fontsize=7.5, color=col, fontweight="bold" if col == st.P2 else "normal")
        ax.text(S[i], 0.2, f"{S[i]:+d}" if S[i] else "0", ha="center", va="bottom", fontsize=6.3, color=st.MUTED)
    for a, b, h in ((-7, 5, 0.72), (-6, 6, 0.92), (-5, 7, 1.12)):
        ax.annotate("", xy=(b, h), xytext=(a, h), arrowprops=dict(arrowstyle="<->", color=st.P2, lw=0.8, shrinkA=0, shrinkB=0))
    ax.text(0, 1.28, "enharmonic twins: distance 0 on the circle, 12 on the line", ha="center", fontsize=7.4, color=st.P2)
    ax.text(-7.6, -0.72, "flat side", fontsize=6.8, color=st.MUTED); ax.text(7.6, -0.72, "sharp side", fontsize=6.8, color=st.MUTED, ha="right")
    ax.set_xlim(-8.2, 8.2); ax.set_ylim(-1.45, 1.5)
    ax.text(0, -1.3, "(b) tonal pitch classes: open line of fifths (signed accidental count $s$)", ha="center", va="top", fontsize=8.4, color=st.INK)
    save(fig, "fig_spaces")

# ---------------------------------------------------------------- Fig 2: corpus kernels
def fig_corpus():
    r = json.load(open("results/corpus/wiki/report.json"))["all"]
    mo = r["months"]; ky = r["major_canon@pmi"]; kp = np.array(ky["kappa"]); d = np.arange(12)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.2))
    ax = axes[0]; ax.plot(d, mo["kappa"], "o-", color=st.P1); ax.set_xlabel("distance $d$ (months)"); ax.set_ylabel(r"kernel $\kappa(d)$"); ax.set_title("(a) months, Karkada $M^*$", loc="left")
    ax = axes[1]; ax.plot(d, kp, "o-", color=st.BASELINE, label="semitone order"); ax.plot(d, kp[(7 * d) % 12], "s-", color=st.P1, label="fifths order"); ax.legend(); ax.set_xlabel("distance $d$"); ax.set_ylabel("PMI kernel"); ax.set_title("(b) 12 major keys, PMI", loc="left")
    ax = axes[2]; lab = ["P1", "P2", "P3", "P4", "P5", "E6"]; x = np.arange(6); w = 0.38
    ax.bar(x - w / 2, mo["profile_abs_lambda"], w, color=st.P1, label="months"); ax.bar(x + w / 2, ky["profile_abs_lambda"], w, color=st.P2, label="keys (PMI)")
    ax.axhline(2 / 11, ls=":", color=st.MUTED, lw=0.8); ax.text(3.0, 2 / 11 + 0.014, "relabeling null", ha="center", fontsize=6.5, color=st.MUTED)
    ax.set_xticks(x); ax.set_xticklabels(lab); ax.set_ylabel(r"share of $|\lambda_k|$"); ax.set_title("(c) paired Fourier energies", loc="left"); ax.legend(loc="upper center")
    save(fig, "fig_corpus")

# ---------------------------------------------------------------- Fig 3: the orthographic alias
def fig_alias():
    x = np.arange(12); black = np.isin(x, [1, 3, 6, 8, 10]).astype(float); f = black - black.mean()
    F = np.fft.fft(f) / np.sqrt(12); E = np.abs(F) ** 2; tot = E[1:].sum()
    prof = np.array([E[1] + E[11], E[2] + E[10], E[3] + E[9], E[4] + E[8], E[5] + E[7], E[6]]) / tot
    assert abs(prof[4] - 0.796) < 0.005, prof
    circ = np.column_stack([np.cos(2 * np.pi * 7 * x / 12), np.sin(2 * np.pi * 7 * x / 12)]); Fc = np.fft.fft(circ - circ.mean(0), axis=0) / np.sqrt(12); Ec = (np.abs(Fc) ** 2).sum(1); tc = Ec[1:].sum()
    pc = np.array([Ec[1] + Ec[11], Ec[2] + Ec[10], Ec[3] + Ec[9], Ec[4] + Ec[8], Ec[5] + Ec[7], Ec[6]]) / tc
    fig = plt.figure(figsize=(7.2, 2.6)); gs = fig.add_gridspec(1, 5, width_ratios=[1.35, 1, 1, 1, 1], wspace=0.55)
    ax = fig.add_subplot(gs[0]); lab = ["P1", "P2", "P3", "P4", "P5", "E6"]; xx = np.arange(6); w = 0.38
    ax.bar(xx - w / 2, pc, w, color=st.P1, label="true fifths circle"); ax.bar(xx + w / 2, prof, w, color=st.NEGATIVE, label="accidental indicator")
    ax.axhline(2 / 11, ls=":", color=st.MUTED, lw=0.8); ax.set_xticks(xx); ax.set_xticklabels(lab); ax.set_ylabel("share of Fourier energy"); ax.set_ylim(0, 1.05); ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=1, fontsize=6.2)
    ax.set_title(f"(a) {100 * prof[4]:.1f}% in P5", loc="left")
    for j, m in enumerate(MODELS):
        ax = fig.add_subplot(gs[j + 1]); D = json.load(open(f"results/decouple/{m}.json")); c = D["canonical"]["rows"]; d = D["decoupled"]["rows"]; L = [r["layer"] for r in c]
        ax.plot(L, [r["rsa_black"] for r in c], color=st.P1, label="standard spelling:\nblack-key = glyph block")
        ax.plot(L, [r["rsa_black"] for r in d], color=st.NEGATIVE, label="respelled: black-key block")
        ax.plot(L, [r["rsa_glyph"] for r in d], "--", color=st.NEGATIVE, label="respelled: glyph block")
        ax.axhline(0, color=st.MUTED, lw=0.5); ax.set_ylim(-0.4, 1.0); ax.set_xlabel("layer"); ax.set_title(("(b) " if j == 0 else "") + MLAB[m], loc="left")
        if j == 0: ax.set_ylabel("RSA with block", labelpad=1)
        else: ax.set_yticklabels([])
        if j == 3: ax.legend(loc="upper center", bbox_to_anchor=(-1.3, -0.28), ncol=3, fontsize=6.2)
    save(fig, "fig_alias")

# ---------------------------------------------------------------- Fig 4: operators (association vs conditional-row, incl. matched same-count operators)
def fig_operators():
    txt = open("results/phase5/operators.txt").read().splitlines(); ops = {}; eci = {}
    keys = {"SYM (PMI, symmetric window)": ("Karkada-window PMI (L=16)", st.BASELINE), "MATCHED PMI (N+N^T), distance = -PMI": ("PMI from the same 40-word counts", "#B9C0C6"),
            "HELP (helper-word factorization)": ("helper-word factorization", st.UNRESOLVED), "COND rows JS (A_win40)": ("40-word conditional p(j|i)", st.P1), "MATCHED sym-conditional JS (N+N^T rows)": ("symmetrized conditional, same counts", "#7A8FB5")}
    for l in txt:
        for key in keys:
            if l.strip().startswith(key) and "ECI" in l: eci[key] = float(re.search(r"ECI ([0-9.]+)", l).group(1))
            if l.startswith(key) and "|" in l and "ECI" not in l: ops[key] = [float(v) for v in l.split("|")[1].split()]
    # columns: keyname 7B, prompt-final 7B, behaviour 7B, behaviour 1B, keyname Qwen, behaviour Qwen, behaviour Gemma
    sel = [(3, "behaviour\nOLMo-2-1B"), (6, "behaviour\nGemma-2-2B"), (5, "behaviour\nQwen2.5-3B"), (2, "behaviour\nOLMo-2-7B"), (1, "prompt-final\ngeometry, 7B"), (0, "key-name\ngeometry, 7B")]
    fig, ax = plt.subplots(figsize=(7.2, 2.6)); w = 0.16; x = np.arange(len(sel))
    for k, key in enumerate(keys):
        lab, col = keys[key]; ax.bar(x + (k - 2) * w, [ops[key][i] for i, _ in sel], w, color=col, label=f"{lab} (twin ECI {eci[key]:.2f})")
    ax.axhline(0, color=st.MUTED, lw=0.5); ax.set_xticks(x); ax.set_xticklabels([t for _, t in sel]); ax.set_ylabel("Spearman ρ over 105 key pairs"); ax.set_ylim(-0.3, 0.95)
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.28), fontsize=6.6)
    save(fig, "fig_operators")

# ---------------------------------------------------------------- Fig 5: held-out prediction
def fig_heldout():
    Jn = json.load(open("results/phase5/fingerprint/wikipedia_v3_neutral.json")); Jr = json.load(open("results/phase5/fingerprint/wikipedia_v3_neutral_rich.json")); Js = json.load(open("results/phase5/fingerprint/wikipedia_v3.json"))
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0)); axes = axes.ravel()
    short = ["OLMo\n1B", "Gemma\n2B", "Qwen\n3B", "OLMo\n7B"]
    for ax, fam, title in ((axes[0], "E_modulation", "(a) modulation family, target-aggregated view"), (axes[1], "C_harmonic", "(b) harmonic family, target-aggregated view")):
        x = np.arange(4); w = 0.2
        for k, (ex, J, hatch, lab) in enumerate((("A_win40", Jn, "", "window conditional, base theory"), ("A_win40", Jr, "///", "window conditional, rich theory"), ("D_doc", Jn, "", "document conditional, base theory"), ("D_doc", Jr, "///", "document conditional, rich theory"))):
            col = st.P1 if ex == "A_win40" else st.P2
            vals = [J[f"{m}|{fam}|{ex}"]["dkl"] for m in MODELS]; ps = [J[f"{m}|{fam}|{ex}"]["dkl_p"] for m in MODELS]
            ax.bar(x + (k - 1.5) * w, vals, w, color=col, alpha=1.0 if not hatch else 0.5, hatch=hatch, edgecolor="white", linewidth=0.4, label=lab if fam == "E_modulation" else None)
            for xi, v, p in zip(x + (k - 1.5) * w, vals, ps):
                if p < .05: ax.text(xi, max(v, 0) + 0.0008, "*", ha="center", va="bottom", fontsize=7, color=st.INK)
        th = [Jn[f"{m}|{fam}|A_win40"]["kl"]["theory"] for m in MODELS]; thr = [Jr[f"{m}|{fam}|A_win40"]["kl"]["theory"] for m in MODELS]
        ax.set_xticks(x); ax.set_xticklabels([f"{s}\nKL$_0$ {t:.2f}\nrich {u:.2f}" for s, t, u in zip(short, th, thr)], fontsize=6.0)
        ax.axhline(0, color=st.MUTED, lw=0.5); ax.set_title(title, loc="left", fontsize=8.2); ax.set_ylim(-0.005, 0.052)
        if fam == "E_modulation": ax.set_ylabel("held-out ΔKL (nats per source row)")
    for ax, key, ylab, title in ((axes[2], "dkl", "held-out ΔKL (nats per row)", "(c) modulation family, spelled view: ΔKL"), (axes[3], "r2gain", "held-out within-row ΔR²", "(d) modulation family, spelled view: ΔR²")):
        x = np.arange(4); w = 0.38
        for k, (ex, lab) in enumerate((("A_win40", "window"), ("D_doc", "document"))):
            vals = [Js[f"{m}|E_modulation|{ex}"][key] for m in MODELS]; ps = [Js[f"{m}|E_modulation|{ex}"][key + "_p"] for m in MODELS]
            ax.bar(x + (k - 0.5) * w, vals, w, color=st.P1 if ex == "A_win40" else st.P2, edgecolor="white", linewidth=0.4, label=lab)
            for xi, v, p in zip(x + (k - 0.5) * w, vals, ps):
                if p < .05: ax.text(xi, max(v, 0) + (0.0004 if key == "dkl" else 0.004), "*", ha="center", va="bottom", fontsize=7)
        ax.axhline(0, color=st.MUTED, lw=0.5); ax.set_xticks(x); ax.set_xticklabels(short, fontsize=6.4); ax.set_ylabel(ylab); ax.set_title(title, loc="left", fontsize=8.2)
        if key == "dkl": ax.set_ylim(-0.0068, 0.0150); ax.legend(fontsize=6.2, loc="upper left")
    h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=4, fontsize=6.4, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.95, bottom=0.17, wspace=0.28, hspace=0.55); st.save(fig, f"{OUT}/fig_heldout.pdf"); plt.close(fig); print("wrote fig_heldout")

# ---------------------------------------------------------------- Fig 6: synthetic controls
def fig_synthetic():
    runs = {}
    for d in sorted(glob.glob("results/phase3/runs/main/circle_*_s*")):
        cfg = json.load(open(d + "/config.json")); T = json.load(open(d + "/trajectory.json"))["trajectory"]; runs.setdefault(cfg["code"], []).append(T)
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.45)); steps = [e["step"] for e in runs["aligned"][0]]
    def band(ax, code, key, where, col, lab):
        arr = np.array([[e[where][key] if where == "behaviour" else e["hidden"][-1][key] for e in T] for T in runs[code]]); m, s = arr.mean(0), arr.std(0)
        ax.plot(steps[1:], m[1:], color=col, label=lab); ax.fill_between(steps[1:], (m - s)[1:], (m + s)[1:], color=col, alpha=0.18, lw=0)
    ax = axes[0]; rows = [l.split("|") for l in open("results/phase3/analysis_uncontrolled.txt") if re.match(r"\s*\d+ \|", l)]
    st_ = [int(r[0]) for r in rows]; al = [float(r[2].split()[0]) for r in rows]; pe = [float(r[2].split()[1]) for r in rows]
    ax.plot(st_, al, "o-", color=st.P1, ms=3, label="line-aligned code"); ax.plot(st_, pe, "o-", color=st.BASELINE, ms=3, label="permuted code")
    ax.axhline(0.439, color=st.P2, ls="--", lw=0.9); ax.text(0.5, 0.445, "oracle", ha="center", va="bottom", fontsize=6.6, color=st.P2, transform=ax.get_yaxis_transform()); ax.set_ylim(0.08, 0.485); ax.set_xscale("log"); ax.set_xlabel("training step"); ax.set_ylabel("behaviour line | circle"); ax.set_title("(a) behaviour", loc="left"); ax.legend(loc="upper center", bbox_to_anchor=(0.3, -0.3), ncol=2, fontsize=6.9)
    ax = axes[1]; band(ax, "aligned", "twin_target_asym", "behaviour", st.P1, ""); band(ax, "permuted", "twin_target_asym", "behaviour", st.BASELINE, ""); ax.set_xscale("log"); ax.set_xlabel("training step"); ax.set_ylabel("twin-target asym. (nats)"); ax.set_title("(b) spelling asymmetry", loc="left")
    ax = axes[2]; band(ax, "aligned", "line_given_circle", "hidden", st.P1, ""); band(ax, "permuted", "line_given_circle", "hidden", st.BASELINE, ""); ax.set_xscale("log"); ax.set_xlabel("training step"); ax.set_ylabel("hidden line | circle"); ax.set_title("(c) representation", loc="left")
    ax = axes[3]; cmap = plt.get_cmap("viridis"); rs = [0.5, 0.1, 0.03, 0.01, 0.003, 0.001]
    for d in sorted(glob.glob("results/phase4/runs/primary/alias_*") + glob.glob("results/phase4/runs/extended/alias_*")):
        cfg = json.load(open(d + "/config.json")); T = json.load(open(d + "/trajectory.json"))["trajectory"]
        ex = np.array([e["exposure_rare"] for e in T], float); js = np.array([e["twin_jsz"] for e in T]); keep = ex > 0
        ax.plot(ex[keep], js[keep], color=cmap(rs.index(cfg["r"]) / 5), lw=0.45, alpha=0.5, ls="-" if cfg["code"] == "aligned" else "--")
    xx = np.logspace(2.3, 5.5, 50); ax.plot(xx, 1.18 * xx ** -0.74, color=st.NEGATIVE, lw=1.3, label=r"$1.18\,N^{-0.74}$")
    for i, r in enumerate(rs): ax.plot([], [], color=cmap(i / 5), lw=1.2, label=f"r = {r}")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("rare-alias exposures $N$"); ax.set_ylabel("latent twin JS"); ax.set_title("(d) sparse alias", loc="left"); ax.legend(fontsize=6.4, ncol=4, loc="upper center", bbox_to_anchor=(-0.9, -0.3), columnspacing=0.9, handlelength=1.4)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.9, bottom=0.36, wspace=0.62); st.save(fig, f"{OUT}/fig_synthetic.pdf"); plt.close(fig); print("wrote fig_synthetic")

# ---------------------------------------------------------------- Fig 7: checkpoint trajectory
def fig_trajectory():
    rows = json.load(open("results/phase5/ckpt_fingerprint.json")); R = {(r["rev"], r["fam"]): r for r in rows}
    s1 = [("stage1-step300-tokens1B", 1), ("stage1-step10000-tokens21B", 21), ("stage1-step23100-tokens49B", 49), ("stage1-step50000-tokens105B", 105), ("stage1-step140000-tokens294B", 294), ("stage1-step480000-tokens1007B", 1007), ("stage1-step950000-tokens1993B", 1993), ("stage1-step1907359-tokens4001B", 4001)]
    s2 = ["stage2-ingredient1-step23852-tokens51B", "stage2-ingredient2-step23852-tokens51B", "stage2-ingredient3-step23852-tokens51B"]
    tok = [t for _, t in s1]; x2 = [4052 * 1.07 ** k for k in (-1, 0, 1)]; xr = 4052 * 2.4
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5))
    ax = axes[0]; fam = "E_modulation"
    for key, col, lab, ls in (("lc", st.P1, "line | circle", "-"), ("cl", st.P2, "circle | line", "--")):
        ax.plot(tok, [R[(rv, fam)][key] for rv, _ in s1], "o" + ls, color=col, label=lab); ax.plot(x2, [R[(rv, fam)][key] for rv in s2], "s", color=col); ax.plot([xr], [R[("main", fam)][key]], "D", color=col, mfc="white")
    ax.plot(tok, [R[(rv, "C_harmonic")]["lc"] for rv, _ in s1], "o-", color=st.BASELINE, ms=3, lw=1, label="line | circle, harmonic family"); ax.plot(x2, [R[(rv, "C_harmonic")]["lc"] for rv in s2], "s", color=st.BASELINE, ms=3); ax.plot([xr], [R[("main", "C_harmonic")]["lc"]], "D", color=st.BASELINE, ms=4, mfc="white")
    ax.set_ylabel("controlled partial Spearman"); ax.set_title("(a) line vs circle in next-key behaviour", loc="left"); ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=3, fontsize=6.2, handletextpad=0.5, columnspacing=1.0)
    ax = axes[1]
    for exx, col, lab in (("D_doc", st.P1, "document conditional"), ("A_win40", st.P2, "window conditional")):
        y1 = [R[(rv, fam)]["resid"][exx][0] for rv, _ in s1]; p1 = [R[(rv, fam)]["resid"][exx][1] for rv, _ in s1]; y2 = [R[(rv, fam)]["resid"][exx][0] for rv in s2]; p2 = [R[(rv, fam)]["resid"][exx][1] for rv in s2]; yr, pr = R[("main", fam)]["resid"][exx]
        ax.plot(tok, y1, "-", color=col, label=lab); ax.scatter(tok, y1, s=[22 if p < .05 else 16 for p in p1], color=[col if p < .05 else "white" for p in p1], edgecolors=col, linewidths=0.9, zorder=3)
        ax.scatter(x2, y2, s=[22 if p < .05 else 16 for p in p2], marker="s", color=[col if p < .05 else "white" for p in p2], edgecolors=col, linewidths=0.9, zorder=3)
        ax.scatter([xr], [yr], s=26, marker="D", color=col if pr < .05 else "white", edgecolors=col, linewidths=0.9, zorder=3)
    ax.set_ylabel("residual Spearman (rich theory model)"); ax.set_title("(b) residual correspondence with corpus conditionals", loc="left")
    ax.scatter([], [], s=22, color=st.INK, label="p < .05"); ax.scatter([], [], s=16, color="white", edgecolors=st.INK, label="n.s."); ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=4, fontsize=6.2, handletextpad=0.5, columnspacing=1.0)
    for ax in axes:
        ax.set_xscale("log"); ax.axhline(0, color=st.MUTED, lw=0.5); ax.set_xlim(0.7, 14000); ax.set_xlabel("stage-1 tokens (B)")
        ax.axvspan(4001 * 1.02, 4052 * 1.17, color=st.RULE, alpha=0.7, lw=0)
        ax.text(4052 * 0.93, 0.02, "S2", ha="right", va="bottom", fontsize=5.8, color=st.MUTED, transform=ax.get_xaxis_transform())
        ax.text(xr, 0.02, "rel.", ha="center", va="bottom", fontsize=5.8, color=st.MUTED, transform=ax.get_xaxis_transform())
    save(fig, "fig_trajectory")

# ---------------------------------------------------------------- Fig 8: cross-corpus
def fig_crosscorpus():
    corpora = [("olmomix_wiki", "OM wiki\n2 sh.\n8.5k"), ("olmomix_dclm", "OM DCLM\n9 sh.\n633"), ("dolmino_dclm", "Dolmino\nDCLM\n1.3k")]
    if os.path.exists("results/phase5/fingerprint/olmomix_dclm_big_neutral.json") and os.path.exists("results/phase5/fingerprint/wikipedia_thin_olmomix_dclm_big_neutral.json"):
        npairs = int(json.load(open("results/phase5/fingerprint/olmomix_dclm_big_neutral.json"))["olmo2_1b|E_modulation|A_win40"]["pairs"])
        corpora.insert(2, ("olmomix_dclm_big", f"OM DCLM\n54 sh.\n{npairs / 1000:.1f}k"))
    cells = [("E_modulation", "D_doc", "(a) modulation × document"), ("E_modulation", "A_win40", "(b) modulation × window"), ("C_harmonic", "A_win40", "(c) harmonic × window")]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), sharey=True)
    for ax, (fam, ex, title) in zip(axes, cells):
        x = 0; ticks = []
        for c, clab in corpora:
            Jc = json.load(open(f"results/phase5/fingerprint/{c}_neutral.json")); Jb = json.load(open(f"results/phase5/fingerprint/wikipedia_thin_{c}_neutral.json"))
            for m in MODELS:
                v = Jc[f"{m}|{fam}|{ex}"]; w = Jb[f"{m}|{fam}|{ex}"]
                ax.bar(x, v["dkl"], 0.42, color=MCOL[m], edgecolor="white", lw=0.3); ax.bar(x + 0.42, w["dkl"], 0.42, color=MCOL[m], alpha=0.38, hatch="////", edgecolor="white", lw=0.3)
                if v["dkl_p"] < .05: ax.text(x, max(v["dkl"], 0) + 0.0008, "*", ha="center", fontsize=6)
                if w["dkl_p"] < .05: ax.text(x + 0.42, max(w["dkl"], 0) + 0.0008, "*", ha="center", fontsize=6)
                x += 1
            ticks.append((x - 2.3, clab)); x += 1.8
        ax.set_xticks([t for t, _ in ticks]); ax.set_xticklabels([c for _, c in ticks], fontsize=5.6); ax.axhline(0, color=st.MUTED, lw=0.5); ax.set_title(title, loc="left")
    axes[0].set_ylim(-0.007, 0.081)
    axes[0].set_ylabel("held-out ΔKL (nats per row)")
    for m in MODELS: axes[0].bar([0], [0], color=MCOL[m], label=MLAB[m])
    axes[0].bar([0], [0], color=st.BASELINE, alpha=0.38, hatch="////", label="Wikipedia thinned to the same pair mass"); axes[0].legend(fontsize=6.3, loc="upper center", bbox_to_anchor=(1.7, -0.32), ncol=5)
    save(fig, "fig_crosscorpus")

if __name__ == "__main__":
    for f in (fig_spaces, fig_corpus, fig_alias, fig_operators, fig_heldout, fig_synthetic, fig_trajectory, fig_crosscorpus): f()
