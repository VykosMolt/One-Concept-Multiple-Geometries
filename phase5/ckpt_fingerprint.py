"""Checkpoint trajectory using the shared v4.1 residual projection.

The checkpoint residual correspondence is explicitly non-held-out: the
shared rich raw builder is full-sample scaled and projected with unpenalised
source-row fixed effects.  This is the same projection used by
``phase5.fingerprint``.  It is labelled as such in the JSON output so it is
not confused with the LOO flagship test.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.stats import rankdata, spearmanr

from phase2.keys15 import ENH_PAIRS, KEYS15, S, candidate_geometries, n
from phase5.theory_features import (
    MASTER_SEED,
    RANK_RTOL,
    SCHEMA_VERSION,
    SEED_ALGORITHM,
    build_raw_features,
    dump_json_strict,
    residual_projection,
    stable_seed,
)


REVISIONS = [
    "stage1-step300-tokens1B",
    "stage1-step10000-tokens21B",
    "stage1-step23100-tokens49B",
    "stage1-step50000-tokens105B",
    "stage1-step140000-tokens294B",
    "stage1-step480000-tokens1007B",
    "stage1-step950000-tokens1993B",
    "stage1-step1907359-tokens4001B",
    "stage2-ingredient3-step23852-tokens51B",
    "stage2-ingredient1-step23852-tokens51B",
    "stage2-ingredient2-step23852-tokens51B",
]
N_PERM = 2000
CHECKPOINT_FAMILIES = ("E_modulation", "C_harmonic")
CHECKPOINT_TEMPLATES = (0, 1, 2, 3)


def _logsumexp(x: np.ndarray, axis: int = 1, keepdims: bool = False) -> np.ndarray:
    m = np.max(x, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))
    return out if keepdims else np.squeeze(out, axis=axis)


def _merge_columns(L: np.ndarray) -> np.ndarray:
    out = np.full((n, 12), -np.inf, dtype=float)
    pc = (7 * S) % 12
    for z in range(12):
        out[:, z] = np.logaddexp.reduce(L[:, pc == z], axis=1)
    return out


def _load_tokens() -> np.ndarray:
    """Load the one unique finite 15-key tokenizer ``n_span`` bank."""

    path = Path("results/phase2/hidden/olmo2_1b_symbol_tokens.json")
    if not path.is_file():
        raise FileNotFoundError(f"required tokenizer file is missing: {path}")
    try:
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read tokenizer file {path}") from exc
    if not isinstance(data, Mapping):
        raise ValueError(f"{path}: tokenizer payload must be an object")
    banks: list[tuple[str, np.ndarray]] = []
    for key, value in data.items():
        if isinstance(value, Mapping) and "n_span" in value:
            try:
                arr = np.asarray(value["n_span"], dtype=float)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{key}: n_span is not numeric") from exc
            if arr.shape != (n,):
                raise ValueError(f"{path}:{key}: n_span must have shape ({n},), got {arr.shape}")
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{path}:{key}: n_span contains non-finite values")
            banks.append((str(key), arr))
    if not banks:
        raise ValueError(f"{path}: no finite ({n},) n_span bank found")
    unique = {arr.tobytes() for _, arr in banks}
    if len(unique) != 1:
        names = ", ".join(key for key, _ in banks)
        raise ValueError(f"{path}: expected exactly one unique finite ({n},) n_span bank; found {len(unique)} across {names}")
    return banks[0][1]


def _load_behavior(path: Path) -> dict[str, list[np.ndarray]]:
    """Load all checkpoint family/template matrices without subset averaging."""

    if not path.is_file():
        raise FileNotFoundError(f"required checkpoint behavior file is missing: {path}")
    try:
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read checkpoint behavior file {path}") from exc
    if not isinstance(data, Mapping):
        raise ValueError(f"{path}: behavior payload must be an object")
    out: dict[str, list[np.ndarray]] = {}
    for family in CHECKPOINT_FAMILIES:
        matrices: list[np.ndarray] = []
        for template in CHECKPOINT_TEMPLATES:
            key = f"{family}__t{template}"
            if key not in data:
                raise ValueError(f"{path}: missing required behavior key {key}")
            value = data[key]
            if not isinstance(value, Mapping) or "total" not in value:
                raise ValueError(f"{path}: behavior key {key} lacks total matrix")
            try:
                matrix = np.asarray(value["total"], dtype=float)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{key}: total matrix is not numeric") from exc
            if matrix.shape != (n, n):
                raise ValueError(f"{path}:{key}: expected shape ({n}, {n}), got {matrix.shape}")
            if not np.all(np.isfinite(matrix)):
                raise ValueError(f"{path}:{key}: total matrix contains non-finite values")
            matrices.append(matrix)
        out[family] = matrices
    return out


def _load_corpus() -> dict[str, np.ndarray]:
    """Load checkpoint corpus inputs with exact finite non-negative shapes."""

    path = Path("results/phase5/cond_wikipedia.npz")
    if not path.is_file():
        raise FileNotFoundError(f"required checkpoint corpus file is missing: {path}")
    try:
        with np.load(path, allow_pickle=False) as archive:
            out: dict[str, np.ndarray] = {}
            for key, shape in (("uni", (n,)), ("A_win40", (n, n)), ("D_doc", (n, n))):
                if key not in archive.files:
                    raise ValueError(f"{path}: missing required input {key}")
                try:
                    value = np.asarray(archive[key], dtype=float)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{path}:{key}: input is not numeric") from exc
                if value.shape != shape:
                    raise ValueError(f"{path}:{key}: expected shape {shape}, got {value.shape}")
                if not np.all(np.isfinite(value)) or np.any(value < 0):
                    raise ValueError(f"{path}:{key}: input must be finite and non-negative")
                out[key] = value
    except OSError as exc:
        raise ValueError(f"cannot read checkpoint corpus file {path}") from exc
    return out


def _partial(values: np.ndarray, target: np.ndarray, controls: list[np.ndarray]) -> float | None:
    y = rankdata(np.asarray(values, dtype=float))
    g = rankdata(np.asarray(target, dtype=float))
    X = np.column_stack([np.ones(len(y))] + [rankdata(np.asarray(c, dtype=float)) for c in controls])
    ry = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    rg = g - X @ np.linalg.lstsq(X, g, rcond=None)[0]
    den = np.linalg.norm(ry) * np.linalg.norm(rg)
    return float(ry @ rg / den) if den > 1e-12 else None


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float | None:
    with np.errstate(all="ignore"):
        r = spearmanr(a, b).correlation
    return float(r) if r is not None and np.isfinite(r) else None


def _mc(null: list[float], observed: float) -> tuple[float | None, int | None]:
    vals = np.asarray(null, dtype=float)
    vals = vals[np.isfinite(vals)]
    if not len(vals) or not np.isfinite(observed):
        return None, None
    b = int(np.sum(vals >= observed))
    return float((b + 1) / (len(vals) + 1)), b


def main() -> int:
    zc = _load_corpus()
    uni = np.asarray(zc["uni"], dtype=float)
    tokens = _load_tokens()
    behavior_by_revision: dict[str, dict[str, list[np.ndarray]]] = {}
    for revision in REVISIONS + ["main"]:
        tag = "olmo2_1b" if revision == "main" else f"olmo2_1b_{revision}"
        behavior_by_revision[revision] = _load_behavior(Path(f"results/phase2/behavior/{tag}.json"))
    # Shared raw rich builder.  This is constructed once per frequency bank;
    # every checkpoint uses exactly these pair labels and feature names.
    fs = build_raw_features("aggregated", uni, rich=True)
    iu = np.triu_indices(n, 1)
    G = candidate_geometries(logfreq=np.log1p(uni))
    pair = {name: matrix[iu] for name, matrix in G.items()}
    controls = [pair["glyph_class"], pair["edit_distance"], pair["same_letter"], pair["alphabet"], pair["commonness"]]
    header = f"{'checkpoint (tokens)':22s} fam | line|circle circle|line | twin ECI asym | neutral residual r (rich+source-effects): A_win40 (p) D_doc (p) | KL rows"
    print(header)
    text_lines = [header]
    rows: list[dict[str, Any]] = []
    for revision in REVISIONS + ["main"]:
        tag = "olmo2_1b" if revision == "main" else f"olmo2_1b_{revision}"
        behavior = behavior_by_revision[revision]
        for family in CHECKPOINT_FAMILIES:
            matrices = behavior[family]
            L = np.mean(matrices, axis=0)
            L = L - _logsumexp(L, axis=1, keepdims=True)
            Sm = -(L + L.T) / 2
            d = Sm[iu]
            # This behavioural diagnostic is over unordered pairs in the
            # original 15-spelling matrix, so it retains the established
            # spelled line/circle geometries.  Only the residual-corpus test
            # below merges target classes, where it uses the shared v4.1
            # set-valued feature builder and never a class centroid.
            lc = _partial(d, pair["line_fifths"], controls + [pair["circle_fifths"]])
            cl = _partial(d, pair["circle_fifths"], controls + [pair["line_fifths"]])
            eci = float(np.mean(rankdata(d)[[np.where((iu[0] == a) & (iu[1] == b))[0][0] for a, b in ENH_PAIRS]] / len(d)))
            asym = float(np.mean([abs(L[i, a] - L[i, b]) for a, b in ENH_PAIRS for i in range(n)]))
            logQ = _merge_columns(L)
            qpair = logQ[fs.source, fs.target]
            qres, projection_meta = residual_projection(qpair, fs, token_counts=tokens, source_effects=True, relative_tolerance=RANK_RTOL)
            residuals: dict[str, Any] = {}
            for extraction in ("A_win40", "D_doc"):
                M = np.asarray(zc[extraction], dtype=float)
                R = (M + 0.5) / np.sum(M + 0.5, axis=1, keepdims=True)
                cpair = _merge_columns(np.log(R))[fs.source, fs.target]
                cres, _ = residual_projection(cpair, fs, token_counts=tokens, source_effects=True, relative_tolerance=RANK_RTOL)
                rr = _safe_corr(qres, cres)
                labels = {"corpus": "wikipedia", "model": tag, "family": family, "extraction": extraction, "view": "aggregated", "templates": "all", "analysis": "checkpoint_nonheldout_rich_source_effects"}
                seed = stable_seed(MASTER_SEED, **labels, stream="checkpoint_residual", replicate=0)
                null: list[float] = []
                for draw in range(N_PERM):
                    rng = np.random.default_rng(stable_seed(MASTER_SEED, **labels, stream="checkpoint_residual", replicate=draw))
                    p = rng.permutation(n)
                    Mp = M[np.ix_(p, p)]
                    Rp = (Mp + 0.5) / np.sum(Mp + 0.5, axis=1, keepdims=True)
                    cp = _merge_columns(np.log(Rp))[fs.source, fs.target]
                    crp, _ = residual_projection(cp, fs, token_counts=tokens, source_effects=True, relative_tolerance=RANK_RTOL)
                    rnull = _safe_corr(qres, crp)
                    if rnull is not None:
                        null.append(rnull)
                pval, bcount = _mc(null, rr if rr is not None else float("nan"))
                residuals[extraction] = {"r": rr, "p": pval, "b": bcount, "B": len(null), "B_requested": N_PERM, "tail": ">=", "estimator": "(b+1)/(B+1)", "seed": seed, "randomization_labels": labels, "projection": projection_meta}
            entropy = float(np.mean([-(np.exp(L[i]) * L[i]).sum() for i in range(n)]))
            label = revision.replace("stage1-step", "s1 ").replace("stage2-ingredient3-step", "s2i3 ").replace("stage2-ingredient1-step", "s2i1 ").replace("stage2-ingredient2-step", "s2i2 ").replace("-tokens", " ") if revision != "main" else "released (other run)"
            line = f"{label:22s} {family[:1]} | {lc if lc is not None else float('nan'):+.2f} {cl if cl is not None else float('nan'):+.2f} | {eci:.2f} {asym:.2f} | " + " ".join(f"{residuals[x]['r'] if residuals[x]['r'] is not None else float('nan'):+.2f} ({residuals[x]['p'] if residuals[x]['p'] is not None else float('nan'):.3f})" for x in ("A_win40", "D_doc")) + f" | row entropy {entropy:.2f}"
            print(line)
            text_lines.append(line)
            available = all(residuals[x]["r"] is not None and residuals[x]["p"] is not None for x in ("A_win40", "D_doc"))
            rows.append({"schema": SCHEMA_VERSION, "status": "OK" if available else "UNAVAILABLE", "revision": revision, "family": family, "line_given_circle": lc, "circle_given_line": cl, "eci": eci, "twin_asymmetry": asym, "residual": residuals, "entropy": entropy, "feature_names": list(fs.names), "projection_label": "non-held-out rich+source-effects", "randomization_labels": labels, "seed_algorithm": SEED_ALGORITHM, "master_seed": MASTER_SEED})
    dump_json_strict(rows, "results/phase5/ckpt_fingerprint_v4.json")
    Path("results/phase5/ckpt_trajectory_v4.txt").write_text("\n".join(text_lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
