"""Deterministic cross-corpus specificity report for the corrected Phase-V run.

The report compares each full corpus with five independently thinned,
size-matched Wikipedia baselines.  The direct and baseline grids are required
to be complete v4 grids (36 cells, each with 15 held-out source rows); a
pre-correction, partial, or stale artifact therefore fails closed before any
summary is written.

With no arguments both the spelled and target-aggregated views are written.
The view can be selected with ``spelled``, ``aggregated``, or
``--view=<view>``.  This module does not generate or alter fingerprint
artifacts; it only reads the final v4 inputs and writes the two report files
for the selected view.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import wilcoxon

from phase5.theory_features import SCHEMA_VERSION, dump_json_strict


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FINGERPRINT_ROOT = PROJECT_ROOT / "results" / "phase5" / "fingerprint"
REPORT_ROOT = PROJECT_ROOT / "results" / "phase5"

CORPORA = (
    "olmomix_wiki",
    "olmomix_dclm",
    "dolmino_dclm",
    "olmomix_dclm_big",
)
MODELS = (
    "olmo2_1b",
    "gemma2_2b",
    "qwen25_3b",
    "olmo2_7b",
)
OLMO_MODELS = ("olmo2_1b", "olmo2_7b")
OTHER_MODELS = ("gemma2_2b", "qwen25_3b")
FAMILIES = ("C_harmonic", "D_chord", "E_modulation")
EXTRACTIONS = ("A_win40", "B_any", "D_doc")
N_CELLS = len(MODELS) * len(FAMILIES) * len(EXTRACTIONS)
N_ROWS = 15
N_BASELINE_SEEDS = 5

EXPECTED_CELLS = tuple(
    f"{model}|{family}|{extraction}"
    for model in MODELS
    for family in FAMILIES
    for extraction in EXTRACTIONS
)


def _reject_json_constant(value: str) -> None:
    """Reject NaN/Infinity in input artifacts instead of silently accepting it."""

    raise ValueError(f"non-standard JSON numeric constant in v4 input: {value}")


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _as_float(value: Any, *, label: str) -> float:
    if not _finite(value):
        raise ValueError(f"{label}: expected a finite numeric value, got {value!r}")
    return float(value)


def _load_json(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing v4 input grid: {path}")
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle, parse_constant=_reject_json_constant)
    if not isinstance(payload, Mapping) or not payload:
        raise ValueError(f"{path}: expected a non-empty object grid")
    return payload


def _finite_rows(record: Mapping[str, Any], *, path: Path, cell: str, component: str) -> np.ndarray:
    loo = record.get("loo")
    if not isinstance(loo, Mapping):
        raise ValueError(f"{path}:{cell}: missing loo object")
    audits = loo.get("rank_audits")
    if not isinstance(audits, list) or len(audits) != N_ROWS:
        raise ValueError(f"{path}:{cell}: missing the {N_ROWS} held-out fold audits needed to align rows")
    heldout = [audit.get("heldout_source") if isinstance(audit, Mapping) else None for audit in audits]
    if heldout != list(range(N_ROWS)):
        raise ValueError(f"{path}:{cell}: held-out row order is not source rows 0..{N_ROWS - 1}")
    kl_rows = loo.get("kl_rows")
    if not isinstance(kl_rows, Mapping):
        raise ValueError(f"{path}:{cell}: missing loo.kl_rows")
    values = kl_rows.get(component)
    if not isinstance(values, (list, tuple)) or len(values) != N_ROWS:
        raise ValueError(f"{path}:{cell}: loo.kl_rows.{component} must contain exactly {N_ROWS} values")
    if not all(_finite(value) for value in values):
        raise ValueError(f"{path}:{cell}: loo.kl_rows.{component} contains a non-finite value")
    return np.asarray(values, dtype=float)


def _validate_grid(path: Path, *, view: str) -> Mapping[str, Any]:
    """Load and validate one direct or thinned v4 fingerprint grid."""

    payload = _load_json(path)
    keys = set(payload)
    expected = set(EXPECTED_CELLS)
    if len(payload) != N_CELLS or keys != expected:
        missing = sorted(expected - keys)
        extra = sorted(keys - expected)
        raise ValueError(
            f"{path}: expected exactly {N_CELLS} v4 cells; got {len(payload)} "
            f"(missing={missing}, extra={extra})"
        )

    for cell in EXPECTED_CELLS:
        record = payload[cell]
        if not isinstance(record, Mapping):
            raise ValueError(f"{path}:{cell}: cell is not an object")
        if record.get("status") != "OK":
            raise ValueError(f"{path}:{cell}: v4 cell is not OK ({record.get('status')!r})")
        if record.get("schema") != SCHEMA_VERSION:
            raise ValueError(
                f"{path}:{cell}: expected corrected schema {SCHEMA_VERSION!r}, "
                f"got {record.get('schema')!r}"
            )
        if record.get("view") != view:
            raise ValueError(f"{path}:{cell}: expected view {view!r}, got {record.get('view')!r}")
        if record.get("rich") is not False or record.get("targetprior") is not False:
            raise ValueError(f"{path}:{cell}: cross-corpus input must be the base v4 grid")

        for field in ("resid_r", "resid_p", "dkl", "dkl_p", "r2gain", "r2gain_p"):
            _as_float(record.get(field), label=f"{path}:{cell}.{field}")
        _finite_rows(record, path=path, cell=cell, component="theory")
        _finite_rows(record, path=path, cell=cell, component="both")

    return payload


def _wilcoxon(values: Sequence[float], *, label: str) -> dict[str, Any]:
    """Return a reproducible paired, two-sided Wilcoxon summary.

    SciPy reports a NaN p-value for an all-zero vector.  In that degenerate
    case the paired difference is exactly zero for every source row, so the
    two-sided test is represented explicitly as statistic=0, p=1 rather than
    propagating an undefined JSON number.
    """

    array = np.asarray(values, dtype=float)
    if array.shape != (N_ROWS,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{label}: Wilcoxon input must be {N_ROWS} finite paired values")
    nonzero = np.abs(array) > 0.0
    if not np.any(nonzero):
        return {
            "statistic": 0.0,
            "p": 1.0,
            "n": int(array.size),
            "n_nonzero": 0,
            "alternative": "two-sided",
            "zero_method": "wilcox",
            "method": "auto",
        }
    result = wilcoxon(array, zero_method="wilcox", alternative="two-sided", method="auto")
    statistic = _as_float(result.statistic, label=f"{label}.statistic")
    p = _as_float(result.pvalue, label=f"{label}.p")
    return {
        "statistic": statistic,
        "p": p,
        "n": int(array.size),
        "n_nonzero": int(np.count_nonzero(nonzero)),
        "alternative": "two-sided",
        "zero_method": "wilcox",
        "method": "auto",
    }


def _row_dkl(record: Mapping[str, Any], *, path: Path, cell: str) -> np.ndarray:
    theory = _finite_rows(record, path=path, cell=cell, component="theory")
    both = _finite_rows(record, path=path, cell=cell, component="both")
    result = theory - both
    if result.shape != (N_ROWS,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{path}:{cell}: direct row DKL is not finite for all {N_ROWS} rows")
    return result


def _baseline_summary(records: Sequence[Mapping[str, Any]], *, cell: str) -> dict[str, Any]:
    values = np.asarray([_as_float(record.get("dkl"), label=f"baseline:{cell}.dkl") for record in records], dtype=float)
    p_values = np.asarray([_as_float(record.get("dkl_p"), label=f"baseline:{cell}.dkl_p") for record in records], dtype=float)
    if values.shape != (N_BASELINE_SEEDS,) or p_values.shape != (N_BASELINE_SEEDS,):
        raise ValueError(f"baseline:{cell}: expected {N_BASELINE_SEEDS} seeds")
    return {
        "n_seeds": N_BASELINE_SEEDS,
        "dkl_values": values.tolist(),
        "dkl_p_values": p_values.tolist(),
        "dkl_mean": float(np.mean(values)),
        "dkl_sample_sd": float(np.std(values, ddof=1)),
        "dkl_min": float(np.min(values)),
        "dkl_max": float(np.max(values)),
        "n_significant_p_lt_0_05": int(np.count_nonzero(p_values < 0.05)),
    }


def _input_paths(corpus: str, view: str) -> tuple[Path, tuple[Path, ...]]:
    suffix = "" if view == "spelled" else "_neutral"
    direct = FINGERPRINT_ROOT / f"{corpus}_v4{suffix}.json"
    seed0 = FINGERPRINT_ROOT / f"wikipedia_thin_{corpus}_v4{suffix}.json"
    seeds = (seed0,) + tuple(
        FINGERPRINT_ROOT / f"wikipedia_thin_{corpus}_v4_s{seed}{suffix}.json"
        for seed in range(1, N_BASELINE_SEEDS)
    )
    return direct, seeds


def _cell_record(
    *,
    corpus: str,
    model: str,
    family: str,
    extraction: str,
    direct: Mapping[str, Any],
    direct_path: Path,
    baseline: Sequence[Mapping[str, Any]],
    baseline_paths: Sequence[Path],
) -> dict[str, Any]:
    cell = f"{model}|{family}|{extraction}"
    direct_rows = _row_dkl(direct, path=direct_path, cell=cell)
    baseline_rows = np.vstack([_row_dkl(record, path=path, cell=cell) for record, path in zip(baseline, baseline_paths)])
    baseline_row_mean = np.mean(baseline_rows, axis=0)
    row_difference = direct_rows - baseline_row_mean
    baseline_summary = _baseline_summary(baseline, cell=cell)
    return {
        "corpus": corpus,
        "model": model,
        "family": family,
        "extraction": extraction,
        "cell": cell,
        "direct": {
            "residual_r": _as_float(direct.get("resid_r"), label=f"{direct_path}:{cell}.resid_r"),
            "residual_p": _as_float(direct.get("resid_p"), label=f"{direct_path}:{cell}.resid_p"),
            "dkl": _as_float(direct.get("dkl"), label=f"{direct_path}:{cell}.dkl"),
            "p": _as_float(direct.get("dkl_p"), label=f"{direct_path}:{cell}.dkl_p"),
            "r2gain": _as_float(direct.get("r2gain"), label=f"{direct_path}:{cell}.r2gain"),
            "r2gain_p": _as_float(direct.get("r2gain_p"), label=f"{direct_path}:{cell}.r2gain_p"),
        },
        "baseline": baseline_summary,
        "specificity": {
            "direct_row_dkl": direct_rows.tolist(),
            "baseline_row_dkl_mean": baseline_row_mean.tolist(),
            "direct_minus_baseline_row_dkl": row_difference.tolist(),
            "mean": float(np.mean(row_difference)),
            "wilcoxon": _wilcoxon(row_difference, label=f"{corpus}:{cell}"),
        },
    }


def _did_record(corpus: str, family: str, extraction: str, by_model: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
    cell = f"{family}|{extraction}"
    olmo_vectors = np.asarray(
        [by_model[model]["specificity"]["direct_minus_baseline_row_dkl"] for model in OLMO_MODELS],
        dtype=float,
    )
    other_vectors = np.asarray(
        [by_model[model]["specificity"]["direct_minus_baseline_row_dkl"] for model in OTHER_MODELS],
        dtype=float,
    )
    if olmo_vectors.shape != (len(OLMO_MODELS), N_ROWS) or other_vectors.shape != (len(OTHER_MODELS), N_ROWS):
        raise ValueError(f"{corpus}:{cell}: DiD requires all four model row-difference vectors")
    olmo_mean = np.mean(olmo_vectors, axis=0)
    other_mean = np.mean(other_vectors, axis=0)
    difference = olmo_mean - other_mean
    return {
        "corpus": corpus,
        "family": family,
        "extraction": extraction,
        "cell": cell,
        "olmo_models": list(OLMO_MODELS),
        "other_models": list(OTHER_MODELS),
        "olmo_row_difference_mean": olmo_mean.tolist(),
        "other_row_difference_mean": other_mean.tolist(),
        "did_row_difference": difference.tolist(),
        "mean": float(np.mean(difference)),
        "wilcoxon": _wilcoxon(difference, label=f"{corpus}:DiD:{cell}"),
    }


def build_report(view: str) -> dict[str, Any]:
    """Read, validate, and summarise one view without writing files."""

    if view not in ("spelled", "aggregated"):
        raise ValueError(f"unknown view {view!r}; expected 'spelled' or 'aggregated'")

    direct_grids: dict[str, Mapping[str, Any]] = {}
    baseline_grids: dict[str, tuple[Mapping[str, Any], ...]] = {}
    input_records: list[dict[str, Any]] = []
    for corpus in CORPORA:
        direct_path, baseline_paths = _input_paths(corpus, view)
        direct_grids[corpus] = _validate_grid(direct_path, view=view)
        baselines = tuple(_validate_grid(path, view=view) for path in baseline_paths)
        if len(baselines) != N_BASELINE_SEEDS:
            raise ValueError(f"{corpus}:{view}: expected {N_BASELINE_SEEDS} baseline grids")
        baseline_grids[corpus] = baselines
        input_records.append(
            {
                "corpus": corpus,
                "view": view,
                "direct": {"path": str(direct_path.relative_to(PROJECT_ROOT)), "cells": N_CELLS, "rows_per_cell": N_ROWS},
                "baseline": [
                    {
                        "seed": seed,
                        "path": str(path.relative_to(PROJECT_ROOT)),
                        "cells": N_CELLS,
                        "rows_per_cell": N_ROWS,
                    }
                    for seed, path in enumerate(baseline_paths)
                ],
            }
        )

    cells: list[dict[str, Any]] = []
    by_corpus_model: dict[str, dict[str, dict[str, Any]]] = {corpus: {} for corpus in CORPORA}
    for corpus in CORPORA:
        direct = direct_grids[corpus]
        baselines = baseline_grids[corpus]
        _, baseline_paths = _input_paths(corpus, view)
        by_corpus_model[corpus] = {}
        for model in MODELS:
            by_corpus_model[corpus][model] = {}
            for family in FAMILIES:
                for extraction in EXTRACTIONS:
                    cell = f"{model}|{family}|{extraction}"
                    result = _cell_record(
                        corpus=corpus,
                        model=model,
                        family=family,
                        extraction=extraction,
                        direct=direct[cell],
                        direct_path=_input_paths(corpus, view)[0],
                        baseline=[grid[cell] for grid in baselines],
                        baseline_paths=baseline_paths,
                    )
                    cells.append(result)
                    by_corpus_model[corpus][model][f"{family}|{extraction}"] = result

    did: list[dict[str, Any]] = []
    for corpus in CORPORA:
        for family in FAMILIES:
            for extraction in EXTRACTIONS:
                by_model = {
                    model: by_corpus_model[corpus][model][f"{family}|{extraction}"]
                    for model in MODELS
                }
                did.append(_did_record(corpus, family, extraction, by_model))

    return {
        "schema": "phase5-crosscorpus-v4.1",
        "feature_schema": SCHEMA_VERSION,
        "view": view,
        "corpora": list(CORPORA),
        "models": list(MODELS),
        "families": list(FAMILIES),
        "extractions": list(EXTRACTIONS),
        "expected_cells_per_grid": N_CELLS,
        "required_rows_per_cell": N_ROWS,
        "row_order": "loo.kl_rows entries are held-out source rows 0..14",
        "baseline_seed_count": N_BASELINE_SEEDS,
        "wilcoxon": {
            "alternative": "two-sided",
            "zero_method": "wilcox",
            "method": "auto",
            "scope": "paired held-out source rows",
        },
        "inputs": input_records,
        "cells": cells,
        "difference_in_differences": did,
    }


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "NA"
    return f"{float(value):+.{digits}f}"


def _write_text(payload: Mapping[str, Any], path: Path) -> None:
    view = str(payload["view"])
    lines = [
        f"Phase-V v4 cross-corpus specificity ({view} view)",
        f"schema={payload['schema']}; feature_schema={payload['feature_schema']}",
        f"grids: {len(payload['inputs'])} corpora x one direct + {payload['baseline_seed_count']} baselines; {payload['expected_cells_per_grid']} cells/grid; {payload['required_rows_per_cell']} rows/cell",
        "",
        "Direct versus five-seed size-matched Wikipedia baseline; specificity is direct row DeltaKL minus the mean baseline row DeltaKL.",
        "corpus | model | family | extraction | direct residual_r(p) dkl(p) r2 | baseline dkl mean sd [min,max] sig/5 | row mean Wilcoxon-p",
    ]
    for item in payload["cells"]:
        direct = item["direct"]
        baseline = item["baseline"]
        specificity = item["specificity"]
        lines.append(
            f"{item['corpus']} | {item['model']} | {item['family']} | {item['extraction']} | "
            f"{_fmt(direct['residual_r'], 4)}({_fmt(direct['residual_p'], 4)}) "
            f"{_fmt(direct['dkl'], 6)}({_fmt(direct['p'], 4)}) "
            f"{_fmt(direct['r2gain'], 6)} | "
            f"{_fmt(baseline['dkl_mean'], 6)} {baseline['dkl_sample_sd']:.6f} "
            f"[{_fmt(baseline['dkl_min'], 6)},{_fmt(baseline['dkl_max'], 6)}] "
            f"{baseline['n_significant_p_lt_0_05']}/5 | "
            f"{_fmt(specificity['mean'], 6)} p={specificity['wilcoxon']['p']:.6g}"
        )
    lines.extend(
        [
            "",
            "Difference-in-differences: mean OLMo row difference minus mean Gemma/Qwen row difference; paired two-sided Wilcoxon over rows.",
            "corpus | family | extraction | DiD mean | Wilcoxon statistic | p",
        ]
    )
    for item in payload["difference_in_differences"]:
        test = item["wilcoxon"]
        lines.append(
            f"{item['corpus']} | {item['family']} | {item['extraction']} | "
            f"{_fmt(item['mean'], 6)} | {test['statistic']:.6g} | {test['p']:.6g}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(view: str) -> tuple[Path, Path]:
    payload = build_report(view)
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    stem = REPORT_ROOT / f"crosscorpus_compare_v4_{view}"
    json_path = stem.with_suffix(".json")
    txt_path = stem.with_suffix(".txt")
    dump_json_strict(payload, json_path)
    _write_text(payload, txt_path)
    return json_path, txt_path


def _parse_views(argv: Sequence[str]) -> tuple[str, ...]:
    if not argv:
        return ("spelled", "aggregated")
    views: list[str] = []
    for argument in argv:
        if argument.startswith("--view="):
            value = argument.split("=", 1)[1]
        elif argument in ("spelled", "aggregated"):
            value = argument
        else:
            raise ValueError(f"unknown argument {argument!r}; use spelled, aggregated, or --view=<view>")
        if value not in ("spelled", "aggregated"):
            raise ValueError(f"unknown view {value!r}; expected spelled or aggregated")
        if value not in views:
            views.append(value)
    return tuple(views)


def main(argv: Sequence[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    views = _parse_views(raw)
    for view in views:
        json_path, txt_path = write_report(view)
        print(f"wrote {json_path}")
        print(f"wrote {txt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
