"""Exhaustive, auditable Phase-V v3-to-v4 result comparison.

The script compares every inherited fingerprint/checkpoint artifact affected by
the corrected theory model and writes both a long-form CSV and a concise
Markdown roll-up.  It deliberately includes null cells and significance losses.

Usage: ``python -m phase5.compare_v3_v4``.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from phase5.validate_v4 import validate_checkpoint_file, validate_checkpoint_twins_file, validate_file


ROOT = Path("results/phase5")
FP = ROOT / "fingerprint"
CSV_PATH = ROOT / "v3_v4_comparison.csv"
MD_PATH = ROOT / "v3_v4_comparison.md"
THINNING_CSV_PATH = ROOT / "thinning_seed_variance.csv"
ALPHA = 0.05


@dataclass(frozen=True)
class PairSpec:
    group: str
    v3: Path
    v4: Path
    dependent_outputs: str


def _specs() -> list[PairSpec]:
    specs = [
        PairSpec("wikipedia_spelled_base", FP / "wikipedia_v3.json", FP / "wikipedia_v4.json", "abstract; Results at a Glance; Section 7; Figure 5; Table C.2"),
        PairSpec("wikipedia_aggregated_base", FP / "wikipedia_v3_neutral.json", FP / "wikipedia_v4_neutral.json", "abstract; Results at a Glance; central claim; Section 7; Figure 5; Table C.2"),
        PairSpec("wikipedia_spelled_rich", FP / "wikipedia_v3_rich.json", FP / "wikipedia_v4_rich.json", "Section 7; Figure 5"),
        PairSpec("wikipedia_aggregated_rich", FP / "wikipedia_v3_neutral_rich.json", FP / "wikipedia_v4_neutral_rich.json", "abstract; Results at a Glance; Section 7; Figure 5; Table C.2"),
        PairSpec("wikipedia_spelled_target_prior", FP / "wikipedia_v3_tp.json", FP / "wikipedia_v4_tp.json", "Section 7 target-prior robustness"),
        PairSpec("wikipedia_aggregated_target_prior", FP / "wikipedia_v3_neutral_tp.json", FP / "wikipedia_v4_neutral_tp.json", "Section 7; Table C.2"),
        PairSpec("wikipedia_aggregated_rich_target_prior", FP / "wikipedia_v3_neutral_rich_tp.json", FP / "wikipedia_v4_neutral_rich_tp.json", "Section 7 rich target-prior robustness"),
        PairSpec("wikipedia_document_bootstrap", FP / "wikipedia_v3_neutral_docboot.json", FP / "wikipedia_v4_neutral_docboot.json", "Section 7 bootstrap intervals; Table C.2"),
    ]
    for kind in ("t", "lo"):
        for index in range(4):
            specs.append(
                PairSpec(
                    f"template_{kind}{index}",
                    FP / f"wikipedia_v3_neutral_{kind}{index}.json",
                    FP / f"wikipedia_v4_neutral_{kind}{index}.json",
                    "template robustness prose; Table C.7",
                )
            )
    for operator in ("sym", "rev", "pmi"):
        for suffix, label in (("", "spelled"), ("_neutral", "aggregated"), ("_neutral_tp", "aggregated_target_prior")):
            specs.append(
                PairSpec(
                    f"matched_{operator}_{label}",
                    FP / f"matched_{operator}{suffix}.json",
                    FP / f"matched_{operator}_v4{suffix}.json",
                    "Section 6 matched-operator comparison",
                )
            )
    corpora = ("olmomix_wiki", "olmomix_dclm", "dolmino_dclm", "olmomix_dclm_big")
    for corpus in corpora:
        for suffix, label in (("", "spelled"), ("_neutral", "aggregated")):
            specs.append(
                PairSpec(
                    f"crosscorpus_{corpus}_{label}",
                    FP / f"{corpus}{suffix}.json",
                    FP / f"{corpus}_v4{suffix}.json",
                    "Section 8; Figure C.1; Table C.3",
                )
            )
            specs.append(
                PairSpec(
                    f"thinned_wikipedia_{corpus}_{label}",
                    FP / f"wikipedia_thin_{corpus}{suffix}.json",
                    FP / f"wikipedia_thin_{corpus}_v4{suffix}.json",
                    "Section 8 size-matched baseline; Figure C.1; Table C.3",
                )
            )
    return specs


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
        return float(value)
    return None


def _nested(record: Mapping[str, Any], *parts: str) -> Any:
    current: Any = record
    for part in parts:
        if not isinstance(current, Mapping):
            return None
        current = current.get(part)
    return current


def _sig(p: float | None) -> bool | None:
    return None if p is None else bool(p < ALPHA)


def _sig_change(old_p: float | None, new_p: float | None) -> str:
    old, new = _sig(old_p), _sig(new_p)
    if old is None or new is None:
        return "not_applicable"
    if old and not new:
        return "lost_significance"
    if not old and new:
        return "gained_significance"
    return "stayed_significant" if old else "stayed_nonsignificant"


def _row(spec: PairSpec, cell: str, metric: str, old: Any, new: Any, old_p: Any = None, new_p: Any = None) -> dict[str, Any] | None:
    v3, v4 = _number(old), _number(new)
    if v3 is None and v4 is None:
        return None
    signed_delta = (v4 - v3) if v3 is not None and v4 is not None else None
    absolute_delta = abs(signed_delta) if signed_delta is not None else None
    relative = (signed_delta / abs(v3)) if signed_delta is not None and abs(v3) > 1e-12 else None
    p3, p4 = _number(old_p), _number(new_p)
    sig_change = _sig_change(p3, p4)
    sign_change = bool(v3 is not None and v4 is not None and v3 * v4 < 0)
    changed = bool(absolute_delta is None or absolute_delta != 0.0 or sig_change in ("lost_significance", "gained_significance"))
    return {
        "group": spec.group,
        "v3_artifact": str(spec.v3),
        "v4_artifact": str(spec.v4),
        "cell": cell,
        "metric": metric,
        "v3_value": v3,
        "v4_value": v4,
        "signed_change": signed_delta,
        "absolute_change": absolute_delta,
        "relative_change": relative,
        "old_p": p3,
        "new_p": p4,
        "old_significant_p_lt_0_05": _sig(p3),
        "new_significant_p_lt_0_05": _sig(p4),
        "significance_change": sig_change,
        "effect_sign_change": sign_change,
        "dependent_outputs": spec.dependent_outputs,
        "claim_action": "REGENERATE_OR_REAUDIT" if changed else "NO_NUMERICAL_CHANGE",
    }


def _twin_rows(spec: PairSpec, cell: str, old: Any, new: Any) -> Iterable[dict[str, Any]]:
    if not isinstance(old, list) or not isinstance(new, list):
        return []
    rows: list[dict[str, Any]] = []
    for index, (a, b) in enumerate(zip(old, new)):
        if isinstance(a, list):
            old_values = {"cosine": a[0], "line_controlled": a[1], "p": a[2], "spearman": a[3]}
        elif isinstance(a, Mapping):
            old_values = a
        else:
            continue
        if isinstance(b, list):
            new_values = {"cosine": b[0], "line_controlled": b[1], "p": b[2], "spearman": b[3]}
        elif isinstance(b, Mapping):
            new_values = b
        else:
            continue
        identity = new_values.get("source_a")
        if identity is not None:
            identity = f"{identity}|{new_values.get('source_b')}"
        else:
            identity = str(index)
        for metric in ("cosine", "line_controlled", "spearman"):
            item = _row(spec, cell, f"twin[{identity}].{metric}", old_values.get(metric), new_values.get(metric), old_values.get("p") if metric == "line_controlled" else None, new_values.get("p") if metric == "line_controlled" else None)
            if item:
                rows.append(item)
    return rows


def _fingerprint_rows(spec: PairSpec) -> list[dict[str, Any]]:
    validate_file(spec.v4)
    old = json.load(spec.v3.open(encoding="utf-8"))
    new = json.load(spec.v4.open(encoding="utf-8"))
    if set(old) != set(new):
        missing = sorted(set(old) - set(new))
        added = sorted(set(new) - set(old))
        raise ValueError(f"{spec.group}: cell mismatch missing={missing}, added={added}")
    rows: list[dict[str, Any]] = []
    effects = (("resid_r", "resid_p"), ("dcv", "dcv_p"), ("r2gain", "r2gain_p"), ("dkl", "dkl_p"))
    for cell in sorted(old):
        a, b = old[cell], new[cell]
        if a.get("status") == "UNAVAILABLE" or b.get("status") == "UNAVAILABLE":
            if a.get("status") != b.get("status"):
                rows.append(
                    {
                        "group": spec.group,
                        "v3_artifact": str(spec.v3),
                        "v4_artifact": str(spec.v4),
                        "cell": cell,
                        "metric": "availability",
                        "v3_value": None,
                        "v4_value": None,
                        "signed_change": None,
                        "absolute_change": None,
                        "relative_change": None,
                        "old_p": None,
                        "new_p": None,
                        "old_significant_p_lt_0_05": None,
                        "new_significant_p_lt_0_05": None,
                        "significance_change": "not_applicable",
                        "effect_sign_change": False,
                        "dependent_outputs": spec.dependent_outputs,
                        "claim_action": f"AVAILABILITY_CHANGED:{a.get('status')}->{b.get('status')}",
                    }
                )
            continue
        for metric, pfield in effects:
            item = _row(spec, cell, metric, a.get(metric), b.get(metric), a.get(pfield), b.get(pfield))
            if item:
                rows.append(item)
        for parent, metrics in (("loo", ("theory", "corpus", "both")), ("kl", ("theory", "corpus", "both"))):
            for metric in metrics:
                item = _row(spec, cell, f"{parent}.{metric}", _nested(a, parent, metric), _nested(b, parent, metric))
                if item:
                    rows.append(item)
        old_rows, new_rows = _nested(a, "loo", "kl_rows"), _nested(b, "loo", "kl_rows")
        if isinstance(old_rows, Mapping) and isinstance(new_rows, Mapping):
            for model in ("theory", "corpus", "both"):
                for index, (x, y) in enumerate(zip(old_rows.get(model, []), new_rows.get(model, []))):
                    item = _row(spec, cell, f"loo.kl_rows.{model}[{index}]", x, y)
                    if item:
                        rows.append(item)
        for metric in ("sd", "q025", "q975"):
            item = _row(spec, cell, f"docboot.{metric}", _nested(a, "docboot", metric), _nested(b, "docboot", metric))
            if item:
                rows.append(item)
        item = _row(spec, cell, "pairs", a.get("pairs"), b.get("pairs"))
        if item:
            rows.append(item)
        rows.extend(_twin_rows(spec, cell, a.get("twins"), b.get("twins")))
    return rows


def _checkpoint_rows() -> list[dict[str, Any]]:
    spec = PairSpec("checkpoint_trajectory", ROOT / "ckpt_fingerprint.json", ROOT / "ckpt_fingerprint_v4.json", "Section 10; Figure 7; Table C.5")
    validate_checkpoint_file(spec.v4)
    old = json.load(spec.v3.open(encoding="utf-8"))
    new = json.load(spec.v4.open(encoding="utf-8"))
    old_by = {(x["rev"], x["fam"]): x for x in old}
    new_by = {(x.get("revision", x.get("rev")), x.get("family", x.get("fam"))): x for x in new}
    if set(old_by) != set(new_by):
        raise ValueError("checkpoint row identity changed")
    rows: list[dict[str, Any]] = []
    for identity in sorted(old_by):
        a, b = old_by[identity], new_by[identity]
        cell = "|".join(identity)
        aliases = (("line_given_circle", "lc"), ("circle_given_line", "cl"), ("eci", "eci"), ("twin_asymmetry", "asym"), ("entropy", "entropy"))
        for new_name, old_name in aliases:
            item = _row(spec, cell, new_name, a.get(old_name), b.get(new_name, b.get(old_name)))
            if item:
                rows.append(item)
        for extraction in ("A_win40", "D_doc"):
            av = a.get("resid", {}).get(extraction)
            bv = b.get("residual", b.get("resid", {})).get(extraction)
            old_r, old_p = (av[0], av[1]) if isinstance(av, list) else (av.get("r"), av.get("p"))
            new_r, new_p = (bv[0], bv[1]) if isinstance(bv, list) else (bv.get("r"), bv.get("p"))
            item = _row(spec, cell, f"residual.{extraction}", old_r, new_r, old_p, new_p)
            if item:
                rows.append(item)
    return rows


def _checkpoint_twin_rows() -> list[dict[str, Any]]:
    spec = PairSpec("checkpoint_twin_differences", ROOT / "ckpt_twins.json", ROOT / "ckpt_twins_v4.json", "Section 10 twin-stability sentence; checkpoint audit")
    validate_checkpoint_twins_file(spec.v4)
    old = json.load(spec.v3.open(encoding="utf-8"))
    new = json.load(spec.v4.open(encoding="utf-8"))
    old_by = {(row["rev"], row["fam"]): row for row in old}
    new_by = {(row["revision"], row["family"]): row for row in new}
    if set(old_by) != set(new_by):
        raise ValueError("checkpoint-twin row identity changed")
    rows: list[dict[str, Any]] = []
    for identity in sorted(old_by):
        a, b = old_by[identity], new_by[identity]
        for pair_name in sorted(a["pairs"]):
            av, bv = a["pairs"][pair_name], b["pairs"][pair_name]
            cell = "|".join(identity + (pair_name,))
            for metric, old_name, new_name in (("magnitude", "mag", "mean_absolute_difference"), ("stability", "stab", "stability_to_released")):
                item = _row(spec, cell, metric, av.get(old_name), bv.get(new_name))
                if item:
                    rows.append(item)
            for extraction in ("A_win40", "D_doc"):
                old_test = av.get(extraction)
                new_test = bv.get("extractions", {}).get(extraction)
                for metric, index, new_name in (("raw_cosine", 0, "raw_cosine"), ("line_controlled_cosine", 1, "line_controlled_cosine")):
                    item = _row(spec, cell, f"{extraction}.{metric}", old_test[index], new_test.get(new_name), old_test[2] if metric == "line_controlled_cosine" else None, new_test.get("p") if metric == "line_controlled_cosine" else None)
                    if item:
                        rows.append(item)
    return rows


def _v4_only_file_rows(path: Path, group: str, dependent_outputs: str, *, expected_cells: int) -> list[dict[str, Any]]:
    validate_file(path, expected_cells=expected_cells)
    payload = json.load(path.open(encoding="utf-8"))
    spec = PairSpec(group, Path("N/A"), path, dependent_outputs)
    rows: list[dict[str, Any]] = []
    for cell, record in sorted(payload.items()):
        for metric, pfield in (("resid_r", "resid_p"), ("dcv", "dcv_p"), ("r2gain", "r2gain_p"), ("dkl", "dkl_p")):
            item = _row(spec, cell, metric, None, record.get(metric), None, record.get(pfield))
            if item:
                rows.append(item)
    return rows


def _v4_only_rows() -> list[dict[str, Any]]:
    rows = _v4_only_file_rows(
        FP / "wikipedia_v4_rich_tp.json",
        "wikipedia_spelled_rich_target_prior_v4_only",
        "Section 7 supplementary robustness audit",
        expected_cells=16,
    )
    rows.extend(
        _v4_only_file_rows(
            FP / "wikipedia_v4_neutral_rich_docboot.json",
            "wikipedia_aggregated_rich_document_bootstrap_v4_only",
            "Section 7 rich-baseline bootstrap intervals; Table C.2",
            expected_cells=16,
        )
    )
    return rows


def _thinning_seed_rows() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for corpus in ("olmomix_wiki", "olmomix_dclm", "dolmino_dclm", "olmomix_dclm_big"):
        for suffix, view in (("", "spelled"), ("_neutral", "aggregated")):
            paths = [FP / f"wikipedia_thin_{corpus}_v4{suffix}.json"] + [FP / f"wikipedia_thin_{corpus}_v4_s{seed}{suffix}.json" for seed in range(1, 5)]
            payloads = []
            for path in paths:
                validate_file(path, expected_cells=36)
                payloads.append(json.load(path.open(encoding="utf-8")))
            cells = set(payloads[0])
            _require_same = all(set(payload) == cells for payload in payloads)
            if not _require_same:
                raise ValueError(f"thinning seed cell mismatch for {corpus} {view}")
            for cell in sorted(cells):
                for metric in ("resid_r", "resid_p", "dcv", "dcv_p", "r2gain", "r2gain_p", "dkl", "dkl_p"):
                    values = [_number(payload[cell].get(metric)) for payload in payloads]
                    finite = [value for value in values if value is not None]
                    if len(finite) != 5:
                        continue
                    mean = sum(finite) / len(finite)
                    variance = sum((value - mean) ** 2 for value in finite) / (len(finite) - 1)
                    records.append(
                        {
                            "corpus": corpus,
                            "view": view,
                            "cell": cell,
                            "metric": metric,
                            "n_seeds": 5,
                            "mean": mean,
                            "sd": math.sqrt(variance),
                            "min": min(finite),
                            "max": max(finite),
                            "seed_artifacts": ";".join(str(path) for path in paths),
                        }
                    )
    with THINNING_CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        fields = ("corpus", "view", "cell", "metric", "n_seeds", "mean", "sd", "min", "max", "seed_artifacts")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    return records


FIELDS = (
    "group", "v3_artifact", "v4_artifact", "cell", "metric", "v3_value", "v4_value", "signed_change", "absolute_change", "relative_change",
    "old_p", "new_p", "old_significant_p_lt_0_05", "new_significant_p_lt_0_05", "significance_change", "effect_sign_change",
    "dependent_outputs", "claim_action",
)


def _format(value: Any) -> str:
    return "—" if value is None else f"{float(value):+.5f}"


def _write_markdown(rows: list[dict[str, Any]], specs: list[PairSpec], thinning: list[dict[str, Any]]) -> None:
    core_groups = ("wikipedia_aggregated_base", "wikipedia_aggregated_rich", "wikipedia_aggregated_target_prior", "wikipedia_aggregated_rich_target_prior", "wikipedia_spelled_rich_target_prior_v4_only")
    core = [r for r in rows if r["group"] in core_groups and r["metric"] == "dkl" and "|E_modulation|" in r["cell"] and r["cell"].endswith(("A_win40", "D_doc"))]
    flips = [r for r in rows if r["significance_change"] in ("lost_significance", "gained_significance")]
    lines = [
        "# Phase-V v3 to v4 comparison",
        "",
        f"Compared {len(specs)} inherited fingerprint artifact pairs, the checkpoint trajectory and twin controls, plus v4-only rich-target-prior and rich document-bootstrap grids: {len(rows)} numerical rows. Significance uses strict p < {ALPHA}.",
        "",
        "## Core target-aggregated modulation cells",
        "",
        "| baseline | cell | v3 DKL | v4 DKL | v3 p | v4 p | status |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in sorted(core, key=lambda x: (x["group"], x["cell"])):
        lines.append(f"| {row['group']} | `{row['cell']}` | {_format(row['v3_value'])} | {_format(row['v4_value'])} | {_format(row['old_p'])} | {_format(row['new_p'])} | {row['significance_change']} |")
    lines.extend(["", "## Significance changes across all inherited cells", ""])
    if flips:
        lines.extend(["| group | cell | metric | v3 | v4 | v3 p | v4 p | change |", "|---|---|---|---:|---:|---:|---:|---|"])
        for row in sorted(flips, key=lambda x: (x["group"], x["cell"], x["metric"])):
            lines.append(f"| {row['group']} | `{row['cell']}` | {row['metric']} | {_format(row['v3_value'])} | {_format(row['v4_value'])} | {_format(row['old_p'])} | {_format(row['new_p'])} | {row['significance_change']} |")
    else:
        lines.append("No significance changes.")
    lines.extend(["", "## Coverage by inherited artifact group", "", "| group | compared rows | value changes | significance losses | significance gains | dependent outputs |", "|---|---:|---:|---:|---:|---|"])
    groups = sorted({r["group"] for r in rows})
    for group in groups:
        subset = [r for r in rows if r["group"] == group]
        changed = sum(r["absolute_change"] is None or r["absolute_change"] != 0 for r in subset)
        losses = sum(r["significance_change"] == "lost_significance" for r in subset)
        gains = sum(r["significance_change"] == "gained_significance" for r in subset)
        lines.append(f"| {group} | {len(subset)} | {changed} | {losses} | {gains} | {subset[0]['dependent_outputs']} |")
    lines.extend(["", "## Five-seed thinning variance", "", "| corpus | view | maximum DKL SD | median DKL SD |", "|---|---|---:|---:|"])
    for corpus in ("olmomix_wiki", "olmomix_dclm", "dolmino_dclm", "olmomix_dclm_big"):
        for view in ("spelled", "aggregated"):
            values = sorted(row["sd"] for row in thinning if row["corpus"] == corpus and row["view"] == view and row["metric"] == "dkl")
            median = values[len(values) // 2] if len(values) % 2 else (values[len(values) // 2 - 1] + values[len(values) // 2]) / 2
            lines.append(f"| {corpus} | {view} | {max(values):.6f} | {median:.6f} |")
    lines.extend(["", f"All per-cell five-seed means, SDs, minima and maxima are in `{THINNING_CSV_PATH}`.", "", "The long-form comparison CSV contains the per-row KL values, null p-values, twin diagnostics, bootstrap limits, and all component scores; no unstable or null cell is omitted.", ""])
    MD_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    specs = _specs()
    missing = [str(path) for spec in specs for path in (spec.v3, spec.v4) if not path.exists()]
    for path in (ROOT / "ckpt_fingerprint.json", ROOT / "ckpt_fingerprint_v4.json", ROOT / "ckpt_twins.json", ROOT / "ckpt_twins_v4.json", FP / "wikipedia_v4_rich_tp.json"):
        if not path.exists():
            missing.append(str(path))
    if missing:
        raise FileNotFoundError("missing comparison artifacts: " + ", ".join(sorted(set(missing))))
    rows: list[dict[str, Any]] = []
    for spec in specs:
        rows.extend(_fingerprint_rows(spec))
    rows.extend(_checkpoint_rows())
    rows.extend(_checkpoint_twin_rows())
    rows.extend(_v4_only_rows())
    thinning = _thinning_seed_rows()
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    _write_markdown(rows, specs, thinning)
    print(f"wrote {CSV_PATH} ({len(rows)} rows), {MD_PATH}, and {THINNING_CSV_PATH} ({len(thinning)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
