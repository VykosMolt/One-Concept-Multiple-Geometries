"""Fail closed on incomplete or malformed Phase-V v4 fingerprint artifacts.

Usage: ``python -m phase5.validate_v4 FILE.json [FILE.json ...]``.
The validator accepts explicitly sparse cells marked ``UNAVAILABLE`` but
requires every computed cell to carry the v4.1 schema, complete finite nulls,
fold-local scaling evidence, full-rank designs, and the recorded master seed.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from phase5.theory_features import MASTER_SEED, SCHEMA_VERSION


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON numeric constant: {value}")


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate_test(test: Mapping[str, Any], label: str) -> None:
    for field in ("observed", "p"):
        _require(_finite(test.get(field)), f"{label}: non-finite {field}")
    B = test.get("B")
    requested = test.get("B_requested")
    b = test.get("b")
    _require(isinstance(B, int) and B > 0, f"{label}: invalid B")
    _require(B == requested, f"{label}: incomplete null B={B}, requested={requested}")
    _require(isinstance(b, int) and 0 <= b <= B, f"{label}: invalid exceedance count")
    expected = (b + 1) / (B + 1)
    _require(math.isclose(float(test["p"]), expected, rel_tol=0.0, abs_tol=1e-15), f"{label}: p is not (b+1)/(B+1)")
    _require(test.get("estimator") == "(b+1)/(B+1)", f"{label}: wrong estimator label")


def validate_record(record: Mapping[str, Any], label: str, *, expected_docboot: int | None = None) -> None:
    status = record.get("status")
    _require(status in ("OK", "UNAVAILABLE"), f"{label}: invalid status {status!r}")
    if status == "UNAVAILABLE":
        _require(record.get("reason") == "too_sparse", f"{label}: computed cell became unavailable")
        return
    _require(record.get("schema") == SCHEMA_VERSION, f"{label}: wrong schema")
    _require(record.get("randomization", {}).get("master_seed") == MASTER_SEED, f"{label}: wrong or missing master seed")
    _require(record.get("standardization") == "per-fold mean/scale fitted on training pair observations only; constant scale=1", f"{label}: wrong standardization record")
    for field in ("resid_r", "resid_p", "dcv", "dcv_p", "r2gain", "r2gain_p", "dkl", "dkl_p"):
        _require(_finite(record.get(field)), f"{label}: non-finite core field {field}")
    docboot = record.get("docboot")
    if expected_docboot is not None:
        _require(isinstance(expected_docboot, int) and not isinstance(expected_docboot, bool) and expected_docboot > 0, f"{label}: invalid expected document-bootstrap B")
        _require(isinstance(docboot, Mapping), f"{label}: missing document bootstrap")
    if docboot is not None:
        _require(isinstance(docboot, Mapping), f"{label}: missing document bootstrap")
        requested_docboot = docboot.get("B_requested")
        observed_docboot = docboot.get("B")
        _require(isinstance(requested_docboot, int) and not isinstance(requested_docboot, bool) and requested_docboot > 0, f"{label}: invalid requested document-bootstrap B")
        _require(isinstance(observed_docboot, int) and not isinstance(observed_docboot, bool) and observed_docboot > 0, f"{label}: invalid document-bootstrap B")
        _require(observed_docboot == requested_docboot, f"{label}: incomplete document bootstrap")
        if expected_docboot is not None:
            _require(observed_docboot == requested_docboot == expected_docboot, f"{label}: wrong document-bootstrap B")
        for field in ("sd", "q025", "q975"):
            _require(_finite(docboot.get(field)), f"{label}: non-finite document-bootstrap {field}")
        _require(
            docboot.get("rebuilds") == ["corpus_matrix", "source_frequency", "raw_features"],
            f"{label}: document bootstrap did not record required rebuilds",
        )
    tests = record.get("tests")
    _require(isinstance(tests, Mapping), f"{label}: missing null-test records")
    for name in ("residual", "dcv", "r2gain", "dkl"):
        _require(isinstance(tests.get(name), Mapping), f"{label}: missing {name} null")
        _validate_test(tests[name], f"{label}.{name}")
    loo = record.get("loo")
    _require(isinstance(loo, Mapping), f"{label}: missing LOO record")
    audits = loo.get("rank_audits")
    scalers = loo.get("fold_scalers")
    _require(isinstance(audits, list) and len(audits) == 15, f"{label}: missing 15 rank audits")
    _require(isinstance(scalers, list) and len(scalers) == 15, f"{label}: missing 15 fold scalers")
    expected_observations = 196 if record.get("view") == "spelled" else 154
    for fold, (audit, scaler) in enumerate(zip(audits, scalers)):
        _require(audit.get("heldout_source") == fold, f"{label}: rank-audit fold ordering changed")
        _require(scaler.get("heldout_source") == fold, f"{label}: scaler fold ordering changed")
        _require(scaler.get("fit_observations") == expected_observations, f"{label}: scaler includes wrong observations")
        ranks = audit.get("rank", {})
        for name in ("base", "rich", "combined"):
            _require(isinstance(ranks.get(name), Mapping) and ranks[name].get("full_rank") is True and not ranks[name].get("omissions"), f"{label}: {name} rank failure in fold {fold}")
        if record.get("targetprior"):
            tp = ranks.get("targetprior")
            _require(isinstance(tp, Mapping) and tp.get("full_rank") is True and not tp.get("omissions"), f"{label}: target-prior rank failure in fold {fold}")
            meta = audit.get("target_prior")
            _require(isinstance(meta, Mapping) and meta.get("heldout_excluded") == expected_observations // 14, f"{label}: target prior did not exclude held-out row")


def validate_file(path: str | Path, *, expected_cells: int | None = None, expected_docboot: int | None = None) -> int:
    source = Path(path)
    with source.open(encoding="utf-8") as handle:
        payload = json.load(handle, parse_constant=_reject_constant)
    _require(isinstance(payload, Mapping) and payload, f"{source}: empty or non-object artifact")
    if expected_cells is not None:
        _require(len(payload) == int(expected_cells), f"{source}: incomplete grid has {len(payload)} cells, expected {expected_cells}")
    for key, record in payload.items():
        _require(isinstance(record, Mapping), f"{source}:{key}: non-object cell")
        validate_record(record, f"{source}:{key}", expected_docboot=expected_docboot)
    return len(payload)


def validate_checkpoint_file(path: str | Path, *, expected_rows: int = 24, expected_B: int = 2000) -> int:
    source = Path(path)
    with source.open(encoding="utf-8") as handle:
        payload = json.load(handle, parse_constant=_reject_constant)
    _require(isinstance(payload, list) and len(payload) == expected_rows, f"{source}: expected {expected_rows} checkpoint rows")
    identities: set[tuple[str, str]] = set()
    for index, row in enumerate(payload):
        label = f"{source}[{index}]"
        _require(isinstance(row, Mapping) and row.get("status") == "OK", f"{label}: unavailable checkpoint row")
        _require(row.get("schema") == SCHEMA_VERSION and row.get("master_seed") == MASTER_SEED, f"{label}: wrong checkpoint provenance")
        identity = (str(row.get("revision")), str(row.get("family")))
        _require(identity not in identities, f"{label}: duplicate checkpoint identity")
        identities.add(identity)
        for extraction in ("A_win40", "D_doc"):
            test = row.get("residual", {}).get(extraction)
            _require(isinstance(test, Mapping), f"{label}: missing {extraction} residual")
            _validate_test({**test, "observed": test.get("r")}, f"{label}.{extraction}")
            _require(test.get("B") == expected_B, f"{label}.{extraction}: wrong checkpoint B")
    return len(payload)


def validate_checkpoint_twins_file(path: str | Path, *, expected_rows: int = 24, expected_B: int = 2000) -> int:
    source = Path(path)
    with source.open(encoding="utf-8") as handle:
        payload = json.load(handle, parse_constant=_reject_constant)
    _require(isinstance(payload, list) and len(payload) == expected_rows, f"{source}: expected {expected_rows} checkpoint-twin rows")
    identities: set[tuple[str, str]] = set()
    for index, row in enumerate(payload):
        label = f"{source}[{index}]"
        _require(isinstance(row, Mapping) and row.get("status") == "OK", f"{label}: unavailable checkpoint-twin row")
        _require(row.get("schema") == SCHEMA_VERSION and row.get("master_seed") == MASTER_SEED, f"{label}: wrong checkpoint-twin provenance")
        identity = (str(row.get("revision")), str(row.get("family")))
        _require(identity not in identities, f"{label}: duplicate checkpoint-twin identity")
        identities.add(identity)
        pairs = row.get("pairs")
        _require(isinstance(pairs, Mapping) and len(pairs) == 3, f"{label}: expected three twin pairs")
        for pair_name, pair in pairs.items():
            _require(_finite(pair.get("mean_absolute_difference")) and _finite(pair.get("stability_to_released")), f"{label}.{pair_name}: non-finite twin summary")
            extractions = pair.get("extractions")
            _require(isinstance(extractions, Mapping) and set(extractions) == {"A_win40", "D_doc"}, f"{label}.{pair_name}: missing extraction")
            for extraction, test in extractions.items():
                _require(_finite(test.get("raw_cosine")), f"{label}.{pair_name}.{extraction}: non-finite raw cosine")
                _validate_test({**test, "observed": test.get("line_controlled_cosine")}, f"{label}.{pair_name}.{extraction}")
                _require(test.get("B") == expected_B, f"{label}.{pair_name}.{extraction}: wrong checkpoint-twin B")
    return len(payload)


def main(argv: Sequence[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    expected_cells = None
    expected_docboot = None
    checkpoint_path = None
    checkpoint_twins_path = None
    paths: list[str] = []
    for arg in raw:
        if arg.startswith("--expected-cells="):
            expected_cells = int(arg.split("=", 1)[1])
        elif arg.startswith("--expected-docboot="):
            expected_docboot = int(arg.split("=", 1)[1])
        elif arg.startswith("--checkpoint="):
            checkpoint_path = arg.split("=", 1)[1]
        elif arg.startswith("--checkpoint-twins="):
            checkpoint_twins_path = arg.split("=", 1)[1]
        else:
            paths.append(arg)
    if not paths and checkpoint_path is None and checkpoint_twins_path is None:
        print(__doc__)
        return 2
    for path in paths:
        count = validate_file(path, expected_cells=expected_cells, expected_docboot=expected_docboot)
        print(f"VALID {path}: {count} cells")
    if checkpoint_path is not None:
        count = validate_checkpoint_file(checkpoint_path)
        print(f"VALID {checkpoint_path}: {count} checkpoint rows")
    if checkpoint_twins_path is not None:
        count = validate_checkpoint_twins_file(checkpoint_twins_path)
        print(f"VALID {checkpoint_twins_path}: {count} checkpoint-twin rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
