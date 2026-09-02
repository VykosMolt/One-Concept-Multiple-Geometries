"""Focused contract tests for the Phase-V v4.1 correction.

These tests are deliberately independent of the large stored result files.
They protect the feature definitions, fold isolation, reproducible streams,
and strict artifact mechanics used by the full recompute.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from phase2.keys15 import KEYS15, S
from phase5.fingerprint import CellConfig, _load_behavior, _merge_behavior, _twin_stats, _v4_fingerprint_output_path, compute_cell, logC_of, loo, main as fingerprint_main
from phase5.ckpt_twins import _controlled as checkpoint_twin_controlled
from phase5.thin_wikipedia import _v4_thinning_output_path
from phase5.v4_provenance import EXPECTED_GENERATED_OUTPUT_RELATIVE, _generated_output_path
from phase5.theory_features import (
    MASTER_SEED,
    SCHEMA_VERSION,
    build_raw_features,
    dump_json_strict,
    fit_scaler,
    fit_tokenizer_irregularity,
    make_fold_feature_set,
    pair_indices,
    stable_seed,
    training_target_prior,
    transform_scaler,
)
from phase5.validate_v4 import validate_record


def _row(fs, source_name: str, target_name: str) -> np.ndarray:
    i = KEYS15.index(source_name)
    if fs.view == "aggregated":
        j = int((7 * int(S[KEYS15.index(target_name)])) % 12)
    else:
        j = KEYS15.index(target_name)
    k = np.flatnonzero((fs.source == i) & (fs.target == j))
    assert len(k) == 1
    return fs.values[k[0]]


def _toy_logq() -> np.ndarray:
    rng = np.random.default_rng(4)
    q = rng.normal(size=(15, 15))
    q -= np.log(np.exp(q).sum(axis=1, keepdims=True))
    return q


def _toy_counts() -> np.ndarray:
    rng = np.random.default_rng(8)
    m = rng.poisson(30, size=(15, 15)).astype(float)
    np.fill_diagonal(m, 0)
    return m


def test_true_circle_is_distinct_from_chromatic_in_both_views():
    for view in ("spelled", "aggregated"):
        fs = build_raw_features(view, np.arange(1, 16, dtype=float))
        c_to_g = _row(fs, "C", "G" if view == "spelled" else "G")
        c_to_db = _row(fs, "C", "Db" if view == "spelled" else "Db")
        assert tuple(c_to_g[:2]) == (1.0, 5.0)
        assert tuple(c_to_db[:2]) == (5.0, 1.0)


def test_twins_have_circle_zero_and_open_line_twelve():
    fs = build_raw_features("spelled", np.ones(15))
    for a, b in (("Cb", "B"), ("Gb", "F#"), ("Db", "C#")):
        row = _row(fs, a, b)
        assert row[0] == 0.0
        assert row[2] == 12.0


def test_aggregate_schema_has_no_centroid_or_class_order_features():
    fs = build_raw_features("aggregated", np.ones(15), rich=True)
    assert not any("centroid" in n.lower() or "class_mean" in n.lower() or "class_size" in n.lower() or "class_span" in n.lower() for n in fs.names)
    assert "line_signed_low" in fs.names and "line_signed_high" in fs.names
    assert "circle_fifths_x_line_min_abs" in fs.names
    assert "circle_fifths_x_line_max_abs" not in fs.names


def test_aggregate_set_descriptors_are_member_order_invariant_and_endpoint_safe():
    # The class {Db,C#} is an exact endpoint/tie-sensitive case for a source
    # at C.  Its min/max extrema must be deterministic and expose both signs.
    fs = build_raw_features("aggregated", np.ones(15))
    row = _row(fs, "C", "Db")
    names = {name: row[k] for k, name in enumerate(fs.names)}
    assert names["line_min_abs"] == 5.0
    assert names["line_max_abs"] == 7.0
    assert names["line_signed_low"] == -5.0
    assert names["line_signed_high"] == 7.0
    assert names["signature_size_min_diff"] == 5.0
    assert names["signature_size_max_diff"] == 7.0
    # Rebuilding is independent of any external target-member enumeration.
    assert np.array_equal(fs.values, build_raw_features("target-aggregated", np.ones(15)).values)


def test_fold_scaler_uses_training_statistics_only_and_constant_scale_one():
    train = np.array([[0.0, 2.0], [2.0, 2.0]])
    test = np.array([[100.0, 4.0]])
    scaler = fit_scaler(train)
    assert np.array_equal(scaler["mean"], np.array([1.0, 2.0]))
    assert np.array_equal(scaler["scale"], np.array([1.0, 1.0]))
    assert np.array_equal(transform_scaler(test, scaler), np.array([[99.0, 2.0]]))


def test_tokenizer_fit_is_fold_local_and_target_prior_excludes_heldout_response():
    tok = np.arange(15, dtype=float) + 2.0
    fit0 = fit_tokenizer_irregularity(tok, np.arange(1, 15))
    fit1 = fit_tokenizer_irregularity(tok + np.eye(15)[0] * 1000.0, np.arange(1, 15))
    # A change at the held-out source cannot alter the fitted coefficients.
    assert np.array_equal(fit0["coefficients"], fit1["coefficients"])
    source, target = pair_indices("aggregated")
    values = np.arange(len(source), dtype=float)
    mask = source != 0
    prior0, _ = training_target_prior(values, source, target, mask, n_targets=12)
    changed = values.copy()
    changed[source == 0] += 10000
    prior1, _ = training_target_prior(changed, source, target, mask, n_targets=12)
    assert np.array_equal(prior0, prior1)


def test_loo_records_fold_ranks_and_heldout_scalers_for_both_views_and_models():
    q = _toy_logq()
    m = _toy_counts()
    uni = np.arange(1, 16, dtype=float) * 10
    tok = np.array([3, 3, 4, 4, 4, 3, 2, 2, 2, 2, 2, 2, 2, 3, 3], dtype=float)
    for view, neutral_q in (("spelled", q), ("aggregated", None)):
        if neutral_q is None:
            neutral_q = np.full((15, 12), -np.inf)
            pc = (7 * S) % 12
            for z in range(12):
                neutral_q[:, z] = np.logaddexp.reduce(q[:, pc == z], axis=1)
        fs = build_raw_features(view, uni, rich=False)
        for rich in (False, True):
            result = loo(neutral_q, fs, logC_of(m, view=view), target_prior=True, tok_counts=tok, frequencies=uni, rich=rich)
            assert len(result["rank_audits"]) == 15
            for fold in result["rank_audits"]:
                assert fold["rank"]["base"]["full_rank"]
                assert fold["rank"]["rich"]["full_rank"]
                assert fold["rank"]["targetprior"]["full_rank"]
                assert fold["rank"]["combined"]["full_rank"]
            assert len(result["fold_scalers"]) == 15
            assert all(x["fit_observations"] in (154, 196) for x in result["fold_scalers"])


def test_stable_seed_is_process_hash_seed_and_identity_order_independent():
    a = stable_seed(MASTER_SEED, corpus="wiki", model="m", stream="loo", replicate=3)
    b = stable_seed(MASTER_SEED, model="m", corpus="wiki", replicate=3, stream="loo")
    c = stable_seed(MASTER_SEED, corpus="wiki", model="m", stream="loo", replicate=4)
    assert a == b and a != c
    code = "from phase5.theory_features import stable_seed; print(stable_seed(20260830, corpus='wiki', model='m', stream='loo', replicate=3))"
    vals = []
    for hashseed in ("1", "999"):
        env = dict(os.environ, PYTHONHASHSEED=hashseed)
        vals.append(subprocess.check_output([sys.executable, "-c", code], env=env, text=True).strip())
    assert vals[0] == vals[1] == str(a)


def test_twin_null_streams_are_pair_order_independent():
    q = _toy_logq()
    c = logC_of(_toy_counts())
    pairs = ((0, 12), (1, 13), (2, 14))
    labels = {
        "corpus": "toy",
        "model": "toy",
        "family": "E_modulation",
        "extraction": "A_win40",
        "view": "spelled",
        "rich": False,
        "targetprior": False,
        "templates": [0],
    }
    forward = _twin_stats(q, c, 13, labels=labels, pairs=pairs)
    reverse = _twin_stats(q, c, 13, labels=labels, pairs=tuple(reversed(pairs)))
    by_pair_forward = {tuple(x["twin_identity"]): (x["b"], x["p"]) for x in forward}
    by_pair_reverse = {tuple(x["twin_identity"]): (x["b"], x["p"]) for x in reverse}
    assert by_pair_forward == by_pair_reverse
    assert all(x["randomization"]["seed_fields"][-1] == "replicate" for x in forward)


def test_nonheldout_residual_and_twins_ignore_loo_only_switches():
    q = _toy_logq()
    counts = _toy_counts()
    frequency = np.arange(1, 16, dtype=float)
    records = []
    for rich, targetprior in ((False, False), (True, False), (False, True), (True, True)):
        config = CellConfig("toy_v4", "toy", "E_modulation", "A_win40", "spelled", rich, targetprior, (0,), 7, 1, 0)
        record, _ = compute_cell(config, q, counts, frequency, None)
        records.append(record)
    residual_identities = {
        (record["resid_r"], record["resid_p"], record["tests"]["residual"]["b"], record["randomization"]["streams"]["residual"])
        for record in records
    }
    assert len(residual_identities) == 1
    twin_identities = {
        tuple((pair["twin_identity"][0], pair["twin_identity"][1], pair["b"], pair["p"]) for pair in record["twins"])
        for record in records
    }
    assert len(twin_identities) == 1
    assert all("rich" not in record["randomization"]["stream_labels"]["residual"] for record in records)
    assert all("targetprior" not in record["twin_randomization"]["labels"] for record in records)
    assert all(record["twin_randomization"]["cell_labels"] == record["twin_randomization"]["labels"] for record in records)


def test_builder_parity_and_bootstrap_rebuild_metadata():
    fs1 = build_raw_features("aggregated", np.arange(1, 16, dtype=float), rich=True)
    fs2 = build_raw_features("aggregated", np.arange(1, 16, dtype=float), rich=True)
    assert fs1.names == fs2.names and np.array_equal(fs1.values, fs2.values)
    cfg = CellConfig("toy", "toy", "E_modulation", "A_win40", "aggregated", False, False, (0,), 1, 1, 2)
    rec, _ = compute_cell(cfg, np.full((15, 12), -np.log(12.0)), _toy_counts(), np.arange(1, 16, dtype=float), None)
    assert rec["schema"] == SCHEMA_VERSION
    assert rec["dkl_bootstrap"]["B"] == 2
    assert rec["dkl_bootstrap"]["rebuilds"] == ["corpus_matrix", "source_frequency", "raw_features"]
    assert len(rec["loo_feature_names_by_fold"]) == 15
    assert "raw builder" in rec["feature_name_scope"]


def test_validator_accepts_complete_cells_and_rejects_truncated_nulls():
    cfg = CellConfig("toy", "toy", "E_modulation", "A_win40", "aggregated", True, True, (0,), 3, 3, 0)
    rec, _ = compute_cell(cfg, _merge_behavior(_toy_logq()), _toy_counts(), np.arange(1, 16, dtype=float), None)
    validate_record(rec, "toy")
    rec["tests"]["dkl"]["B"] -= 1
    try:
        validate_record(rec, "toy")
    except ValueError as error:
        assert "incomplete null" in str(error)
    else:
        raise AssertionError("truncated null was accepted")
    rec["tests"]["dkl"]["B"] += 1
    rec["docboot"] = {
        "sd": 0.01,
        "q025": 0.001,
        "q975": 0.03,
        "B": 3,
        "B_requested": 3,
        "rebuilds": ["corpus_matrix", "source_frequency", "raw_features"],
    }
    validate_record(rec, "toy", expected_docboot=3)
    rec["docboot"]["B"] = 2
    try:
        validate_record(rec, "toy", expected_docboot=3)
    except ValueError as error:
        assert "incomplete document bootstrap" in str(error)
    else:
        raise AssertionError("truncated document bootstrap was accepted")


def test_validator_rejects_missing_document_bootstrap_when_expected():
    cfg = CellConfig("toy", "toy", "E_modulation", "A_win40", "aggregated", True, True, (0,), 3, 3, 0)
    rec, _ = compute_cell(cfg, _merge_behavior(_toy_logq()), _toy_counts(), np.arange(1, 16, dtype=float), None)
    try:
        validate_record(rec, "toy", expected_docboot=3)
    except ValueError as error:
        assert "missing document bootstrap" in str(error)
    else:
        raise AssertionError("missing document bootstrap was accepted")


def test_checkpoint_twin_null_is_stable_and_uses_finite_estimator():
    keep = np.arange(2, 15)
    dq = np.random.default_rng(17).normal(size=len(keep))
    dc = np.random.default_rng(18).normal(size=len(keep))
    labels = {"corpus": "toy", "model": "toy", "family": "E_modulation", "extraction": "A_win40", "view": "spelled_twin_difference", "revision": "toy", "twin_identity": "Cb|B"}
    first = checkpoint_twin_controlled(dq, dc, keep, 0, 1, labels=labels, nperm=11)
    second = checkpoint_twin_controlled(dq, dc, keep, 0, 1, labels=labels, nperm=11)
    assert first == second
    assert first["B"] == first["B_requested"] == 11
    assert first["p"] == (first["b"] + 1) / (first["B"] + 1)
    assert first["estimator"] == "(b+1)/(B+1)"


def test_strict_json_converts_nonfinite_to_null_with_status_available():
    path = Path("/tmp/phase5-v4-strict-test.json")
    dump_json_strict({"x": float("nan"), "y": float("inf")}, path)
    loaded = json.load(open(path))
    assert loaded == {"x": None, "y": None}


def test_phase5_sources_do_not_use_builtin_hash():
    for path in (Path("phase5/fingerprint.py"), Path("phase5/theory_features.py"), Path("phase5/ckpt_fingerprint.py"), Path("phase5/ckpt_twins.py"), Path("phase5/thin_wikipedia.py")):
        source = path.read_text()
        assert "hash(" not in source.replace("hashlib", "")


def test_cli_protects_v3_and_requested_templates_fail_closed():
    try:
        fingerprint_main(["missing-input.npz", "wikipedia_v3"])
    except ValueError as error:
        assert "paper-v1" in str(error) and "non-v4" in str(error)
    else:
        raise AssertionError("v4 CLI accepted a v3 output identity")
    try:
        fingerprint_main(["missing-input.npz", "v4/../wikipedia_v3"])
    except ValueError as error:
        assert "unsafe" in str(error)
    else:
        raise AssertionError("traversal output identity was accepted")
    try:
        _load_behavior("olmo2_1b", ("E_modulation",), (0, 99))
    except ValueError as error:
        assert "E_modulation__t99" in str(error)
    else:
        raise AssertionError("missing requested template was silently averaged away")


def test_v4_output_paths_reject_symlinks_and_non_direct_children():
    fingerprint_link = Path("results/phase5/fingerprint/path_guard_v4.json")
    thinning_link = Path("results/phase5/path_guard_v4.npz")
    assert not fingerprint_link.exists() and not fingerprint_link.is_symlink()
    assert not thinning_link.exists() and not thinning_link.is_symlink()
    try:
        fingerprint_link.symlink_to("/tmp/paper-v1-fingerprint-target.json")
        thinning_link.symlink_to("/tmp/paper-v1-thinning-target.npz")
        for function, argument in (
            (_v4_fingerprint_output_path, "path_guard_v4"),
            (_v4_thinning_output_path, thinning_link),
        ):
            try:
                function(argument)
            except ValueError as error:
                assert "symlink" in str(error)
            else:
                raise AssertionError(f"{function.__name__} accepted a symlink output")
        try:
            _v4_thinning_output_path("results/phase5/nested/../path_guard_v4.npz")
        except ValueError as error:
            assert "traversal" in str(error)
        else:
            raise AssertionError("thinning path traversal was accepted")
    finally:
        if fingerprint_link.is_symlink():
            fingerprint_link.unlink()
        if thinning_link.is_symlink():
            thinning_link.unlink()


def test_provenance_generated_output_allowlist_is_exact():
    fingerprints = [path for path in EXPECTED_GENERATED_OUTPUT_RELATIVE if path.parent == Path("results/phase5/fingerprint")]
    thinnings = [path for path in EXPECTED_GENERATED_OUTPUT_RELATIVE if path.name.startswith("cond_wikipedia_thin_")]
    assert len(fingerprints) == 75
    assert len(thinnings) == 20
    assert _generated_output_path(Path("results/phase5/fingerprint/wikipedia_v4.json"))
    assert _generated_output_path(Path("results/phase5/cond_wikipedia_thin_olmomix_wiki_v4_s4.npz"))
    assert not _generated_output_path(Path("results/phase5/fingerprint/unrelated_v4.json"))
    assert not _generated_output_path(Path("results/phase5/cond_wikipedia_thin_unplanned_v4.npz"))
