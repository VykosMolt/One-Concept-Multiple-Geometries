"""Phase-V fingerprint tests under the corrected v4.1 theory schema.

The command-line interface is compatible with the original Phase-V runner,
but all feature construction, fold scaling, rank auditing, and random-seed
derivation are delegated to :mod:`phase5.theory_features`.  v3 result files
are never read for computation and are never overwritten.
"""

from __future__ import annotations

import json
import os
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr

from phase2.keys15 import ENH_PAIRS, GLYPH, KEYS15, LETTER, S, n
from phase5.theory_features import (
    MASTER_SEED,
    RANK_RTOL,
    SCHEMA_VERSION,
    SEED_ALGORITHM,
    FeatureSet,
    build_raw_features,
    cyclic_distance,
    dump_json_strict,
    feature_definitions,
    finite_status,
    fit_scaler,
    make_fold_feature_set,
    normalise_view,
    residual_projection,
    stable_rng,
    stable_seed,
    training_target_prior,
    transform_scaler,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FINGERPRINT_OUTPUT_DIR = PROJECT_ROOT / "results/phase5/fingerprint"
_SAFE_OUTPUT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*\Z")


def _v4_fingerprint_output_path(output_name: str) -> Path:
    """Return the protected direct-child path for one v4 fingerprint JSON."""

    if not _SAFE_OUTPUT_NAME.fullmatch(output_name) or "v4" not in output_name.casefold():
        raise ValueError(
            f"refusing unsafe or non-v4 output identity {output_name!r}; "
            "historical reproduction must use the immutable paper-v1 snapshot"
        )
    if FINGERPRINT_OUTPUT_DIR.is_symlink():
        raise ValueError(f"fingerprint output directory is a symlink: {FINGERPRINT_OUTPUT_DIR}")
    FINGERPRINT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FINGERPRINT_OUTPUT_DIR / f"{output_name}.json"
    if output_path.parent != FINGERPRINT_OUTPUT_DIR:
        raise ValueError(f"fingerprint output is not a direct child of {FINGERPRINT_OUTPUT_DIR}: {output_path}")
    if output_path.is_symlink():
        raise ValueError(f"refusing symlink fingerprint output: {output_path}")
    if output_path.exists() and not output_path.is_file():
        raise ValueError(f"fingerprint output is not a regular file: {output_path}")
    return output_path


# Compatibility alias retained for small diagnostics that imported the old
# module-level helper.  It is the generic cyclic distance, not the fifths
# distance; the named v4.1 builder uses ``circle_fifths`` explicitly.
circ12 = cyclic_distance


def ridge_fit_predict(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, lam: float = 1.0) -> np.ndarray:
    """Ridge prediction with an unpenalised intercept."""

    train = np.asarray(Xtr, dtype=float)
    test = np.asarray(Xte, dtype=float)
    y = np.asarray(ytr, dtype=float)
    train1 = np.column_stack([np.ones(len(train)), train])
    test1 = np.column_stack([np.ones(len(test)), test])
    penalty = np.diag([0.0] + [float(lam)] * train.shape[1])
    A = train1.T @ train1 + penalty
    rhs = train1.T @ y
    try:
        coef = np.linalg.solve(A, rhs)
    except np.linalg.LinAlgError:
        coef = np.linalg.lstsq(A, rhs, rcond=None)[0]
    return test1 @ coef


def logC_of(M: np.ndarray, *, view: str = "spelled") -> np.ndarray:
    """Add-0.5-smoothed corpus log conditionals in the requested view."""

    counts = np.asarray(M, dtype=float)
    if counts.shape != (n, n):
        raise ValueError(f"corpus matrix must have shape ({n}, {n})")
    if np.any(counts < 0) or not np.all(np.isfinite(counts)):
        raise ValueError("corpus counts must be finite and non-negative")
    R = (counts + 0.5) / np.sum(counts + 0.5, axis=1, keepdims=True)
    L = np.log(R)
    if normalise_view(view) == "spelled":
        return L
    out = np.full((n, 12), -np.inf, dtype=float)
    pc = (7 * S) % 12
    for z in range(12):
        out[:, z] = np.logaddexp.reduce(L[:, pc == z], axis=1)
    return out


def perm_keys(M: np.ndarray, permutation: Sequence[int]) -> np.ndarray:
    """Jointly relabel corpus rows and columns."""

    p = np.asarray(permutation, dtype=int)
    if p.shape != (n,) or set(p.tolist()) != set(range(n)):
        raise ValueError("permutation must be a permutation of the 15 keys")
    return np.asarray(M)[np.ix_(p, p)]


def mc_test(null: Sequence[float], observed: float, *, tail: str = ">=") -> tuple[float | None, int | None, int]:
    """Finite-sample Monte-Carlo test ``(b+1)/(B+1)``."""

    values = np.asarray(null, dtype=float)
    values = values[np.isfinite(values)]
    B = int(len(values))
    if B == 0 or not np.isfinite(observed):
        return None, None, B
    if tail == ">=":
        b = int(np.sum(values >= float(observed)))
    elif tail == "<=":
        b = int(np.sum(values <= float(observed)))
    else:
        raise ValueError(f"unsupported Monte-Carlo tail: {tail}")
    return float((b + 1) / (B + 1)), b, B


def mc_p(null: Sequence[float], obs: float, ge: bool = True) -> float | None:
    return mc_test(null, obs, tail=">=" if ge else "<=")[0]


def _safe_corr(a: Sequence[float], b: Sequence[float]) -> float | None:
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        r = spearmanr(np.asarray(a, dtype=float), np.asarray(b, dtype=float)).correlation
    return float(r) if r is not None and np.isfinite(r) else None


def _logsumexp(x: np.ndarray, axis: int | None = None, keepdims: bool = False) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mx = np.max(x, axis=axis, keepdims=True)
    out = mx + np.log(np.sum(np.exp(x - mx), axis=axis, keepdims=True))
    if not keepdims and axis is not None:
        out = np.squeeze(out, axis=axis)
    return out


def _feature_set_from_input(features: FeatureSet | np.ndarray, view: str | None = None, frequencies: Sequence[float] | None = None) -> FeatureSet:
    if isinstance(features, FeatureSet):
        return features
    X = np.asarray(features, dtype=float)
    v = normalise_view(view or ("spelled" if len(X) == n * (n - 1) else "aggregated"))
    from phase5.theory_features import pair_indices

    source, target = pair_indices(v)
    if X.shape[0] != len(source):
        raise ValueError("feature matrix row count does not match view")
    names = tuple(f"x{k}" for k in range(X.shape[1]))
    freq = np.ones(n) if frequencies is None else np.asarray(frequencies, dtype=float)
    return FeatureSet(X, names, source, target, v, False, freq, {"schema": SCHEMA_VERSION, "legacy_input": True})


def feats(tokcount: Sequence[float] | None, uni_: Sequence[float], *, neutral: bool = False, rich: bool = False) -> np.ndarray:
    """Compatibility helper returning raw v4.1 features."""

    fs = build_raw_features("aggregated" if neutral else "spelled", uni_, rich=rich)
    if tokcount is not None:
        from phase5.theory_features import fit_tokenizer_irregularity

        fit = fit_tokenizer_irregularity(tokcount)
        fs = build_raw_features(fs.view, uni_, rich=rich, tokenizer_residual=fit["residual"])
    return fs.values


def _matrix_for_pairs(matrix: np.ndarray, source: np.ndarray, target: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim == 1:
        if arr.shape != source.shape:
            raise ValueError("pair vector is not aligned to feature rows")
        return arr
    if arr.ndim != 2:
        raise ValueError("matrix must be one- or two-dimensional")
    if arr.shape[0] != n:
        raise ValueError("matrix has the wrong source-row count")
    return arr[source, target]


def _fit_train_test(Xtr: np.ndarray, Xte: np.ndarray, relative_tolerance: float) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    scaler = fit_scaler(Xtr, relative_tolerance=relative_tolerance)
    return transform_scaler(Xtr, scaler), transform_scaler(Xte, scaler), scaler


def _scaler_json(scaler: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "mean": np.asarray(scaler["mean"], dtype=float).tolist(),
        "scale": np.asarray(scaler["scale"], dtype=float).tolist(),
        "constant": np.asarray(scaler["constant"], dtype=bool).tolist(),
        "relative_tolerance": float(scaler["relative_tolerance"]),
    }


@dataclass
class _PreparedLooFold:
    heldout: int
    train: np.ndarray
    Xtr: np.ndarray
    Xte: np.ndarray
    y: np.ndarray
    theory_prediction: np.ndarray
    pred_names: list[str]
    base_tokenizer_audit: dict[str, Any]
    rich_tokenizer_audit: dict[str, Any]
    rank_audits: dict[str, Any]
    target_prior_meta: dict[str, Any] | None
    theory_scaler: dict[str, Any]


@dataclass
class _PreparedLoo:
    feature_set: FeatureSet
    response_pairs: np.ndarray
    folds: list[_PreparedLooFold]
    rich: bool
    target_prior: bool
    relative_tolerance: float


def _prepare_loo(
    logQ: np.ndarray,
    features: FeatureSet | np.ndarray,
    *,
    target_prior: bool = False,
    tok_counts: Sequence[float] | None = None,
    frequencies: Sequence[float] | None = None,
    rich: bool | None = None,
    relative_tolerance: float = RANK_RTOL,
    audit: bool = True,
) -> _PreparedLoo:
    """Prepare the response/theory side shared by every corpus relabeling.

    The held-out source never enters a fold's tokenizer fit or scaler.  This
    preparation depends on the response only through training targets and the
    optional training-row target prior; it is therefore invariant across the
    corpus permutations used by the null.  Bootstrap resamples do not reuse it
    because their frequencies and raw theory features are rebuilt.
    """

    from phase5.theory_features import assert_expected_rank, matrix_rank_audit

    fs0 = _feature_set_from_input(features, frequencies=frequencies)
    q = np.asarray(logQ, dtype=float)
    if q.shape[0] != n:
        raise ValueError("logQ must have 15 source rows")
    qv = _matrix_for_pairs(q, fs0.source, fs0.target)
    if not np.all(np.isfinite(qv)):
        raise ValueError("LOO response must be finite")
    freq = fs0.frequencies if frequencies is None else np.asarray(frequencies, dtype=float)
    use_rich = bool(fs0.rich if rich is None else rich)
    folds: list[_PreparedLooFold] = []

    for heldout in range(n):
        base_fs, base_tok_audit = make_fold_feature_set(
            fs0.view, freq, tok_counts, heldout, rich=False, relative_tolerance=relative_tolerance
        )
        rich_fs, rich_tok_audit = make_fold_feature_set(
            fs0.view, freq, tok_counts, heldout, rich=True, relative_tolerance=relative_tolerance
        )
        pred_fs = rich_fs if use_rich else base_fs
        train = pred_fs.source != heldout
        if not np.array_equal(pred_fs.source, fs0.source) or not np.array_equal(pred_fs.target, fs0.target):
            raise RuntimeError("fold feature builder changed pair ordering")
        audits: dict[str, Any] = {
            "base": matrix_rank_audit(
                base_fs.values[train], base_fs.names, relative_tolerance=relative_tolerance, context=f"fold{heldout}:base"
            ),
            "rich": matrix_rank_audit(
                rich_fs.values[train], rich_fs.names, relative_tolerance=relative_tolerance, context=f"fold{heldout}:rich"
            ),
            "targetprior": None,
        }
        pred_raw = pred_fs.values
        pred_names = list(pred_fs.names)
        target_prior_meta = None
        if target_prior:
            prior, target_prior_meta = training_target_prior(
                qv, fs0.source, fs0.target, train, n_targets=(12 if fs0.view == "aggregated" else n)
            )
            pred_raw = np.column_stack([pred_raw, prior[pred_fs.target]])
            pred_names.append("target_prior_train")
            audits["targetprior"] = matrix_rank_audit(
                pred_raw[train], pred_names, relative_tolerance=relative_tolerance, context=f"fold{heldout}:targetprior"
            )
        if audit:
            for key in ("base", "rich", "targetprior"):
                if audits[key] is not None:
                    assert_expected_rank(audits[key])
        Xtr, Xte, xscale = _fit_train_test(pred_raw[train], pred_raw[~train], relative_tolerance)
        folds.append(
            _PreparedLooFold(
                heldout=int(heldout),
                train=train,
                Xtr=Xtr,
                Xte=Xte,
                y=qv[~train],
                theory_prediction=ridge_fit_predict(Xtr, qv[train], Xte),
                pred_names=pred_names,
                base_tokenizer_audit=base_tok_audit,
                rich_tokenizer_audit=rich_tok_audit,
                rank_audits=audits,
                target_prior_meta=target_prior_meta,
                theory_scaler=xscale,
            )
        )
    return _PreparedLoo(fs0, qv, folds, use_rich, bool(target_prior), float(relative_tolerance))


def _score_prepared_loo(prepared: _PreparedLoo, logC: np.ndarray, *, audit: bool = False) -> dict[str, Any]:
    """Score one corpus matrix against a prepared response/theory design.

    Crucially, the corpus predictor is still refit and standardized inside
    every fold for every permutation.  Only invariant response/theory work is
    reused.
    """

    from phase5.theory_features import assert_expected_rank, matrix_rank_audit

    fs0 = prepared.feature_set
    cv = _matrix_for_pairs(logC, fs0.source, fs0.target)
    if not np.all(np.isfinite(cv)):
        raise ValueError("LOO corpus predictor must be finite")
    scores: dict[str, list[float | None]] = {"theory": [], "corpus": [], "both": []}
    sse = {"theory": 0.0, "corpus": 0.0, "both": 0.0}
    kl_rows: dict[str, list[float | None]] = {"theory": [], "corpus": [], "both": []}
    fold_audits: list[dict[str, Any]] = []
    fold_scalers: list[dict[str, Any]] = []

    for fold in prepared.folds:
        train = fold.train
        ctr, cte, cscale = _fit_train_test(cv[train, None], cv[~train, None], prepared.relative_tolerance)
        combined_tr = np.column_stack([fold.Xtr, ctr])
        combined_te = np.column_stack([fold.Xte, cte])
        combined_audit = matrix_rank_audit(
            combined_tr,
            tuple(fold.pred_names + ["corpus_logconditional"]),
            relative_tolerance=prepared.relative_tolerance,
            context=f"fold{fold.heldout}:combined",
        )
        if audit:
            assert_expected_rank(combined_audit)
        pred_map = {
            "theory": fold.theory_prediction,
            "corpus": ridge_fit_predict(ctr, prepared.response_pairs[train], cte),
            "both": ridge_fit_predict(combined_tr, prepared.response_pairs[train], combined_te),
        }
        py = np.exp(fold.y - _logsumexp(fold.y))
        for name, pred in pred_map.items():
            scores[name].append(_safe_corr(pred, fold.y))
            sse[name] += float(np.sum(((pred - pred.mean()) - (fold.y - fold.y.mean())) ** 2))
            pp = pred - _logsumexp(pred)
            kval = float(np.sum(py * (np.log(py + 1e-12) - pp)))
            kl_rows[name].append(kval if np.isfinite(kval) else None)
        fold_audits.append(
            {
                "heldout_source": fold.heldout,
                "feature_names": fold.pred_names,
                "tokenizer": {"base": fold.base_tokenizer_audit, "rich": fold.rich_tokenizer_audit},
                "rank": {**fold.rank_audits, "combined": combined_audit},
                "target_prior": fold.target_prior_meta,
            }
        )
        fold_scalers.append(
            {
                "heldout_source": fold.heldout,
                "theory": _scaler_json(fold.theory_scaler),
                "corpus": _scaler_json(cscale),
                "fit_observations": int(np.sum(train)),
            }
        )

    def avg(values: Sequence[float | None]) -> float | None:
        finite = [float(x) for x in values if x is not None and np.isfinite(x)]
        return float(np.mean(finite)) if finite else None

    out: dict[str, Any] = {k: avg(v) for k, v in scores.items()}
    out["r2gain"] = float(1.0 - sse["both"] / sse["theory"]) if sse["theory"] else None
    out["kl"] = {k: avg(v) for k, v in kl_rows.items()}
    out["dkl"] = out["kl"]["theory"] - out["kl"]["both"] if out["kl"]["theory"] is not None and out["kl"]["both"] is not None else None
    out["kl_rows"] = kl_rows
    out["rank_audits"] = fold_audits
    out["fold_scalers"] = fold_scalers
    out["feature_schema"] = SCHEMA_VERSION
    out["feature_view"] = fs0.view
    out["rich"] = prepared.rich
    out["targetprior"] = prepared.target_prior
    out["standardization"] = "per-fold mean/scale fitted on training pair observations only; constant scale=1"
    out["preparation_reuse"] = "response/theory prepared once per cell; corpus scaler refit in every fold and permutation"
    return out


def loo(
    logQ: np.ndarray,
    features: FeatureSet | np.ndarray,
    logC: np.ndarray,
    *,
    target_prior: bool = False,
    tok_counts: Sequence[float] | None = None,
    frequencies: Sequence[float] | None = None,
    rich: bool | None = None,
    relative_tolerance: float = RANK_RTOL,
    audit: bool = True,
) -> dict[str, Any]:
    """Leave-one-source-row-out prediction with fold-local preprocessing."""

    prepared = _prepare_loo(
        logQ,
        features,
        target_prior=target_prior,
        tok_counts=tok_counts,
        frequencies=frequencies,
        rich=rich,
        relative_tolerance=relative_tolerance,
        audit=audit,
    )
    return _score_prepared_loo(prepared, logC, audit=audit)


def _null_record(null: Sequence[float], observed: float, *, B_requested: int, stream: str) -> dict[str, Any]:
    p, b, B = mc_test(null, observed)
    return {"observed": observed, "b": b, "B": B, "B_requested": int(B_requested), "p": p, "tail": ">=", "estimator": "(b+1)/(B+1)", "stream": stream}


def _row_values(logQ: np.ndarray, fs: FeatureSet) -> np.ndarray:
    return np.asarray(logQ, dtype=float)[fs.source, fs.target]


def _twin_stats(
    logQ: np.ndarray,
    logC: np.ndarray,
    nperm: int,
    *,
    master_seed: int = MASTER_SEED,
    labels: Mapping[str, Any] | None = None,
    pairs: Sequence[tuple[int, int]] | None = None,
) -> list[dict[str, Any]]:
    """Compute enharmonic-twin controls with one identity-derived stream per draw.

    The seed contains the complete cell label, a canonical (unordered) twin
    identity, and the replicate number.  Thus changing the order in which
    twin pairs or null draws are visited cannot change any other pair's null.
    ``pairs`` is exposed for the order-invariance test and small diagnostics;
    production callers use the fixed :data:`ENH_PAIRS` bank.
    """

    out: list[dict[str, Any]] = []
    q, c = np.asarray(logQ, dtype=float), np.asarray(logC, dtype=float)
    cell_labels = dict(labels or {})
    pair_bank = tuple(ENH_PAIRS if pairs is None else pairs)
    for a, b in pair_bank:
        if not (0 <= int(a) < n and 0 <= int(b) < n) or int(a) == int(b):
            raise ValueError("twin pair indices must be distinct keys in the 15-key bank")
        a, b = int(a), int(b)
        twin_identity = tuple(sorted((KEYS15[a], KEYS15[b])))
        keep = np.array([j for j in range(n) if j not in (a, b)], dtype=int)
        dq, dc = q[a, keep] - q[b, keep], c[a, keep] - c[b, keep]
        dq0, dc0 = dq - dq.mean(), dc - dc.mean()
        raw = float(dq0 @ dc0 / (np.linalg.norm(dq0) * np.linalg.norm(dc0) + 1e-12))
        Xd = np.column_stack([np.ones(len(keep)), S[keep], np.abs(S[keep] - S[a]) - np.abs(S[keep] - S[b]), (GLYPH[keep] == -1).astype(float), (GLYPH[keep] == 1).astype(float)])
        rq = dq - Xd @ np.linalg.lstsq(Xd, dq, rcond=None)[0]
        rc = dc - Xd @ np.linalg.lstsq(Xd, dc, rcond=None)[0]
        denom = np.linalg.norm(rq) * np.linalg.norm(rc)
        controlled = float(rq @ rc / (denom + 1e-12)) if denom > 1e-12 else float("nan")
        null = []
        for replicate in range(int(nperm)):
            twin_rng = stable_rng(
                master_seed,
                **cell_labels,
                stream="twins",
                twin_identity=twin_identity,
                replicate=replicate,
            )
            perm = twin_rng.permutation(len(keep))
            rp = dc[perm] - Xd @ np.linalg.lstsq(Xd, dc[perm], rcond=None)[0]
            den = np.linalg.norm(rq) * np.linalg.norm(rp)
            null.append(float(rq @ rp / (den + 1e-12)) if den > 1e-12 else float("nan"))
        p, bb, BB = mc_test(null, controlled)
        out.append(
            {
                "source_a": KEYS15[a],
                "source_b": KEYS15[b],
                "twin_identity": list(twin_identity),
                "cosine": raw,
                "line_controlled": controlled if np.isfinite(controlled) else None,
                "p": p,
                "b": bb,
                "B": BB,
                "B_requested": int(nperm),
                "tail": ">=",
                "estimator": "(b+1)/(B+1)",
                "spearman": _safe_corr(dq, dc),
                "randomization": {
                    "schema": SCHEMA_VERSION,
                    "master_seed": int(master_seed),
                    "algorithm": SEED_ALGORITHM,
                    "labels": cell_labels,
                    "stream": "twins",
                    "twin_identity": list(twin_identity),
                    "B": int(nperm),
                    "seed_fields": ["master_seed", "cell_labels", "stream", "twin_identity", "replicate"],
                },
            }
        )
    return out


@dataclass
class CellConfig:
    corpus_name: str
    model: str
    family: str
    extraction: str
    view: str
    rich: bool
    targetprior: bool
    templates: tuple[int, ...]
    nperm_res: int
    nperm_loo: int
    nboot: int
    master_seed: int = MASTER_SEED


def compute_cell(
    config: CellConfig,
    logQ: np.ndarray,
    corpus_counts: np.ndarray,
    frequencies: Sequence[float],
    token_counts: Sequence[float] | None,
    *,
    perdoc: Mapping[str, Any] | None = None,
    n_docboot: int = 0,
) -> tuple[dict[str, Any] | None, str]:
    """Compute one cell and its concise human-readable line."""

    fs = build_raw_features(config.view, frequencies, rich=config.rich)
    M = np.asarray(corpus_counts, dtype=float)
    if M.shape != (n, n):
        raise ValueError("corpus matrix must be 15x15")
    if float(np.sum(M)) < 100:
        rec = {"status": "UNAVAILABLE", "reason": "too_sparse", "pairs": int(np.sum(M)), "schema": SCHEMA_VERSION}
        return rec, f"{config.model:10s} {config.family:13s} {config.extraction:13s} | unavailable: too sparse ({int(M.sum())} pairs)"
    logC = logC_of(M, view=config.view)
    # Non-held-out residual correspondence always uses the same rich builder
    # and OLS source-effects projection, independent of the LOO --rich flag.
    proj_fs = build_raw_features(config.view, frequencies, rich=True)
    qpair = _row_values(logQ, proj_fs)
    cpair = _row_values(logC, proj_fs)
    qres, qmeta = residual_projection(qpair, proj_fs, token_counts=token_counts, source_effects=True)
    cres, _ = residual_projection(cpair, proj_fs, token_counts=token_counts, source_effects=True)
    observed_resid = _safe_corr(qres, cres)

    # These are the complete cell labels for LOO/bootstrap streams.  The
    # non-held-out projection and twin controls use the explicit analysis
    # labels below so their identities do not vary with LOO-only switches.
    labels = {
        "corpus": config.corpus_name,
        "model": config.model,
        "family": config.family,
        "extraction": config.extraction,
        "view": config.view,
        "rich": bool(config.rich),
        "targetprior": bool(config.targetprior),
        "templates": list(config.templates),
    }
    residual_labels = {
        "corpus": config.corpus_name,
        "model": config.model,
        "family": config.family,
        "extraction": config.extraction,
        "view": config.view,
        "templates": list(config.templates),
        "analysis": "nonheldout_rich_source_effects",
    }
    twin_labels = {
        "corpus": config.corpus_name,
        "model": config.model,
        "family": config.family,
        "extraction": config.extraction,
        "view": config.view,
        "templates": list(config.templates),
        "analysis": "enharmonic_twin_control",
    }
    stream_labels: dict[str, Mapping[str, Any]] = {
        "residual": residual_labels,
        "loo": labels,
        "poisson_bootstrap_matrix": labels,
        "poisson_bootstrap_frequency": labels,
        "document_bootstrap": labels,
        "twins": twin_labels,
    }
    streams = ("residual", "loo", "poisson_bootstrap_matrix", "poisson_bootstrap_frequency", "document_bootstrap", "twins")
    seeds = {stream: stable_seed(config.master_seed, **stream_labels[stream], stream=stream, replicate=0) for stream in streams}
    null_resid: list[float] = []
    for draw in range(config.nperm_res):
        # A fresh identity-derived generator per draw prevents one stream's
        # consumption order from affecting another stream or process.
        residual_rng = stable_rng(config.master_seed, **residual_labels, stream="residual", replicate=draw)
        p = residual_rng.permutation(n)
        cp, _ = residual_projection(_row_values(logC_of(perm_keys(M, p), view=config.view), proj_fs), proj_fs, token_counts=token_counts, source_effects=True)
        r = _safe_corr(qres, cp)
        if r is not None:
            null_resid.append(r)
    resid_test = _null_record(null_resid, observed_resid if observed_resid is not None else float("nan"), B_requested=config.nperm_res, stream="residual")

    prepared_loo = _prepare_loo(
        logQ,
        fs,
        target_prior=config.targetprior,
        tok_counts=token_counts,
        frequencies=frequencies,
        rich=config.rich,
        audit=True,
    )
    score = _score_prepared_loo(prepared_loo, logC, audit=True)
    dcv = score["both"] - score["theory"] if score.get("both") is not None and score.get("theory") is not None else None
    nullcv: list[float] = []
    nullr2: list[float] = []
    nullkl: list[float] = []
    for draw in range(config.nperm_loo):
        loo_rng = stable_rng(config.master_seed, **labels, stream="loo", replicate=draw)
        p = loo_rng.permutation(n)
        s2 = _score_prepared_loo(prepared_loo, logC_of(perm_keys(M, p), view=config.view), audit=False)
        if s2.get("both") is not None and s2.get("theory") is not None:
            nullcv.append(float(s2["both"] - s2["theory"]))
        if s2.get("r2gain") is not None:
            nullr2.append(float(s2["r2gain"]))
        if s2.get("dkl") is not None:
            nullkl.append(float(s2["dkl"]))
    dcv_test = _null_record(nullcv, dcv if dcv is not None else float("nan"), B_requested=config.nperm_loo, stream="loo_dcv")
    r2_test = _null_record(nullr2, score.get("r2gain") if score.get("r2gain") is not None else float("nan"), B_requested=config.nperm_loo, stream="loo_r2")
    dkl_test = _null_record(nullkl, score.get("dkl") if score.get("dkl") is not None else float("nan"), B_requested=config.nperm_loo, stream="loo_dkl")

    # Rebuild frequencies and theory features in every Poisson draw.
    boot_dkl: list[float] = []
    for draw in range(config.nboot):
        mb_rng = stable_rng(config.master_seed, **labels, stream="poisson_bootstrap_matrix", replicate=draw)
        ub_rng = stable_rng(config.master_seed, **labels, stream="poisson_bootstrap_frequency", replicate=draw)
        mb = mb_rng.poisson(np.maximum(M, 0)).astype(float)
        ub = ub_rng.poisson(np.maximum(np.asarray(frequencies, dtype=float), 0)).astype(float)
        fb = build_raw_features(config.view, ub, rich=config.rich)
        sb = loo(logQ, fb, logC_of(mb, view=config.view), target_prior=config.targetprior, tok_counts=token_counts, frequencies=ub, rich=config.rich, audit=False)
        if sb.get("dkl") is not None:
            boot_dkl.append(float(sb["dkl"]))

    docboot = None
    if perdoc is not None and n_docboot > 0 and config.extraction in ("A_win40", "D_doc"):
        d_id = np.asarray(perdoc[f"{config.extraction}_doc"], dtype=int)
        ii = np.asarray(perdoc[f"{config.extraction}_i"], dtype=int)
        jj = np.asarray(perdoc[f"{config.extraction}_j"], dtype=int)
        cc = np.asarray(perdoc[f"{config.extraction}_c"], dtype=float)
        U = np.asarray(perdoc["uni_docs"], dtype=float)
        nd = int(U.shape[0])
        values: list[float] = []
        for draw in range(n_docboot):
            d_rng = stable_rng(config.master_seed, **labels, stream="document_bootstrap", replicate=draw)
            weights = d_rng.multinomial(nd, np.ones(nd) / nd).astype(float)
            mb = np.zeros((n, n), dtype=float)
            np.add.at(mb, (ii, jj), cc * weights[d_id])
            ub = weights @ U
            fb = build_raw_features(config.view, ub, rich=config.rich)
            sb = loo(logQ, fb, logC_of(mb, view=config.view), target_prior=config.targetprior, tok_counts=token_counts, frequencies=ub, rich=config.rich, audit=False)
            if sb.get("dkl") is not None:
                values.append(float(sb["dkl"]))
        if values:
            arr = np.asarray(values)
            docboot = {"sd": float(np.std(arr)), "q025": float(np.percentile(arr, 2.5)), "q975": float(np.percentile(arr, 97.5)), "B": len(values), "B_requested": int(n_docboot), "rebuilds": ["corpus_matrix", "source_frequency", "raw_features"]}
        else:
            docboot = {"status": "UNAVAILABLE", "B": 0, "B_requested": int(n_docboot), "rebuilds": ["corpus_matrix", "source_frequency", "raw_features"]}

    twins = (
        _twin_stats(
            logQ,
            logC,
            config.nperm_res,
            master_seed=config.master_seed,
            labels=twin_labels,
        )
        if config.view == "spelled"
        else []
    )
    fitted_definitions = feature_definitions(config.view, rich=config.rich, tokenizer=token_counts is not None)
    if config.targetprior:
        fitted_definitions["target_prior_train"] = "target mean log response over the 14 training source rows only"
    rec: dict[str, Any] = {
        "status": "OK",
        "schema": SCHEMA_VERSION,
        "feature_names": list(fs.names),
        "feature_name_scope": "raw builder before fold-local tokenizer selection; exact fitted design is recorded in loo_feature_names_by_fold",
        "loo_feature_names_by_fold": [list(fold["feature_names"]) for fold in score.get("rank_audits", [])],
        "feature_definitions": fitted_definitions,
        "view": config.view,
        "rich": config.rich,
        "targetprior": config.targetprior,
        "resid_r": observed_resid,
        "resid_p": resid_test["p"],
        "loo": score,
        "dcv": dcv,
        "dcv_p": dcv_test["p"],
        "dcv_boot_sd": float(np.std(boot_dkl)) if boot_dkl else None,
        "dkl_bootstrap": {"B": len(boot_dkl), "q025": float(np.percentile(boot_dkl, 2.5)), "q975": float(np.percentile(boot_dkl, 97.5)), "values": boot_dkl, "rebuilds": ["corpus_matrix", "source_frequency", "raw_features"]} if boot_dkl else {"B": 0, "rebuilds": ["corpus_matrix", "source_frequency", "raw_features"]},
        "r2gain": score.get("r2gain"),
        "r2gain_p": r2_test["p"],
        "kl": score.get("kl"),
        "dkl": score.get("dkl"),
        "dkl_p": dkl_test["p"],
        "twins": twins,
        "pairs": int(np.sum(M)),
        "nperm_res": config.nperm_res,
        "nperm_loo": config.nperm_loo,
        "docboot": docboot,
        "templates": list(config.templates),
        "residual_projection": qmeta,
        "randomization": {
            "master_seed": config.master_seed,
            "algorithm": SEED_ALGORITHM,
            "labels": labels,
            "streams": seeds,
            "stream_labels": stream_labels,
            "per_draw_seed_fields": ["master_seed", "cell_labels", "stream", "replicate"],
        },
        "twin_randomization": {
            "schema": SCHEMA_VERSION,
            "master_seed": int(config.master_seed),
            "algorithm": SEED_ALGORITHM,
            "labels": twin_labels,
            "cell_labels": twin_labels,
            "stream": "twins",
            "B": int(config.nperm_res),
            "enabled": bool(config.view == "spelled"),
            "seed_fields": ["master_seed", "cell_labels", "stream", "twin_identity", "replicate"],
        },
        "tests": {"residual": resid_test, "dcv": dcv_test, "r2gain": r2_test, "dkl": dkl_test},
        "standardization": score.get("standardization"),
    }
    safe, available, paths = finite_status(rec)
    safe["status"] = "OK" if available else "UNAVAILABLE"
    if paths:
        safe["unavailable_fields"] = paths
    def fmt(value: Any, spec: str) -> str:
        return format(float(value), spec) if value is not None and np.isfinite(value) else "nan"
    line = f"{config.model:10s} {config.family:13s} {config.extraction:13s} | {fmt(observed_resid, '+.2f')} ({fmt(resid_test['p'], '.4f')}) | {fmt(score.get('theory'), '+.2f')} {fmt(score.get('corpus'), '+.2f')} {fmt(score.get('both'), '+.2f')} Δ {fmt(dcv, '+.3f')} ({fmt(dcv_test['p'], '.4f')}) ΔR² {fmt(score.get('r2gain'), '+.3f')} ({fmt(r2_test['p'], '.4f')}) ΔKL {fmt(score.get('dkl'), '+.4f')} ({fmt(dkl_test['p'], '.4f')})"
    return safe, line


def _parse_flag(args: list[str], name: str, default: Any = None) -> Any:
    key = f"--{name}"
    for arg in args:
        if arg == key:
            return True
        if arg.startswith(key + "="):
            return arg.split("=", 1)[1]
    return default


def _require_finite_matrix(value: Any, *, shape: tuple[int, int], source: str, nonnegative: bool = False) -> np.ndarray:
    """Convert one named matrix and reject missing, malformed, or non-finite data."""

    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source}: matrix is not numeric") from exc
    if arr.shape != shape:
        raise ValueError(f"{source}: expected shape {shape}, got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{source}: matrix contains non-finite values")
    if nonnegative and np.any(arr < 0):
        raise ValueError(f"{source}: matrix contains negative values")
    return arr


def _load_behavior(model: str, families: Sequence[str], templates: Sequence[int]) -> dict[str, np.ndarray]:
    """Load every requested behavior template, failing closed before workers start."""

    path = Path(f"results/phase2/behavior/{model}.json")
    if not path.is_file():
        raise FileNotFoundError(f"required model behavior file is missing: {path}")
    try:
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read model behavior file {path}") from exc
    if not isinstance(data, Mapping):
        raise ValueError(f"{path}: behavior payload must be an object")
    if not templates:
        raise ValueError("at least one behavior template is required")
    if len(set(templates)) != len(templates):
        raise ValueError(f"duplicate behavior templates requested: {list(templates)}")
    out: dict[str, np.ndarray] = {}
    for family in families:
        rows: list[np.ndarray] = []
        for template in templates:
            key = f"{family}__t{template}"
            if key not in data:
                raise ValueError(f"{path}: missing required behavior key {key}")
            value = data[key]
            if not isinstance(value, Mapping) or "total" not in value:
                raise ValueError(f"{path}: behavior key {key} lacks total matrix")
            rows.append(_require_finite_matrix(value["total"], shape=(n, n), source=f"{path}:{key}:total"))
        L = np.mean(rows, axis=0)
        L = L - _logsumexp(L, axis=1, keepdims=True)
        if not np.all(np.isfinite(L)):
            raise ValueError(f"{path}: normalized behavior for {family} is non-finite")
        out[family] = L
    return out


def _load_tokens(model: str) -> np.ndarray:
    """Load the one unique finite 15-key tokenizer ``n_span`` bank."""

    token_model = model if not model.startswith("olmo") else "olmo2_1b"
    path = Path(f"results/phase2/hidden/{token_model}_symbol_tokens.json")
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
        if not isinstance(value, Mapping) or "n_span" not in value:
            continue
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


def _merge_behavior(L: np.ndarray) -> np.ndarray:
    out = np.full((n, 12), -np.inf, dtype=float)
    pc = (7 * S) % 12
    for z in range(12):
        out[:, z] = np.logaddexp.reduce(L[:, pc == z], axis=1)
    return out


def _load_corpus(corpus_npz: str, extractions: Sequence[str]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Load and validate the requested corpus matrices before multiprocessing."""

    path = Path(corpus_npz)
    if not path.is_file():
        raise FileNotFoundError(f"required corpus file is missing: {path}")
    try:
        with np.load(path, allow_pickle=False) as archive:
            if "uni" not in archive.files:
                raise ValueError(f"{path}: missing required frequency vector uni")
            frequencies = np.asarray(archive["uni"], dtype=float)
            if frequencies.shape != (n,):
                raise ValueError(f"{path}: uni must have shape ({n},), got {frequencies.shape}")
            if not np.all(np.isfinite(frequencies)) or np.any(frequencies < 0):
                raise ValueError(f"{path}: uni must be finite and non-negative")
            zc: dict[str, np.ndarray] = {"uni": frequencies}
            for extraction in extractions:
                if extraction not in archive.files:
                    raise ValueError(f"{path}: missing requested corpus extraction {extraction}")
                zc[extraction] = _require_finite_matrix(
                    archive[extraction],
                    shape=(n, n),
                    source=f"{path}:{extraction}",
                    nonnegative=True,
                )
    except (OSError, ValueError) as exc:
        if isinstance(exc, ValueError):
            raise
        raise ValueError(f"cannot read corpus file {path}") from exc
    return zc, frequencies


def _load_perdoc(path_value: str, extractions: Sequence[str]) -> dict[str, np.ndarray]:
    """Load every document-cluster array and validate aligned document indices."""

    path = Path(path_value)
    if not path.is_file():
        raise FileNotFoundError(f"required per-document bootstrap file is missing: {path}")
    try:
        with np.load(path, allow_pickle=False) as archive:
            if "uni_docs" not in archive.files:
                raise ValueError(f"{path}: missing required per-document frequency matrix uni_docs")
            uni_docs = np.asarray(archive["uni_docs"], dtype=float)
            if uni_docs.ndim != 2 or uni_docs.shape[1] != n or uni_docs.shape[0] == 0:
                raise ValueError(f"{path}: uni_docs must have shape (documents, {n}), got {uni_docs.shape}")
            if not np.all(np.isfinite(uni_docs)) or np.any(uni_docs < 0):
                raise ValueError(f"{path}: uni_docs must be finite and non-negative")
            out: dict[str, np.ndarray] = {"uni_docs": uni_docs}
            ndocs = int(uni_docs.shape[0])
            for extraction in extractions:
                names = tuple(f"{extraction}_{suffix}" for suffix in ("doc", "i", "j", "c"))
                missing = [name for name in names if name not in archive.files]
                if missing:
                    raise ValueError(f"{path}: missing per-document fields for {extraction}: {missing}")
                doc = np.asarray(archive[names[0]], dtype=int)
                source = np.asarray(archive[names[1]], dtype=int)
                target = np.asarray(archive[names[2]], dtype=int)
                count = np.asarray(archive[names[3]], dtype=float)
                if any(value.ndim != 1 for value in (doc, source, target, count)) or not (len(doc) == len(source) == len(target) == len(count)):
                    raise ValueError(f"{path}:{extraction}: per-document arrays must be aligned one-dimensional vectors")
                if np.any(doc < 0) or np.any(doc >= ndocs):
                    raise ValueError(f"{path}:{extraction}: document index outside 0..{ndocs - 1}")
                if np.any(source < 0) or np.any(source >= n) or np.any(target < 0) or np.any(target >= n):
                    raise ValueError(f"{path}:{extraction}: key index outside 0..{n - 1}")
                if not np.all(np.isfinite(count)) or np.any(count < 0):
                    raise ValueError(f"{path}:{extraction}: counts must be finite and non-negative")
                out.update({names[0]: doc, names[1]: source, names[2]: target, names[3]: count})
    except OSError as exc:
        raise ValueError(f"cannot read per-document bootstrap file {path}") from exc
    return out


def main(argv: Sequence[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    args = [a for a in raw if not a.startswith("--")]
    if len(args) < 2:
        print(__doc__)
        return 2
    corpus_npz, corpus_name = args[:2]
    if "v4" not in corpus_name.casefold():
        raise ValueError(
            f"refusing non-v4 corpus name {corpus_name!r}; historical reproduction must use the immutable paper-v1 snapshot"
        )
    models = args[2].split(",") if len(args) > 2 else ["olmo2_1b", "gemma2_2b", "qwen25_3b", "olmo2_7b"]
    families = args[3].split(",") if len(args) > 3 else ["C_harmonic", "D_chord", "E_modulation"]
    extractions = args[4].split(",") if len(args) > 4 else ["A_win40", "B_any", "D_doc"]
    neutral = bool(_parse_flag(raw, "neutral", False))
    rich = bool(_parse_flag(raw, "rich", False))
    targetprior = bool(_parse_flag(raw, "targetprior", False))
    templates = tuple(int(x) for x in str(_parse_flag(raw, "templates", "0,1,2,3")).split(",") if x != "")
    nperm_res = int(_parse_flag(raw, "nperm-res", 5000))
    nperm_loo = int(_parse_flag(raw, "nperm-loo", 5000))
    nboot = int(_parse_flag(raw, "nboot", 0))
    jobs = int(_parse_flag(raw, "jobs", 1))
    docboot_arg = _parse_flag(raw, "docboot", None)
    tag = str(_parse_flag(raw, "tag", "") or "")
    view = "aggregated" if neutral else "spelled"
    output_name = f"{corpus_name}{'_neutral' if neutral else ''}{'_rich' if rich else ''}{'_tp' if targetprior else ''}{tag}"
    output_path = _v4_fingerprint_output_path(output_name)
    zc, frequencies = _load_corpus(corpus_npz, extractions)
    perdoc = None
    n_docboot = 0
    if docboot_arg:
        try:
            p, b = str(docboot_arg).rsplit(":", 1)
            n_docboot = int(b)
        except (ValueError, TypeError) as exc:
            raise ValueError("--docboot must be PATH:positive-integer-B") from exc
        if n_docboot <= 0:
            raise ValueError("--docboot B must be positive")
        perdoc = _load_perdoc(p, extractions)
    tasks: list[tuple[CellConfig, np.ndarray, np.ndarray, np.ndarray | None]] = []
    for model in models:
        q_by_family = _load_behavior(model, families, templates)
        tok = _load_tokens(model)
        for family in families:
            q = q_by_family[family]
            if view == "aggregated":
                q = _merge_behavior(q)
            for extraction in extractions:
                config = CellConfig(corpus_name, model, family, extraction, view, rich, targetprior, templates, nperm_res, nperm_loo, nboot)
                tasks.append((config, q, np.asarray(zc[extraction], dtype=float), tok))
    print(f"corpus={corpus_name}; source counts {frequencies.astype(int).tolist()}; view={view} rich={rich} targetprior={targetprior} templates={list(templates)} nperm_res={nperm_res} nperm_loo={nperm_loo} nboot={nboot} jobs={jobs}")
    print(f"{'model':10s} {'family':13s} {'extract':13s} | residual r (p) | LOO theory corpus both ΔCV(p) ΔR²(p) ΔKL(p)")
    # A doc-cluster mapping contains numpy arrays and is intentionally handled
    # serially.  Ordinary cells can be evaluated in a process pool; each cell
    # has identity-derived streams, so scheduling cannot alter values.
    if jobs > 1 and len(tasks) > 1 and perdoc is None:
        import multiprocessing as mp

        with mp.Pool(jobs) as pool:
            results = pool.starmap(compute_cell, [(cfg, q, M, frequencies, tok) for cfg, q, M, tok in tasks])
    else:
        results = [compute_cell(cfg, q, M, frequencies, tok, perdoc=perdoc, n_docboot=n_docboot) for cfg, q, M, tok in tasks]
    out: dict[str, Any] = {}
    for (cfg, _, _, _), (rec, line) in zip(tasks, results):
        print(line, flush=True)
        out[f"{cfg.model}|{cfg.family}|{cfg.extraction}"] = rec
    dump_json_strict(out, output_path)
    print("saved", output_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
