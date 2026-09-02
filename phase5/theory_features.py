"""Shared, auditable Phase-V theory features and fold utilities.

The Phase-V correction uses one feature definition for the spelled (15 target)
and target-aggregated (12 target-class) views.  This module deliberately keeps
feature construction separate from standardisation: raw features are built
once, while every held-out fit obtains its own mean and scale from training
pair observations.

Schema ``phase5-theory-v4.1`` is intentionally small and explicit.  In
particular, a target class is a *set* of spellings.  No class centroid, class
size/span, or order-dependent nearest spelling is used as a feature.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from phase2.keys15 import GLYPH, KEYS15, LETTER, PC, S, levenshtein, n


SCHEMA_VERSION = "phase5-theory-v4.1"
MASTER_SEED = 20260830
SEED_ALGORITHM = "sha256(canonical-json(master_seed, labels))[0:8]"
# This tolerance is relative to the largest singular value.  It is shared by
# rank audits and by the tokenizer-column inclusion rule.
RANK_RTOL = 1.0e-10

SPELLING_VIEWS = frozenset(("spelled", "spell", "15", "raw"))
AGGREGATED_VIEWS = frozenset(("aggregated", "neutral", "target_aggregated", "target-aggregated", "12"))


def normalise_view(view: str) -> str:
    """Return the canonical view name used in metadata and tests."""

    v = str(view).strip().lower()
    if v in SPELLING_VIEWS:
        return "spelled"
    if v in AGGREGATED_VIEWS:
        return "aggregated"
    raise ValueError(f"unknown Phase-V view: {view!r}")


def cyclic_distance(a: Any, b: Any, period: int = 12) -> np.ndarray:
    """Vectorised unsigned cyclic distance on ``Z_period``."""

    aa = np.asarray(a, dtype=int)
    bb = np.asarray(b, dtype=int)
    d = (aa - bb) % period
    return np.minimum(d, period - d).astype(float)


def circle_of_fifths_coordinate(pitch_class: Any) -> np.ndarray:
    """The fifth-coordinate class identity ``f(z) = 7 z mod 12``.

    ``PC`` is semitone pitch class.  Multiplication by seven is its inverse
    modulo twelve, so C (pc 0) and G (pc 7) map to fifth coordinates 0 and 1.
    """

    return (7 * np.asarray(pitch_class, dtype=int)) % 12


def circle_fifths_distance(source_pc: Any, target_pc: Any) -> np.ndarray:
    """True periodic fifths-circle distance, distinct from chromatic distance."""

    return cyclic_distance(circle_of_fifths_coordinate(source_pc), circle_of_fifths_coordinate(target_pc))


def open_line_fifths_distance(source_s: Any, target_s: Any) -> np.ndarray:
    """Absolute distance on the signed/open fifths coordinate line."""

    return np.abs(np.asarray(source_s, dtype=float) - np.asarray(target_s, dtype=float))


# Descriptive aliases make the two coordinate systems difficult to confuse in
# small diagnostics: ``circle_fifths_distance`` takes pitch classes, whereas
# this helper takes the signed key-signature coordinates directly.
true_circle_fifths_distance = circle_fifths_distance
fifths_line_distance = open_line_fifths_distance


def chromatic_cyclic_distance(source_pc: Any, target_pc: Any) -> np.ndarray:
    """Periodic semitone distance (kept separate from :func:`circle_fifths_distance`)."""

    return cyclic_distance(source_pc, target_pc)


def _class_members() -> tuple[np.ndarray, ...]:
    return tuple(np.flatnonzero(np.asarray(PC) == z) for z in range(12))


CLASS_MEMBERS = _class_members()
CLASS_FIFTH_COORDINATE = circle_of_fifths_coordinate(np.arange(12))


def pair_indices(view: str) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic row-major ``(source, target)`` pair indices.

    For ``aggregated``, target indices are pitch-class classes ``0..11`` and
    the source's own class is excluded.  The order is an implementation detail
    only; all set-valued descriptors below are order invariant.
    """

    v = normalise_view(view)
    if v == "spelled":
        mask = ~np.eye(n, dtype=bool)
        return np.where(mask)
    source, target = np.where(np.ones((n, 12), dtype=bool))
    keep = np.asarray(PC, dtype=int)[source] != target
    return source[keep], target[keep]


def _as_frequency(frequencies: Sequence[float] | None) -> np.ndarray:
    if frequencies is None:
        return np.ones(n, dtype=float)
    f = np.asarray(frequencies, dtype=float)
    if f.shape != (n,):
        raise ValueError(f"frequencies must have shape ({n},), got {f.shape}")
    if not np.all(np.isfinite(f)) or np.any(f < 0):
        raise ValueError("frequencies must be finite and non-negative")
    return f


def _as_pair_indices(view: str, source: Sequence[int] | None, target: Sequence[int] | None) -> tuple[np.ndarray, np.ndarray]:
    if source is None and target is None:
        i, j = pair_indices(view)
    elif source is None or target is None:
        raise ValueError("source and target must be supplied together")
    else:
        i, j = np.asarray(source, dtype=int), np.asarray(target, dtype=int)
        if i.shape != j.shape or i.ndim != 1:
            raise ValueError("source and target must be one-dimensional arrays with equal shape")
        v = normalise_view(view)
        if np.any(i < 0) or np.any(i >= n):
            raise ValueError("source index outside the 15-key bank")
        if v == "spelled":
            if np.any(j < 0) or np.any(j >= n) or np.any(i == j):
                raise ValueError("spelled target index outside bank or diagonal pair")
        elif np.any(j < 0) or np.any(j >= 12) or np.any(np.asarray(PC)[i] == j):
            raise ValueError("aggregated target class outside bank or own class")
    return i, j


def _bool(value: Any) -> float:
    return float(bool(value))


def _spelled_rows(i: int, j: int, freq: np.ndarray) -> list[float]:
    d5 = float(circle_fifths_distance(PC[i], PC[j]))
    chromatic = float(chromatic_cyclic_distance(PC[i], PC[j]))
    line = float(abs(int(S[j]) - int(S[i])))
    signed = float(int(S[j]) - int(S[i]))
    return [
        d5,
        chromatic,
        line,
        signed,
        _bool(GLYPH[i] == GLYPH[j]),
        _bool(GLYPH[i] == -1),
        _bool(GLYPH[i] == 1),
        _bool(GLYPH[j] == -1),
        _bool(GLYPH[j] == 1),
        _bool(LETTER[i] == LETTER[j]),
        float(abs(int(LETTER[i]) - int(LETTER[j]))),
        float(abs(abs(int(S[i])) - abs(int(S[j])))),
        float(np.log1p(freq[i])),
        float(np.log1p(freq[j])),
    ]


def _aggregated_rows(i: int, z: int, freq: np.ndarray) -> list[float]:
    members = np.asarray(CLASS_MEMBERS[int(z)], dtype=int)
    # The members are selected by class identity, not by an order-dependent
    # nearest spelling.  min/max and any() make the representation invariant
    # to the order in which a caller enumerates T.
    fifth = float(cyclic_distance(circle_of_fifths_coordinate(PC[i]), CLASS_FIFTH_COORDINATE[z]))
    chromatic = float(chromatic_cyclic_distance(PC[i], z))
    line_abs = np.abs(np.asarray(S[members], dtype=float) - float(S[i]))
    signed = np.asarray(S[members], dtype=float) - float(S[i])
    alphabet = np.abs(np.asarray(LETTER[members], dtype=float) - float(LETTER[i]))
    signature_size = np.abs(np.abs(np.asarray(S[members], dtype=float)) - abs(float(S[i])))
    edits = np.asarray([levenshtein(KEYS15[i], KEYS15[k]) for k in members], dtype=float)
    return [
        fifth,
        chromatic,
        float(np.min(line_abs)),
        float(np.max(line_abs)),
        float(np.min(signed)),
        float(np.max(signed)),
        float(np.min(alphabet)),
        float(np.min(signature_size)),
        float(np.max(signature_size)),
        float(np.min(edits)),
        float(np.max(edits)),
        _bool(GLYPH[i] == -1),
        _bool(GLYPH[i] == 1),
        _bool(np.any(GLYPH[members] == -1)),
        _bool(np.any(GLYPH[members] == 1)),
        _bool(np.any(GLYPH[i] == GLYPH[members])),
        _bool(np.any(LETTER[i] == LETTER[members])),
        float(np.log1p(freq[i])),
        float(np.log1p(np.sum(freq[members]))),
    ]


SPELLED_BASE_NAMES = (
    "circle_fifths",
    "chromatic_cyclic",
    "line_abs",
    "line_signed",
    "same_glyph",
    "source_is_flat",
    "source_is_sharp",
    "target_is_flat",
    "target_is_sharp",
    "same_root",
    "alphabet_distance",
    "signature_size_diff",
    "source_logfreq",
    "target_logfreq",
)
AGGREGATED_BASE_NAMES = (
    "circle_fifths",
    "chromatic_cyclic",
    "line_min_abs",
    "line_max_abs",
    "line_signed_low",
    "line_signed_high",
    "alphabet_min_distance",
    "signature_size_min_diff",
    "signature_size_max_diff",
    "edit_min",
    "edit_max",
    "source_is_flat",
    "source_is_sharp",
    "target_contains_flat",
    "target_contains_sharp",
    "any_same_glyph",
    "any_same_root",
    "source_logfreq",
    "target_logfreq_sum",
)

SPELLED_RICH_NAMES = (
    "circle_fifths_sq",
    "line_abs_sq",
    "circle_fifths_x_line_abs",
    "chromatic_x_line_abs",
    "circle_fifths_cos1",
    "circle_fifths_cos2",
)
AGGREGATED_RICH_NAMES = (
    "circle_fifths_sq",
    "line_max_abs_sq",
    "circle_fifths_x_line_min_abs",
    "chromatic_x_line_min_abs",
    "circle_fifths_cos1",
    "circle_fifths_cos2",
)


@dataclass
class FeatureSet:
    """Raw feature matrix plus its pair labels and audit metadata.

    ``FeatureSet`` can be unpacked as ``values, names`` for small callers,
    while named access is preferred in the pipeline.
    """

    values: np.ndarray
    names: tuple[str, ...]
    source: np.ndarray
    target: np.ndarray
    view: str
    rich: bool
    frequencies: np.ndarray
    metadata: dict[str, Any]

    @property
    def X(self) -> np.ndarray:
        return self.values

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self.names

    @property
    def matrix(self) -> np.ndarray:
        return self.values

    def __getitem__(self, key: str | int | slice):
        if isinstance(key, str):
            return self.values[:, self.names.index(key)]
        return self.values[key]

    def __iter__(self):
        yield self.values
        yield self.names


def build_raw_features(
    view: str,
    frequencies: Sequence[float] | None = None,
    *,
    uni: Sequence[float] | None = None,
    frequency: Sequence[float] | None = None,
    rich: bool = False,
    tokenizer_residual: Sequence[float] | None = None,
    token_counts: Sequence[float] | None = None,
    source: Sequence[int] | None = None,
    target: Sequence[int] | None = None,
) -> FeatureSet:
    """Build the ordered, *raw* v4.1 feature matrix.

    ``tokenizer_residual`` is optional because its coefficients are fit in a
    fold.  Pass the 15-key residual vector to add the appropriate source and
    target controls.  No standardisation happens here.
    """

    if frequencies is None:
        frequencies = uni if uni is not None else frequency
    elif uni is not None or frequency is not None:
        raise ValueError("supply frequencies once (not frequencies plus uni/frequency)")
    if tokenizer_residual is not None and token_counts is not None:
        raise ValueError("supply tokenizer_residual or token_counts, not both")
    if token_counts is not None:
        # This all-key convenience is intended for non-held-out diagnostics;
        # LOO callers use make_fold_feature_set so the fit is fold-local.
        tokenizer_residual = fit_tokenizer_irregularity(token_counts)["residual"]
    v = normalise_view(view)
    freq = _as_frequency(frequencies)
    i, j = _as_pair_indices(v, source, target)
    if tokenizer_residual is not None:
        tr = np.asarray(tokenizer_residual, dtype=float)
        if tr.shape != (n,) or not np.all(np.isfinite(tr)):
            raise ValueError(f"tokenizer_residual must be finite with shape ({n},)")
    else:
        tr = None

    base = np.asarray(
        [_spelled_rows(a, b, freq) if v == "spelled" else _aggregated_rows(a, b, freq) for a, b in zip(i, j)],
        dtype=float,
    )
    names = list(SPELLED_BASE_NAMES if v == "spelled" else AGGREGATED_BASE_NAMES)
    if rich:
        if v == "spelled":
            d5, chromatic, line = base[:, 0], base[:, 1], base[:, 2]
            additions = np.column_stack(
                [d5**2, line**2, d5 * line, chromatic * line, np.cos(2 * np.pi * d5 / 12), np.cos(4 * np.pi * d5 / 12)]
            )
            add_names = list(SPELLED_RICH_NAMES)
        else:
            d5, chromatic, nearest, farthest = base[:, 0], base[:, 1], base[:, 2], base[:, 3]
            additions = np.column_stack(
                [d5**2, farthest**2, d5 * nearest, chromatic * nearest, np.cos(2 * np.pi * d5 / 12), np.cos(4 * np.pi * d5 / 12)]
            )
            add_names = list(AGGREGATED_RICH_NAMES)
        base = np.column_stack([base, additions])
        names.extend(add_names)

    if tr is not None:
        if v == "spelled":
            tok_values = np.column_stack([tr[i], tr[j]])
            tok_names = ["tokenizer_residual_source", "tokenizer_residual_target"]
        else:
            means = np.asarray([float(np.mean(tr[members])) for members in CLASS_MEMBERS])
            tok_values = np.column_stack([tr[i], means[j]])
            tok_names = ["tokenizer_residual_source", "tokenizer_residual_target_mean"]
        base = np.column_stack([base, tok_values])
        names.extend(tok_names)

    return FeatureSet(
        values=base,
        names=tuple(names),
        source=i.astype(int, copy=False),
        target=j.astype(int, copy=False),
        view=v,
        rich=bool(rich),
        frequencies=freq.copy(),
        metadata={
            "schema": SCHEMA_VERSION,
            "view": v,
            "rich": bool(rich),
            "feature_definitions": feature_definitions(v, rich=rich, tokenizer=tr is not None),
        },
    )


# Convenient aliases used by focused tests and downstream scripts.
build_features = build_raw_features
build_feature_matrix = build_raw_features
build_theory_features = build_raw_features


def feature_definitions(view: str, *, rich: bool = False, tokenizer: bool = False) -> dict[str, str]:
    """Return human-readable definitions for the exact ordered feature schema."""

    v = normalise_view(view)
    d: dict[str, str] = {
        "circle_fifths": "min(|f(PC_i)-f(PC_j)| mod 12, 12-|...|), f(z)=7z mod 12",
        "chromatic_cyclic": "cyclic semitone distance between PC_i and PC_j",
    }
    if v == "spelled":
        d.update(
            {
                "line_abs": "|s_j-s_i| on the open signed fifths line",
                "line_signed": "s_j-s_i",
                "same_glyph": "1[ glyph_i = glyph_j ]",
                "source_is_flat": "1[ glyph_i is flat ]",
                "source_is_sharp": "1[ glyph_i is sharp ]",
                "target_is_flat": "1[ glyph_j is flat ]",
                "target_is_sharp": "1[ glyph_j is sharp ]",
                "same_root": "1[ root-letter_i = root-letter_j ]",
                "alphabet_distance": "absolute root-letter alphabet-index difference",
                "signature_size_diff": "||s_i|-|s_j||",
                "source_logfreq": "log(1+frequency_i)",
                "target_logfreq": "log(1+frequency_j)",
            }
        )
    else:
        d.update(
            {
                "line_min_abs": "min_{t in T}|s_t-s_i|",
                "line_max_abs": "max_{t in T}|s_t-s_i|",
                "line_signed_low": "min_{t in T}s_t-s_i",
                "line_signed_high": "max_{t in T}s_t-s_i",
                "alphabet_min_distance": "min_{t in T}|alphabet(root_t)-alphabet(root_i)|",
                "signature_size_min_diff": "min_{t in T}||s_t|-|s_i||",
                "signature_size_max_diff": "max_{t in T}||s_t|-|s_i||",
                "edit_min": "min_{t in T}Levenshtein(key_i,key_t)",
                "edit_max": "max_{t in T}Levenshtein(key_i,key_t)",
                "source_is_flat": "1[ glyph_i is flat ]",
                "source_is_sharp": "1[ glyph_i is sharp ]",
                "target_contains_flat": "1[ exists t in T with flat glyph ]",
                "target_contains_sharp": "1[ exists t in T with sharp glyph ]",
                "any_same_glyph": "1[ exists t in T with glyph_t=glyph_i ]",
                "any_same_root": "1[ exists t in T with root_t=root_i ]",
                "source_logfreq": "log(1+frequency_i)",
                "target_logfreq_sum": "log(1+sum_{t in T}frequency_t)",
            }
        )
    if rich:
        if v == "spelled":
            d.update(
                {
                    "circle_fifths_sq": "circle_fifths^2",
                    "line_abs_sq": "line_abs^2",
                    "circle_fifths_x_line_abs": "circle_fifths*line_abs",
                    "chromatic_x_line_abs": "chromatic_cyclic*line_abs",
                    "circle_fifths_cos1": "cos(2*pi*circle_fifths/12)",
                    "circle_fifths_cos2": "cos(4*pi*circle_fifths/12)",
                }
            )
        else:
            d.update(
                {
                    "circle_fifths_sq": "circle_fifths^2",
                    "line_max_abs_sq": "line_max_abs^2 (farthest line distance)",
                    "circle_fifths_x_line_min_abs": "circle_fifths*line_min_abs (nearest distance)",
                    "chromatic_x_line_min_abs": "chromatic_cyclic*line_min_abs",
                    "circle_fifths_cos1": "cos(2*pi*circle_fifths/12)",
                    "circle_fifths_cos2": "cos(4*pi*circle_fifths/12)",
                }
            )
    if tokenizer:
        d.update(
            {
                "tokenizer_residual_source": "r(i), residual from token_count~1+is_flat+is_sharp fit on training source keys",
                "tokenizer_residual_target": "r(j), same fold fit",
                "tokenizer_residual_target_mean": "mean_{t in T}r(t), same fold fit",
            }
        )
    return d


def _canonical_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _canonical_json(value[k]) for k in sorted(value, key=lambda x: str(x))}
    if isinstance(value, (list, tuple)):
        return [_canonical_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_canonical_json(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def stable_seed(master_seed: int = MASTER_SEED, *parts: Any, **labels: Any) -> int:
    """Derive a process/order-independent 63-bit seed via SHA-256.

    Keyword labels are sorted before hashing.  This function intentionally
    never calls Python's salted :func:`hash`.
    """

    payload: dict[str, Any] = {"master_seed": int(master_seed), "parts": list(parts)}
    if labels:
        payload["labels"] = {str(k): labels[k] for k in sorted(labels)}
    blob = json.dumps(_canonical_json(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    # Keep the result in numpy's portable non-negative signed-integer range.
    return int.from_bytes(hashlib.sha256(blob).digest()[:8], "big") & ((1 << 63) - 1)


derive_seed = stable_seed


def stable_rng(master_seed: int = MASTER_SEED, *parts: Any, **labels: Any) -> np.random.Generator:
    return np.random.default_rng(stable_seed(master_seed, *parts, **labels))


def fit_scaler(X_train: np.ndarray, *, relative_tolerance: float = RANK_RTOL) -> dict[str, np.ndarray | float]:
    """Fit a mean/scale transform on training observations only.

    Constant predictors get exactly scale 1.  ``transform_scaler`` always
    applies these training-derived values to held-out observations.
    """

    X = np.asarray(X_train, dtype=float)
    if X.ndim != 2 or X.shape[0] == 0:
        raise ValueError("X_train must be a non-empty two-dimensional array")
    if not np.all(np.isfinite(X)):
        raise ValueError("X_train contains non-finite predictors")
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    maxabs = np.maximum(1.0, np.max(np.abs(X), axis=0))
    constant = std <= float(relative_tolerance) * maxabs
    scale = np.where(constant, 1.0, std)
    return {"mean": mean, "scale": scale, "constant": constant, "relative_tolerance": float(relative_tolerance)}


def transform_scaler(X: np.ndarray, scaler: Mapping[str, Any]) -> np.ndarray:
    return (np.asarray(X, dtype=float) - np.asarray(scaler["mean"], dtype=float)) / np.asarray(scaler["scale"], dtype=float)


def fit_transform_train_test(X_train: np.ndarray, X_test: np.ndarray, *, relative_tolerance: float = RANK_RTOL) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    scaler = fit_scaler(X_train, relative_tolerance=relative_tolerance)
    return transform_scaler(X_train, scaler), transform_scaler(X_test, scaler), scaler


def _rank_from_singular_values(X: np.ndarray, relative_tolerance: float) -> tuple[int, np.ndarray, float]:
    if X.size == 0:
        return 0, np.zeros(0), 0.0
    singular = np.linalg.svd(np.asarray(X, dtype=float), full_matrices=False, compute_uv=False)
    threshold = float(relative_tolerance) * (float(singular[0]) if singular.size else 0.0)
    return int(np.sum(singular > threshold)), singular, threshold


def matrix_rank_audit(
    X: np.ndarray,
    names: Sequence[str] | None = None,
    *,
    relative_tolerance: float = RANK_RTOL,
    include_intercept: bool = True,
    context: str = "design",
) -> dict[str, Any]:
    """Audit zero variance and strict numerical rank without changing ``X``.

    ``rank`` is the rank of the actual design (including an intercept by
    default).  ``independent_feature_names`` greedily identifies a maximal
    independent feature subset, and ``omissions`` records zero-variance or
    dependent columns.  The caller decides whether an omission is expected.
    """

    A = np.asarray(X, dtype=float)
    if A.ndim != 2:
        raise ValueError("design must be two-dimensional")
    if not np.all(np.isfinite(A)):
        raise ValueError(f"{context} contains non-finite predictors")
    p = A.shape[1]
    nm = tuple(str(x) for x in (names if names is not None else [f"x{k}" for k in range(p)]))
    if len(nm) != p:
        raise ValueError("feature names do not match design columns")
    maxabs = np.maximum(1.0, np.max(np.abs(A), axis=0)) if p else np.zeros(0)
    zero = np.std(A, axis=0) <= float(relative_tolerance) * maxabs if p else np.zeros(0, dtype=bool)
    design = np.column_stack([np.ones(A.shape[0]), A]) if include_intercept else A
    rank, singular, threshold = _rank_from_singular_values(design, relative_tolerance)
    # Greedy audit: append only columns that raise the strict rank.  This is
    # used for reporting and tokenizer selection, never to silently alter a
    # requested model.
    independent: list[str] = []
    selected: list[int] = []
    current = np.ones((A.shape[0], 1), dtype=float) if include_intercept else np.empty((A.shape[0], 0), dtype=float)
    current_rank = _rank_from_singular_values(current, relative_tolerance)[0] if current.size else 0
    dependent: list[str] = []
    for k, name in enumerate(nm):
        trial = np.column_stack([current, A[:, k]])
        trial_rank = _rank_from_singular_values(trial, relative_tolerance)[0]
        if not zero[k] and trial_rank > current_rank:
            independent.append(name)
            selected.append(k)
            current, current_rank = trial, trial_rank
        else:
            dependent.append(name)
    zero_names = [nm[k] for k in range(p) if zero[k]]
    dependent_nonzero = [name for name in dependent if name not in zero_names]
    omissions = list(dict.fromkeys(zero_names + dependent_nonzero))
    return {
        "context": context,
        "relative_tolerance": float(relative_tolerance),
        "n_observations": int(A.shape[0]),
        "n_features": int(p),
        "rank": int(rank),
        "full_rank_expected": int((1 if include_intercept else 0) + len(independent)),
        "full_rank": bool(rank == ((1 if include_intercept else 0) + len(independent))),
        "zero_variance": zero_names,
        "dependent": dependent_nonzero,
        "omissions": omissions,
        "feature_names": list(nm),
        "independent_feature_names": independent,
        "singular_values": [float(x) for x in singular],
        "threshold": float(threshold),
        "include_intercept": bool(include_intercept),
    }


def assert_expected_rank(audit: Mapping[str, Any], *, allow_omissions: Iterable[str] = ()) -> None:
    """Raise on rank loss not explicitly recorded as an allowed omission."""

    allowed = set(str(x) for x in allow_omissions)
    unexpected = [x for x in audit.get("omissions", []) if x not in allowed]
    if unexpected:
        raise np.linalg.LinAlgError(
            f"{audit.get('context', 'design')} rank deficiency at relative tolerance "
            f"{audit.get('relative_tolerance')}: unexpected omissions {unexpected}"
        )


def fit_tokenizer_irregularity(
    token_counts: Sequence[float],
    train_sources: Sequence[int] | None = None,
    *,
    relative_tolerance: float = RANK_RTOL,
) -> dict[str, Any]:
    """Fit ``token_count ~ 1 + is_flat + is_sharp`` on training source keys.

    Coefficients are fit only on ``train_sources`` but residuals are returned
    for all 15 keys, making application to a held-out source explicit.
    """

    y = np.asarray(token_counts, dtype=float)
    if y.shape != (n,) or not np.all(np.isfinite(y)):
        raise ValueError(f"token_counts must be finite with shape ({n},)")
    train = np.arange(n, dtype=int) if train_sources is None else np.asarray(train_sources, dtype=int)
    if train.ndim != 1 or len(train) == 0 or np.any(train < 0) or np.any(train >= n):
        raise ValueError("train_sources must contain valid non-empty key indices")
    design_all = np.column_stack([np.ones(n), (GLYPH == -1).astype(float), (GLYPH == 1).astype(float)])
    design_train = design_all[train]
    audit = matrix_rank_audit(design_train[:, 1:], ("is_flat", "is_sharp"), relative_tolerance=relative_tolerance, context="tokenizer_fit")
    # The 14-key bank retains all three glyph groups in every ordinary LOO
    # fold.  For a deliberately pathological caller, lstsq remains defined
    # and the audit is preserved rather than hidden.
    coef = np.linalg.lstsq(design_train, y[train], rcond=None)[0]
    fitted = design_all @ coef
    residual = y - fitted
    maxabs = max(1.0, float(np.max(np.abs(residual))))
    nonzero = bool(np.max(np.abs(residual)) > float(relative_tolerance) * maxabs)
    return {
        "coefficients": coef,
        "fitted": fitted,
        "residual": residual,
        "train_sources": train,
        "nonzero": nonzero,
        "rank_audit": audit,
        "relative_tolerance": float(relative_tolerance),
    }


def tokenizer_pair_columns(view: str, source: Sequence[int], target: Sequence[int], residual: Sequence[float]) -> tuple[np.ndarray, tuple[str, ...]]:
    """Return candidate pair-level tokenizer residual columns."""

    v = normalise_view(view)
    i, j = np.asarray(source, dtype=int), np.asarray(target, dtype=int)
    r = np.asarray(residual, dtype=float)
    if r.shape != (n,):
        raise ValueError("residual must have one value per key")
    if v == "spelled":
        return np.column_stack([r[i], r[j]]), ("tokenizer_residual_source", "tokenizer_residual_target")
    means = np.asarray([np.mean(r[members]) for members in CLASS_MEMBERS])
    return np.column_stack([r[i], means[j]]), ("tokenizer_residual_source", "tokenizer_residual_target_mean")


def _token_nonzero(column: np.ndarray, *, relative_tolerance: float = RANK_RTOL) -> bool:
    c = np.asarray(column, dtype=float)
    return bool(np.max(np.abs(c)) > float(relative_tolerance) * max(1.0, float(np.max(np.abs(c)))))


def append_tokenizer_columns_if_identifiable(
    feature_set: FeatureSet,
    residual: Sequence[float],
    train_mask: Sequence[bool],
    *,
    relative_tolerance: float = RANK_RTOL,
) -> tuple[FeatureSet, dict[str, Any]]:
    """Append only nonzero tokenizer columns that raise training rank.

    The rank check includes an intercept and uses the same strict relative
    tolerance as all other audits.  It is deliberately sequential so each
    retained column is independently justified.
    """

    mask = np.asarray(train_mask, dtype=bool)
    if mask.shape != (len(feature_set.values),):
        raise ValueError("train_mask has the wrong number of pair observations")
    candidates, candidate_names = tokenizer_pair_columns(feature_set.view, feature_set.source, feature_set.target, residual)
    selected: list[int] = []
    current = feature_set.values.copy()
    base_audit = matrix_rank_audit(current[mask], feature_set.names, relative_tolerance=relative_tolerance, context="tokenizer_base")
    current_rank = int(base_audit["rank"])
    omissions: list[dict[str, Any]] = []
    for k, name in enumerate(candidate_names):
        col = candidates[:, k]
        if not _token_nonzero(col, relative_tolerance=relative_tolerance):
            omissions.append({"name": name, "reason": "zero"})
            continue
        trial = np.column_stack([current, col])
        trial_audit = matrix_rank_audit(trial[mask], tuple(list(feature_set.names) + [name] if not selected else list(feature_set.names) + [candidate_names[x] for x in selected] + [name]), relative_tolerance=relative_tolerance, context=f"tokenizer_candidate:{name}")
        if int(trial_audit["rank"]) > current_rank:
            current = trial
            current_rank = int(trial_audit["rank"])
            selected.append(k)
        else:
            omissions.append({"name": name, "reason": "no_rank_increase"})
    names = tuple(list(feature_set.names) + [candidate_names[k] for k in selected])
    result = FeatureSet(
        values=current,
        names=names,
        source=feature_set.source,
        target=feature_set.target,
        view=feature_set.view,
        rich=feature_set.rich,
        frequencies=feature_set.frequencies,
        metadata={**feature_set.metadata, "tokenizer_selected": list(names[len(feature_set.names) :]), "tokenizer_omissions": omissions},
    )
    return result, {"base": base_audit, "selected": [candidate_names[k] for k in selected], "omissions": omissions}


def make_fold_feature_set(
    view: str,
    frequencies: Sequence[float],
    token_counts: Sequence[float] | None,
    heldout_source: int,
    *,
    rich: bool = False,
    relative_tolerance: float = RANK_RTOL,
) -> tuple[FeatureSet, dict[str, Any]]:
    """Build the feature set used by one source-row LOO fold."""

    base = build_raw_features(view, frequencies, rich=rich)
    if token_counts is None:
        return base, {"tokenizer": None}
    train_mask = base.source != int(heldout_source)
    train_sources = np.arange(n, dtype=int)[np.arange(n) != int(heldout_source)]
    fit = fit_tokenizer_irregularity(token_counts, train_sources, relative_tolerance=relative_tolerance)
    augmented, audit = append_tokenizer_columns_if_identifiable(base, fit["residual"], train_mask, relative_tolerance=relative_tolerance)
    audit["fit"] = {k: v for k, v in fit.items() if k not in ("coefficients", "fitted", "residual")}
    return augmented, audit


def training_target_prior(values: Sequence[float], source: Sequence[int], target: Sequence[int], train_mask: Sequence[bool], n_targets: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    """Compute target means from training responses only."""

    y = np.asarray(values, dtype=float)
    i, j, mask = np.asarray(source, dtype=int), np.asarray(target, dtype=int), np.asarray(train_mask, dtype=bool)
    if y.ndim != 1 or i.shape != y.shape or j.shape != y.shape or mask.shape != y.shape:
        raise ValueError("values, source, target, and train_mask must be aligned vectors")
    if n_targets is None:
        # Infer class labels only when the observed target indices are all in
        # 0..11 and do not require the 15-key spelling bank.  Production LOO
        # calls pass n_targets explicitly, making the view boundary explicit.
        nt = 12 if len(j) and np.max(j) < 12 and set(np.unique(j)).issubset(set(range(12))) else n
    else:
        nt = int(n_targets)
    prior = np.zeros(nt, dtype=float)
    counts = np.zeros(nt, dtype=int)
    for z in range(nt):
        take = mask & (j == z)
        if np.any(take):
            prior[z] = float(np.mean(y[take]))
            counts[z] = int(np.sum(take))
        else:
            prior[z] = 0.0
    return prior, {"n_targets": nt, "training_observations": counts.tolist(), "heldout_excluded": int(np.sum(~mask))}


_PROJECTION_CACHE: dict[tuple[Any, ...], tuple[FeatureSet, np.ndarray, np.ndarray, dict[str, Any]]] = {}


def residual_projection(
    values: Sequence[float],
    feature_set: FeatureSet,
    *,
    token_counts: Sequence[float] | None = None,
    source_effects: bool = True,
    relative_tolerance: float = RANK_RTOL,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Full-sample, unpenalised rich+source-effects residual projection.

    This is intentionally non-held-out and is shared by fingerprint residual
    correspondence and checkpoint correspondence.  Theory features are
    standardised with full-sample pair statistics.  Source-only columns are
    omitted when source fixed effects are included because they lie exactly in
    the fixed-effect span; all omissions are recorded.  A source reference
    (key index 0) is dropped, so the intercept and dummies have a full-rank
    parameterisation.
    """

    y = np.asarray(values, dtype=float)
    if y.shape != (len(feature_set.values),) or not np.all(np.isfinite(y)):
        raise ValueError("values must be finite and aligned to feature_set")
    # Null relabelings reuse the same full-sample design thousands of times.
    # Cache its unpenalised projection by the immutable feature-set identity;
    # response values never participate in this key.
    cache_key = (id(feature_set), id(token_counts) if token_counts is not None else None, bool(source_effects), float(relative_tolerance))
    cached = _PROJECTION_CACHE.get(cache_key)
    if cached is not None:
        _, design, pinv, cached_meta = cached
        return y - design @ (pinv @ y), cached_meta
    fs = feature_set
    tok_meta = None
    if token_counts is not None:
        fit = fit_tokenizer_irregularity(token_counts, relative_tolerance=relative_tolerance)
        fs, tok_meta = append_tokenizer_columns_if_identifiable(fs, fit["residual"], np.ones(len(fs.values), dtype=bool), relative_tolerance=relative_tolerance)
    # Omit source-constant theory predictors before adding source effects.  A
    # source-row intercept absorbs them, and retaining them creates a silent
    # singular OLS design.  The omission is semantic, not a numerical hack.
    keep: list[int] = []
    omitted: list[dict[str, str]] = []
    for k, name in enumerate(fs.names):
        vals = fs.values[:, k]
        varying = False
        for src in range(n):
            row = vals[fs.source == src]
            if len(row) and np.std(row) > relative_tolerance * max(1.0, float(np.max(np.abs(row)))):
                varying = True
                break
        if source_effects and not varying:
            omitted.append({"name": name, "reason": "source_fixed_effect_span"})
        else:
            keep.append(k)
    Xraw = fs.values[:, keep]
    kept_names = tuple(fs.names[k] for k in keep)
    if Xraw.shape[1]:
        scaler = fit_scaler(Xraw, relative_tolerance=relative_tolerance)
        X = transform_scaler(Xraw, scaler)
    else:
        scaler = {"mean": np.zeros(0), "scale": np.ones(0), "constant": np.zeros(0, dtype=bool), "relative_tolerance": relative_tolerance}
        X = np.empty((len(y), 0), dtype=float)
    names = list(kept_names)
    if source_effects:
        # Key 0 is the explicit reference.  The all-zero reference row is
        # represented by the intercept.
        dummies = np.column_stack([(fs.source == src).astype(float) for src in range(1, n)])
        X = np.column_stack([X, dummies])
        names.extend([f"source_effect_{src}" for src in range(1, n)])
    design = np.column_stack([np.ones(len(y)), X])
    audit = matrix_rank_audit(X, names, relative_tolerance=relative_tolerance, context="full_sample_rich_source_effects")
    # At this point every source-only duplicate has been removed.  Any
    # remaining omission is unexpected and must stop the analysis.
    assert_expected_rank(audit)
    coef = np.linalg.lstsq(design, y, rcond=None)[0]
    resid = y - design @ coef
    meta = {
        "schema": SCHEMA_VERSION,
        "label": "non-held-out rich+source-effects",
        "source_effects": bool(source_effects),
        "source_reference": 0 if source_effects else None,
        "feature_names": names,
        "theory_feature_names": list(kept_names),
        "omissions": omitted,
        "rank_audit": audit,
        "scaler": {
            "mean": np.asarray(scaler["mean"]).tolist(),
            "scale": np.asarray(scaler["scale"]).tolist(),
            "constant": np.asarray(scaler["constant"]).astype(bool).tolist(),
            "fit_scope": "all pair observations (non-held-out)",
        },
        "tokenizer": tok_meta,
    }
    _PROJECTION_CACHE[cache_key] = (feature_set, design, np.linalg.pinv(design, rcond=relative_tolerance), meta)
    return resid, meta


def finite_or_none(value: Any) -> Any:
    """Recursively convert non-finite numerics to JSON null."""

    if isinstance(value, Mapping):
        return {str(k): finite_or_none(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_or_none(v) for v in value]
    if isinstance(value, np.ndarray):
        return finite_or_none(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def dump_json_strict(payload: Any, path: str | Path) -> Any:
    """Atomically write strict JSON, replacing non-finite values.

    A same-directory replacement prevents partial files and replaces a
    pre-existing symlink itself rather than following it.  Callers that own a
    protected output namespace additionally reject symlink paths before this
    helper is reached.
    """

    safe = finite_or_none(payload)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=str(destination.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(safe, fh, indent=1, allow_nan=False)
            fh.write("\n")
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return safe


def finite_status(payload: Any) -> tuple[Any, bool, list[str]]:
    """Return sanitised payload, availability status, and null field paths."""

    paths: list[str] = []

    def walk(x: Any, path: str) -> Any:
        if isinstance(x, Mapping):
            return {str(k): walk(v, f"{path}.{k}" if path else str(k)) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [walk(v, f"{path}[{k}]") for k, v in enumerate(x)]
        if isinstance(x, np.ndarray):
            return walk(x.tolist(), path)
        if isinstance(x, (np.floating, float)):
            if not math.isfinite(float(x)):
                paths.append(path)
                return None
            return float(x)
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, np.bool_):
            return bool(x)
        return x

    result = walk(payload, "")
    return result, not paths, paths


__all__ = [
    "AGGREGATED_BASE_NAMES",
    "AGGREGATED_RICH_NAMES",
    "CLASS_MEMBERS",
    "FeatureSet",
    "MASTER_SEED",
    "RANK_RTOL",
    "SCHEMA_VERSION",
    "SEED_ALGORITHM",
    "SPELLED_BASE_NAMES",
    "SPELLED_RICH_NAMES",
    "append_tokenizer_columns_if_identifiable",
    "build_feature_matrix",
    "build_features",
    "build_raw_features",
    "build_theory_features",
    "circle_fifths_distance",
    "circle_of_fifths_coordinate",
    "chromatic_cyclic_distance",
    "cyclic_distance",
    "derive_seed",
    "dump_json_strict",
    "feature_definitions",
    "finite_or_none",
    "finite_status",
    "fit_scaler",
    "fit_fold_scaler",
    "fit_fold_tokenizer",
    "fit_tokenizer_irregularity",
    "fit_transform_train_test",
    "make_fold_feature_set",
    "matrix_rank_audit",
    "normalise_view",
    "open_line_fifths_distance",
    "pair_indices",
    "get_pair_indices",
    "residual_projection",
    "stable_rng",
    "stable_seed",
    "training_target_prior",
    "target_prior_from_training",
    "transform_scaler",
    "tokenizer_pair_columns",
    "true_circle_fifths_distance",
    "fifths_line_distance",
]

# Small compatibility aliases for diagnostics that prefer verb-based names.
fit_fold_scaler = fit_scaler
fit_fold_tokenizer = fit_tokenizer_irregularity
get_pair_indices = pair_indices
target_prior_from_training = training_target_prior
