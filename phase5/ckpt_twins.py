"""Deterministic checkpoint twin-difference controls for the v4 correction.

For each OLMo checkpoint, family, enharmonic source pair, and extraction, the
target-permutation null uses an identity-derived SHA-256 stream and the
finite-sample estimator ``(b+1)/(B+1)``.  The historical v3 JSON is never
overwritten; v4 writes ``ckpt_twins_v4.json`` and ``ckpt_twins_v4.txt``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from phase2.keys15 import ENH_PAIRS, GLYPH, KEYS15, S, n
from phase5.theory_features import MASTER_SEED, SCHEMA_VERSION, SEED_ALGORITHM, dump_json_strict, stable_rng, stable_seed


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
    "main",
]
N_PERM = 2000


def _logc(counts: np.ndarray) -> np.ndarray:
    matrix = np.asarray(counts, dtype=float)
    conditional = (matrix + 0.5) / np.sum(matrix + 0.5, axis=1, keepdims=True)
    return np.log(conditional)


def _load(revision: str, family: str) -> np.ndarray:
    tag = "olmo2_1b" if revision == "main" else f"olmo2_1b_{revision}"
    path = Path(f"results/phase2/behavior/{tag}.json")
    if not path.is_file():
        raise FileNotFoundError(f"required checkpoint behavior file is missing: {path}")
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: behavior payload must be an object")
    rows: list[np.ndarray] = []
    for template in range(4):
        key = f"{family}__t{template}"
        value = payload.get(key)
        if not isinstance(value, dict) or "total" not in value:
            raise ValueError(f"{path}: missing required behavior matrix {key}")
        matrix = np.asarray(value["total"], dtype=float)
        if matrix.shape != (n, n) or not np.all(np.isfinite(matrix)):
            raise ValueError(f"{path}:{key}: total must be a finite ({n}, {n}) matrix")
        rows.append(matrix)
    matrix = np.mean(rows, axis=0)
    return matrix - np.logaddexp.reduce(matrix, axis=1, keepdims=True)


def _load_corpus() -> dict[str, np.ndarray]:
    path = Path("results/phase5/cond_wikipedia.npz")
    if not path.is_file():
        raise FileNotFoundError(f"required checkpoint corpus file is missing: {path}")
    out: dict[str, np.ndarray] = {}
    with np.load(path, allow_pickle=False) as archive:
        for extraction in ("A_win40", "D_doc"):
            if extraction not in archive.files:
                raise ValueError(f"{path}: missing required extraction {extraction}")
            matrix = np.asarray(archive[extraction], dtype=float)
            if matrix.shape != (n, n) or not np.all(np.isfinite(matrix)) or np.any(matrix < 0):
                raise ValueError(f"{path}:{extraction}: counts must be a finite non-negative ({n}, {n}) matrix")
            out[extraction] = matrix
    return out


def _controlled(
    dq: np.ndarray,
    dc: np.ndarray,
    keep: np.ndarray,
    source_a: int,
    source_b: int,
    *,
    labels: dict[str, Any],
    nperm: int,
) -> dict[str, Any]:
    design = np.column_stack(
        [
            np.ones(len(keep)),
            S[keep],
            np.abs(S[keep] - S[source_a]) - np.abs(S[keep] - S[source_b]),
            (GLYPH[keep] == -1).astype(float),
            (GLYPH[keep] == 1).astype(float),
        ]
    )
    rq = dq - design @ np.linalg.lstsq(design, dq, rcond=None)[0]
    rc = dc - design @ np.linalg.lstsq(design, dc, rcond=None)[0]
    dq0, dc0 = dq - np.mean(dq), dc - np.mean(dc)
    raw_den = np.linalg.norm(dq0) * np.linalg.norm(dc0)
    controlled_den = np.linalg.norm(rq) * np.linalg.norm(rc)
    raw = float(dq0 @ dc0 / raw_den) if raw_den > 1e-12 else None
    observed = float(rq @ rc / controlled_den) if controlled_den > 1e-12 else None
    null: list[float] = []
    for replicate in range(int(nperm)):
        rng = stable_rng(MASTER_SEED, **labels, stream="checkpoint_twins", replicate=replicate)
        permuted = dc[rng.permutation(len(keep))]
        residual = permuted - design @ np.linalg.lstsq(design, permuted, rcond=None)[0]
        denominator = np.linalg.norm(rq) * np.linalg.norm(residual)
        if denominator > 1e-12:
            null.append(float(rq @ residual / denominator))
    if observed is None or len(null) != int(nperm):
        raise ValueError(f"incomplete checkpoint-twin null for {labels}")
    b = int(np.sum(np.asarray(null) >= observed))
    return {
        "raw_cosine": raw,
        "line_controlled_cosine": observed,
        "p": float((b + 1) / (len(null) + 1)),
        "b": b,
        "B": len(null),
        "B_requested": int(nperm),
        "tail": ">=",
        "estimator": "(b+1)/(B+1)",
        "seed": stable_seed(MASTER_SEED, **labels, stream="checkpoint_twins", replicate=0),
    }


def main(argv: Sequence[str] | None = None) -> int:
    del argv
    corpus = _load_corpus()
    behaviors = {
        (revision, family): _load(revision, family)
        for revision in REVISIONS
        for family in ("E_modulation", "C_harmonic")
    }
    corpus_log = {extraction: _logc(corpus[extraction]) for extraction in ("A_win40", "D_doc")}
    out: list[dict[str, Any]] = []
    text_lines: list[str] = []
    for family in ("E_modulation", "C_harmonic"):
        final = behaviors[("main", family)]
        header = f"== {family}: magnitude; stability to released model; line-controlled Wikipedia twin correspondence"
        print(header)
        text_lines.append(header)
        for revision in REVISIONS:
            behavior = behaviors[(revision, family)]
            record: dict[str, Any] = {
                "schema": SCHEMA_VERSION,
                "status": "OK",
                "revision": revision,
                "family": family,
                "master_seed": MASTER_SEED,
                "seed_algorithm": SEED_ALGORITHM,
                "pairs": {},
            }
            cells: list[str] = []
            for source_a, source_b in ENH_PAIRS:
                keep = np.asarray([target for target in range(n) if target not in (source_a, source_b)], dtype=int)
                dq = behavior[source_a, keep] - behavior[source_b, keep]
                dq_final = final[source_a, keep] - final[source_b, keep]
                centered, centered_final = dq - np.mean(dq), dq_final - np.mean(dq_final)
                denominator = np.linalg.norm(centered) * np.linalg.norm(centered_final)
                stability = float(centered @ centered_final / denominator) if denominator > 1e-12 else None
                pair_name = f"{KEYS15[source_a]}|{KEYS15[source_b]}"
                extraction_results: dict[str, Any] = {}
                for extraction, logc in corpus_log.items():
                    labels = {
                        "corpus": "wikipedia",
                        "model": "olmo2_1b",
                        "family": family,
                        "extraction": extraction,
                        "view": "spelled_twin_difference",
                        "revision": revision,
                        "twin_identity": pair_name,
                    }
                    extraction_results[extraction] = _controlled(
                        dq,
                        logc[source_a, keep] - logc[source_b, keep],
                        keep,
                        source_a,
                        source_b,
                        labels=labels,
                        nperm=N_PERM,
                    )
                record["pairs"][pair_name] = {
                    "mean_absolute_difference": float(np.mean(np.abs(dq))),
                    "stability_to_released": stability,
                    "extractions": extraction_results,
                }
                cells.append(
                    f"{pair_name}: {np.mean(np.abs(dq)):.2f}; stab {stability if stability is not None else float('nan'):+.2f}; "
                    + " ".join(
                        f"{extraction} {values['line_controlled_cosine']:+.2f} (p={values['p']:.4f})"
                        for extraction, values in extraction_results.items()
                    )
                )
            line = f"{revision:44s} | " + " || ".join(cells)
            print(line)
            text_lines.append(line)
            out.append(record)
    dump_json_strict(out, "results/phase5/ckpt_twins_v4.json")
    Path("results/phase5/ckpt_twins_v4.txt").write_text("\n".join(text_lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
