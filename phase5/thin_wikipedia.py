"""Deterministically thin Wikipedia conditional matrices for v4 cross-corpus tests.

Usage::

    python -m phase5.thin_wikipedia <target-corpus.npz> <new-v4-output.npz> [replicate]

The output path must be a v4 path.  This guard prevents an accidental
overwrite of the tracked v3 seed-0 files.  Every extraction receives a
stable SHA-256-derived stream and the NPZ carries the seed/provenance
metadata needed to reproduce it.
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
from pathlib import Path

import numpy as np

from phase5.theory_features import MASTER_SEED, SCHEMA_VERSION, SEED_ALGORITHM, stable_seed


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "results/phase5"
_SAFE_OUTPUT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*\.npz\Z")


def _v4_thinning_output_path(path: str | Path) -> Path:
    """Validate a direct-child v4 thinning path without following symlinks."""

    supplied = Path(path)
    if ".." in supplied.parts:
        raise ValueError(f"refusing traversal in thinning output path: {supplied}")
    candidate = supplied if supplied.is_absolute() else PROJECT_ROOT / supplied
    if not _SAFE_OUTPUT_NAME.fullmatch(candidate.name) or "v4" not in candidate.name.casefold():
        raise ValueError(f"refusing unsafe or non-v4 thinning output path {supplied}; v3 thinning files are immutable")
    if OUTPUT_DIR.is_symlink():
        raise ValueError(f"thinning output directory is a symlink: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if candidate.parent != OUTPUT_DIR:
        raise ValueError(f"thinning output must be a direct child of {OUTPUT_DIR}: {supplied}")
    if candidate.is_symlink():
        raise ValueError(f"refusing symlink thinning output: {candidate}")
    if candidate.exists() and not candidate.is_file():
        raise ValueError(f"thinning output is not a regular file: {candidate}")
    return candidate


def _flag(argv: list[str], name: str, default: str | None = None) -> str | None:
    key = f"--{name}="
    for arg in argv:
        if arg.startswith(key):
            return arg[len(key) :]
    return default


def _target_name(path: Path) -> str:
    stem = path.stem
    return stem[5:] if stem.startswith("cond_") else stem


def thin(target_path: str | Path, output_path: str | Path, replicate: int = 0, *, master_seed: int = MASTER_SEED) -> dict[str, object]:
    target = Path(target_path)
    output = _v4_thinning_output_path(output_path)
    if target.resolve() == output.resolve():
        raise ValueError("target and output paths must differ")
    if int(replicate) < 0:
        raise ValueError("replicate must be non-negative")
    tgt = np.load(target)
    wiki_path = Path("results/phase5/cond_wikipedia.npz")
    wiki = np.load(wiki_path)
    target_corpus = _target_name(target)
    source_corpus = _target_name(wiki_path)
    out: dict[str, np.ndarray] = {}
    stream_seeds: dict[str, int] = {}
    for key in wiki.files:
        value = np.asarray(wiki[key])
        if value.ndim == 0:
            out[key] = value.copy()
            continue
        if key in tgt.files and value.size and float(np.sum(value)) > 0 and float(np.sum(tgt[key])) < float(np.sum(value)):
            fraction = float(np.sum(tgt[key])) / float(np.sum(value))
            seed = stable_seed(master_seed, source_corpus=source_corpus, target_corpus=target_corpus, replicate=int(replicate), stream=f"thin:{key}")
            stream_seeds[key] = seed
            rng = np.random.default_rng(seed)
            counts = np.rint(value).astype(np.int64)
            out[key] = rng.binomial(counts, fraction).astype(float)
            print(f"{key}: wiki {int(value.sum())} -> {int(out[key].sum())} (target {int(np.sum(tgt[key]))})")
        else:
            out[key] = value.copy()
    # Metadata values are 1-element arrays so they remain ordinary NPZ fields
    # and are readable without a pickle/object array.
    out["v4_schema"] = np.asarray(SCHEMA_VERSION)
    out["v4_master_seed"] = np.asarray(int(master_seed), dtype=np.int64)
    out["v4_replicate"] = np.asarray(int(replicate), dtype=np.int64)
    out["v4_source_corpus"] = np.asarray(source_corpus)
    out["v4_target_corpus"] = np.asarray(target_corpus)
    out["v4_seed_algorithm"] = np.asarray(SEED_ALGORITHM)
    out["v4_stream_seeds_json"] = np.asarray(__import__("json").dumps(stream_seeds, sort_keys=True, separators=(",", ":")))
    fd, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".npz", dir=str(output.parent))
    os.close(fd)
    try:
        np.savez(temporary_name, **out)
        os.replace(temporary_name, output)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return {"schema": SCHEMA_VERSION, "master_seed": int(master_seed), "replicate": int(replicate), "source_corpus": source_corpus, "target_corpus": target_corpus, "stream_seeds": stream_seeds, "output": str(output)}


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    positional = [x for x in args if not x.startswith("--")]
    if len(positional) < 2:
        print(__doc__)
        return 2
    replicate = int(positional[2]) if len(positional) > 2 else 0
    master = int(_flag(args, "master-seed", str(MASTER_SEED)))
    thin(positional[0], positional[1], replicate, master_seed=master)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
