"""Fail-closed provenance binding for the Phase-V v4 recompute.

The multi-hour v4 runner reads a fixed, explicit set of source and input
files.  This module records their SHA-256 digests together with the current
Git HEAD/status before any computation, and verifies that snapshot before
each later stage.  Generated v4 outputs are deliberately not part of the
manifest: they are the products whose provenance is being bound.

The manifest contains no timestamp or machine-local absolute paths, so the
same repository state produces deterministic manifest bytes.  A changed
source/input, Git HEAD, file mode, or unrelated working-tree status causes
verification to fail closed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_RELATIVE = Path("results/phase5/v4_provenance_manifest.json")
MANIFEST_SCHEMA = "phase5-v4-provenance-1"


CHECKPOINT_REVISIONS = (
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
)


# These are the v4 implementation files imported or executed by the runner.
# Keep this list explicit: silently discovering files from a directory would
# make a typo or an accidental extra dependency invisible to the audit.
V4_SOURCE_RELATIVE = (
    Path("phase5/ckpt_fingerprint.py"),
    Path("phase5/ckpt_twins.py"),
    Path("phase5/compare_v3_v4.py"),
    Path("phase5/crosscorpus_compare_v4.py"),
    Path("phase5/fingerprint.py"),
    Path("phase5/rerun_v4.sh"),
    Path("phase5/theory_features.py"),
    Path("phase5/thin_wikipedia.py"),
    Path("phase5/v4_provenance.py"),
    Path("phase5/validate_v4.py"),
)


def _checkpoint_behavior_paths() -> tuple[Path, ...]:
    tags = ("olmo2_1b",) + tuple(f"olmo2_1b_{revision}" for revision in CHECKPOINT_REVISIONS)
    return tuple(Path("results/phase2/behavior") / f"{tag}.json" for tag in tags)


def _v4_input_paths() -> tuple[Path, ...]:
    # Main-model behaviour files read by phase5.fingerprint.
    main_behaviour = tuple(
        Path("results/phase2/behavior") / f"{model}.json"
        for model in ("gemma2_2b", "olmo2_1b", "olmo2_7b", "qwen25_3b")
    )
    # Token counts read by phase5.fingerprint and the checkpoint scripts.  The
    # OLMo 7B path intentionally reuses the OLMo 1B tokenizer artifact, as the
    # implementation does.
    tokenisers = tuple(
        Path("results/phase2/hidden") / f"{model}_symbol_tokens.json"
        for model in ("gemma2_2b", "olmo2_1b", "qwen25_3b")
    )
    corpus_npz = (
        Path("results/phase5/cond_wikipedia.npz"),
        Path("results/phase5/cond_wikipedia_perdoc.npz"),
        Path("results/phase5/cond_wikipedia_matched_pmi.npz"),
        Path("results/phase5/cond_wikipedia_matched_rev.npz"),
        Path("results/phase5/cond_wikipedia_matched_sym.npz"),
        Path("results/phase5/cond_dolmino_dclm.npz"),
        Path("results/phase5/cond_olmomix_dclm.npz"),
        Path("results/phase5/cond_olmomix_dclm_big.npz"),
        Path("results/phase5/cond_olmomix_wiki.npz"),
    )
    return main_behaviour + _checkpoint_behavior_paths() + tokenisers + corpus_npz


def _v3_comparison_paths() -> tuple[Path, ...]:
    """Immutable v3 artifacts consumed by the exhaustive comparison."""

    paths = [
        Path("results/phase5/fingerprint/wikipedia_v3.json"),
        Path("results/phase5/fingerprint/wikipedia_v3_neutral.json"),
        Path("results/phase5/fingerprint/wikipedia_v3_rich.json"),
        Path("results/phase5/fingerprint/wikipedia_v3_neutral_rich.json"),
        Path("results/phase5/fingerprint/wikipedia_v3_tp.json"),
        Path("results/phase5/fingerprint/wikipedia_v3_neutral_tp.json"),
        Path("results/phase5/fingerprint/wikipedia_v3_neutral_rich_tp.json"),
        Path("results/phase5/fingerprint/wikipedia_v3_neutral_docboot.json"),
    ]
    for kind in ("t", "lo"):
        paths.extend(Path("results/phase5/fingerprint") / f"wikipedia_v3_neutral_{kind}{index}.json" for index in range(4))
    for operator in ("sym", "rev", "pmi"):
        paths.extend(
            Path("results/phase5/fingerprint") / f"matched_{operator}{suffix}.json"
            for suffix in ("", "_neutral", "_neutral_tp")
        )
    for corpus in ("olmomix_wiki", "olmomix_dclm", "dolmino_dclm", "olmomix_dclm_big"):
        paths.extend(
            Path("results/phase5/fingerprint") / f"{corpus}{suffix}.json"
            for suffix in ("", "_neutral")
        )
        paths.extend(
            Path("results/phase5/fingerprint") / f"wikipedia_thin_{corpus}{suffix}.json"
            for suffix in ("", "_neutral")
        )
    paths.extend(
        (
            Path("results/phase5/ckpt_fingerprint.json"),
            Path("results/phase5/ckpt_twins.json"),
            Path("phase2/keys15.py"),
        )
    )
    return tuple(paths)


def expected_snapshot_paths() -> tuple[Path, ...]:
    """Return the sorted, de-duplicated relative manifest path list."""

    paths = tuple(V4_SOURCE_RELATIVE) + _v4_input_paths() + _v3_comparison_paths()
    unique = {path.as_posix(): path for path in paths}
    return tuple(unique[name] for name in sorted(unique))


def expected_generated_output_paths() -> frozenset[Path]:
    """Return the exact output namespace produced by ``rerun_v4.sh``."""

    fingerprint_names = [
        "wikipedia_v4",
        "wikipedia_v4_neutral",
        "wikipedia_v4_neutral_rich",
        "wikipedia_v4_rich",
        "wikipedia_v4_neutral_tp",
        "wikipedia_v4_neutral_rich_tp",
        "wikipedia_v4_rich_tp",
        "wikipedia_v4_tp",
        "wikipedia_v4_neutral_docboot",
        "wikipedia_v4_neutral_rich_docboot",
    ]
    for operator in ("sym", "rev", "pmi"):
        fingerprint_names.extend(
            (f"matched_{operator}_v4", f"matched_{operator}_v4_neutral", f"matched_{operator}_v4_neutral_tp")
        )
    for index in range(4):
        fingerprint_names.extend((f"wikipedia_v4_neutral_t{index}", f"wikipedia_v4_neutral_lo{index}"))

    thinning_names: list[str] = []
    for corpus in ("olmomix_wiki", "olmomix_dclm", "dolmino_dclm", "olmomix_dclm_big"):
        fingerprint_names.extend((f"{corpus}_v4", f"{corpus}_v4_neutral"))
        for replicate in range(5):
            suffix = "v4" if replicate == 0 else f"v4_s{replicate}"
            base = f"wikipedia_thin_{corpus}_{suffix}"
            fingerprint_names.extend((base, f"{base}_neutral"))
            thinning_names.append(f"cond_{base}.npz")

    if len(fingerprint_names) != 75 or len(set(fingerprint_names)) != 75:
        raise RuntimeError("internal v4 fingerprint output inventory is not exactly 75 unique files")
    if len(thinning_names) != 20 or len(set(thinning_names)) != 20:
        raise RuntimeError("internal v4 thinning output inventory is not exactly 20 unique files")

    paths = {Path("results/phase5/fingerprint") / f"{name}.json" for name in fingerprint_names}
    paths.update(Path("results/phase5") / name for name in thinning_names)
    paths.update(
        Path(path)
        for path in (
            MANIFEST_RELATIVE.as_posix(),
            "results/phase5/rerun_v4.log",
            "results/phase5/V4_COMPUTE_AND_COMPARISON_COMPLETE.marker",
            "results/phase5/v3_v4_comparison.csv",
            "results/phase5/v3_v4_comparison.md",
            "results/phase5/thinning_seed_variance.csv",
            "results/phase5/ckpt_fingerprint_v4.json",
            "results/phase5/ckpt_trajectory_v4.txt",
            "results/phase5/ckpt_twins_v4.json",
            "results/phase5/ckpt_twins_v4.txt",
            "results/phase5/crosscorpus_compare_v4_spelled.json",
            "results/phase5/crosscorpus_compare_v4_spelled.txt",
            "results/phase5/crosscorpus_compare_v4_aggregated.json",
            "results/phase5/crosscorpus_compare_v4_aggregated.txt",
        )
    )
    return frozenset(paths)


EXPECTED_GENERATED_OUTPUT_RELATIVE = expected_generated_output_paths()


def _generated_output_path(relative: Path) -> bool:
    """Whether a status entry is an expected v4/comparison output.

    Outputs are excluded from the immutable file list and from the status
    comparison because they are created during the run.  The allow-list is
    deliberately narrow; an unrelated new or modified working-tree path
    remains a provenance failure.
    """

    return relative in EXPECTED_GENERATED_OUTPUT_RELATIVE


def _git_command(root: Path, *args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(f"git command failed: git -C {root} {' '.join(args)}") from error
    try:
        return completed.stdout.decode("utf-8")
    except UnicodeDecodeError as error:
        raise RuntimeError("git output was not valid UTF-8") from error


def _git_head(root: Path) -> str:
    head = _git_command(root, "rev-parse", "HEAD").strip()
    if not head or any(char not in "0123456789abcdefABCDEF" for char in head):
        raise RuntimeError(f"invalid Git HEAD {head!r}")
    return head


def _git_status(root: Path) -> tuple[str, ...]:
    raw = _git_command(root, "status", "--porcelain=v1", "--untracked-files=all")
    if "\x00" in raw:
        raise RuntimeError("unexpected NUL in Git status")
    return tuple(sorted(line.rstrip("\r") for line in raw.splitlines() if line.strip()))


def _status_paths(line: str) -> tuple[Path, ...]:
    if len(line) < 3 or line[2] != " ":
        raise RuntimeError(f"malformed Git status line {line!r}")
    payload = line[3:]
    if " -> " in payload:
        old, new = payload.split(" -> ", 1)
        return Path(old), Path(new)
    return (Path(payload),)


def _relevant_status(status: Iterable[str]) -> tuple[str, ...]:
    kept: list[str] = []
    for line in status:
        paths = _status_paths(line)
        if not all(_generated_output_path(path) for path in paths):
            kept.append(line)
    return tuple(sorted(kept))


def _relative_path(path: Path, root: Path) -> Path:
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"path escapes project root: {path}") from error
    if relative == Path("."):
        raise ValueError("project root is not a file")
    return relative


def _regular_file(path: Path, *, label: str) -> None:
    if path.is_symlink():
        raise FileNotFoundError(f"{label} is a symlink; refusing non-immutable input: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")


def _digest_file(path: Path, *, label: str) -> dict[str, Any]:
    _regular_file(path, label=label)
    before = path.stat()
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
    except OSError as error:
        raise RuntimeError(f"could not read {label}: {path}") from error
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_mode & 0o777) != (
        after.st_size,
        after.st_mtime_ns,
        after.st_mode & 0o777,
    ):
        raise RuntimeError(f"{label} changed while being hashed: {path}")
    return {
        "path": _relative_path(path, PROJECT_ROOT).as_posix(),
        "bytes": int(after.st_size),
        "mode": int(after.st_mode & 0o777),
        "sha256": digest.hexdigest(),
    }


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON numeric constant: {value}")


def _load_manifest(path: Path) -> dict[str, Any]:
    _regular_file(path, label="provenance manifest")
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle, parse_constant=_reject_json_constant)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid provenance manifest: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"provenance manifest is not an object: {path}")
    return payload


def _manifest_path(path: str | Path | None) -> Path:
    candidate = PROJECT_ROOT / MANIFEST_RELATIVE if path is None else Path(path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve()


def write_manifest(path: str | Path | None = None) -> Path:
    """Hash every required source/input and write a deterministic manifest."""

    manifest_path = _manifest_path(path)
    if _relative_path(manifest_path, PROJECT_ROOT) != MANIFEST_RELATIVE:
        raise ValueError(f"manifest must be {MANIFEST_RELATIVE}, got {manifest_path}")
    paths = expected_snapshot_paths()
    records = tuple(_digest_file(PROJECT_ROOT / relative, label="snapshot input") for relative in paths)
    if tuple(record["path"] for record in records) != tuple(relative.as_posix() for relative in paths):
        raise RuntimeError("manifest path ordering is not deterministic")
    status = _git_status(PROJECT_ROOT)
    payload: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "project_root": ".",
        "git_head": _git_head(PROJECT_ROOT),
        "git_status": list(status),
        "git_status_relevant": list(_relevant_status(status)),
        "files": list(records),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    # A same-directory replace prevents readers from observing a partial JSON
    # document.  The temporary filename is not part of the resulting bytes.
    fd, temporary_name = tempfile.mkstemp(prefix=f".{manifest_path.name}.", dir=str(manifest_path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temporary_name, manifest_path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return manifest_path


def verify_manifest(path: str | Path | None = None) -> Path:
    """Verify the exact source/input snapshot recorded in ``path``."""

    manifest_path = _manifest_path(path)
    payload = _load_manifest(manifest_path)
    if payload.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"wrong provenance manifest schema: {manifest_path}")
    if payload.get("project_root") != ".":
        raise ValueError("provenance manifest has a non-portable project root")
    recorded_head = payload.get("git_head")
    if recorded_head != _git_head(PROJECT_ROOT):
        raise RuntimeError(f"Git HEAD changed: recorded {recorded_head!r}, current {_git_head(PROJECT_ROOT)!r}")
    status = _git_status(PROJECT_ROOT)
    if payload.get("git_status_relevant") != list(_relevant_status(status)):
        raise RuntimeError("working-tree status changed outside expected v4 outputs")
    recorded_files = payload.get("files")
    expected_paths = expected_snapshot_paths()
    if not isinstance(recorded_files, list) or len(recorded_files) != len(expected_paths):
        raise ValueError("provenance manifest has an incomplete file list")
    expected_names = tuple(relative.as_posix() for relative in expected_paths)
    actual_names = tuple(record.get("path") if isinstance(record, dict) else None for record in recorded_files)
    if actual_names != expected_names:
        raise ValueError("provenance manifest file list/order changed")
    for recorded, relative in zip(recorded_files, expected_paths):
        if not isinstance(recorded, dict):
            raise ValueError(f"invalid provenance record for {relative}")
        current = _digest_file(PROJECT_ROOT / relative, label="snapshot input")
        if current != recorded:
            raise RuntimeError(f"provenance mismatch for {relative}")
    return manifest_path


def semantic_preflight() -> dict[str, int]:
    """Parse and validate every late-stage input before the multi-hour run."""

    from phase5 import ckpt_fingerprint, ckpt_twins, fingerprint

    models = ("olmo2_1b", "gemma2_2b", "qwen25_3b", "olmo2_7b")
    families = ("C_harmonic", "D_chord", "E_modulation")
    templates = (0, 1, 2, 3)
    for model in models:
        fingerprint._load_behavior(model, families, templates)
        fingerprint._load_tokens(model)

    corpus_specs = {
        "results/phase5/cond_wikipedia.npz": ("A_win40", "B_any", "D_doc"),
        "results/phase5/cond_wikipedia_matched_sym.npz": ("A_win40",),
        "results/phase5/cond_wikipedia_matched_rev.npz": ("A_win40",),
        "results/phase5/cond_wikipedia_matched_pmi.npz": ("A_win40",),
        "results/phase5/cond_olmomix_wiki.npz": ("A_win40", "B_any", "D_doc"),
        "results/phase5/cond_olmomix_dclm.npz": ("A_win40", "B_any", "D_doc"),
        "results/phase5/cond_dolmino_dclm.npz": ("A_win40", "B_any", "D_doc"),
        "results/phase5/cond_olmomix_dclm_big.npz": ("A_win40", "B_any", "D_doc"),
    }
    for path, extractions in corpus_specs.items():
        fingerprint._load_corpus(path, extractions)
    fingerprint._load_perdoc("results/phase5/cond_wikipedia_perdoc.npz", ("A_win40", "D_doc"))

    ckpt_fingerprint._load_tokens()
    ckpt_fingerprint._load_corpus()
    for revision in ckpt_fingerprint.REVISIONS + ["main"]:
        tag = "olmo2_1b" if revision == "main" else f"olmo2_1b_{revision}"
        ckpt_fingerprint._load_behavior(Path(f"results/phase2/behavior/{tag}.json"))
    ckpt_twins._load_corpus()
    for revision in ckpt_twins.REVISIONS:
        for family in ("E_modulation", "C_harmonic"):
            ckpt_twins._load(revision, family)

    json_inputs = [path for path in _v3_comparison_paths() if path.suffix == ".json"]
    for relative in json_inputs:
        with (PROJECT_ROOT / relative).open(encoding="utf-8") as handle:
            # Historical v3 artifacts predate the strict-JSON correction and
            # a few contain Python JSON NaN tokens.  They are immutable
            # comparison evidence, not v4 computational inputs; parse them
            # with their original semantics and bind their exact bytes.
            json.load(handle)
    return {
        "models": len(models),
        "corpora": len(corpus_specs),
        "checkpoint_revisions": len(ckpt_fingerprint.REVISIONS) + 1,
        "v3_json_inputs": len(json_inputs),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", metavar="MANIFEST", help="write a new deterministic snapshot manifest")
    group.add_argument("--verify", metavar="MANIFEST", help="verify an existing snapshot manifest")
    group.add_argument("--preflight", action="store_true", help="parse and validate every late-stage immutable input")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.preflight:
        print("PREFLIGHT", json.dumps(semantic_preflight(), sort_keys=True))
    elif args.write is not None:
        path = write_manifest(args.write)
        print(f"WROTE {path}")
    else:
        path = verify_manifest(args.verify)
        print(f"VALID {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
