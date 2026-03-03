"""Timestamped experiment-run directory management.

Directory convention
--------------------
runs/<experiment_name>/train_YYYYMMDD_HHMMSS/          <- one training run
                       train_.../models/<env>/          <- saved checkpoints
                       train_.../infer/YYYYMMDD_HHMMSS/ <- inference output
                       latest.txt                       <- points to most recent run

Provides
--------
- resolve_experiment_dir()      Bare name -> runs/<name>/; path -> as-is.
- create_run_dir()              Create a new timestamped run directory.
- latest_run_dir()              Find latest run (via latest.txt or timestamp sort).
- latest_run_dir_with_models()  Find latest run that contains models/.
- resolve_models_dir()          Flexible resolution from experiment name / run / models path.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


_RUN_DIR_RE = re.compile(r"^(?:(?P<prefix>[A-Za-z]+)_)?(?P<ts>\d{8}_\d{6})(?:_(?P<n>\d+))?$")


@dataclass(frozen=True)
class RunPaths:
    experiment_dir: Path
    run_dir: Path


def _run_dir_sort_key(name: str) -> tuple[str, int] | None:
    m = _RUN_DIR_RE.match(name)
    if not m:
        return None
    ts = m.group("ts")
    n = int(m.group("n") or 0)
    return ts, n


def resolve_experiment_dir(out: Path, *, runs_root: Path = Path("runs")) -> Path:
    """Resolve an output experiment directory.

    Convention:
    - If `out` is a bare name like "outputs_repro_1000", store under `runs/<name>/`.
    - If `out` is a path (contains separators / starts with '.' / is absolute), use as-is.
    """
    out = Path(out)
    if out.is_absolute():
        return out

    out_str = out.as_posix()
    runs_root = Path(runs_root)
    is_bare_name = len(out.parts) == 1 and not out_str.startswith(".") and out.name != runs_root.name
    if is_bare_name:
        return runs_root / out
    return out


def _iter_run_dirs(experiment_dir: Path) -> list[Path]:
    if not experiment_dir.exists():
        return []
    runs: list[tuple[tuple[str, int], Path]] = []
    for p in experiment_dir.iterdir():
        if not p.is_dir():
            continue
        key = _run_dir_sort_key(p.name)
        if key is None:
            continue
        runs.append((key, p))
    runs.sort(key=lambda pair: pair[0])
    return [p for _, p in runs]


def latest_run_dir(experiment_dir: Path) -> Path | None:
    """Return the latest timestamped run directory under `experiment_dir`."""
    latest_file = experiment_dir / "latest.txt"
    if latest_file.exists():
        name = latest_file.read_text(encoding="utf-8").strip()
        if name:
            candidate = experiment_dir / name
            if candidate.is_dir():
                return candidate

    runs = _iter_run_dirs(experiment_dir)
    return runs[-1] if runs else None


def latest_run_dir_with_models(experiment_dir: Path) -> Path | None:
    """Return the latest run directory under `experiment_dir` that contains `models/`."""
    candidate = latest_run_dir(experiment_dir)
    if candidate is not None and (candidate / "models").is_dir():
        return candidate

    for run_dir in reversed(_iter_run_dirs(experiment_dir)):
        if (run_dir / "models").is_dir():
            return run_dir
    return None


def create_run_dir(
    experiment_dir: Path,
    *,
    timestamp_runs: bool = True,
    now: datetime | None = None,
    prefix: str | None = None,
) -> RunPaths:
    experiment_dir = Path(experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)

    if not timestamp_runs:
        return RunPaths(experiment_dir=experiment_dir, run_dir=experiment_dir)

    ts = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    stem = f"{prefix}_{ts}" if prefix else ts
    run_dir = experiment_dir / stem
    n = 0
    while run_dir.exists():
        n += 1
        run_dir = experiment_dir / f"{stem}_{n}"

    run_dir.mkdir(parents=True, exist_ok=False)
    (experiment_dir / "latest.txt").write_text(run_dir.name, encoding="utf-8")
    return RunPaths(experiment_dir=experiment_dir, run_dir=run_dir)


def resolve_models_dir(models: Path, *, runs_root: Path = Path("runs")) -> Path:
    """Resolve a models directory for inference.

    Accepts:
    - experiment name (bare): uses latest run under `runs/<name>/models`
    - experiment dir path: uses latest run under `<dir>/models`
    - run dir path: uses `<run>/models`
    - models dir path: uses it directly
    """
    raw = Path(models)

    candidates: list[Path] = []
    mapped = resolve_experiment_dir(raw, runs_root=runs_root)
    candidates.append(mapped)

    if raw != mapped:
        candidates.append(raw)
    else:
        # Back-compat: if user passes e.g. "outputs_repro_1000/models", prefer checking "runs/..." too.
        if not raw.is_absolute() and raw.parts and raw.parts[0] != Path(runs_root).name:
            candidates.append(Path(runs_root) / raw)

    tried: list[str] = []
    for base in candidates:
        tried.append(str(base))

        if base.is_dir():
            if base.name == "models":
                return base
            if (base / "models").is_dir():
                return base / "models"

            lr = latest_run_dir_with_models(base)
            if lr is not None:
                return lr / "models"

        if base.name == "models" and base.parent.is_dir():
            lr = latest_run_dir_with_models(base.parent)
            if lr is not None:
                return lr / "models"

    raise FileNotFoundError(
        "Could not resolve models directory.\n"
        f"- models={raw}\n"
        f"- tried={tried}"
    )
