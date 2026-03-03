"""Shared configuration and utilities for DQN8 plotting."""
from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Algorithm ordering & colours (no relabel / swap / rescale)
# ---------------------------------------------------------------------------
ALG_ORDER = [
    "CNN-PDDQN",
    "CNN-DDQN",
    "CNN-DQN",
    "MLP-PDDQN",
    "MLP-DDQN",
    "MLP-DQN",
    "Hybrid A*",
    "SS-RRT*",
]

ALG_COLORS: dict[str, str] = {
    "CNN-PDDQN": "#e377c2",
    "CNN-DDQN": "#d62728",
    "CNN-DQN": "#2ca02c",
    "MLP-PDDQN": "#17becf",
    "MLP-DDQN": "#ff7f0e",
    "MLP-DQN": "#1f77b4",
    "Hybrid A*": "#9467bd",
    "SS-RRT*": "#8c564b",
}

# ---------------------------------------------------------------------------
# Plot style constants
# ---------------------------------------------------------------------------
OUTPUT_DPI = 300
BASE_FONT_SIZE = 14
TICK_FONT_SIZE = 12
LEGEND_FONT_SIZE = 12
TABLE_FONT_SIZE = 12
COLLISION_BOX_SCALE = 1.0


def apply_rcparams() -> None:
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": BASE_FONT_SIZE,
        "axes.labelsize": BASE_FONT_SIZE,
        "xtick.labelsize": TICK_FONT_SIZE,
        "ytick.labelsize": TICK_FONT_SIZE,
        "legend.fontsize": LEGEND_FONT_SIZE,
        "figure.titlesize": BASE_FONT_SIZE,
    })


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------
def find_trace_files(
    traces_dir: str | Path,
    env_case: str,
    run_idx: int,
) -> dict[str, Path]:
    """Return {algorithm: csv_path} for a given env_case and run index."""
    traces_dir = Path(traces_dir)
    result: dict[str, Path] = {}
    slug = _safe_slug(env_case)
    for p in sorted(traces_dir.glob(f"{slug}__*__run{run_idx}.csv")):
        algo = _extract_algo_from_filename(p.name, slug, run_idx)
        if algo:
            result[algo] = p
    return result


def find_all_run_indices(traces_dir: str | Path, env_case: str) -> list[int]:
    """Return sorted list of all run indices available for an env_case."""
    traces_dir = Path(traces_dir)
    slug = _safe_slug(env_case)
    indices: set[int] = set()
    for p in traces_dir.glob(f"{slug}__*__run*.csv"):
        m = re.search(r"__run(\d+)\.csv$", p.name)
        if m:
            indices.add(int(m.group(1)))
    return sorted(indices)


def _safe_slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s)).strip("_")


def _extract_algo_from_filename(fname: str, env_slug: str, run_idx: int) -> str | None:
    suffix = f"__run{run_idx}.csv"
    if not fname.endswith(suffix):
        return None
    mid = fname[len(env_slug) + 2: -len(suffix)]
    # Reverse slug mapping: exact match first
    for algo in ALG_ORDER:
        if _safe_slug(algo) == mid:
            return algo
    # Legacy baseline names from infer.py
    _LEGACY = {"Hybrid_A": "Hybrid A*", "RRT": "SS-RRT*", "SS-RRT": "SS-RRT*"}
    if mid in _LEGACY:
        return _LEGACY[mid]
    # Fallback: return raw
    return mid if mid else None


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------
def load_trace(csv_path: str | Path) -> pd.DataFrame:
    """Load a trace CSV with all columns."""
    return pd.read_csv(csv_path)


def load_trace_xy(csv_path: str | Path, cell_size_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Load x, y arrays in cell coordinates."""
    df = pd.read_csv(csv_path, usecols=["x_m", "y_m"])
    x = df["x_m"].to_numpy(dtype=float) / float(cell_size_m)
    y = df["y_m"].to_numpy(dtype=float) / float(cell_size_m)
    return x, y


def load_trace_pose(
    csv_path: str | Path, cell_size_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load x, y (cells) and theta (rad)."""
    try:
        df = pd.read_csv(csv_path, usecols=["x_m", "y_m", "theta_rad"])
        theta = df["theta_rad"].to_numpy(dtype=float)
    except ValueError:
        df = pd.read_csv(csv_path, usecols=["x_m", "y_m"])
        dx = np.diff(df["x_m"].to_numpy(dtype=float), append=0.0)
        dy = np.diff(df["y_m"].to_numpy(dtype=float), append=0.0)
        theta = np.arctan2(dy, dx)
    x = df["x_m"].to_numpy(dtype=float) / float(cell_size_m)
    y = df["y_m"].to_numpy(dtype=float) / float(cell_size_m)
    return x, y, theta


def load_map(
    meta_path: str | Path,
) -> tuple[np.ndarray, float, dict]:
    """Load obstacle grid and map metadata.

    Returns (obstacles_bool, cell_size_m, meta_dict).
    """
    meta_path = Path(meta_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    cell_size = float(meta["cell_size_m"])
    npz_path = meta_path.parent / f"{meta['env_base']}__grid_y0_bottom.npz"
    data = np.load(str(npz_path))
    key = "obstacle_grid" if "obstacle_grid" in data.files else data.files[0]
    grid = data[key]
    return grid != 0, cell_size, meta


def load_kpi(csv_path: str | Path) -> pd.DataFrame:
    """Read KPI CSV with encoding fallback."""
    for enc in ("utf-8", "utf-8-sig", "gbk", "latin-1"):
        try:
            return pd.read_csv(csv_path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(csv_path, encoding="latin-1")


def parse_environment(env_case: str) -> tuple[str, str | None]:
    """Split 'forest_a::short' → ('forest_a', 'short')."""
    if "::" in env_case:
        base, suite = env_case.split("::", 1)
        return base.strip(), suite.strip() or None
    return env_case.strip(), None


def get_start_goal_from_json(
    json_path: str | Path,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Read start_xy and goal_xy from trace metadata JSON."""
    with open(json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    sx, sy = meta["start_xy"]
    gx, gy = meta["goal_xy"]
    return (float(sx), float(sy)), (float(gx), float(gy))


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------
def darken_hex(hex_color: str, factor: float = 0.65) -> str:
    s = str(hex_color).strip()
    if not s.startswith("#") or len(s) != 7:
        return hex_color
    r = max(0, min(255, int(int(s[1:3], 16) * factor)))
    g = max(0, min(255, int(int(s[3:5], 16) * factor)))
    b = max(0, min(255, int(int(s[5:7], 16) * factor)))
    return f"#{r:02x}{g:02x}{b:02x}"


def hex_to_rgba(hex_color: str, alpha: float = 1.0) -> tuple[float, ...]:
    s = str(hex_color).strip()
    if not s.startswith("#") or len(s) != 7:
        return (0.0, 0.0, 0.0, float(alpha))
    r = int(s[1:3], 16) / 255.0
    g = int(s[3:5], 16) / 255.0
    b = int(s[5:7], 16) / 255.0
    return (r, g, b, float(alpha))
