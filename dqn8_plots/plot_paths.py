"""Path comparison plots: obstacle grid + trajectory overlays."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle
from matplotlib.transforms import Affine2D

from common import (
    ALG_COLORS,
    ALG_ORDER,
    COLLISION_BOX_SCALE,
    LEGEND_FONT_SIZE,
    OUTPUT_DPI,
    apply_rcparams,
    find_all_run_indices,
    find_trace_files,
    get_start_goal_from_json,
    hex_to_rgba,
    load_map,
    load_trace_pose,
)


def add_collision_boxes(
    ax,
    x_cells: np.ndarray,
    y_cells: np.ndarray,
    theta_rad: np.ndarray,
    footprint: dict | None,
    cell_size_m: float,
    color: str,
) -> None:
    if not footprint or footprint.get("kind") != "two_circle":
        return
    radius_m = float(footprint["radius_m"])
    x1_m = float(footprint["x1_m"])
    x2_m = float(footprint["x2_m"])
    length_m = (x2_m - x1_m) + 2 * radius_m
    width_m = 2 * radius_m
    center_offset_m = 0.5 * (x1_m + x2_m)
    length = length_m / float(cell_size_m) * float(COLLISION_BOX_SCALE)
    width = width_m / float(cell_size_m) * float(COLLISION_BOX_SCALE)
    center_offset = center_offset_m / float(cell_size_m)

    n = len(x_cells)
    if n == 0:
        return
    step = max(n // 20, 1)
    idxs = list(range(0, n, step))
    if idxs[-1] != n - 1:
        idxs.append(n - 1)

    edge_c = hex_to_rgba(color, 0.25)
    face_c = hex_to_rgba(color, 0.0)
    for i in idxs:
        theta = float(theta_rad[i])
        cx = float(x_cells[i]) + center_offset * math.cos(theta)
        cy = float(y_cells[i]) + center_offset * math.sin(theta)
        rect = Rectangle(
            (-length / 2, -width / 2), length, width,
            linewidth=0.8, edgecolor=edge_c, facecolor=face_c, zorder=1,
        )
        rect.set_transform(Affine2D().rotate(theta).translate(cx, cy) + ax.transData)
        ax.add_patch(rect)


def plot_paths_axis(
    ax,
    obstacles: np.ndarray,
    trace_files: dict[str, Path],
    cell_size_m: float,
    goal_tol_cells: float | None,
    collision_footprint: dict | None,
) -> None:
    """Draw obstacle grid + all algorithm paths on *ax*."""
    h, w = obstacles.shape
    ax.imshow(
        obstacles.astype(int), origin="lower", cmap="gray_r",
        interpolation="nearest", extent=(-0.5, w - 0.5, -0.5, h - 0.5), alpha=0.9,
    )
    if not trace_files:
        return

    for algo in ALG_ORDER:
        csv_path = trace_files.get(algo)
        if csv_path is None:
            continue
        x, y, theta = load_trace_pose(csv_path, cell_size_m)
        add_collision_boxes(ax, x, y, theta, collision_footprint, cell_size_m,
                            ALG_COLORS.get(algo, "#333333"))
        ax.plot(x, y, label=algo, color=ALG_COLORS.get(algo, "#333333"),
                linewidth=2, zorder=3)

    # Start / goal markers from first available trace JSON
    first_csv = next(iter(trace_files.values()))
    json_path = first_csv.with_suffix(".json")
    if json_path.exists():
        start_xy, goal_xy = get_start_goal_from_json(json_path)
    else:
        start_xy, goal_xy = (0, 0), (0, 0)

    ax.scatter([start_xy[0]], [start_xy[1]], marker="*", s=140,
               color="royalblue", label="Start", zorder=5)
    ax.scatter([goal_xy[0]], [goal_xy[1]], marker="*", s=140,
               color="red", label="Goal", zorder=5)
    if goal_tol_cells and goal_tol_cells > 0:
        ax.add_patch(Circle(
            (goal_xy[0], goal_xy[1]), radius=goal_tol_cells,
            edgecolor="red", facecolor="none", linestyle="-",
            linewidth=1.5, alpha=0.7, zorder=4, label="_nolegend_",
        ))
    ax.set_aspect("equal", "box")
    ax.set_xlabel("X (cells)")
    ax.set_ylabel("Y (cells)")
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(-0.5, h - 0.5)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0),
              frameon=True, framealpha=0.9, fontsize=LEGEND_FONT_SIZE)


def plot_paths(
    base_dir: str | Path,
    out_dir: str | Path,
    env_cases: list[str] | None = None,
    run_idxs: list[int] | None = None,
) -> list[Path]:
    """Generate path comparison figures for each env_case × run_idx."""
    apply_rcparams()
    base_dir = Path(base_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    traces_dir = base_dir / "traces"
    maps_dir = base_dir / "maps"
    saved: list[Path] = []

    # Discover env_cases from trace files
    if env_cases is None:
        env_cases = _discover_env_cases(traces_dir)

    for env_case in env_cases:
        env_base, _suite = env_case.split("::", 1) if "::" in env_case else (env_case, None)
        meta_path = maps_dir / f"{env_base}__meta.json"
        if not meta_path.exists():
            print(f"[plot_paths] skip {env_case}: no map meta at {meta_path}")
            continue
        obstacles, cell_size_m, meta = load_map(meta_path)
        goal_tol_cells = meta.get("goal", {}).get("position_tolerance_cells")
        footprint = meta.get("collision_footprint")

        idxs = run_idxs if run_idxs else find_all_run_indices(traces_dir, env_case)
        for ri in idxs:
            trace_files = find_trace_files(traces_dir, env_case, ri)
            if not trace_files:
                continue
            fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=OUTPUT_DPI)
            slug = env_case.replace("::", "_")
            plot_paths_axis(ax, obstacles, trace_files, cell_size_m,
                            goal_tol_cells, footprint)
            ax.set_title(f"{env_case}  run {ri}")
            out_path = out_dir / f"fig_paths_{slug}_run{ri:02d}.png"
            fig.savefig(str(out_path), bbox_inches="tight", dpi=OUTPUT_DPI)
            plt.close(fig)
            saved.append(out_path)
            print(f"  saved {out_path.name}")
    return saved


def _discover_env_cases(traces_dir: Path) -> list[str]:
    """Discover unique env_case values from trace JSON files."""
    import json as _json
    cases: set[str] = set()
    for p in traces_dir.glob("*.json"):
        try:
            with open(p, "r", encoding="utf-8") as f:
                d = _json.load(f)
            if "env_case" in d:
                cases.add(d["env_case"])
        except Exception:
            continue
    return sorted(cases)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot path comparison figures.")
    ap.add_argument("--base-dir", type=str, required=True, help="Inference output dir (contains traces/ and maps/).")
    ap.add_argument("--out-dir", type=str, default="figures", help="Output directory for PNGs.")
    ap.add_argument("--env", type=str, nargs="*", default=None, help="Filter env_cases (e.g., forest_a::short).")
    ap.add_argument("--run-idxs", type=int, nargs="*", default=None, help="Specific run indices to plot.")
    args = ap.parse_args()
    plot_paths(args.base_dir, args.out_dir, env_cases=args.env, run_idxs=args.run_idxs)


if __name__ == "__main__":
    main()
