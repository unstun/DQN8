"""RealMap overview: visualise occupancy grid with start/goal annotations."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from common import OUTPUT_DPI, apply_rcparams, load_map


def plot_map_overview(
    maps_dir: str | Path,
    out_dir: str | Path,
    env_bases: list[str] | None = None,
) -> list[Path]:
    apply_rcparams()
    maps_dir = Path(maps_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    meta_files = sorted(maps_dir.glob("*__meta.json"))
    for meta_path in meta_files:
        env_base = meta_path.stem.replace("__meta", "")
        if env_bases and env_base not in env_bases:
            continue

        obstacles, cell_size_m, meta = load_map(meta_path)
        h, w = obstacles.shape
        start = meta.get("canonical_start_xy", [0, 0])
        goal = meta.get("canonical_goal_xy", [0, 0])
        goal_tol = meta.get("goal", {}).get("position_tolerance_cells", 10)

        fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=OUTPUT_DPI)
        ax.imshow(
            obstacles.astype(int), origin="lower", cmap="gray_r",
            interpolation="nearest", extent=(-0.5, w - 0.5, -0.5, h - 0.5), alpha=0.9,
        )
        ax.scatter([start[0]], [start[1]], marker="*", s=200,
                   color="royalblue", label="Start", zorder=5)
        ax.scatter([goal[0]], [goal[1]], marker="*", s=200,
                   color="red", label="Goal", zorder=5)
        if goal_tol and goal_tol > 0:
            ax.add_patch(Circle(
                (goal[0], goal[1]), radius=float(goal_tol),
                edgecolor="red", facecolor="none", linestyle="--",
                linewidth=1.5, alpha=0.7, zorder=4,
            ))
        ax.set_aspect("equal", "box")
        ax.set_xlabel("X (cells)")
        ax.set_ylabel("Y (cells)")
        ax.set_title(f"Map: {env_base}  ({h}×{w}, cell={cell_size_m}m)")
        ax.legend(loc="upper right", frameon=True, framealpha=0.9)
        fig.tight_layout()

        out_path = out_dir / f"fig_map_{env_base}.png"
        fig.savefig(str(out_path), bbox_inches="tight", dpi=OUTPUT_DPI)
        plt.close(fig)
        saved.append(out_path)
        print(f"  saved {out_path.name}")
    return saved


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot map overview figures.")
    ap.add_argument("--maps-dir", type=str, required=True, help="Directory containing *__meta.json and *__grid_y0_bottom.npz.")
    ap.add_argument("--out-dir", type=str, default="figures")
    ap.add_argument("--env", type=str, nargs="*", default=None, help="Filter by env_base (e.g., realmap_a).")
    args = ap.parse_args()
    plot_map_overview(args.maps_dir, args.out_dir, env_bases=args.env)


if __name__ == "__main__":
    main()
