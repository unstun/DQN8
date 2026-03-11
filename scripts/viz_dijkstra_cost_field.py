#!/usr/bin/env python3
"""Visualize the Dijkstra cost-to-goal field overlaid on a grid occupancy map.

Usage:
    conda run --cwd /home/sun/phdproject/dqn/DQN8 -n ros2py310 \
        python scripts/viz_dijkstra_cost_field.py [--seed 42] [--env forest_s42]

Outputs a multi-panel figure showing:
  (a) Binary occupancy grid
  (b) Dijkstra cost-to-goal heatmap (obstacles masked)
  (c) Overlay: cost field with obstacle contours + start/goal markers
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from amr_dqn.env import dijkstra_cost_to_goal_m
from amr_dqn.maps import get_map_spec


def main() -> None:
    ap = argparse.ArgumentParser(description="Dijkstra cost field visualization")
    ap.add_argument("--env", default="forest_a", help="Map name (default: forest_a)")
    ap.add_argument("--out", default=None, help="Output PNG path (default: auto)")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--show", action="store_true", help="plt.show() instead of saving")
    args = ap.parse_args()

    # --- Load map ---
    spec = get_map_spec(args.env)
    grid = spec.obstacle_grid()  # (H, W), y=0 bottom, 1=obstacle
    start_xy = spec.start_xy  # (x, y) in cell coords
    goal_xy = spec.goal_xy
    cell_size_m = 0.1  # forest default

    H, W = grid.shape
    print(f"Map: {args.env}  shape={H}x{W}  cell={cell_size_m}m")
    print(f"Start: {start_xy}  Goal: {goal_xy}")

    # --- Dijkstra cost field ---
    traversable = (grid == 0).astype(np.uint8)
    cost = dijkstra_cost_to_goal_m(traversable, goal_xy=goal_xy, cell_size_m=cell_size_m)

    # Mask obstacles and unreachable cells for plotting
    cost_masked = np.ma.masked_where(~np.isfinite(cost) | (grid == 1), cost)
    vmax = float(np.nanmax(cost[np.isfinite(cost)])) if np.any(np.isfinite(cost)) else 1.0

    print(f"Cost range: 0 ~ {vmax:.2f} m")
    print(f"Goal cost: {cost[goal_xy[1], goal_xy[0]]:.4f} m (should be 0)")
    start_cost = cost[start_xy[1], start_xy[0]]
    print(f"Start cost: {start_cost:.2f} m")

    # --- Extent for imshow (meters) ---
    extent = [0, W * cell_size_m, 0, H * cell_size_m]
    sx_m = (start_xy[0] + 0.5) * cell_size_m
    sy_m = (start_xy[1] + 0.5) * cell_size_m
    gx_m = (goal_xy[0] + 0.5) * cell_size_m
    gy_m = (goal_xy[1] + 0.5) * cell_size_m

    # --- Custom colormap: blue(low/near goal) -> yellow(mid) -> red(far) ---
    cmap_cost = LinearSegmentedColormap.from_list(
        "dijkstra",
        ["#0d0887", "#5b02a3", "#9a179b", "#cb4679", "#eb7852", "#fbb32f", "#f0f921"],
    )
    cmap_cost.set_bad(color="black")  # obstacles / unreachable

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), constrained_layout=True)

    # (a) Occupancy grid
    ax = axes[0]
    ax.set_title("(a) Occupancy Grid", fontsize=13, fontweight="bold")
    ax.imshow(grid, origin="lower", extent=extent, cmap="gray_r", vmin=0, vmax=1)
    ax.plot(sx_m, sy_m, "go", ms=10, label="Start", zorder=5)
    ax.plot(gx_m, gy_m, "r*", ms=14, label="Goal", zorder=5)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    # (b) Dijkstra cost field
    ax = axes[1]
    ax.set_title("(b) Dijkstra Cost-to-Goal (m)", fontsize=13, fontweight="bold")
    im = ax.imshow(cost_masked, origin="lower", extent=extent, cmap=cmap_cost, vmin=0, vmax=vmax)
    ax.plot(gx_m, gy_m, "r*", ms=14, zorder=5)
    fig.colorbar(im, ax=ax, label="Distance to goal (m)", shrink=0.85)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    # (c) Overlay: cost field + obstacle contours
    ax = axes[2]
    ax.set_title("(c) Cost Field + Obstacles", fontsize=13, fontweight="bold")
    ax.imshow(cost_masked, origin="lower", extent=extent, cmap=cmap_cost, vmin=0, vmax=vmax)
    # Obstacle contour
    ax.contour(
        grid, levels=[0.5], origin="lower", extent=extent,
        colors="white", linewidths=0.6, alpha=0.8,
    )
    # Fill obstacles semi-transparent
    obs_mask = np.ma.masked_where(grid == 0, grid)
    ax.imshow(obs_mask, origin="lower", extent=extent, cmap="gray_r", alpha=0.7)
    ax.plot(sx_m, sy_m, "go", ms=10, label="Start", zorder=5)
    ax.plot(gx_m, gy_m, "r*", ms=14, label="Goal", zorder=5)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    fig.suptitle(
        f"Dijkstra Cost-to-Goal Field on Grid Map ({args.env})",
        fontsize=15, fontweight="bold", y=1.02,
    )

    if args.show:
        plt.show()
    else:
        out_path = args.out or f"dqn8_plots/figs/dijkstra_cost_field_{args.env}.png"
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
        print(f"Saved to {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
