"""Offline precomputation of Hybrid A* expert paths for forest maps.

Generates JSON files in amr_dqn/maps/precomputed/ that are loaded at
training time to provide demonstration trajectories for DQfD prefill.
Each JSON contains the path as a list of (x_cells, y_cells) waypoints
plus timing and planner stats.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from amr_dqn.baselines.pathplan import (
    default_ackermann_params,
    forest_two_circle_footprint,
    grid_map_from_obstacles,
    plan_hybrid_astar,
)
from amr_dqn.maps import FOREST_ENV_ORDER, get_map_spec


def main() -> int:
    ap = argparse.ArgumentParser(description="Precompute Hybrid A* reference paths for forest_* maps.")
    ap.add_argument("--envs", nargs="*", default=list(FOREST_ENV_ORDER), help="Subset of envs: forest_a forest_b forest_c forest_d")
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent.parent / "maps" / "precomputed")
    ap.add_argument("--timeout-s", type=float, default=60.0)
    ap.add_argument("--max-nodes", type=int, default=1_200_000)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for env_name in args.envs:
        spec = get_map_spec(str(env_name))
        grid = spec.obstacle_grid()
        cell_size_m = 0.1

        grid_map = grid_map_from_obstacles(grid_y0_bottom=grid, cell_size_m=cell_size_m)
        params = default_ackermann_params()
        footprint = forest_two_circle_footprint()

        res = plan_hybrid_astar(
            grid_map=grid_map,
            footprint=footprint,
            params=params,
            start_xy=spec.start_xy,
            goal_xy=spec.goal_xy,
            goal_theta_rad=0.0,
            start_theta_rad=None,
            goal_xy_tol_m=0.5,
            goal_theta_tol_rad=float(math.pi),
            timeout_s=float(args.timeout_s),
            max_nodes=int(args.max_nodes),
        )

        payload = {
            "cell_size_m": float(cell_size_m),
            "env": str(env_name),
            "goal_xy": [int(spec.goal_xy[0]), int(spec.goal_xy[1])],
            "goal_xy_tol_m": 0.5,
            "path_xy_cells": [[float(x), float(y)] for x, y in res.path_xy_cells],
            "stats": res.stats,
            "success": bool(res.success),
            "time_s": float(res.time_s),
        }

        out_path = out_dir / f"{env_name}_hybrid_astar_path.json"
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[ok] {env_name}: success={res.success} time_s={res.time_s:.3f} -> {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
