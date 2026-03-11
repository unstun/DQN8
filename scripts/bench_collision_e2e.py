#!/usr/bin/env python3
"""End-to-end benchmark: RRT* & LO-HA* with exact vs EDT collision detection.

Same map, same start/goal pairs, compare:
  - Planning time
  - Success rate
  - Path length

Usage:
    PYTHONPATH=/home/sun/phdproject/dqn/DQN8 conda run -n ros2py310 \
        python scripts/bench_collision_e2e.py
"""
from __future__ import annotations

import math
import time
import types

import cv2
import numpy as np

from amr_dqn.baselines.pathplan import (
    default_ackermann_params,
    forest_two_circle_footprint,
    grid_map_from_obstacles,
    plan_lo_hybrid_astar,
    plan_rrt_star,
)
from amr_dqn.env import bilinear_sample_2d
from amr_dqn.maps import get_map_spec
from amr_dqn.third_party.pathplan.geometry import GridFootprintChecker

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ENV_NAMES = ["forest_a", "forest_b", "forest_c", "forest_d"]
CELL_SIZE_M = 1.0
NUM_PAIRS = 20  # start/goal pairs per map
TIMEOUT_S = 5.0
SEED = 123

params = default_ackermann_params()
fp = forest_two_circle_footprint()


# ---------------------------------------------------------------------------
# EDT monkey-patch for GridFootprintChecker
# ---------------------------------------------------------------------------
def _precompute_edt(grid: np.ndarray, cell_size_m: float) -> np.ndarray:
    """Compute EDT distance field in meters (same as RL env)."""
    grid_top = grid[::-1, :]
    free = (grid_top == 0).astype(np.uint8) * 255
    dist_top = cv2.distanceTransform(
        free, distanceType=cv2.DIST_L2, maskSize=cv2.DIST_MASK_PRECISE
    ).astype(np.float32)
    return (dist_top[::-1, :] * float(cell_size_m)).astype(np.float32, copy=False)


def _make_edt_collides_circle(edt_m: np.ndarray, cell_size_m: float, radius: float, half_cell_m: float):
    """Create an EDT-based _collides_circle_world replacement."""
    r_col = radius + half_cell_m

    def _collides_circle_world_edt(self, x: float, y: float) -> bool:
        xi = x / cell_size_m
        yi = y / cell_size_m
        d = bilinear_sample_2d(edt_m, x=xi, y=yi, default=0.0)
        return d <= r_col

    return _collides_circle_world_edt


def _patch_checker_to_edt(checker: GridFootprintChecker, edt_m: np.ndarray, cell_size_m: float):
    """Monkey-patch a GridFootprintChecker to use EDT instead of exact circle-AABB."""
    half_cell_m = 0.5 * cell_size_m
    radius = checker._circle_radius  # already includes padding
    fn = _make_edt_collides_circle(edt_m, cell_size_m, radius, half_cell_m)
    checker._collides_circle_world = types.MethodType(fn, checker)


# ---------------------------------------------------------------------------
# Patch planner factories to inject EDT checker
# ---------------------------------------------------------------------------
_orig_rrt_init = type(None)  # placeholder
_orig_ha_init = type(None)

# We patch at the GridFootprintChecker level after planner construction.
# Wrap the plan functions to patch the checker post-init.


def _plan_rrt_star_edt(*, edt_m, **kwargs):
    """Run RRT* with EDT collision detection."""
    # We need to intercept the checker creation inside RRTStarPlanner.__init__.
    # Strategy: run the planner constructor, then patch its checker before planning.
    from amr_dqn.third_party.pathplan.rrt.rrt_star import RRTStarPlanner

    cell_size_m = float(kwargs["grid_map"].resolution)
    st_rad = kwargs.get("start_theta_rad")
    if st_rad is None:
        sx, sy = kwargs["start_xy"]
        gx, gy = kwargs["goal_xy"]
        st_rad = math.atan2(
            (gy - sy) * cell_size_m, (gx - sx) * cell_size_m
        )

    from amr_dqn.third_party.pathplan.robot import AckermannState

    start = AckermannState(
        float(kwargs["start_xy"][0]) * cell_size_m,
        float(kwargs["start_xy"][1]) * cell_size_m,
        float(st_rad),
    )
    goal = AckermannState(
        float(kwargs["goal_xy"][0]) * cell_size_m,
        float(kwargs["goal_xy"][1]) * cell_size_m,
        float(kwargs.get("goal_theta_rad", 0.0)),
    )

    planner = RRTStarPlanner(
        kwargs["grid_map"],
        kwargs["footprint"],
        kwargs["params"],
        rng_seed=kwargs.get("seed", 0),
        goal_xy_tol=kwargs.get("goal_xy_tol_m", 0.1),
        goal_theta_tol=kwargs.get("goal_theta_tol_rad", math.pi),
    )
    _patch_checker_to_edt(planner.collision_checker, edt_m, cell_size_m)

    t0 = time.perf_counter()
    path, stats = planner.plan(
        start, goal,
        timeout=kwargs.get("timeout_s", 5.0),
        max_iter=kwargs.get("max_iter", 5000),
        self_check=False,
    )
    dt = time.perf_counter() - t0

    success = bool(stats.get("success", bool(path)))
    path_len = float(stats.get("path_length", 0.0))
    return success, dt, path_len, stats


def _plan_loha_edt(*, edt_m, **kwargs):
    """Run LO-HA* with EDT collision detection."""
    from amr_dqn.third_party.pathplan.hybrid_a_star.planner import HybridAStarPlanner

    cell_size_m = float(kwargs["grid_map"].resolution)
    st_rad = kwargs.get("start_theta_rad")
    if st_rad is None:
        sx, sy = kwargs["start_xy"]
        gx, gy = kwargs["goal_xy"]
        st_rad = math.atan2(
            (gy - sy) * cell_size_m, (gx - sx) * cell_size_m
        )

    from amr_dqn.third_party.pathplan.robot import AckermannState

    start = AckermannState(
        float(kwargs["start_xy"][0]) * cell_size_m,
        float(kwargs["start_xy"][1]) * cell_size_m,
        float(st_rad),
    )
    goal = AckermannState(
        float(kwargs["goal_xy"][0]) * cell_size_m,
        float(kwargs["goal_xy"][1]) * cell_size_m,
        float(kwargs.get("goal_theta_rad", 0.0)),
    )

    planner = HybridAStarPlanner(
        kwargs["grid_map"],
        kwargs["footprint"],
        kwargs["params"],
        goal_xy_tol=kwargs.get("goal_xy_tol_m", 0.1),
        goal_theta_tol=kwargs.get("goal_theta_tol_rad", math.pi),
    )
    _patch_checker_to_edt(planner.collision_checker, edt_m, cell_size_m)

    t0 = time.perf_counter()
    path, stats = planner.plan(
        start, goal,
        timeout=kwargs.get("timeout_s", 5.0),
        max_nodes=kwargs.get("max_nodes", 200_000),
        self_check=False,
    )
    dt = time.perf_counter() - t0

    success = bool(path)
    path_len = float(stats.get("path_length", 0.0))
    return success, dt, path_len, stats


# ---------------------------------------------------------------------------
# Generate random valid start/goal pairs
# ---------------------------------------------------------------------------
def _random_pairs(grid: np.ndarray, n: int, rng: np.random.RandomState, min_dist_cells: int = 10):
    """Sample n start/goal pairs on free cells with minimum distance."""
    h, w = grid.shape
    free_ys, free_xs = np.where(grid == 0)
    pairs = []
    attempts = 0
    while len(pairs) < n and attempts < n * 200:
        attempts += 1
        i1, i2 = rng.randint(0, len(free_xs), size=2)
        sx, sy = int(free_xs[i1]), int(free_ys[i1])
        gx, gy = int(free_xs[i2]), int(free_ys[i2])
        if math.hypot(gx - sx, gy - sy) >= min_dist_cells:
            pairs.append(((sx, sy), (gx, gy)))
    return pairs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    rng = np.random.RandomState(SEED)

    # Collect all results
    all_results = []  # list of dicts

    for env_name in ENV_NAMES:
        spec = get_map_spec(env_name)
        grid = spec.obstacle_grid()
        grid_map = grid_map_from_obstacles(grid_y0_bottom=grid, cell_size_m=CELL_SIZE_M)
        edt_m = _precompute_edt(grid, CELL_SIZE_M)

        pairs = _random_pairs(grid, NUM_PAIRS, rng, min_dist_cells=10)
        print(f"\n{'='*70}")
        print(f"  {env_name}: grid {grid.shape}, {len(pairs)} pairs")
        print(f"{'='*70}")

        for algo_name, plan_exact_fn, plan_edt_fn in [
            ("RRT*", "rrt", "rrt_edt"),
            ("LO-HA*", "loha", "loha_edt"),
        ]:
            times_exact, times_edt = [], []
            succ_exact, succ_edt = 0, 0
            plen_exact, plen_edt = [], []

            for pi, (start_xy, goal_xy) in enumerate(pairs):
                common_base = dict(
                    grid_map=grid_map,
                    footprint=fp,
                    params=params,
                    start_xy=start_xy,
                    goal_xy=goal_xy,
                    timeout_s=TIMEOUT_S,
                )

                # --- Exact ---
                if algo_name == "RRT*":
                    res_exact = plan_rrt_star(**common_base, seed=SEED + pi)
                    ok_e, t_e, pl_e = res_exact.success, res_exact.time_s, float(res_exact.stats.get("path_length", 0.0))
                else:
                    res_exact = plan_lo_hybrid_astar(**common_base, lo_iterations=0)
                    ok_e, t_e, pl_e = res_exact.success, res_exact.time_s, float(res_exact.stats.get("path_length", 0.0))

                # --- EDT ---
                if algo_name == "RRT*":
                    ok_d, t_d, pl_d, _ = _plan_rrt_star_edt(edt_m=edt_m, seed=SEED + pi, **common_base)
                else:
                    ok_d, t_d, pl_d, _ = _plan_loha_edt(edt_m=edt_m, **common_base)

                times_exact.append(t_e)
                times_edt.append(t_d)
                if ok_e:
                    succ_exact += 1
                    plen_exact.append(pl_e)
                if ok_d:
                    succ_edt += 1
                    plen_edt.append(pl_d)

            n = len(pairs)
            avg_t_exact = sum(times_exact) / n
            avg_t_edt = sum(times_edt) / n
            speedup = avg_t_exact / avg_t_edt if avg_t_edt > 0 else float("inf")
            avg_pl_exact = sum(plen_exact) / len(plen_exact) if plen_exact else 0.0
            avg_pl_edt = sum(plen_edt) / len(plen_edt) if plen_edt else 0.0

            row = dict(
                env=env_name,
                algo=algo_name,
                n_pairs=n,
                sr_exact=f"{succ_exact}/{n}",
                sr_edt=f"{succ_edt}/{n}",
                time_exact_s=avg_t_exact,
                time_edt_s=avg_t_edt,
                speedup=speedup,
                plen_exact_m=avg_pl_exact,
                plen_edt_m=avg_pl_edt,
            )
            all_results.append(row)
            print(f"\n  [{algo_name}]")
            print(f"    Exact : SR={succ_exact}/{n}, avg_time={avg_t_exact:.4f}s, avg_plen={avg_pl_exact:.2f}m")
            print(f"    EDT   : SR={succ_edt}/{n}, avg_time={avg_t_edt:.4f}s, avg_plen={avg_pl_edt:.2f}m")
            print(f"    Speedup: {speedup:.2f}x")

    # ---------------------------------------------------------------------------
    # Summary table
    # ---------------------------------------------------------------------------
    print(f"\n\n{'='*90}")
    print(f"  SUMMARY TABLE")
    print(f"{'='*90}")
    header = f"{'Env':<12} {'Algo':<8} {'SR(exact)':<10} {'SR(EDT)':<10} {'Time(exact)':<12} {'Time(EDT)':<12} {'Speedup':<8} {'PLen(exact)':<12} {'PLen(EDT)':<12}"
    print(header)
    print("-" * 90)
    for r in all_results:
        print(
            f"{r['env']:<12} {r['algo']:<8} {r['sr_exact']:<10} {r['sr_edt']:<10} "
            f"{r['time_exact_s']:<12.4f} {r['time_edt_s']:<12.4f} {r['speedup']:<8.2f} "
            f"{r['plen_exact_m']:<12.2f} {r['plen_edt_m']:<12.2f}"
        )
    print(f"{'='*90}")

    # Aggregate
    total_time_exact = sum(r["time_exact_s"] for r in all_results)
    total_time_edt = sum(r["time_edt_s"] for r in all_results)
    print(f"\n  Total planning time (exact): {total_time_exact:.2f}s")
    print(f"  Total planning time (EDT)  : {total_time_edt:.2f}s")
    print(f"  Overall speedup            : {total_time_exact / total_time_edt:.2f}x")


if __name__ == "__main__":
    main()
