"""Hybrid A* and RRT* planning wrappers for baseline evaluation.

Provides
--------
- PlannerResult             Dataclass: path + timing + success flag + stats.
- default_ackermann_params  Default Ackermann kinematic params (wheelbase 0.6m, delta_max 27deg).
- grid_map_from_obstacles   Convert numpy obstacle grid to GridMap for planners.
- forest_two_circle_footprint / forest_oriented_box_footprint
                            Vehicle collision geometry matching the bicycle env.
- plan_hybrid_astar()       Run Hybrid A* with timeout and node limit.
- plan_rrt_star()           Run RRT* with multi-restart strategy.

These functions are called by cli/infer.py to produce classical-planner paths
that serve as baselines for the DQN agent comparison (Table II in the paper).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from amr_dqn.third_party.pathplan import (
    AckermannParams,
    AckermannState,
    GridMap,
    HybridAStarPlanner,
    OrientedBoxFootprint,
    RRTStarPlanner,
    TwoCircleFootprint,
)


@dataclass(frozen=True)
class PlannerResult:
    path_xy_cells: list[tuple[float, float]]
    time_s: float
    success: bool
    stats: dict[str, Any]


def default_ackermann_params(
    *,
    wheelbase_m: float = 0.6,
    delta_max_rad: float = math.radians(27.0),
    v_max_m_s: float = 2.0,
) -> AckermannParams:
    min_turn_radius_m = float(wheelbase_m) / max(1e-9, float(math.tan(float(delta_max_rad))))
    return AckermannParams(
        wheelbase=float(wheelbase_m),
        min_turn_radius=float(min_turn_radius_m),
        v_max=float(v_max_m_s),
    )


def grid_map_from_obstacles(*, grid_y0_bottom: np.ndarray, cell_size_m: float) -> GridMap:
    g = np.asarray(grid_y0_bottom, dtype=np.uint8)
    if g.ndim != 2:
        raise ValueError("grid_y0_bottom must be a (H,W) array")
    if not (float(cell_size_m) > 0.0):
        raise ValueError("cell_size_m must be > 0")
    return GridMap(g, resolution=float(cell_size_m), origin=(0.0, 0.0))


def point_footprint(*, cell_size_m: float) -> OrientedBoxFootprint:
    # Keep a small non-zero size so the unified collision checker does not
    # degenerate on strict comparisons.
    s = max(1e-3, 0.1 * float(cell_size_m))
    return OrientedBoxFootprint(length=s, width=s)


def forest_oriented_box_footprint() -> OrientedBoxFootprint:
    # Matches the forest env's nominal vehicle dimensions used by the two-circle approximation:
    # length=0.924m, width=0.740m.
    return OrientedBoxFootprint(length=0.924, width=0.740)


def forest_two_circle_footprint(*, wheelbase_m: float = 0.6) -> TwoCircleFootprint:
    # Use the same nominal vehicle dimensions as the forest env and convert to a conservative
    # two-circle approximation (robust for grid collision checks at arbitrary headings).
    #
    # IMPORTANT: The forest env's bicycle model state is the rear-axle center. To match that
    # reference, shift the footprint forward by wheelbase/2 so the two-circle model covers
    # the vehicle body centered around the axle midpoint.
    box = forest_oriented_box_footprint()
    return TwoCircleFootprint.from_box(
        length=float(box.length),
        width=float(box.width),
        center_shift=0.5 * float(wheelbase_m),
    )


def _default_start_theta(start_xy: tuple[int, int], goal_xy: tuple[int, int], *, cell_size_m: float) -> float:
    dx = float(goal_xy[0] - start_xy[0]) * float(cell_size_m)
    dy = float(goal_xy[1] - start_xy[1]) * float(cell_size_m)
    return float(math.atan2(dy, dx))


def plan_hybrid_astar(
    *,
    grid_map: GridMap,
    footprint: OrientedBoxFootprint | TwoCircleFootprint,
    params: AckermannParams,
    start_xy: tuple[int, int],
    goal_xy: tuple[int, int],
    goal_theta_rad: float = 0.0,
    start_theta_rad: float | None = None,
    goal_xy_tol_m: float = 0.1,
    goal_theta_tol_rad: float = math.pi,
    timeout_s: float = 5.0,
    max_nodes: int = 200_000,
) -> PlannerResult:
    cell_size_m = float(grid_map.resolution)
    st = float(start_theta_rad) if start_theta_rad is not None else _default_start_theta(start_xy, goal_xy, cell_size_m=cell_size_m)
    start = AckermannState(float(start_xy[0]) * cell_size_m, float(start_xy[1]) * cell_size_m, st)
    goal = AckermannState(float(goal_xy[0]) * cell_size_m, float(goal_xy[1]) * cell_size_m, float(goal_theta_rad))

    planner = HybridAStarPlanner(
        grid_map,
        footprint,
        params,
        goal_xy_tol=float(goal_xy_tol_m),
        goal_theta_tol=float(goal_theta_tol_rad),
    )

    t0 = time.perf_counter()
    path, stats = planner.plan(start, goal, timeout=float(timeout_s), max_nodes=int(max_nodes), self_check=False)
    t1 = time.perf_counter()
    dt = float(stats.get("time", t1 - t0))

    if path:
        pts = [(float(s.x) / cell_size_m, float(s.y) / cell_size_m) for s in path]
        return PlannerResult(path_xy_cells=pts, time_s=dt, success=True, stats=stats)
    return PlannerResult(
        path_xy_cells=[(float(start_xy[0]), float(start_xy[1]))],
        time_s=dt,
        success=False,
        stats=stats,
    )


def plan_rrt_star(
    *,
    grid_map: GridMap,
    footprint: OrientedBoxFootprint | TwoCircleFootprint,
    params: AckermannParams,
    start_xy: tuple[int, int],
    goal_xy: tuple[int, int],
    seed: int = 0,
    goal_theta_rad: float = 0.0,
    start_theta_rad: float | None = None,
    goal_xy_tol_m: float = 0.1,
    goal_theta_tol_rad: float = math.pi,
    timeout_s: float = 5.0,
    max_iter: int = 5_000,
) -> PlannerResult:
    cell_size_m = float(grid_map.resolution)
    st = float(start_theta_rad) if start_theta_rad is not None else _default_start_theta(start_xy, goal_xy, cell_size_m=cell_size_m)
    start = AckermannState(float(start_xy[0]) * cell_size_m, float(start_xy[1]) * cell_size_m, st)
    goal = AckermannState(float(goal_xy[0]) * cell_size_m, float(goal_xy[1]) * cell_size_m, float(goal_theta_rad))

    # RRT* is stochastic; for tight per-run budgets we do a few restarts with different RNG
    # seeds. We intentionally keep planner hyperparameters at their library defaults because
    # they were tuned together; changing them can *reduce* success in some scenes.
    base_seed = int(seed)
    max_restarts = 2  # total attempts = 1 + max_restarts
    # Allocate most budget to the first attempt, but keep a small reserve for retries.
    time_fracs = (0.85, 0.10, 0.05)
    iter_fracs = (0.85, 0.10, 0.05)

    start_time = time.perf_counter()
    last_stats: dict[str, Any] = {}
    for attempt in range(1 + int(max_restarts)):
        remaining_time = float(timeout_s) - float(time.perf_counter() - start_time)
        if remaining_time <= 0.0:
            break
        if int(max_iter) <= 0:
            break

        attempt_timeout = min(float(remaining_time), float(timeout_s) * float(time_fracs[int(attempt)]))
        attempt_max_iter = max(1, int(round(float(max_iter) * float(iter_fracs[int(attempt)]))))

        planner = RRTStarPlanner(
            grid_map,
            footprint,
            params,
            rng_seed=base_seed + 1_000_003 * int(attempt),
            goal_xy_tol=float(goal_xy_tol_m),
            goal_theta_tol=float(goal_theta_tol_rad),
        )

        path, stats = planner.plan(
            start,
            goal,
            timeout=float(attempt_timeout),
            max_iter=int(attempt_max_iter),
            self_check=False,
        )
        last_stats = dict(stats)
        last_stats["attempt"] = int(attempt) + 1
        last_stats["attempt_seed"] = base_seed + 1_000_003 * int(attempt)
        last_stats["attempt_timeout_s"] = float(attempt_timeout)
        last_stats["attempt_max_iter"] = int(attempt_max_iter)

        success = bool(stats.get("success", bool(path)))
        if success and path:
            pts = [(float(s.x) / cell_size_m, float(s.y) / cell_size_m) for s in path]
            return PlannerResult(
                path_xy_cells=pts,
                time_s=float(time.perf_counter() - start_time),
                success=True,
                stats=last_stats,
            )

    return PlannerResult(
        path_xy_cells=[(float(start_xy[0]), float(start_xy[1]))],
        time_s=float(time.perf_counter() - start_time),
        success=False,
        stats=last_stats,
    )
