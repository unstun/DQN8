#!/usr/bin/env python3
"""Benchmark: EDT distance-field collision vs exact circle-AABB collision.

Generates a random forest map, builds both checkers, and times N random pose
collision queries.  Reports per-query time and agreement rate.

Usage:
    conda run --cwd /home/sun/phdproject/dqn/DQN8 -n ros2py310 \
        python scripts/bench_collision_edt_vs_exact.py
"""
from __future__ import annotations

import math
import time

import cv2
import numpy as np

from amr_dqn.third_party.pathplan.geometry import GridFootprintChecker, TwoCircleFootprint
from amr_dqn.third_party.pathplan.map_utils import GridMap


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MAP_SIZE = 50  # cells
CELL_SIZE_M = 1.0
OBSTACLE_DENSITY = 0.15
NUM_QUERIES = 200_000
SEED = 42
WHEELBASE_M = 0.6

# ---------------------------------------------------------------------------
# Build random forest map
# ---------------------------------------------------------------------------
rng = np.random.RandomState(SEED)
grid = (rng.rand(MAP_SIZE, MAP_SIZE) < OBSTACLE_DENSITY).astype(np.uint8)
# Clear a border so poses near edges are valid
grid[0, :] = 1; grid[-1, :] = 1; grid[:, 0] = 1; grid[:, -1] = 1

grid_map = GridMap(data=grid, resolution=CELL_SIZE_M, origin=(0.0, 0.0))

# ---------------------------------------------------------------------------
# Footprint (same as forest env)
# ---------------------------------------------------------------------------
fp = TwoCircleFootprint.from_box(length=0.924, width=0.740, center_shift=0.5 * WHEELBASE_M)
print(f"TwoCircleFootprint: radius={fp.radius:.4f} m, "
      f"center_offset={fp.center_offset:.4f} m, center_shift={fp.center_shift:.4f} m")

# ---------------------------------------------------------------------------
# Method A: exact circle-AABB (GridFootprintChecker)
# ---------------------------------------------------------------------------
exact_checker = GridFootprintChecker(grid_map, fp, theta_bins=72)

# ---------------------------------------------------------------------------
# Method B: EDT distance field (same as RL env)
# ---------------------------------------------------------------------------
# Precompute EDT
grid_top = grid[::-1, :]
free = (grid_top == 0).astype(np.uint8) * 255
dist_top = cv2.distanceTransform(
    free, distanceType=cv2.DIST_L2, maskSize=cv2.DIST_MASK_PRECISE
).astype(np.float32)
# Convert to meters, subtract half-cell boundary correction
dist_m = (dist_top[::-1, :] * CELL_SIZE_M).astype(np.float32)
# Note: RL env subtracts 0.5 in cell space during preprocessing;
# we keep raw EDT here and apply the margin in the collision check below.

half_cell_m = 0.5 * CELL_SIZE_M


def _bilinear_sample(arr: np.ndarray, x: float, y: float) -> float:
    """Bilinear sample at index coords (x, y) on (H, W) array."""
    h, w = arr.shape
    if not (0.0 <= x <= w - 1 and 0.0 <= y <= h - 1):
        return 0.0
    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    x1 = min(x0 + 1, w - 1)
    y1 = min(y0 + 1, h - 1)
    fx = x - x0
    fy = y - y0
    v00 = float(arr[y0, x0])
    v10 = float(arr[y0, x1])
    v01 = float(arr[y1, x0])
    v11 = float(arr[y1, x1])
    return (v00 * (1 - fx) + v10 * fx) * (1 - fy) + (v01 * (1 - fx) + v11 * fx) * fy


def edt_collides(x_m: float, y_m: float, theta: float) -> bool:
    """EDT-based collision check, matching RL env logic."""
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    s = fp.center_shift
    d = fp.center_offset
    mid_x = x_m + cos_t * s
    mid_y = y_m + sin_t * s
    # front circle
    fx = mid_x + cos_t * d
    fy = mid_y + sin_t * d
    # rear circle
    rx = mid_x - cos_t * d
    ry = mid_y - sin_t * d

    r = fp.radius
    r_col = r + half_cell_m

    # Convert to index coords for sampling
    d1 = _bilinear_sample(dist_m, fx / CELL_SIZE_M, fy / CELL_SIZE_M)
    d2 = _bilinear_sample(dist_m, rx / CELL_SIZE_M, ry / CELL_SIZE_M)
    return (d1 <= r_col) or (d2 <= r_col)


# ---------------------------------------------------------------------------
# Generate random poses
# ---------------------------------------------------------------------------
margin = 2.0  # stay away from map boundary
xs = rng.uniform(margin, (MAP_SIZE - margin) * CELL_SIZE_M, NUM_QUERIES)
ys = rng.uniform(margin, (MAP_SIZE - margin) * CELL_SIZE_M, NUM_QUERIES)
thetas = rng.uniform(-math.pi, math.pi, NUM_QUERIES)

# ---------------------------------------------------------------------------
# Benchmark exact
# ---------------------------------------------------------------------------
print(f"\nBenchmarking {NUM_QUERIES:,} collision queries...")

t0 = time.perf_counter()
exact_results = np.empty(NUM_QUERIES, dtype=bool)
for i in range(NUM_QUERIES):
    exact_results[i] = exact_checker.collides_pose(xs[i], ys[i], thetas[i])
t_exact = time.perf_counter() - t0

# ---------------------------------------------------------------------------
# Benchmark EDT
# ---------------------------------------------------------------------------
t0 = time.perf_counter()
edt_results = np.empty(NUM_QUERIES, dtype=bool)
for i in range(NUM_QUERIES):
    edt_results[i] = edt_collides(xs[i], ys[i], thetas[i])
t_edt = time.perf_counter() - t0

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
agree = np.sum(exact_results == edt_results)
disagree = NUM_QUERIES - agree
# Where they disagree, check direction
edt_miss = np.sum(exact_results & ~edt_results)  # exact=collision, EDT=no collision (漏检)
edt_false_alarm = np.sum(~exact_results & edt_results)  # exact=free, EDT=collision (误报)

print(f"\n{'='*60}")
print(f"  Exact (circle-AABB)  : {t_exact:.3f} s  ({t_exact/NUM_QUERIES*1e6:.2f} µs/query)")
print(f"  EDT (distance field) : {t_edt:.3f} s  ({t_edt/NUM_QUERIES*1e6:.2f} µs/query)")
print(f"  Speedup              : {t_exact/t_edt:.2f}x")
print(f"{'='*60}")
print(f"  Agreement            : {agree:,}/{NUM_QUERIES:,} ({100*agree/NUM_QUERIES:.2f}%)")
print(f"  EDT misses (漏检)    : {edt_miss:,} ({100*edt_miss/NUM_QUERIES:.4f}%)")
print(f"  EDT false alarm(误报): {edt_false_alarm:,} ({100*edt_false_alarm/NUM_QUERIES:.4f}%)")
print(f"  Exact collision rate : {100*exact_results.sum()/NUM_QUERIES:.2f}%")
print(f"  EDT collision rate   : {100*edt_results.sum()/NUM_QUERIES:.2f}%")
print(f"{'='*60}")
