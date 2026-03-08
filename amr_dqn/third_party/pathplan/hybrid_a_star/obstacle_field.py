"""Obstacle distance field for penalty-graph search.

Computes a Euclidean distance transform of the occupancy grid so that
each free cell stores its distance (in metres) to the nearest obstacle.
Used by LOHybridAStarPlanner for proximity penalties and clearance evaluation.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt

from ..map_utils import GridMap


def compute_obstacle_distance_field(grid_map: GridMap) -> np.ndarray:
    """Return (H, W) float32 array: distance in metres from each cell to nearest obstacle.

    Obstacle cells have distance 0.  Free cells get Euclidean distance scaled
    by the grid resolution.
    """
    free_mask = grid_map.data == 0
    dist_cells = distance_transform_edt(free_mask)
    return (dist_cells * grid_map.resolution).astype(np.float32)


def query_distance(
    dist_field: np.ndarray,
    grid_map: GridMap,
    x: float,
    y: float,
) -> float:
    """Look up obstacle distance (metres) for a world-frame position.

    Out-of-bounds positions return a large distance (no penalty) rather than
    zero, because the absence of map data does not imply an obstacle.
    """
    gx, gy = grid_map.world_to_grid(x, y)
    if not grid_map.in_bounds(gx, gy):
        return 1e6  # far from any obstacle
    return float(dist_field[gy, gx])


def path_min_clearance(
    dist_field: np.ndarray,
    grid_map: GridMap,
    poses: list[Tuple[float, float, float]],
) -> float:
    """Return the minimum obstacle clearance (metres) along a sequence of poses."""
    if not poses:
        return 0.0
    min_d = math.inf
    for x, y, _theta in poses:
        d = query_distance(dist_field, grid_map, x, y)
        if d < min_d:
            min_d = d
    return float(min_d)
