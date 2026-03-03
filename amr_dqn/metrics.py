"""Path quality KPI helpers for evaluation.

Computes geometric metrics on polyline paths (list of (x, y) tuples):
- path_length()           Total Euclidean length.
- corner_angles_deg()     Turning angle at each interior vertex.
- num_path_corners()      Count of vertices exceeding an angle threshold.
- max_corner_degree()     Maximum turning angle.
- avg_abs_curvature()     Length-weighted mean |kappa| via circumcircle formula.
- KPI dataclass           Aggregated result bundle used by infer.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def path_length(path: list[tuple[float, float]]) -> float:
    if len(path) < 2:
        return 0.0
    total = 0.0
    for (x0, y0), (x1, y1) in zip(path[:-1], path[1:], strict=False):
        total += math.hypot(float(x1) - float(x0), float(y1) - float(y0))
    return float(total)


def corner_angles_deg(path: list[tuple[float, float]]) -> list[float]:
    if len(path) < 3:
        return []
    angles: list[float] = []
    for (x0, y0), (x1, y1), (x2, y2) in zip(path[:-2], path[1:-1], path[2:], strict=False):
        v1 = (float(x1) - float(x0), float(y1) - float(y0))
        v2 = (float(x2) - float(x1), float(y2) - float(y1))
        if abs(v1[0]) + abs(v1[1]) < 1e-9 or abs(v2[0]) + abs(v2[1]) < 1e-9:
            continue
        dot = float(v1[0] * v2[0] + v1[1] * v2[1])
        n1 = math.hypot(v1[0], v1[1])
        n2 = math.hypot(v2[0], v2[1])
        if n1 < 1e-12 or n2 < 1e-12:
            continue
        cos = max(-1.0, min(1.0, dot / (n1 * n2)))
        ang = float(math.degrees(math.acos(cos)))
        angles.append(ang)
    return angles


def num_path_corners(path: list[tuple[float, float]], *, angle_threshold_deg: float = 1.0) -> int:
    th = float(angle_threshold_deg)
    if th < 0:
        raise ValueError("angle_threshold_deg must be >= 0")
    return int(sum(1 for a in corner_angles_deg(path) if a >= th))


def max_corner_degree(path: list[tuple[float, float]]) -> float:
    angles = corner_angles_deg(path)
    return float(max(angles) if angles else 0.0)


def avg_abs_curvature(path_m: list[tuple[float, float]]) -> float:
    """Length-weighted average absolute curvature (1/m) for a polyline in meters."""
    if len(path_m) < 3:
        return 0.0
    num = 0.0
    denom = 0.0
    for (x0, y0), (x1, y1), (x2, y2) in zip(path_m[:-2], path_m[1:-1], path_m[2:], strict=False):
        x0 = float(x0)
        y0 = float(y0)
        x1 = float(x1)
        y1 = float(y1)
        x2 = float(x2)
        y2 = float(y2)
        a = math.hypot(x1 - x0, y1 - y0)
        b = math.hypot(x2 - x1, y2 - y1)
        c = math.hypot(x2 - x0, y2 - y0)
        if a < 1e-9 or b < 1e-9 or c < 1e-9:
            continue
        # 2*triangle_area = |(p1-p0) x (p2-p0)|
        area2 = abs((x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0))
        # Curvature magnitude of the circumcircle: kappa = 4A/(abc) = 2*(2A)/(abc) = 2*area2/(abc)
        kappa = (2.0 * area2) / (a * b * c)
        w = 0.5 * (a + b)
        num += abs(kappa) * w
        denom += w
    if denom < 1e-12:
        return 0.0
    return float(num / denom)


@dataclass(frozen=True)
class KPI:
    avg_path_length: float
    path_time_s: float
    avg_curvature_1_m: float
    planning_time_s: float
    tracking_time_s: float
    inference_time_s: float
    num_corners: float
    max_corner_deg: float
