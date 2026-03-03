"""Path smoothing via Chaikin corner-cutting.

Repeatedly replaces each segment (p0, p1) with two new points at 25%/75%
along the segment, preserving start and end points.  Used to post-process
DQN paths and classical planner outputs before KPI evaluation.
"""

from __future__ import annotations

import numpy as np


def chaikin_smooth(points_xy: np.ndarray, *, iterations: int = 2) -> np.ndarray:
    """Chaikin corner-cutting algorithm for polyline smoothing.

    Args:
        points_xy: (N, 2) array.
        iterations: number of refinement iterations.

    Returns:
        (M, 2) array of smoothed points.
    """
    pts = np.asarray(points_xy, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points_xy must have shape (N, 2)")
    if len(pts) < 3 or iterations <= 0:
        return pts

    out = pts
    for _ in range(int(iterations)):
        new_pts = [out[0]]
        for p0, p1 in zip(out[:-1], out[1:], strict=False):
            q = 0.75 * p0 + 0.25 * p1
            r = 0.25 * p0 + 0.75 * p1
            new_pts.extend([q, r])
        new_pts.append(out[-1])
        out = np.vstack(new_pts).astype(np.float32, copy=False)
    return out

