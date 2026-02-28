"""Load ROS-convention PGM/YAML occupancy maps into ArrayGridMapSpec."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from amr_dqn.maps import ArrayGridMapSpec


def load_pgm_map(
    pgm_path: str | Path,
    start_xy: tuple[int, int],
    goal_xy: tuple[int, int],
    *,
    name: str = "pgm_map",
    occupied_thresh: float = 0.65,
    negate: bool = False,
) -> ArrayGridMapSpec:
    """Load a ROS-convention PGM occupancy map into ArrayGridMapSpec.

    ROS convention (negate=False):
      - Brighter pixel  →  more free  (p = pixel/255, low p = occupied)
      - pixel==0   → p=0.0 < free_thresh  → occupied
      - pixel==254 → p≈1.0 > free_thresh  → free

    Conversion to grid (y=0 at bottom, 1=obstacle):
      occupied if  (1 - pixel/255) > occupied_thresh  (i.e. pixel < (1-thresh)*255)

    Args:
        pgm_path: path to the .pgm file
        start_xy: (x, y) in grid coords (y=0 at bottom)
        goal_xy:  (x, y) in grid coords (y=0 at bottom)
        name:     map identifier
        occupied_thresh: ROS occupied_thresh from .yaml (default 0.65)
        negate:   ROS negate flag (default False)
    """
    pgm_path = Path(pgm_path)
    img = cv2.imread(str(pgm_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load PGM: {pgm_path}")

    H, W = img.shape
    # Compute occupancy probability
    p = img.astype(np.float32) / 255.0
    if negate:
        p = 1.0 - p

    # ROS: prob of occupancy = 1 - p (brighter = more free)
    occ_prob = 1.0 - p
    obstacle = (occ_prob > float(occupied_thresh)).astype(np.uint8)

    # Flip vertically: image row-0 is top; grid y=0 is bottom
    grid_y0_bottom = np.flipud(obstacle).copy()

    return ArrayGridMapSpec(
        name=name,
        grid_y0_bottom=grid_y0_bottom,
        start_xy=start_xy,
        goal_xy=goal_xy,
    )
