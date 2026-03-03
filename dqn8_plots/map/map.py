"""LAS/LAZ point cloud → PGM occupancy grid.

Adapted from dqn4作图/map/map.py with parameterised input paths.
Run: python map.py --las-path /path/to/scan.las --out-dir grid_out/
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

try:
    import laspy
except ImportError:
    laspy = None  # type: ignore[assignment]

from scipy import ndimage as ndi

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False


# ---------------------------------------------------------------------------
# ROS PGM / YAML I/O
# ---------------------------------------------------------------------------
def write_pgm(pgm_path: Path, occ_grid: np.ndarray) -> None:
    unk, free, occ = 205, 254, 0
    img = np.full(occ_grid.shape, unk, dtype=np.uint8)
    img[occ_grid == 0] = free
    img[occ_grid == 100] = occ
    img = np.flipud(img)
    h, w = img.shape
    header = f"P5\n{w} {h}\n255\n".encode("ascii")
    with open(pgm_path, "wb") as f:
        f.write(header)
        f.write(img.tobytes())


def write_ros_yaml(yaml_path: Path, pgm_filename: str, resolution: float, origin_xy: Tuple[float, float]) -> None:
    ox, oy = origin_xy
    text = (
        f"image: {pgm_filename}\n"
        f"resolution: {resolution}\n"
        f"origin: [{ox}, {oy}, 0.0]\n"
        f"negate: 0\n"
        f"occupied_thresh: 0.65\n"
        f"free_thresh: 0.196\n"
    )
    yaml_path.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Grid construction from LAS
# ---------------------------------------------------------------------------
def build_grids_from_las(
    las_path: Path,
    resolution: float,
    z_min: float = -999,
    z_max: float = 999,
    rotate_z_deg: float = 0.0,
) -> dict:
    if laspy is None:
        raise ImportError("laspy is required: pip install laspy")
    las = laspy.read(str(las_path))
    x = np.asarray(las.x, dtype=np.float64)
    y = np.asarray(las.y, dtype=np.float64)
    z = np.asarray(las.z, dtype=np.float64)

    if rotate_z_deg != 0.0:
        rad = np.radians(rotate_z_deg)
        c, s = np.cos(rad), np.sin(rad)
        x, y = c * x - s * y, s * x + c * y

    mask = (z >= z_min) & (z <= z_max)
    x, y, z = x[mask], y[mask], z[mask]

    min_x, min_y = x.min(), y.min()
    max_x, max_y = x.max(), y.max()
    nx = int(np.ceil((max_x - min_x) / resolution)) + 1
    ny = int(np.ceil((max_y - min_y) / resolution)) + 1

    ix = np.clip(((x - min_x) / resolution).astype(int), 0, nx - 1)
    iy = np.clip(((y - min_y) / resolution).astype(int), 0, ny - 1)

    count = np.zeros((ny, nx), dtype=np.int32)
    mean_z = np.full((ny, nx), np.nan, dtype=np.float32)
    z_min_arr = np.full((ny, nx), np.nan, dtype=np.float32)
    z_max_arr = np.full((ny, nx), np.nan, dtype=np.float32)
    roughness = np.full((ny, nx), np.nan, dtype=np.float32)

    np.add.at(count, (iy, ix), 1)
    for i in range(len(x)):
        r, c_ = int(iy[i]), int(ix[i])
        zv = float(z[i])
        if np.isnan(mean_z[r, c_]):
            mean_z[r, c_] = zv
            z_min_arr[r, c_] = zv
            z_max_arr[r, c_] = zv
        else:
            mean_z[r, c_] += zv
            z_min_arr[r, c_] = min(z_min_arr[r, c_], zv)
            z_max_arr[r, c_] = max(z_max_arr[r, c_], zv)

    valid = count > 0
    mean_z[valid] /= count[valid]
    roughness[valid] = z_max_arr[valid] - z_min_arr[valid]

    return {
        "shape": (ny, nx),
        "count": count,
        "mean_z": mean_z,
        "z_min": z_min_arr,
        "z_max": z_max_arr,
        "roughness": roughness,
        "min_x": float(min_x),
        "min_y": float(min_y),
        "max_x": float(max_x),
        "max_y": float(max_y),
        "resolution": float(resolution),
    }


def occupancy_from_stats(
    count: np.ndarray,
    roughness: np.ndarray,
    min_points: int = 1,
    roughness_thresh: float = 0.2,
    points_as_free: bool = True,
) -> np.ndarray:
    ny, nx = count.shape
    occ = np.full((ny, nx), -1, dtype=np.int8)
    seen = count >= min_points
    if points_as_free:
        occ[seen] = 0
    else:
        occ[seen] = 0
        rough = seen & (roughness > roughness_thresh)
        occ[rough] = 100
    return occ


def clean_occupancy(
    occ: np.ndarray,
    morph_open: int = 1,
    morph_close: int = 1,
    min_obs_cells: int = 16,
) -> np.ndarray:
    occ = occ.copy()
    binary = occ == 100
    if morph_open > 0:
        binary = ndi.binary_opening(binary, iterations=morph_open)
    if morph_close > 0:
        binary = ndi.binary_closing(binary, iterations=morph_close)
    if min_obs_cells > 0:
        labeled, n = ndi.label(binary)
        for i in range(1, n + 1):
            if np.sum(labeled == i) < min_obs_cells:
                binary[labeled == i] = False
    occ[(occ == 100) & ~binary] = 0
    occ[(occ != -1) & binary] = 100
    return occ


def main() -> None:
    ap = argparse.ArgumentParser(description="LAS/LAZ → PGM occupancy grid.")
    ap.add_argument("--las-path", type=str, required=True, help="Input LAS/LAZ file.")
    ap.add_argument("--out-dir", type=str, default="grid_out", help="Output directory.")
    ap.add_argument("--resolution", type=float, default=0.1, help="Grid cell size (m).")
    ap.add_argument("--z-min", type=float, default=-999, help="Min Z filter.")
    ap.add_argument("--z-max", type=float, default=999, help="Max Z filter.")
    ap.add_argument("--rotate-z", type=float, default=0.0, help="Rotation around Z (degrees).")
    ap.add_argument("--min-points", type=int, default=1)
    ap.add_argument("--roughness-thresh", type=float, default=0.20)
    ap.add_argument("--morph-open", type=int, default=1)
    ap.add_argument("--morph-close", type=int, default=1)
    ap.add_argument("--min-obs-cells", type=int, default=16)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    grids = build_grids_from_las(
        Path(args.las_path),
        resolution=args.resolution,
        z_min=args.z_min,
        z_max=args.z_max,
        rotate_z_deg=args.rotate_z,
    )
    occ = occupancy_from_stats(
        grids["count"], grids["roughness"],
        min_points=args.min_points,
        roughness_thresh=args.roughness_thresh,
        points_as_free=False,
    )
    occ = clean_occupancy(occ, morph_open=args.morph_open, morph_close=args.morph_close,
                          min_obs_cells=args.min_obs_cells)

    pgm_name = "map.pgm"
    write_pgm(out_dir / pgm_name, occ)
    write_ros_yaml(out_dir / "map.yaml", pgm_name, args.resolution,
                   (grids["min_x"], grids["min_y"]))
    np.save(out_dir / "occupancy.npy", occ)
    print(f"Done. Outputs in {out_dir.resolve()}")


if __name__ == "__main__":
    main()
