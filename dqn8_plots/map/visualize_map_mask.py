"""3D visualisation of occupancy grid + point cloud overlay.

Adapted from dqn4作图/map/visualize_map_mask.py with parameterised input paths.
Run: python visualize_map_mask.py --map-yaml grid_out/map.yaml --las-path scan.las --out-dir viz_out/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    import laspy
except ImportError:
    laspy = None  # type: ignore[assignment]

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    HAS_MPL = True
except Exception:
    HAS_MPL = False

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False

MAX_PLOT_POINTS = 200_000


def load_pgm(pgm_path: Path) -> np.ndarray:
    with open(pgm_path, "rb") as f:
        header = b""
        while True:
            line = f.readline()
            if line.startswith(b"#"):
                continue
            header += line
            parts = header.split()
            if len(parts) >= 4:
                break
        w, h, maxval = int(parts[1]), int(parts[2]), int(parts[3])
        data = np.frombuffer(f.read(w * h), dtype=np.uint8).reshape(h, w)
    return np.flipud(data)


def load_ros_yaml(yaml_path: Path) -> dict:
    meta: dict = {}
    for line in yaml_path.read_text(encoding="utf-8").splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    return meta


def classify_points(
    pgm: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    resolution: float,
    origin_x: float,
    origin_y: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Classify points into black (obstacle), white (free), and out-of-bounds masks."""
    ny, nx = pgm.shape
    ix = ((x - origin_x) / resolution).astype(int)
    iy = ((y - origin_y) / resolution).astype(int)
    in_bounds = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    black = np.zeros(len(x), dtype=bool)
    white = np.zeros(len(x), dtype=bool)
    black[in_bounds] = pgm[iy[in_bounds], ix[in_bounds]] < 50
    white[in_bounds] = pgm[iy[in_bounds], ix[in_bounds]] > 200
    return black, white, ~in_bounds


def plot_3d_scatter(
    las_path: Path,
    pgm: np.ndarray,
    resolution: float,
    origin_x: float,
    origin_y: float,
    out_dir: Path,
    max_points: int = MAX_PLOT_POINTS,
) -> None:
    if laspy is None:
        raise ImportError("laspy required")
    las = laspy.read(str(las_path))
    x = np.asarray(las.x, dtype=np.float64)
    y = np.asarray(las.y, dtype=np.float64)
    z = np.asarray(las.z, dtype=np.float64)

    black, white, oob = classify_points(pgm, x, y, resolution, origin_x, origin_y)

    # Downsample if too many points
    if len(x) > max_points:
        idx = np.random.default_rng(42).choice(len(x), max_points, replace=False)
        x, y, z, black, white, oob = x[idx], y[idx], z[idx], black[idx], white[idx], oob[idx]

    if HAS_MPL:
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection="3d")
        if black.any():
            ax.scatter(x[black], y[black], z[black], s=0.3, c="red", alpha=0.5, label="obstacle pts")
        if white.any():
            ax.scatter(x[white], y[white], z[white], s=0.1, c="green", alpha=0.2, label="free pts")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend()
        png_path = out_dir / "map_obstacles_3d.png"
        fig.savefig(str(png_path), dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {png_path.name}")

    if HAS_PLOTLY:
        traces = []
        if black.any():
            traces.append(go.Scatter3d(
                x=x[black], y=y[black], z=z[black],
                mode="markers", marker=dict(size=1, color="red", opacity=0.5),
                name="obstacle",
            ))
        if white.any():
            traces.append(go.Scatter3d(
                x=x[white], y=y[white], z=z[white],
                mode="markers", marker=dict(size=1, color="green", opacity=0.2),
                name="free",
            ))
        fig = go.Figure(data=traces)
        html_path = out_dir / "map_obstacles_3d.html"
        fig.write_html(str(html_path))
        print(f"  saved {html_path.name}")


def main() -> None:
    ap = argparse.ArgumentParser(description="3D visualisation of map + point cloud.")
    ap.add_argument("--map-yaml", type=str, required=True, help="ROS map YAML file.")
    ap.add_argument("--las-path", type=str, required=True, help="LAS/LAZ point cloud.")
    ap.add_argument("--out-dir", type=str, default="viz_out")
    ap.add_argument("--max-points", type=int, default=MAX_PLOT_POINTS)
    args = ap.parse_args()

    yaml_path = Path(args.map_yaml)
    meta = load_ros_yaml(yaml_path)
    pgm_path = yaml_path.parent / meta.get("image", "map.pgm")
    pgm = load_pgm(pgm_path)
    resolution = float(meta.get("resolution", 0.1))
    origin = meta.get("origin", "[0, 0, 0]")
    origin = origin.strip("[]").split(",")
    ox, oy = float(origin[0]), float(origin[1])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_3d_scatter(Path(args.las_path), pgm, resolution, ox, oy, out_dir, args.max_points)


if __name__ == "__main__":
    main()
