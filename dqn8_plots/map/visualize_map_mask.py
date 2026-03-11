"""3-D visualisation of occupancy grid + LiDAR point-cloud overlay.

Ported from dqn4作图/map/visualize_map_mask.py with full feature parity:
  - Z-rotation support (--rotate-z-deg or auto from --meta-txt)
  - Z-alignment (percentile-based vertical shift to map plane)
  - Occupancy-grid 3-D surface floor + classified point-cloud scatter
  - Plotly interactive HTML: raw point cloud + grid overlay
  - Matplotlib 3-D PNG (ortho / perspective)
  - Black-margin cropping for cleaner visualisation
  - White-cell high-point filtering (trees / structures above threshold)
  - Custom Z colour map  blue → green → yellow → red
  - Filtered obstacle LAS export

Usage examples:
  # Minimal (uses defaults for rotation / alignment / thresholds):
  python visualize_map_mask.py \\
      --map-yaml grid_out/map_a.yaml \\
      --las-path scans6-sor2.las \\
      --out-dir viz_out/

  # Full control:
  python visualize_map_mask.py \\
      --map-yaml grid_out/map_a.yaml \\
      --las-path scans6-sor2.las \\
      --raw-las-path ../scans6.laz \\
      --meta-txt grid_out/meta.txt \\
      --out-dir viz_out/ \\
      --max-points 200000 \\
      --white-cell-min-z 1.0 \\
      --z-align-percentile 5.0 \\
      --crop-pad 2
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Literal

import numpy as np

try:
    import laspy
except ImportError:
    laspy = None  # type: ignore[assignment]

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    HAS_MPL = True
except Exception:
    HAS_MPL = False

try:
    import plotly.graph_objects as go
    import plotly.io as pio

    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False

# ── defaults ──────────────────────────────────────────────────────────
MAX_PLOT_POINTS = 200_000
MAX_RAW_POINTS = 200_000
WHITE_CELL_MIN_Z = 1.0
MAP_PLANE_Z = 0.0
Z_ALIGN_PERCENTILE: float | None = 5.0
CROP_PAD_CELLS = 2
PLOTLY_JS: Literal["cdn", "embed"] = "cdn"

# Z colour maps
MPL_Z_CMAP = (
    LinearSegmentedColormap.from_list(
        "bgyr", ["#0000ff", "#00ff00", "#ffff00", "#ff0000"]
    )
    if HAS_MPL
    else None
)
PLOTLY_Z_COLORSCALE = [
    [0.0, "rgb(0,0,255)"],
    [0.33, "rgb(0,255,0)"],
    [0.66, "rgb(255,255,0)"],
    [1.0, "rgb(255,0,0)"],
]

# Plotly raw-HTML styling
RAW_HTML_MARKER_SIZE = 2.0
RAW_HTML_MARKER_OPACITY = 1.0
RAW_HTML_BG = "white"
RAW_HTML_HIDE_AXES = True

# Matplotlib PNG camera
PNG_VIEW_ELEV = 85
PNG_VIEW_AZIM = -90
PNG_PROJ_TYPE: Literal["persp", "ortho"] = "ortho"


# ── helpers ───────────────────────────────────────────────────────────
def parse_simple_yaml(yaml_path: Path) -> tuple[float, tuple[float, float], Path]:
    """Parse resolution, origin, and image path from a ROS map YAML."""
    data: dict[str, str] = {}
    for line in yaml_path.read_text(encoding="utf-8").splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            data[k.strip()] = v.strip()
    resolution = float(data["resolution"])
    origin = ast.literal_eval(data["origin"])
    image_path = yaml_path.parent / data["image"]
    return resolution, (float(origin[0]), float(origin[1])), image_path


def parse_meta(meta_path: Path) -> float:
    """Return rotation_deg from meta.txt; 0 if missing."""
    if not meta_path.exists():
        return 0.0
    for line in meta_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("rotation_z_deg:"):
            return float(line.split(":", 1)[1].strip())
    return 0.0


def read_pgm(path: Path) -> np.ndarray:
    """Minimal P5 (binary) PGM reader returning a uint8 image array."""
    with open(path, "rb") as f:
        if f.readline().strip() != b"P5":
            raise ValueError("Only P5 PGM supported.")

        def _next_token() -> bytes:
            token = b""
            while True:
                ch = f.read(1)
                if ch == b"":
                    break
                if ch.isspace():
                    if token:
                        break
                    continue
                if ch == b"#":
                    f.readline()
                    continue
                token += ch
            return token

        width = int(_next_token())
        height = int(_next_token())
        _maxval = int(_next_token())
        if _maxval > 255:
            raise ValueError("Only 8-bit PGM supported.")
        img = np.frombuffer(f.read(width * height), dtype=np.uint8)
        return img.reshape((height, width))


def _compute_crop_bbox(
    mask: np.ndarray, pad: int
) -> tuple[int, int, int, int] | None:
    """Return (x0, x1, y0, y1) bbox for True region in *mask*, with padding."""
    if mask.size == 0:
        return None
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return None
    h, w = mask.shape
    y0 = int(np.argmax(rows))
    y1 = int(h - 1 - np.argmax(rows[::-1]))
    x0 = int(np.argmax(cols))
    x1 = int(w - 1 - np.argmax(cols[::-1]))
    if pad > 0:
        x0 = max(0, x0 - pad)
        y0 = max(0, y0 - pad)
        x1 = min(w - 1, x1 + pad)
        y1 = min(h - 1, y1 + pad)
    return x0, x1, y0, y1


def _z_color_range(*arrays: np.ndarray) -> tuple[float, float]:
    z_vals = [a[:, 2] for a in arrays if a is not None and len(a) > 0]
    if not z_vals:
        return 0.0, 1.0
    z_all = np.concatenate(z_vals)
    return float(np.nanmin(z_all)), float(np.nanmax(z_all))


def _rotate_xy(
    x: np.ndarray, y: np.ndarray, theta: float, cx: float, cy: float
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate (x, y) around (cx, cy) by *theta* radians."""
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    dx, dy = x - cx, y - cy
    return cos_t * dx - sin_t * dy + cx, sin_t * dx + cos_t * dy + cy


def _downsample_idx(n: int, max_pts: int, seed: int = 42) -> np.ndarray:
    return np.random.default_rng(seed).choice(n, max_pts, replace=False)


# ── Plotly: raw point-cloud HTML ──────────────────────────────────────
def save_plotly_raw_html(
    html_path: Path,
    raw_plot: np.ndarray,
    x_lim: tuple[float, float],
    y_lim: tuple[float, float],
    title: str,
    z_color_range: tuple[float, float],
) -> None:
    if not HAS_PLOTLY:
        print("  [skip] plotly not available – raw HTML not saved")
        return

    html_path.parent.mkdir(parents=True, exist_ok=True)
    z_cmin, z_cmax = z_color_range
    x_min, x_max = x_lim
    y_min, y_max = y_lim

    if len(raw_plot) > 0:
        z_min = float(np.nanmin(raw_plot[:, 2]))
        z_max = float(np.nanmax(raw_plot[:, 2]))
        z_pad = 0.05 * max(1.0, z_max - z_min)
        z_min -= z_pad
        z_max += z_pad
    else:
        z_min, z_max = 0.0, 1.0

    if len(raw_plot) > 0:
        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=raw_plot[:, 0],
                    y=raw_plot[:, 1],
                    z=raw_plot[:, 2],
                    mode="markers",
                    name="All points",
                    marker=dict(
                        size=RAW_HTML_MARKER_SIZE,
                        color=raw_plot[:, 2],
                        coloraxis="coloraxis",
                        opacity=RAW_HTML_MARKER_OPACITY,
                        showscale=False,
                    ),
                )
            ]
        )
    else:
        fig = go.Figure()

    # 3-D colour bar aligned with point-cloud Z axis
    xr = float(x_max - x_min)
    yr = float(y_max - y_min)
    pad_x = 0.18 * xr if xr > 0 else 1.0
    bar_x = float(x_max + 0.06 * xr)
    bar_y = float(0.5 * (y_min + y_max))
    z_bar = np.linspace(float(z_cmin), float(z_cmax), num=128, dtype=np.float64)
    fig.add_trace(
        go.Scatter3d(
            x=np.full_like(z_bar, bar_x),
            y=np.full_like(z_bar, bar_y),
            z=z_bar,
            mode="markers",
            marker=dict(
                size=6,
                symbol="square",
                color=z_bar,
                coloraxis="coloraxis",
                opacity=1.0,
                showscale=False,
            ),
            hoverinfo="skip",
            showlegend=False,
            name="Z color scale",
        )
    )

    xr_full = float((x_max + pad_x) - x_min)
    maxr = max(xr_full, yr, float(max(z_max - z_min, 1e-6)))
    aspect = dict(
        x=xr_full / maxr,
        y=yr / maxr,
        z=float(max(z_max - z_min, 1e-6)) / maxr,
    )

    fig.update_layout(
        title=title,
        scene=dict(
            bgcolor=RAW_HTML_BG,
            xaxis=dict(
                title="X (m)",
                range=[x_min, x_max + pad_x],
                visible=not RAW_HTML_HIDE_AXES,
                showbackground=False,
                showgrid=False,
                zeroline=False,
            ),
            yaxis=dict(
                title="Y (m)",
                range=[y_min, y_max],
                visible=not RAW_HTML_HIDE_AXES,
                showbackground=False,
                showgrid=False,
                zeroline=False,
            ),
            zaxis=dict(
                title="Height (m)",
                range=[z_min, z_max],
                visible=not RAW_HTML_HIDE_AXES,
                showbackground=False,
                showgrid=False,
                zeroline=False,
            ),
            aspectmode="manual",
            aspectratio=aspect,
        ),
        coloraxis=dict(
            colorscale=PLOTLY_Z_COLORSCALE,
            cmin=z_cmin,
            cmax=z_cmax,
            showscale=False,
        ),
        paper_bgcolor=RAW_HTML_BG,
        font=dict(color="black"),
        margin=dict(l=0, r=0, b=0, t=70),
        showlegend=not RAW_HTML_HIDE_AXES,
        legend=dict(
            x=0.01,
            y=0.99,
            bgcolor="rgba(255,255,255,0.6)",
            font=dict(color="black"),
        ),
    )
    pio.write_html(fig, file=str(html_path), include_plotlyjs=PLOTLY_JS, auto_open=False)
    print(f"  saved {html_path.name}")


# ── Plotly: grid overlay HTML ─────────────────────────────────────────
def save_plotly_grid_html(
    html_path: Path,
    occ_grid: np.ndarray,
    black_keep_mask: np.ndarray | None,
    resolution: float,
    plane_z: float,
    obstacle_plot: np.ndarray,
    white_plot: np.ndarray,
    x_lim: tuple[float, float],
    y_lim: tuple[float, float],
    z_lim: tuple[float, float],
    title: str,
    z_color_range: tuple[float, float],
    white_cell_min_z: float = WHITE_CELL_MIN_Z,
) -> None:
    if not HAS_PLOTLY:
        print("  [skip] plotly not available – grid HTML not saved")
        return

    html_path.parent.mkdir(parents=True, exist_ok=True)
    z_cmin, z_cmax = z_color_range
    x_min, x_max = x_lim

    traces: list = []

    # ── occupancy grid mesh on the map plane ──
    h, w = occ_grid.shape
    cell = float(resolution)
    x_edges = x_min + np.arange(w + 1, dtype=np.float64) * cell
    y_min_g = y_lim[0]
    y_edges = y_min_g + np.arange(h + 1, dtype=np.float64) * cell
    Xv, Yv = np.meshgrid(x_edges, y_edges)
    Zv = np.full_like(Xv, float(plane_z), dtype=np.float64)

    rr = np.repeat(np.arange(h, dtype=np.int64), w)
    cc = np.tile(np.arange(w, dtype=np.int64), h)
    v00 = rr * (w + 1) + cc
    v01 = v00 + 1
    v10 = v00 + (w + 1)
    v11 = v10 + 1
    tri_i = np.concatenate([v00, v00])
    tri_j = np.concatenate([v01, v11])
    tri_k = np.concatenate([v11, v10])

    occ_flat = occ_grid.astype(np.int16).ravel()
    cell_intensity = np.full(h * w, 0.5, dtype=np.float32)
    cell_intensity[occ_flat == 0] = 0.0
    cell_intensity[occ_flat >= 250] = 1.0
    if black_keep_mask is not None:
        keep_flat = black_keep_mask.ravel()
        cell_intensity[(occ_flat == 0) & (~keep_flat)] = 0.25
    tri_intensity = np.concatenate([cell_intensity, cell_intensity])

    traces.append(
        go.Mesh3d(
            x=Xv.ravel(),
            y=Yv.ravel(),
            z=Zv.ravel(),
            i=tri_i,
            j=tri_j,
            k=tri_k,
            intensity=tri_intensity,
            intensitymode="cell",
            colorscale=[
                [0.0, "rgba(0,0,0,1.0)"],
                [0.249, "rgba(0,0,0,1.0)"],
                [0.25, "rgba(0,0,0,0.0)"],
                [0.5, "rgba(128,128,128,0.15)"],
                [1.0, "rgba(255,255,255,0.05)"],
            ],
            showscale=False,
            flatshading=True,
            lighting=dict(
                ambient=1.0, diffuse=0.0, specular=0.0, roughness=1.0, fresnel=0.0
            ),
            name="Occupancy grid",
            hoverinfo="skip",
            opacity=1.0,
        )
    )

    # ── obstacle points ──
    if len(obstacle_plot) > 0:
        traces.append(
            go.Scatter3d(
                x=obstacle_plot[:, 0],
                y=obstacle_plot[:, 1],
                z=obstacle_plot[:, 2],
                mode="markers",
                name="Masked points",
                marker=dict(
                    size=1.5,
                    color=obstacle_plot[:, 2],
                    coloraxis="coloraxis",
                    opacity=0.85,
                    showscale=False,
                ),
            )
        )

    # ── white-cell high points ──
    if len(white_plot) > 0:
        traces.append(
            go.Scatter3d(
                x=white_plot[:, 0],
                y=white_plot[:, 1],
                z=white_plot[:, 2],
                mode="markers",
                name=f"Free points (z >= {white_cell_min_z} m)",
                marker=dict(
                    size=1.5,
                    color=white_plot[:, 2],
                    coloraxis="coloraxis",
                    opacity=0.65,
                    showscale=False,
                ),
            )
        )

    # ── 3-D colour bar ──
    xr = float(x_max - x_min)
    yr = float(y_lim[1] - y_lim[0])
    pad_x = 0.18 * xr if xr > 0 else 1.0
    bar_x = float(x_max + 0.06 * xr)
    bar_y = float(0.5 * (y_lim[0] + y_lim[1]))
    z_bar = np.linspace(float(z_cmin), float(z_cmax), num=128, dtype=np.float64)
    traces.append(
        go.Scatter3d(
            x=np.full_like(z_bar, bar_x),
            y=np.full_like(z_bar, bar_y),
            z=z_bar,
            mode="markers",
            name="Z color scale",
            marker=dict(
                size=6,
                symbol="square",
                color=z_bar,
                coloraxis="coloraxis",
                opacity=1.0,
                showscale=False,
            ),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    z_min_l, z_max_l = z_lim
    zr = float(max(z_max_l - z_min_l, 1e-6))
    xr_full = float((x_max + pad_x) - x_min)
    maxr = max(xr_full, yr, zr)
    aspect = dict(x=xr_full / maxr, y=yr / maxr, z=zr / maxr)

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(
                title="X (m)",
                range=[x_min, x_max + pad_x],
                visible=False,
                showbackground=False,
                showgrid=False,
                zeroline=False,
            ),
            yaxis=dict(
                title="Y (m)",
                range=[y_lim[0], y_lim[1]],
                visible=False,
                showbackground=False,
                showgrid=False,
                zeroline=False,
            ),
            zaxis=dict(
                title="Height (m)",
                range=[min(float(plane_z), z_min_l), z_max_l],
                visible=False,
                showbackground=False,
                showgrid=False,
                zeroline=False,
            ),
            aspectmode="manual",
            aspectratio=aspect,
        ),
        coloraxis=dict(
            colorscale=PLOTLY_Z_COLORSCALE,
            cmin=z_cmin,
            cmax=z_cmax,
            showscale=False,
        ),
        margin=dict(l=0, r=0, b=0, t=70),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.6)"),
    )
    pio.write_html(fig, file=str(html_path), include_plotlyjs=PLOTLY_JS, auto_open=False)
    print(f"  saved {html_path.name}")


# ── Matplotlib: 3-D PNG ──────────────────────────────────────────────
def save_mpl_png(
    out_png: Path,
    occ_vis: np.ndarray,
    keep_black_cell_vis: np.ndarray,
    resolution: float,
    x_min: float,
    y_min: float,
    z_floor: float,
    z_ceiling: float,
    plot_points: np.ndarray,
    plot_labels: np.ndarray,
    n_obstacle: int,
    n_white_high: int,
    white_cell_min_z: float,
    rotation_deg: float,
    x_max: float,
    y_max: float,
) -> None:
    if not HAS_MPL:
        print("  [skip] matplotlib not available – PNG not saved")
        return

    fig = plt.figure(figsize=(11, 8))
    ax3d = fig.add_subplot(1, 1, 1, projection="3d")

    # Down-sample occupancy grid for faster surface rendering
    target_pixels = 600
    hv, wv = occ_vis.shape
    ds = max(1, int(np.ceil(max(hv, wv) / target_pixels)))
    occ_small = occ_vis[::ds, ::ds]

    xs = x_min + (np.arange(occ_small.shape[1]) + 0.5) * resolution * ds
    ys = y_min + (np.arange(occ_small.shape[0]) + 0.5) * resolution * ds
    Xs, Ys = np.meshgrid(xs, ys)
    Zs = np.full_like(Xs, z_floor, dtype=np.float64)

    colors = plt.cm.gray(occ_small.astype(np.float64) / 255.0)
    colors[..., 3] = 0.35
    keep_small = keep_black_cell_vis[::ds, ::ds]
    hide = (occ_small == 0) & (~keep_small)
    colors[hide, 3] = 0.0
    ax3d.plot_surface(
        Xs,
        Ys,
        Zs,
        rstride=1,
        cstride=1,
        facecolors=colors,
        linewidth=0,
        antialiased=False,
        shade=False,
    )

    if len(plot_points) > 0:
        sc = ax3d.scatter(
            plot_points[:, 0],
            plot_points[:, 1],
            plot_points[:, 2],
            s=0.8,
            c=plot_points[:, 2],
            cmap=MPL_Z_CMAP,
            alpha=0.85,
            linewidths=0,
            depthshade=False,
        )
        fig.colorbar(sc, ax=ax3d, shrink=0.6, pad=0.08, label="Z (m)")

    ax3d.set_xlim(x_min, x_max)
    ax3d.set_ylim(y_min, y_max)
    ax3d.set_zlim(z_floor, z_ceiling)
    ax3d.set_box_aspect((x_max - x_min, y_max - y_min, z_ceiling - z_floor))
    ax3d.set_xlabel("X (m)")
    ax3d.set_ylabel("Y (m)")
    ax3d.set_zlabel("Height (m)")
    ax3d.set_title(
        "Occupancy with obstacle + white-cell-high points\n"
        f"{n_obstacle:,} black-cell pts; {n_white_high:,} white-cell pts "
        f"(>= {white_cell_min_z} m)\n"
        f"rotation_z_deg={rotation_deg:.3f}"
    )
    if hasattr(ax3d, "set_proj_type"):
        ax3d.set_proj_type(PNG_PROJ_TYPE)
    ax3d.view_init(elev=PNG_VIEW_ELEV, azim=PNG_VIEW_AZIM)

    plt.tight_layout()
    fig.savefig(str(out_png), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_png.name}")


# ── main pipeline ─────────────────────────────────────────────────────
def run(
    map_yaml: Path,
    las_path: Path,
    out_dir: Path,
    raw_las_path: Path | None = None,
    meta_txt: Path | None = None,
    rotate_z_deg: float | None = None,
    max_points: int = MAX_PLOT_POINTS,
    max_raw_points: int = MAX_RAW_POINTS,
    white_cell_min_z: float = WHITE_CELL_MIN_Z,
    z_align_percentile: float | None = Z_ALIGN_PERCENTILE,
    crop_pad: int = CROP_PAD_CELLS,
    save_las: bool = True,
) -> None:
    if laspy is None:
        raise ImportError("laspy required: pip install laspy")

    # ── load map metadata ──
    resolution, origin_xy, pgm_path = parse_simple_yaml(map_yaml)
    origin_x, origin_y = origin_xy
    pgm_img = read_pgm(pgm_path)

    # ── rotation ──
    if rotate_z_deg is not None:
        rotation_deg = rotate_z_deg
    elif meta_txt is not None:
        rotation_deg = parse_meta(meta_txt)
    else:
        rotation_deg = 0.0
    theta = np.deg2rad(rotation_deg)
    use_rotation = abs(theta) > 1e-9

    # ── load scan ──
    las = laspy.read(str(las_path))
    x = np.asarray(las.x, dtype=np.float64)
    y = np.asarray(las.y, dtype=np.float64)
    z = np.asarray(las.z, dtype=np.float64)

    cx = 0.5 * (float(las.header.mins[0]) + float(las.header.maxs[0]))
    cy = 0.5 * (float(las.header.mins[1]) + float(las.header.maxs[1]))

    if use_rotation:
        x, y = _rotate_xy(x, y, theta, cx, cy)

    occ_grid = np.flipud(pgm_img)
    obstacle_mask = occ_grid == 0
    free_mask = occ_grid >= 250

    ix = np.floor((x - origin_x) / resolution).astype(np.int64)
    iy = np.floor((y - origin_y) / resolution).astype(np.int64)

    h, w = occ_grid.shape
    x_map_min, x_map_max = origin_x, origin_x + w * resolution
    y_map_min, y_map_max = origin_y, origin_y + h * resolution
    valid = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
    ixv, iyv = ix[valid], iy[valid]

    in_obstacle = np.zeros_like(valid)
    in_obstacle[valid] = obstacle_mask[iyv, ixv]
    in_free = np.zeros_like(valid)
    in_free[valid] = free_mask[iyv, ixv]
    white_high = in_free & (z >= white_cell_min_z)

    # ── Z alignment ──
    z_shift = 0.0
    z_align_note = ""
    if z_align_percentile is not None:
        if np.any(in_obstacle):
            z_ref = float(np.percentile(z[in_obstacle], z_align_percentile))
            z_shift = MAP_PLANE_Z - z_ref
            z_align_note = "obstacle points"
        elif np.any(valid):
            z_ref = float(np.percentile(z[valid], z_align_percentile))
            z_shift = MAP_PLANE_Z - z_ref
            z_align_note = "valid points"
        else:
            z_ref = float(np.percentile(z, z_align_percentile))
            z_shift = MAP_PLANE_Z - z_ref
            z_align_note = "all points"

    # ── extract classified point arrays ──
    obstacle_points = (
        np.column_stack((x[in_obstacle], y[in_obstacle], z[in_obstacle]))
        if np.any(in_obstacle)
        else np.empty((0, 3))
    )
    white_high_points = (
        np.column_stack((x[white_high], y[white_high], z[white_high]))
        if np.any(white_high)
        else np.empty((0, 3))
    )

    n_total = len(x)
    n_obstacle = len(obstacle_points)
    n_white_high = len(white_high_points)

    # ── crop visualisation extents ──
    plot_mask = in_obstacle | white_high
    plot_cell_mask = np.zeros((h, w), dtype=bool)
    if np.any(plot_mask):
        plot_cell_mask[iy[plot_mask], ix[plot_mask]] = True

    keep_mask = (occ_grid != 0) | plot_cell_mask
    bbox = _compute_crop_bbox(keep_mask, pad=crop_pad)
    if bbox is None:
        x0, x1, y0, y1 = 0, w - 1, 0, h - 1
    else:
        x0, x1, y0, y1 = bbox
    occ_vis = occ_grid[y0 : y1 + 1, x0 : x1 + 1]

    non_black_bbox = _compute_crop_bbox(occ_grid != 0, pad=0)
    if non_black_bbox is None:
        keep_black_cell_vis = np.ones_like(occ_vis, dtype=bool)
    else:
        nb_x0, nb_x1, nb_y0, nb_y1 = non_black_bbox
        ys_g = np.arange(y0, y1 + 1, dtype=np.int64)
        xs_g = np.arange(x0, x1 + 1, dtype=np.int64)
        keep_black_cell_vis = np.outer(
            (ys_g >= nb_y0) & (ys_g <= nb_y1),
            (xs_g >= nb_x0) & (xs_g <= nb_x1),
        )

    x_grid_min = origin_x + x0 * resolution
    x_grid_max = origin_x + (x1 + 1) * resolution
    y_grid_min = origin_y + y0 * resolution
    y_grid_max = origin_y + (y1 + 1) * resolution

    # ── apply Z shift ──
    out_dir.mkdir(parents=True, exist_ok=True)

    if save_las:
        filtered = laspy.LasData(las.header)
        filtered.points = las.points[in_obstacle]

    if abs(z_shift) > 1e-9:
        if n_obstacle:
            obstacle_points[:, 2] += z_shift
            if save_las:
                filtered.z = filtered.z + z_shift
        if n_white_high:
            white_high_points[:, 2] += z_shift

    if save_las:
        out_las = out_dir / "obstacles_filtered.las"
        filtered.write(str(out_las))
        print(f"  saved {out_las.name}")

    # ── combine & downsample for plotting ──
    parts, labels = [], []
    if n_obstacle:
        parts.append(obstacle_points)
        labels.append(np.zeros(n_obstacle, dtype=np.uint8))
    if n_white_high:
        parts.append(white_high_points)
        labels.append(np.ones(n_white_high, dtype=np.uint8))

    if parts:
        plot_points = np.vstack(parts)
        plot_labels = np.concatenate(labels)
        if max_points and len(plot_points) > max_points:
            idx = _downsample_idx(len(plot_points), max_points, seed=0)
            plot_points, plot_labels = plot_points[idx], plot_labels[idx]
    else:
        plot_points = np.empty((0, 3))
        plot_labels = np.empty((0,), dtype=np.uint8)

    obstacle_plot = plot_points[plot_labels == 0]
    white_plot = plot_points[plot_labels == 1]

    # ── z limits for vis ──
    if len(plot_points) > 0:
        z_max_pts = float(np.nanmax(plot_points[:, 2]))
        z_pad = 0.05 * max(1.0, z_max_pts - MAP_PLANE_Z)
        z_floor, z_ceiling = MAP_PLANE_Z, z_max_pts + z_pad
    else:
        z_floor, z_ceiling = MAP_PLANE_Z, MAP_PLANE_Z + 1.0

    # ── 1. Matplotlib PNG ──
    out_png = out_dir / "map_obstacles_3d.png"
    save_mpl_png(
        out_png,
        occ_vis,
        keep_black_cell_vis,
        resolution,
        x_grid_min,
        y_grid_min,
        z_floor,
        z_ceiling,
        plot_points,
        plot_labels,
        n_obstacle,
        n_white_high,
        white_cell_min_z,
        rotation_deg,
        x_grid_max,
        y_grid_max,
    )

    # ── 2. Plotly grid HTML ──
    raw_z_grid = np.asarray(z, dtype=np.float64) + float(z_shift)
    raw_plot_grid = np.column_stack(
        (np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64), raw_z_grid)
    )
    if max_raw_points and n_total > max_raw_points:
        idx = _downsample_idx(n_total, max_raw_points, seed=1)
        raw_plot_grid = raw_plot_grid[idx]

    z_cr_grid = _z_color_range(raw_plot_grid, obstacle_plot, white_plot)
    title_grid = (
        "Occupancy with masked + free-cell-high points<br>"
        f"{n_obstacle:,} black-cell pts; {n_white_high:,} white-cell pts "
        f"(>= {white_cell_min_z} m)<br>"
        f"rotation_z_deg={rotation_deg:.3f}"
    )
    save_plotly_grid_html(
        html_path=out_dir / "map_obstacles_3d.html",
        occ_grid=occ_vis,
        black_keep_mask=keep_black_cell_vis,
        resolution=resolution,
        plane_z=float(z_floor),
        obstacle_plot=obstacle_plot,
        white_plot=white_plot,
        x_lim=(x_grid_min, x_grid_max),
        y_lim=(y_grid_min, y_grid_max),
        z_lim=(z_floor, z_ceiling),
        title=title_grid,
        z_color_range=z_cr_grid,
        white_cell_min_z=white_cell_min_z,
    )

    # ── 3. Plotly raw HTML ──
    if raw_las_path is not None and raw_las_path.exists():
        raw_las = laspy.read(str(raw_las_path))
        raw_x = np.asarray(raw_las.x, dtype=np.float64)
        raw_y = np.asarray(raw_las.y, dtype=np.float64)
        raw_z = np.asarray(raw_las.z, dtype=np.float64)
        if use_rotation:
            raw_x, raw_y = _rotate_xy(raw_x, raw_y, theta, cx, cy)
        raw_z = raw_z + float(z_shift)
        n_raw = len(raw_x)
        if max_raw_points and n_raw > max_raw_points:
            idx = _downsample_idx(n_raw, max_raw_points, seed=1)
            raw_x, raw_y, raw_z = raw_x[idx], raw_y[idx], raw_z[idx]
        raw_plot = np.column_stack((raw_x, raw_y, raw_z))
    else:
        # Fall back: use the filtered scan as "raw"
        raw_plot = raw_plot_grid

    z_cr_raw = _z_color_range(raw_plot)
    stem = raw_las_path.stem if raw_las_path else las_path.stem
    title_raw = (
        f"{stem} point cloud (aligned)<br>"
        f"rotation_z_deg={rotation_deg:.3f}, z_shift={z_shift:.3f} m<br>"
        f"raw points shown: {len(raw_plot):,}"
    )
    save_plotly_raw_html(
        html_path=out_dir / "raw_pointcloud.html",
        raw_plot=raw_plot,
        x_lim=(x_map_min, x_map_max),
        y_lim=(y_map_min, y_map_max),
        title=title_raw,
        z_color_range=z_cr_raw,
    )

    # ── summary ──
    print(f"\nTotal points: {n_total:,}")
    print(f"In black cells (obstacle): {n_obstacle:,}")
    print(f"In white cells >= {white_cell_min_z} m: {n_white_high:,}")
    print(f"Plotted points (downsampled): {len(plot_points):,}")
    if z_align_percentile is None:
        print("Vertical shift: 0.000 m (disabled)")
    else:
        print(
            f"Vertical shift: {z_shift:.3f} m "
            f"(align p{z_align_percentile:g} of {z_align_note} to map plane)"
        )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="3-D visualisation of occupancy grid + LiDAR point cloud."
    )
    ap.add_argument(
        "--map-yaml", type=str, required=True, help="ROS map YAML file."
    )
    ap.add_argument(
        "--las-path",
        type=str,
        required=True,
        help="LAS/LAZ point cloud (used for masking & grid overlay).",
    )
    ap.add_argument(
        "--raw-las-path",
        type=str,
        default=None,
        help="Original (unfiltered) LAS/LAZ for raw HTML view.",
    )
    ap.add_argument(
        "--meta-txt",
        type=str,
        default=None,
        help="meta.txt with rotation_z_deg (auto-detected if omitted).",
    )
    ap.add_argument(
        "--rotate-z-deg",
        type=float,
        default=None,
        help="Explicit Z rotation (overrides --meta-txt).",
    )
    ap.add_argument("--out-dir", type=str, default="viz_out")
    ap.add_argument("--max-points", type=int, default=MAX_PLOT_POINTS)
    ap.add_argument("--max-raw-points", type=int, default=MAX_RAW_POINTS)
    ap.add_argument("--white-cell-min-z", type=float, default=WHITE_CELL_MIN_Z)
    ap.add_argument(
        "--z-align-percentile",
        type=float,
        default=Z_ALIGN_PERCENTILE,
        help="Percentile for Z alignment (None to disable).",
    )
    ap.add_argument("--crop-pad", type=int, default=CROP_PAD_CELLS)
    ap.add_argument(
        "--no-las-export",
        action="store_true",
        help="Skip filtered LAS export.",
    )
    args = ap.parse_args()

    run(
        map_yaml=Path(args.map_yaml),
        las_path=Path(args.las_path),
        out_dir=Path(args.out_dir),
        raw_las_path=Path(args.raw_las_path) if args.raw_las_path else None,
        meta_txt=Path(args.meta_txt) if args.meta_txt else None,
        rotate_z_deg=args.rotate_z_deg,
        max_points=args.max_points,
        max_raw_points=args.max_raw_points,
        white_cell_min_z=args.white_cell_min_z,
        z_align_percentile=args.z_align_percentile,
        crop_pad=args.crop_pad,
        save_las=not args.no_las_export,
    )


if __name__ == "__main__":
    main()
