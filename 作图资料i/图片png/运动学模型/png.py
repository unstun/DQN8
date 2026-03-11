from typing import Optional, Tuple
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np


# =======================
# 统一风格参数（所有线都用同一个磅值）
# =======================
LW = 2.4                 # ✅ 全局统一线宽
ARROW_SCALE = 18         # ✅ 统一箭头头部大小
FONT = 14                # ✅ 统一字号（你也可以调大/调小）
LABEL_PAD = 0.55         # ✅ 文本候选偏移的基准尺度（数据单位）


@dataclass
class VehicleGeometry:
    wheelbase: float = 4.0      # L
    track_width: float = 3.0
    wheel_length: float = 1.2
    wheel_width: float = 0.5


def rotate_points(x: np.ndarray, y: np.ndarray, angle_rad: float) -> Tuple[np.ndarray, np.ndarray]:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return x * c - y * s, x * s + y * c


def wheel_polygon(center, heading_rad: float, steer_rad: float, geom: VehicleGeometry):
    half_l, half_w = geom.wheel_length / 2, geom.wheel_width / 2
    base_x = np.array([-half_l, half_l, half_l, -half_l, -half_l])
    base_y = np.array([-half_w, -half_w, half_w, half_w, -half_w])
    rx, ry = rotate_points(base_x, base_y, heading_rad + steer_rad)
    x = center[0] + rx
    y = center[1] + ry
    return x, y


def draw_wheel(
    ax,
    center,
    heading_rad: float,
    steer_rad: float,
    geom: VehicleGeometry,
    color: str = "black",
    linestyle: str = "-",
):
    x, y = wheel_polygon(center, heading_rad, steer_rad, geom)
    ax.plot(x, y, color=color, lw=LW, ls=linestyle, zorder=3)
    return np.c_[x, y]  # 返回采样点用于“避障”


def draw_world_frame(
    ax,
    origin=(-0.5, -0.5),
    x_len=4.2,
    y_len=4.2,
    color="black",
):
    ox, oy = origin

    # 轴箭头（线宽统一）
    ax.annotate(
        "", xy=(ox + x_len, oy), xytext=(ox, oy),
        arrowprops=dict(arrowstyle="->", lw=LW, color=color, mutation_scale=ARROW_SCALE),
        zorder=2
    )
    ax.annotate(
        "", xy=(ox, oy + y_len), xytext=(ox, oy),
        arrowprops=dict(arrowstyle="->", lw=LW, color=color, mutation_scale=ARROW_SCALE),
        zorder=2
    )

    # ✅ 原点：画实心点，避免“空”的感觉
    ax.plot([ox], [oy], marker="o", markersize=6, color=color, zorder=4)

    # ✅ 坐标轴标签：改为 X / Y（不要 x_g / y_g）
    ax.text(ox + x_len + 0.18, oy - 0.12, r"$X$", fontsize=FONT, color=color)
    ax.text(ox - 0.18, oy + y_len + 0.18, r"$Y$", fontsize=FONT, color=color)


def sample_segment(p0, p1, n=25):
    p0 = np.array(p0, dtype=float)
    p1 = np.array(p1, dtype=float)
    t = np.linspace(0.0, 1.0, n)[:, None]
    return (1 - t) * p0 + t * p1


def arc_points(center, radius, ang_start, ang_end, n=80):
    c = np.array(center, dtype=float)
    ang = np.linspace(ang_start, ang_end, n)
    xs = c[0] + radius * np.cos(ang)
    ys = c[1] + radius * np.sin(ang)
    return np.c_[xs, ys]


def draw_arc_with_arrow(ax, center, radius, ang_start, ang_end, color="black", linestyle="-"):
    pts = arc_points(center, radius, ang_start, ang_end, n=90)
    ax.plot(pts[:, 0], pts[:, 1], color=color, lw=LW, ls=linestyle, zorder=4)

    # 末端箭头（线宽统一）
    ax.annotate(
        "",
        xy=(pts[-1, 0], pts[-1, 1]),
        xytext=(pts[-3, 0], pts[-3, 1]),
        arrowprops=dict(arrowstyle="-|>", lw=LW, color=color, mutation_scale=ARROW_SCALE),
        zorder=5,
    )
    return pts


def best_label_position(anchor, avoid_pts: np.ndarray, candidates: np.ndarray, xlim, ylim):
    """
    在多个候选位置中选一个：与 avoid_pts 的“最小距离”最大（尽量不撞线/轮子）
    """
    ax0, ay0 = anchor
    best = None
    best_score = -1e18

    for dx, dy in candidates:
        x = ax0 + dx
        y = ay0 + dy

        # 视野外强惩罚
        if not (xlim[0] + 0.1 <= x <= xlim[1] - 0.1 and ylim[0] + 0.1 <= y <= ylim[1] - 0.1):
            score = -1e9
        else:
            d = avoid_pts - np.array([x, y])[None, :]
            dist2 = d[:, 0] ** 2 + d[:, 1] ** 2
            score = float(np.min(dist2))  # 最小距离越大越好

        if score > best_score:
            best_score = score
            best = (x, y)

    return best


def place_text_smart(ax, text, anchor, avoid_pts, scale=LABEL_PAD, fontsize=FONT):
    # 8方向 + 斜向 + 远一点的备选（更稳）
    base = scale
    candidates = np.array([
        [ base, 0.0], [-base, 0.0], [0.0,  base], [0.0, -base],
        [ base,  base], [ base, -base], [-base,  base], [-base, -base],
        [1.6*base, 0.0], [-1.6*base, 0.0], [0.0, 1.6*base], [0.0, -1.6*base],
        [1.2*base, 0.6*base], [1.2*base, -0.6*base], [-1.2*base, 0.6*base], [-1.2*base, -0.6*base],
    ], dtype=float)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x, y = best_label_position(anchor, avoid_pts, candidates, xlim, ylim)
    ax.text(x, y, text, fontsize=fontsize, color="black", zorder=10)


def draw_vehicle_model(
    rear=(2.0, 2.0),
    heading_deg: float = 35,
    steering_deg: float = -20,
    geometry: Optional[VehicleGeometry] = None,
    save_path: str = "vehicle_kinematic_model.png",
    dpi: int = 300,
    show: bool = True,
):
    geom = geometry or VehicleGeometry()
    rear_x, rear_y = rear
    theta = np.deg2rad(heading_deg)
    delta = np.deg2rad(steering_deg)

    fig, ax = plt.subplots(figsize=(7.0, 6.0), dpi=180)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlim(-1, 10)
    ax.set_ylim(-1, 10)

    # 关键点：后轴中心、前轴中心
    front_x = rear_x + geom.wheelbase * np.cos(theta)
    front_y = rear_y + geom.wheelbase * np.sin(theta)

    # 左右轮位置
    axle_dx = (geom.track_width / 2) * np.cos(theta + np.pi / 2)
    axle_dy = (geom.track_width / 2) * np.sin(theta + np.pi / 2)

    # -----------------------
    # 先准备“避障采样点集合”
    # -----------------------
    avoid = []

    # 车身中心线（实线）
    ax.plot([rear_x, front_x], [rear_y, front_y], color="black", lw=LW, ls="-", zorder=2)
    avoid.append(sample_segment((rear_x, rear_y), (front_x, front_y), n=35))

    # 前/后轴（实线）
    rear_L = (rear_x - axle_dx, rear_y - axle_dy)
    rear_R = (rear_x + axle_dx, rear_y + axle_dy)
    front_L = (front_x - axle_dx, front_y - axle_dy)
    front_R = (front_x + axle_dx, front_y + axle_dy)

    ax.plot([rear_L[0], rear_R[0]], [rear_L[1], rear_R[1]], color="black", lw=LW, ls="-", zorder=2)
    ax.plot([front_L[0], front_R[0]], [front_L[1], front_R[1]], color="black", lw=LW, ls="-", zorder=2)
    avoid.append(sample_segment(rear_L, rear_R, n=25))
    avoid.append(sample_segment(front_L, front_R, n=25))

    # 四个外轮（实线）
    avoid.append(draw_wheel(ax, rear_L, theta, 0.0, geom, linestyle="-"))
    avoid.append(draw_wheel(ax, rear_R, theta, 0.0, geom, linestyle="-"))
    avoid.append(draw_wheel(ax, front_L, theta, delta, geom, linestyle="-"))
    avoid.append(draw_wheel(ax, front_R, theta, delta, geom, linestyle="-"))

    # 自行车模型（虚线）
    ax.plot([rear_x, front_x], [rear_y, front_y], color="black", lw=LW, ls="--", zorder=3)
    avoid.append(sample_segment((rear_x, rear_y), (front_x, front_y), n=20))

    bicycle_geom = VehicleGeometry(
        wheelbase=geom.wheelbase,
        track_width=0.0,
        wheel_length=geom.wheel_length * 0.9,
        wheel_width=geom.wheel_width * 0.9,
    )
    avoid.append(draw_wheel(ax, (rear_x, rear_y), theta, 0.0, bicycle_geom, linestyle="--"))
    avoid.append(draw_wheel(ax, (front_x, front_y), theta, delta, bicycle_geom, linestyle="--"))

    # theta 参考线（也统一线宽）
    ref_len = 2.4
    ax.plot([rear_x, rear_x + ref_len], [rear_y, rear_y], color="black", lw=LW, ls="-", zorder=1)
    avoid.append(sample_segment((rear_x, rear_y), (rear_x + ref_len, rear_y), n=20))

    # 前轮方向箭头（线宽统一）
    ext_len = 2.0
    wheel_dir_x = front_x + ext_len * np.cos(theta + delta)
    wheel_dir_y = front_y + ext_len * np.sin(theta + delta)
    ax.annotate(
        "",
        xy=(wheel_dir_x, wheel_dir_y),
        xytext=(front_x, front_y),
        arrowprops=dict(arrowstyle="->", lw=LW, color="black", mutation_scale=ARROW_SCALE),
        zorder=5,
    )
    avoid.append(sample_segment((front_x, front_y), (wheel_dir_x, wheel_dir_y), n=25))

    # theta 弧线 + 箭头
    theta_r = 1.25
    theta_pts = draw_arc_with_arrow(ax, (rear_x, rear_y), theta_r, 0.0, theta, linestyle="-")
    avoid.append(theta_pts)

    # delta 弧线 + 箭头
    delta_r = 1.05
    delta_pts = draw_arc_with_arrow(ax, (front_x, front_y), delta_r, theta, theta + delta, linestyle="-")
    avoid.append(delta_pts)

    # 轴距 L 尺寸线（统一线宽）
    n_x, n_y = np.cos(theta + np.pi / 2), np.sin(theta + np.pi / 2)
    off = 0.85  # 尺寸线离车身更远，减少碰撞
    p0 = (rear_x + off * n_x, rear_y + off * n_y)
    p1 = (front_x + off * n_x, front_y + off * n_y)

    ax.annotate(
        "",
        xy=p1, xytext=p0,
        arrowprops=dict(arrowstyle="<->", lw=LW, color="black", mutation_scale=ARROW_SCALE),
        zorder=5,
    )
    avoid.append(sample_segment(p0, p1, n=25))

    # 合并避障点
    avoid_pts = np.vstack(avoid)

    # -----------------------
    # ✅ 智能放置数学符号（避免与线段/轮子碰撞）
    # -----------------------
    # theta：以弧线中点方向为 anchor
    mid_theta = 0.5 * (0.0 + theta)
    theta_anchor = (rear_x + (theta_r * 1.15) * np.cos(mid_theta),
                    rear_y + (theta_r * 1.15) * np.sin(mid_theta))
    place_text_smart(ax, r"$\theta$", theta_anchor, avoid_pts, scale=LABEL_PAD, fontsize=FONT)

    # delta
    mid_delta = 0.5 * (theta + theta + delta)
    delta_anchor = (front_x + (delta_r * 1.20) * np.cos(mid_delta),
                    front_y + (delta_r * 1.20) * np.sin(mid_delta))
    place_text_smart(ax, r"$\delta$", delta_anchor, avoid_pts, scale=LABEL_PAD, fontsize=FONT)

    # L：以尺寸线中点为 anchor
    mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
    place_text_smart(ax, r"$L$", (mx, my), avoid_pts, scale=LABEL_PAD, fontsize=FONT)

    # -----------------------
    # ✅ 大地坐标系（更长、更粗、有原点、X/Y）
    # -----------------------
    draw_world_frame(ax, origin=(-0.6, -0.6), x_len=4.6, y_len=4.6)

    # 清理画面
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ["top", "right", "left", "bottom"]:
        ax.spines[s].set_visible(False)

    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)

    if show:
        plt.show()

    return fig, ax


if __name__ == "__main__":
    draw_vehicle_model()
