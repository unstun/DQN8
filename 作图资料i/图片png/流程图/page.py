# -*- coding: utf-8 -*-
"""
用素材图生成“论文风流程图”（中文、简洁、适合论文排版）。
流程：
1) 实验设备
2) 点云地图建立
3) 可通行性分析与栅格地图生成
4) 路径规划算法设计
5) 路径规划结果
"""

import os
from typing import Optional, Dict

import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")  # ✅ 无GUI后端：只保存，不弹窗
import matplotlib.pyplot as plt
from matplotlib import rcParams, font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


# =========================
# 0) 全局可调样式（论文风：简洁、可打印、尽量“矢量线稿”感）
# =========================
PANEL_TITLE_FS = 13.5
LABEL_FS = 10.5

TEXT_COLOR = "#111827"        # 深灰黑
MUTED_TEXT_COLOR = "#374151"  # 次级文字
ARROW_COLOR = "#374151"       # 箭头灰
PANEL_EDGE_COLOR = "#4b5563"  # 轮廓线（避免纯黑太“硬”）
PANEL_BG_COLOR = "#ffffff"
IMG_FRAME_COLOR = "#9ca3af"


# =========================
# 1) 中文字体设置（避免 Glyph missing）
# =========================
def setup_chinese_font(font_file: Optional[str] = None) -> Optional[str]:
    """
    font_file: 可选，传入字体文件路径（.ttf/.otf），最稳（不依赖系统已安装字体）
    """
    if font_file and os.path.exists(font_file):
        font_manager.fontManager.addfont(font_file)
        name = font_manager.FontProperties(fname=font_file).get_name()
        rcParams["font.family"] = "sans-serif"
        rcParams["font.sans-serif"] = [name]
        rcParams["axes.unicode_minus"] = False
        print(f"[Font] 使用字体文件: {name}  ({font_file})")
        return name

    preferred = [
        "Microsoft YaHei", "SimHei", "SimSun", "NSimSun",    # Windows
        "PingFang SC", "Heiti SC",                           # macOS
        "Noto Sans CJK SC", "Noto Sans CJK JP",              # Noto
        "Source Han Sans SC", "Source Han Sans CN",          # 思源
        "WenQuanYi Micro Hei", "AR PL UMing CN", "AR PL SungtiL GB",  # Linux
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in preferred:
        if name in available:
            rcParams["font.family"] = "sans-serif"
            rcParams["font.sans-serif"] = [name]
            rcParams["axes.unicode_minus"] = False
            print(f"[Font] 使用系统字体: {name}")
            return name

    print("[Font] 未找到可用中文字体：建议安装 Microsoft YaHei/SimHei，或下载思源黑体/ NotoSansSC 并传入 font_file。")
    return None


# ✅ 如果你想“100%不翻车”，把字体文件放脚本同目录然后启用：
# setup_chinese_font("./SourceHanSansSC-Regular.otf")
setup_chinese_font()


# =========================
# 2) 素材图路径（改成你自己的）
# =========================
paths: Dict[str, str] = {
    "equipment": "设备图.jpg",
    "pointcloud_map": "点云图.jpg",
    "trav_pointcloud": "可通行分割后的点云.jpg",
    "grid_map": "格栅地图.jpg",
    "planning_algo": "强化学习路径规划的设计.jpg",
    "planning_result": "路径规划结果.png",
}


# =========================
# 3) 工具函数：读图 / 裁剪 / 中文按字符换行（避免太长溢出）
# =========================
def load_img(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到图片：{path}")
    return np.asarray(Image.open(path).convert("RGB"))


def center_crop_to_aspect(
    img: np.ndarray,
    aspect_w_over_h: float,
    *,
    align_x: float = 0.5,
    align_y: float = 0.5,
) -> np.ndarray:
    """
    裁剪到指定宽高比。
    align_x/align_y: 0~1，决定裁剪偏移（0=靠左/下，0.5=居中，1=靠右/上）。
    """
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        return img

    align_x = min(1.0, max(0.0, float(align_x)))
    align_y = min(1.0, max(0.0, float(align_y)))

    cur = w / h
    if cur > aspect_w_over_h:
        new_w = int(round(h * aspect_w_over_h))
        new_w = max(1, min(w, new_w))
        x0 = int(round((w - new_w) * align_x))
        x0 = max(0, min(w - new_w, x0))
        return img[:, x0 : x0 + new_w]

    new_h = int(round(w / aspect_w_over_h))
    new_h = max(1, min(h, new_h))
    y0 = int(round((h - new_h) * align_y))
    y0 = max(0, min(h - new_h, y0))
    return img[y0 : y0 + new_h, :]


def wrap_zh(text: str, max_chars_per_line: int = 16) -> str:
    """按中文字符数强制换行，防止注释条文字溢出（更稳定）。"""
    if not text:
        return text
    s = text.strip()
    lines = [s[i:i + max_chars_per_line] for i in range(0, len(s), max_chars_per_line)]
    return "\n".join(lines)


def files_identical(path_a: str, path_b: str, chunk_size: int = 1024 * 1024) -> bool:
    """比较两文件是否内容完全一致（用于判断素材是否误放/重复）。"""
    try:
        if os.path.abspath(path_a) == os.path.abspath(path_b):
            return True
        if os.path.getsize(path_a) != os.path.getsize(path_b):
            return False
        with open(path_a, "rb") as fa, open(path_b, "rb") as fb:
            while True:
                ba = fa.read(chunk_size)
                bb = fb.read(chunk_size)
                if not ba and not bb:
                    return True
                if ba != bb:
                    return False
    except OSError:
        return False


def load_assets():
    img_equipment = load_img(paths["equipment"])
    img_pc_map = load_img(paths["pointcloud_map"])
    img_trav = load_img(paths["trav_pointcloud"])
    img_grid = load_img(paths["grid_map"])
    img_result = load_img(paths["planning_result"])

    algo_path = paths.get("planning_algo", "")
    use_algo_img = (
        bool(algo_path)
        and os.path.exists(algo_path)
        and not files_identical(algo_path, paths["grid_map"])
    )
    img_algo = load_img(algo_path) if use_algo_img else None

    return {
        "equipment": img_equipment,
        "pointcloud_map": img_pc_map,
        "trav_pointcloud": img_trav,
        "grid_map": img_grid,
        "planning_algo": img_algo,
        "planning_algo_is_image": use_algo_img,
        "planning_result": img_result,
    }


def draw_planning_schematic(ax):
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    def rbox(x, y, w, h, text, *, edge, face, fs=11, weight="regular", color=TEXT_COLOR):
        patch = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="square,pad=0.02",
            linewidth=1.0,
            edgecolor=edge,
            facecolor=face,
            transform=ax.transAxes,
            clip_on=False,
        )
        ax.add_patch(patch)
        ax.text(
            x + w / 2, y + h / 2,
            text,
            ha="center", va="center",
            fontsize=fs,
            fontweight=weight,
            color=color,
            transform=ax.transAxes,
        )

    def arrow(p0, p1):
        ax.add_patch(
            FancyArrowPatch(
                p0, p1,
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=1.4,
                color=ARROW_COLOR,
                transform=ax.transAxes,
                clip_on=False,
            )
        )

    light = "#ffffff"
    planner_bg = "#f3f4f6"

    # Inputs
    rbox(0.05, 0.64, 0.25, 0.18, "栅格/代价地图", edge=PANEL_EDGE_COLOR, face=light, fs=11)
    rbox(0.05, 0.40, 0.25, 0.18, "起点/终点", edge=PANEL_EDGE_COLOR, face=light, fs=11)
    rbox(0.05, 0.16, 0.25, 0.18, "运动学约束", edge=PANEL_EDGE_COLOR, face=light, fs=11)

    # Planner
    rbox(
        0.36, 0.22, 0.38, 0.58,
        "路径规划器\nAPF | Hybrid A*\nD-Hybrid A* | Informed RRT*",
        edge=PANEL_EDGE_COLOR,
        face=planner_bg,
        fs=11,
        weight="bold",
    )

    # Output
    rbox(0.78, 0.38, 0.17, 0.28, "可行路径\n平滑/优化", edge=PANEL_EDGE_COLOR, face=light, fs=11)

    # Connections
    arrow((0.30, 0.73), (0.36, 0.73))
    arrow((0.30, 0.49), (0.36, 0.51))
    arrow((0.30, 0.25), (0.36, 0.35))
    arrow((0.74, 0.51), (0.78, 0.52))


# =========================
# 4) 画“论文风流程框”（无阴影/少装饰：更像期刊里的线框流程图）
# =========================
def add_panel(fig, rect, step_no: int, title: str, blocks):
    ax = fig.add_axes(rect)
    ax.set_axis_off()

    panel = FancyBboxPatch(
        (0, 0), 1, 1,
        boxstyle="square,pad=0.010",
        linewidth=1.2,
        edgecolor=PANEL_EDGE_COLOR,
        facecolor=PANEL_BG_COLOR,
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.add_patch(panel)

    ax.text(
        0.03, 0.96, f"{step_no}. {title}",
        ha="left", va="top",
        fontsize=PANEL_TITLE_FS, fontweight="bold",
        color=TEXT_COLOR,
        transform=ax.transAxes,
    )
    ax.plot(
        [0.02, 0.98], [0.86, 0.86],
        color=PANEL_EDGE_COLOR,
        linewidth=1.0,
        transform=ax.transAxes,
    )

    for b in blocks:
        x, y, w, h = b["pos"]
        iax = ax.inset_axes([x, y, w, h])
        iax.set_axis_off()

        img = b.get("img")
        if img is not None:
            crop_aspect = b.get("aspect", max(1e-6, w / max(1e-6, h)))
            align_x = float(b.get("align_x", 0.5))
            align_y = float(b.get("align_y", 0.5))
            img = center_crop_to_aspect(img, crop_aspect, align_x=align_x, align_y=align_y)
            iax.imshow(img, interpolation="bilinear")
        else:
            draw_fn = b.get("draw")
            if callable(draw_fn):
                draw_fn(iax)

        # 图片边框（线条要“克制”）
        frame = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="square,pad=0.003",
            linewidth=0.9,
            edgecolor=IMG_FRAME_COLOR,
            facecolor="none",
            transform=ax.transAxes,
            clip_on=False,
        )
        ax.add_patch(frame)

        label = b.get("label", "")
        if label:
            iax.text(
                0.01, 0.99, label,
                ha="left", va="top",
                fontsize=LABEL_FS,
                color=TEXT_COLOR,
                transform=iax.transAxes,
                bbox=dict(
                    boxstyle="square,pad=0.15",
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.9,
                ),
            )

    return ax


# =========================
# 5) 主流程：排版 + 箭头
# =========================
def main(output_path: str = "workflow_overview_cn_sci_v2.png"):
    assets = load_assets()
    img_equipment = assets["equipment"]
    img_pc_map = assets["pointcloud_map"]
    img_trav = assets["trav_pointcloud"]
    img_grid = assets["grid_map"]
    img_algo = assets["planning_algo"]
    img_result = assets["planning_result"]
    use_algo_img = assets["planning_algo_is_image"]

    fig = plt.figure(figsize=(16, 9), dpi=300)
    fig.patch.set_facecolor("white")

    M, G = 0.03, 0.03
    top_y0, top_y1 = 0.64, 0.95
    bot_y0, bot_y1 = 0.06, 0.62

    top_h = top_y1 - top_y0
    bot_h = bot_y1 - bot_y0
    top_w = (1 - 2*M - 2*G) / 3
    bot_total_w = (1 - 2*M - 1*G)
    bot_w4 = bot_total_w * 0.38
    bot_w5 = bot_total_w - bot_w4

    p1 = [M + 0*(top_w+G), top_y0, top_w, top_h]
    p2 = [M + 1*(top_w+G), top_y0, top_w, top_h]
    p3 = [M + 2*(top_w+G), top_y0, top_w, top_h]
    p4 = [M, bot_y0, bot_w4, bot_h]
    p5 = [M + bot_w4 + G, bot_y0, bot_w5, bot_h]

    add_panel(fig, p1, 1, "实验设备", [
        dict(img=img_equipment, pos=[0.07, 0.07, 0.86, 0.75]),
    ])

    add_panel(fig, p2, 2, "点云地图建立", [
        dict(img=img_pc_map, pos=[0.07, 0.09, 0.86, 0.72]),
    ])

    add_panel(fig, p3, 3, "可通行性分析与栅格地图生成", [
        dict(img=img_grid, pos=[0.07, 0.09, 0.86, 0.72], label="(a)"),
        dict(img=img_trav, pos=[0.10, 0.12, 0.34, 0.20], label="(b)"),
    ])

    algo_block = (
        dict(img=img_algo, pos=[0.07, 0.07, 0.86, 0.75])
        if use_algo_img
        else dict(draw=draw_planning_schematic, pos=[0.07, 0.07, 0.86, 0.75])
    )
    add_panel(fig, p4, 4, "路径规划算法设计", [algo_block])

    add_panel(fig, p5, 5, "路径规划结果", [
        dict(img=img_result, pos=[0.03, 0.12, 0.94, 0.55]),
    ])

    # --------- 全局箭头层
    overlay = fig.add_axes([0, 0, 1, 1])
    overlay.set_axis_off()
    overlay.set_xlim(0, 1)
    overlay.set_ylim(0, 1)

    def right_mid(rect):
        x, y, w, h = rect
        return x + w, y + h*0.52

    def left_mid(rect):
        x, y, w, h = rect
        return x, y + h*0.52

    def bottom_mid(rect):
        x, y, w, h = rect
        return x + w*0.50, y

    def top_mid(rect, dx=0.0):
        x, y, w, h = rect
        return x + w*(0.50 + dx), y + h

    arrow_kw = dict(
        arrowstyle="-|>",
        mutation_scale=16,
        linewidth=1.8,
        color=ARROW_COLOR,
    )

    overlay.add_patch(FancyArrowPatch(
        (right_mid(p1)[0] + 0.006, right_mid(p1)[1]),
        (left_mid(p2)[0]  - 0.006, left_mid(p2)[1]),
        **arrow_kw
    ))
    overlay.add_patch(FancyArrowPatch(
        (right_mid(p2)[0] + 0.006, right_mid(p2)[1]),
        (left_mid(p3)[0]  - 0.006, left_mid(p3)[1]),
        **arrow_kw
    ))

    # 3 -> 4：折线路径，避免跨越内容区
    start = (bottom_mid(p3)[0], bottom_mid(p3)[1] - 0.008)
    end = (top_mid(p4, dx=0.22)[0], top_mid(p4, dx=0.22)[1] + 0.006)
    overlay.add_patch(FancyArrowPatch(
        start, end,
        connectionstyle="angle3,angleA=-90,angleB=180",
        **arrow_kw,
    ))

    overlay.add_patch(FancyArrowPatch(
        (right_mid(p4)[0] + 0.006, right_mid(p4)[1]),
        (left_mid(p5)[0]  - 0.006, left_mid(p5)[1]),
        **arrow_kw
    ))

    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.10)
    plt.close(fig)
    print(f"[OK] 已保存：{output_path}")


def main_horizontal(output_path: str = "workflow_overview_cn_sci_v3.png"):
    assets = load_assets()
    img_equipment = assets["equipment"]
    img_pc_map = assets["pointcloud_map"]
    img_trav = assets["trav_pointcloud"]
    img_grid = assets["grid_map"]
    img_algo = assets["planning_algo"]
    img_result = assets["planning_result"]
    use_algo_img = assets["planning_algo_is_image"]

    fig = plt.figure(figsize=(18, 4.8), dpi=300)
    fig.patch.set_facecolor("white")

    M, G = 0.02, 0.015
    y0, y1 = 0.14, 0.92
    h = y1 - y0

    weights = [1.00, 1.05, 1.15, 1.05, 1.55]
    usable_w = 1 - 2 * M - (len(weights) - 1) * G
    unit = usable_w / sum(weights)
    widths = [w * unit for w in weights]

    rects = []
    x = M
    for w in widths:
        rects.append([x, y0, w, h])
        x += w + G

    p1, p2, p3, p4, p5 = rects

    add_panel(fig, p1, 1, "实验设备", [
        dict(img=img_equipment, pos=[0.13, 0.06, 0.74, 0.74]),
    ])

    add_panel(fig, p2, 2, "点云地图建立", [
        dict(img=img_pc_map, pos=[0.04, 0.14, 0.92, 0.58]),
    ])

    add_panel(fig, p3, 3, "可通行性分析与栅格地图生成", [
        dict(img=img_grid, pos=[0.04, 0.10, 0.92, 0.70], label="(a)"),
        dict(img=img_trav, pos=[0.06, 0.13, 0.34, 0.20], label="(b)"),
    ])

    algo_block = (
        dict(img=img_algo, pos=[0.06, 0.10, 0.88, 0.70])
        if use_algo_img
        else dict(draw=draw_planning_schematic, pos=[0.06, 0.10, 0.88, 0.70])
    )
    add_panel(fig, p4, 4, "路径规划算法设计", [algo_block])

    add_panel(fig, p5, 5, "路径规划结果", [
        dict(img=img_result, pos=[0.04, 0.18, 0.92, 0.52]),
    ])

    overlay = fig.add_axes([0, 0, 1, 1])
    overlay.set_axis_off()
    overlay.set_xlim(0, 1)
    overlay.set_ylim(0, 1)

    def right_mid(rect):
        x0, y0_, w0, h0 = rect
        return x0 + w0, y0_ + h0 * 0.52

    def left_mid(rect):
        x0, y0_, w0, h0 = rect
        return x0, y0_ + h0 * 0.52

    arrow_kw = dict(
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=1.4,
        color=ARROW_COLOR,
    )

    for a, b in [(p1, p2), (p2, p3), (p3, p4), (p4, p5)]:
        overlay.add_patch(FancyArrowPatch(
            (right_mid(a)[0] + 0.004, right_mid(a)[1]),
            (left_mid(b)[0] - 0.004, left_mid(b)[1]),
            **arrow_kw,
        ))

    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"[OK] 已保存：{output_path}")


if __name__ == "__main__":
    main("workflow_overview_cn_sci_v2.png")
    main("workflow_overview_cn_sci_v2.pdf")
    main_horizontal("workflow_overview_cn_sci_v3.png")
    main_horizontal("workflow_overview_cn_sci_v3.pdf")
