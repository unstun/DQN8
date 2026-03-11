import os, json, math
from PIL import Image, ImageDraw, ImageFont

# ============== 你可调的主参数（都以“最终输出图像像素”为单位） ==============
RENDER_SCALE = 2          # 2 = 更平滑（内部2倍画再缩小）；1 = 更直接

# ✅ 圆圈/字母：加大
BADGE_RADIUS = 40         # 小圈半径（原来 18）
BADGE_FONT_SIZE = 52      # 数字/字母字号（原来 22）
BADGE_TO_THUMB_GAP = 60   # 圆心到缩略图左边距离（圆大了要更远，避免贴住）
ARROW_PAD = 10            # 箭头停在圆外余量（可选调大一点）

GAP_MAIN_TO_THUMBS = 8    # 主图到右侧缩略图区的间隙（越小越“无缝”）
ROW_GAP = 14              # 缩略图行间距（越小越紧凑）

# ============== 图片路径 ==============
paths = {
    "main": "大车整体图.png",
    "monitor": "显示器.png",
    "lidar": "雷达mid360.png",
    "ipc": "工控机.png",
    "power": "电源.png",
    "chassis": "无人小车底盘.png",
}

# 组件顺序就是你点选锚点的顺序：1~5
components = [
    {"label": "a", "img": paths["monitor"]},
    {"label": "b", "img": paths["lidar"]},
    {"label": "c", "img": paths["ipc"]},
    {"label": "d", "img": paths["power"]},
    {"label": "e", "img": paths["chassis"]},
]

ANCHORS_JSON = "anchors.json"

# 如果你已经有 anchor_rel，可以直接在这里写死，例如：
# ANCHOR_REL = [(0.80,0.19),(0.50,0.43),(0.62,0.31),(0.52,0.62),(0.48,0.86)]
# 否则设为 None，程序会让你鼠标点选生成
ANCHOR_REL = None


# ----------------- 字体：强制TTF，避免字号不生效 -----------------
def load_font_strict(size: int, bold: bool = True):
    candidates = []
    if bold:
        candidates += [
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "C:/Windows/Fonts/msyhbd.ttc",   # Windows：微软雅黑粗体（有就用）
            "C:/Windows/Fonts/arialbd.ttf",
        ]
    candidates += [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/arial.ttf",
    ]

    for p in candidates:
        if os.path.exists(p):
            return ImageFont.truetype(p, size=size), p

    raise RuntimeError(
        "找不到可用的 TTF/TTC 字体，导致字号可能不生效。\n"
        "请安装 NotoSansCJK / DejaVuSans，或在 load_font_strict() 的 candidates 里填入你机器上的字体路径。"
    )

# ----------------- 基础工具 -----------------
def sc(x):  # internal scale
    return int(round(x * RENDER_SCALE))

def resize_contain(im: Image.Image, target_w: int, target_h: int) -> Image.Image:
    im = im.convert("RGBA")
    w, h = im.size
    scale = min(target_w / w, target_h / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    return im.resize((nw, nh), Image.Resampling.LANCZOS)

# ✅ 更稳的“真正居中”版本：用 bbox 手算中心（各 Pillow 版本都稳）
def draw_number_badge(draw, center, label, r: int, font):
    cx, cy = center
    stroke_w = max(sc(3), int(r * 0.12))  # 圈大时描边也更协调

    # 圆
    draw.ellipse([cx - r, cy - r, cx + r, cy + r],
                 outline=(0, 0, 0, 255),
                 width=stroke_w,
                 fill=(255, 255, 255, 255))

    txt = str(label)

    # 文本 bbox（优先 font.getbbox，更准确）
    if hasattr(font, "getbbox"):
        l, t, r2, b2 = font.getbbox(txt)
    else:
        l, t, r2, b2 = draw.textbbox((0, 0), txt, font=font)

    x = int(round(cx - (l + r2) / 2))
    y = int(round(cy - (t + b2) / 2))

    draw.text((x, y), txt, fill=(0, 0, 0, 255), font=font)

def shorten_to_circle(start, circle_center, radius, pad):
    sx, sy = start
    cx, cy = circle_center
    dx, dy = cx - sx, cy - sy
    dist = math.hypot(dx, dy) or 1.0
    back = radius + pad
    t = max(0.0, (dist - back) / dist)
    return (sx + dx * t, sy + dy * t)

def draw_arrow(draw, start, end):
    width = sc(2)
    head_len = sc(12)
    head_angle_deg = 28

    draw.line([start, end], fill=(0, 0, 0, 255), width=width)
    sx, sy = start
    ex, ey = end
    ang = math.atan2(ey - sy, ex - sx)
    ha = math.radians(head_angle_deg)

    p1 = (ex, ey)
    p2 = (ex - head_len * math.cos(ang - ha), ey - head_len * math.sin(ang - ha))
    p3 = (ex - head_len * math.cos(ang + ha), ey - head_len * math.sin(ang + ha))
    draw.polygon([p1, p2, p3], fill=(0, 0, 0, 255))

# ----------------- 交互点选锚点（matplotlib） -----------------
def pick_anchor_rel(main_path: str, n_points: int, save_json: str):
    import matplotlib.pyplot as plt

    img = Image.open(main_path).convert("RGB")
    w, h = img.size

    plt.figure(figsize=(10, 8))
    plt.imshow(img)
    plt.title(f"按顺序点击 {n_points} 个锚点（1→{n_points}），点在主图对应实物上；点完关闭窗口")
    plt.axis("off")

    pts = plt.ginput(n_points, timeout=0)  # [(x,y),...]
    for i, (x, y) in enumerate(pts, start=1):
        plt.scatter([x], [y])
        plt.text(x + 5, y + 5, str(i), color="red", fontsize=14)
    plt.show()

    anchor_rel = [(float(x) / w, float(y) / h) for (x, y) in pts]
    with open(save_json, "w", encoding="utf-8") as f:
        json.dump(anchor_rel, f, ensure_ascii=False, indent=2)

    return anchor_rel

def load_anchor_rel(json_path: str):
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

# ----------------- 渲染函数（无文字、紧凑布局） -----------------
def render(main_path: str, comps, anchor_rel, out_path: str):
    main = Image.open(main_path).convert("RGBA")
    main_fit = resize_contain(main, target_w=sc(900), target_h=sc(860))
    mw, mh = main_fit.size

    n = len(comps)

    left_margin = sc(40)
    right_margin = sc(40)
    top_margin = sc(40)
    bottom_margin = sc(40)

    gap_main_to_thumbs = sc(GAP_MAIN_TO_THUMBS)
    row_gap = sc(ROW_GAP)

    # 缩略图高度自适应：尽量塞满主图高度，减少空白
    usable_h = mh - sc(40)
    thumb_h = int((usable_h - (n - 1) * row_gap) / n)
    thumb_h = max(thumb_h, sc(90))
    thumb_w = int(thumb_h * (260 / 150))  # 维持大致比例

    thumb_x = left_margin + mw + gap_main_to_thumbs
    canvas_w = thumb_x + thumb_w + right_margin

    thumb_total_h = n * thumb_h + (n - 1) * row_gap
    canvas_h = max(top_margin + mh + bottom_margin,
                   top_margin + thumb_total_h + bottom_margin)

    canvas = Image.new("RGBA", (canvas_w, canvas_h), (255, 255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    main_x = left_margin
    main_y = (canvas_h - mh) // 2
    canvas.alpha_composite(main_fit, (main_x, main_y))

    thumb_y0 = main_y + (mh - thumb_total_h) // 2
    thumb_y0 = max(top_margin, thumb_y0)

    badge_r = sc(BADGE_RADIUS)
    badge_font, font_path = load_font_strict(sc(BADGE_FONT_SIZE), bold=True)

    badge_to_thumb_gap = sc(BADGE_TO_THUMB_GAP)
    arrow_pad = sc(ARROW_PAD)

    print("Using font:", font_path)
    print("Badge radius(px final):", BADGE_RADIUS, "Font size(px final):", BADGE_FONT_SIZE)

    for i, c in enumerate(comps):
        ty = thumb_y0 + i * (thumb_h + row_gap)

        im = Image.open(c["img"]).convert("RGBA")
        thumb = resize_contain(im, thumb_w - sc(12), thumb_h - sc(12))

        frame = Image.new("RGBA", (thumb_w, thumb_h), (255, 255, 255, 255))
        fdraw = ImageDraw.Draw(frame)
        fdraw.rectangle([0, 0, thumb_w - 1, thumb_h - 1], outline=(0, 0, 0, 255), width=sc(2))
        frame.alpha_composite(thumb, ((thumb_w - thumb.size[0]) // 2, (thumb_h - thumb.size[1]) // 2))
        canvas.alpha_composite(frame, (thumb_x, ty))

        badge_center = (thumb_x - badge_to_thumb_gap, ty + thumb_h // 2)
        draw_number_badge(draw, badge_center, c["label"], badge_r, badge_font)

        # 箭头起点：来自 anchor_rel（指到实物）
        ax = main_x + int(mw * anchor_rel[i][0])
        ay = main_y + int(mh * anchor_rel[i][1])

        arrow_end = shorten_to_circle((ax, ay), badge_center, badge_r, pad=arrow_pad)
        draw_arrow(draw, (ax, ay), arrow_end)

    # 缩回输出
    final_img = canvas.resize((canvas_w // RENDER_SCALE, canvas_h // RENDER_SCALE), Image.Resampling.LANCZOS)
    final_img.convert("RGB").save(out_path, quality=95)
    print("saved:", out_path)


if __name__ == "__main__":
    # 1) 获取 anchor_rel（优先：代码里写死；其次：读 anchors.json；最后：交互点选）
    anchor_rel = ANCHOR_REL or load_anchor_rel(ANCHORS_JSON)

    if anchor_rel is None:
        anchor_rel = pick_anchor_rel(paths["main"], len(components), ANCHORS_JSON)

    # 2) 生成图
    render(paths["main"], components, anchor_rel, "robot_ai_style_no_text.png")
