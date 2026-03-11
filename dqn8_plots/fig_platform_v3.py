"""fig_platform_v3 — rembg 抠图 + 白底重排版"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
from PIL import Image
from rembg import remove as rembg_remove

# ── SCI style ──────────────────────────────────────────
text_color = '#000000'
line_color = '#000000'
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'text.color': text_color,
    'axes.labelcolor': text_color,
    'font.weight': 'regular',
    'font.size': 9,
})

# ── Paths ──────────────────────────────────────────────
base_dir = "/home/sun/phdproject/dqn/DQN8/作图资料i/图片png/设备图"
main_img_path = os.path.join(base_dir, "大车整体图.png")
sub1_path = os.path.join(base_dir, "雷达mid360.png")
sub2_path = os.path.join(base_dir, "工控机.png")
sub3_path = os.path.join(base_dir, "无人小车底盘.png")

out_dir = "/home/sun/phdproject/dqn/DQN8/paperdqn8.3/media"
os.makedirs(out_dir, exist_ok=True)

# ── rembg helper ───────────────────────────────────────
def remove_bg(img_path, crop_box=None):
    """Remove background → RGBA, optional crop, then paste on white."""
    img = Image.open(img_path)
    if crop_box:
        img = img.crop(crop_box)
    # rembg expects RGB or RGBA PIL image
    img_nobg = rembg_remove(img)  # returns RGBA
    # paste onto white
    white = Image.new("RGB", img_nobg.size, (255, 255, 255))
    white.paste(img_nobg, mask=img_nobg.split()[3])
    return white

print("Removing backgrounds (this may take a moment on first run)...")
img_main = remove_bg(main_img_path, crop_box=(0, 70, 405, 545))
img_sub1 = remove_bg(sub1_path)
img_sub2 = remove_bg(sub2_path)
img_sub3 = remove_bg(sub3_path)
print("Background removal done.")

# ── Figure ─────────────────────────────────────────────
fig = plt.figure(figsize=(3.5, 3.8), dpi=600)
fig.patch.set_facecolor('white')

gs = gridspec.GridSpec(3, 2, width_ratios=[1.7, 1.1], wspace=0.15, hspace=0.25)
ax_main = fig.add_subplot(gs[:, 0])
ax_sub1 = fig.add_subplot(gs[0, 1])
ax_sub2 = fig.add_subplot(gs[1, 1])
ax_sub3 = fig.add_subplot(gs[2, 1])

# Main image
ax_main.imshow(img_main)
ax_main.axis('off')
rect_main = patches.Rectangle(
    (0, 0), img_main.width - 1, img_main.height - 1,
    linewidth=0.5, edgecolor=line_color, facecolor='none', zorder=10,
)
ax_main.add_patch(rect_main)

# Sub-image helper
def show_sub(img, ax, label):
    # pad to square
    size = max(img.size)
    bg_size = int(size * 1.05)
    bg = Image.new('RGB', (bg_size, bg_size), (255, 255, 255))
    offset = ((bg_size - img.size[0]) // 2, (bg_size - img.size[1]) // 2)
    bg.paste(img, offset)
    ax.imshow(bg)
    ax.axis('off')
    rect = patches.Rectangle(
        (0, 0), bg_size - 1, bg_size - 1,
        linewidth=0.5, edgecolor=line_color, facecolor='none', zorder=10,
    )
    ax.add_patch(rect)
    ax.text(0.5, -0.1, label,
            transform=ax.transAxes, ha='center', va='top',
            fontsize=9, fontweight='bold', color=text_color)

show_sub(img_sub1, ax_sub1, "(a) Livox Mid-360")
show_sub(img_sub2, ax_sub2, "(b) Jetson AGX Orin")
show_sub(img_sub3, ax_sub3, "(c) Ackermann chassis")

plt.tight_layout()
fig.canvas.draw()

# ── Callout lines & anchor dots ────────────────────────
def add_callout(ax_src, xy_data, ax_dst):
    trans_data = ax_src.transData
    trans_fig = fig.transFigure.inverted()
    pt_src = trans_fig.transform(trans_data.transform(xy_data))
    bbox = ax_dst.get_position()
    pt_dst = (bbox.x0 - 0.015, bbox.y0 + bbox.height / 2.0)
    mid_x = pt_src[0] + (pt_dst[0] - pt_src[0]) * 0.5
    line = mlines.Line2D(
        [pt_src[0], mid_x, mid_x, pt_dst[0]],
        [pt_src[1], pt_src[1], pt_dst[1], pt_dst[1]],
        transform=fig.transFigure,
        linewidth=1.0, color=line_color,
        solid_joinstyle='miter', solid_capstyle='butt', zorder=20,
    )
    fig.add_artist(line)

def add_dot(ax, xy):
    dot_halo = patches.Circle(xy, radius=10, color='white', zorder=29)
    dot = patches.Circle(xy, radius=6, color=line_color, zorder=30)
    ax.add_patch(dot_halo)
    ax.add_patch(dot)

anchors = [(220, 160), (165, 255), (210, 420)]
for (xy, ax_dst) in zip(anchors, [ax_sub1, ax_sub2, ax_sub3]):
    add_callout(ax_main, xy, ax_dst)
    add_dot(ax_main, xy)

# ── Save ───────────────────────────────────────────────
for ext in ('pdf', 'png'):
    path = os.path.join(out_dir, f"fig_platform_v3.{ext}")
    fig.savefig(path, dpi=600, bbox_inches='tight', facecolor='white', pad_inches=0.02)
    print(f"Saved: {path}")
