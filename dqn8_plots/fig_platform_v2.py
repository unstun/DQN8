import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
from PIL import Image

# ==========================================
# SCI (Nature/IEEE) Standard Plot Settings
# ==========================================
# Use pure black for clear contrast in printing
text_color = '#000000'
line_color = '#000000'

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['text.color'] = text_color
plt.rcParams['axes.labelcolor'] = text_color
plt.rcParams['xtick.color'] = text_color
plt.rcParams['ytick.color'] = text_color
# Academic papers usually use regular font weight, bold only for emphasis (like subfigure labels)
plt.rcParams['font.weight'] = 'regular'
# Font size for typical 2-column IEEE papers is 8-10 pt
plt.rcParams['font.size'] = 9

# Create figure. Width=3.5 inches fits exactly in one column of an IEEE 2-column paper
fig = plt.figure(figsize=(3.5, 3.8), dpi=600)
fig.patch.set_facecolor('white')

# Use standard grid spec
import matplotlib.gridspec as gridspec
gs = gridspec.GridSpec(3, 2, width_ratios=[1.7, 1.1], wspace=0.15, hspace=0.25)

ax_main = fig.add_subplot(gs[:, 0])
ax_sub1 = fig.add_subplot(gs[0, 1])
ax_sub2 = fig.add_subplot(gs[1, 1])
ax_sub3 = fig.add_subplot(gs[2, 1])

# Image Paths
base_dir = "/home/sun/phdproject/dqn/DQN8/作图资料i/图片png/设备图"
main_img_path = os.path.join(base_dir, "大车整体图.png")
sub1_path = os.path.join(base_dir, "雷达mid360.png")
sub2_path = os.path.join(base_dir, "工控机.png")
sub3_path = os.path.join(base_dir, "无人小车底盘.png")

# Main Image (Crop as before, but without any fancy borders)
img_main = Image.open(main_img_path)
CROP_BOX = (0, 70, 405, 545)
img_main = img_main.crop(CROP_BOX)
ax_main.imshow(img_main)
ax_main.axis('off')

# Simple, professional thin border around main image if needed, or just let it be.
# We'll use a thin, sharp black border for the bounding box.
rect_main = patches.Rectangle(
    (0, 0), img_main.width-1, img_main.height-1,
    linewidth=0.5, edgecolor=line_color, facecolor='none', zorder=10
)
ax_main.add_patch(rect_main)

# Subfigure processing function (Clean academic style)
def process_subimg(img_path, ax, label):
    img = Image.open(img_path)
    size = max(img.size)
    bg_size = int(size * 1.05) # minimal padding
    img.thumbnail((size, size), Image.Resampling.LANCZOS)
    
    bg = Image.new('RGB', (bg_size, bg_size), (255, 255, 255))
    offset = ((bg_size - img.size[0]) // 2, (bg_size - img.size[1]) // 2)
    
    if img.mode == 'RGBA':
        bg.paste(img, offset, img)
    else:
        bg.paste(img, offset)
        
    ax.imshow(bg)
    ax.axis('off')
    
    # Sharp, standard rectangular border
    rect = patches.Rectangle(
        (0, 0), bg_size-1, bg_size-1,
        linewidth=0.5, edgecolor=line_color, facecolor='none', zorder=10
    )
    ax.add_patch(rect)
    
    # SCI labels are usually bold e.g., "(a)", text normal e.g., "Livox Mid-360"
    # We'll just position it neatly under or next to the box.
    ax.text(0.5, -0.1, label, 
            transform=ax.transAxes, 
            ha='center', va='top', 
            fontsize=9, fontweight='bold', color=text_color)

# Add sub-images with bold labels
process_subimg(sub1_path, ax_sub1, "(a) Livox Mid-360")
process_subimg(sub2_path, ax_sub2, "(b) Jetson AGX Orin")
process_subimg(sub3_path, ax_sub3, "(c) Ackermann chassis")

plt.tight_layout()
fig.canvas.draw()

# ==========================================
# Professional Right-Angle Callout Lines
# ==========================================
def add_callout(ax_src, xy_data, ax_dst):
    # Transform src coordinates from data to fig
    trans_data = ax_src.transData
    trans_fig = fig.transFigure.inverted()
    pt_src_fig = trans_fig.transform(trans_data.transform(xy_data))
    
    # Transform dst coordinates
    bbox_dst = ax_dst.get_position()
    pt_dst_fig = (bbox_dst.x0 - 0.015, bbox_dst.y0 + bbox_dst.height / 2.0)
    
    # Draw a line with a bend (elbow)
    # E.g., from src, go horizontally a bit, then vertically, then horizontally to dst
    mid_x_fig = pt_src_fig[0] + (pt_dst_fig[0] - pt_src_fig[0]) * 0.5
    
    line = matplotlib.lines.Line2D(
        [pt_src_fig[0], mid_x_fig, mid_x_fig, pt_dst_fig[0]],
        [pt_src_fig[1], pt_src_fig[1], pt_dst_fig[1], pt_dst_fig[1]],
        transform=fig.transFigure,
        linewidth=1.0, color=line_color, solid_joinstyle='miter', solid_capstyle='butt', zorder=20
    )
    fig.add_artist(line)

# Add standard SCI geometric callout lines instead of curved arrows
# Tweak the starting positions slightly if needed to match the new image mapping
add_callout(ax_main, (220, 160), ax_sub1)
add_callout(ax_main, (165, 255), ax_sub2)
add_callout(ax_main, (210, 420), ax_sub3)

# Professional Anchor Dots
def add_anchor_dot(ax, xy):
    # Pure black dot with white clean edge
    dot_halo = patches.Circle(xy, radius=10, color='white', alpha=1.0, zorder=29)
    dot = patches.Circle(xy, radius=6, color=line_color, alpha=1.0, zorder=30)
    ax.add_patch(dot_halo)
    ax.add_patch(dot)

add_anchor_dot(ax_main, (220, 160))
add_anchor_dot(ax_main, (165, 255))
add_anchor_dot(ax_main, (210, 420))

# Save PDF (High Quality 600dpi for publication) and PNG (preview)
out_dir = "/home/sun/phdproject/dqn/DQN8/paperdqn8.3/media"
os.makedirs(out_dir, exist_ok=True)
out_path_pdf = os.path.join(out_dir, "fig_platform_v2.pdf")
out_path_png = os.path.join(out_dir, "fig_platform_v2.png")
fig.savefig(out_path_pdf, dpi=600, bbox_inches='tight', facecolor='white', pad_inches=0.02)
fig.savefig(out_path_png, dpi=600, bbox_inches='tight', facecolor='white', pad_inches=0.02)
print(f"Professional SCI standard Figure saved to {out_path_pdf} and {out_path_png}")
