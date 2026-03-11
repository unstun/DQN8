import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
from PIL import Image

# 基础设置
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 8

# 画布，宽 3.5 英寸 = 单栏宽度
fig = plt.figure(figsize=(3.5, 3.2), dpi=300)
fig.patch.set_facecolor('white')

# 布局：左 1，右 3
import matplotlib.gridspec as gridspec
gs = gridspec.GridSpec(3, 2, width_ratios=[1.6, 1], wspace=0.1, hspace=0.4)

ax_main = fig.add_subplot(gs[:, 0])
ax_sub1 = fig.add_subplot(gs[0, 1])
ax_sub2 = fig.add_subplot(gs[1, 1])
ax_sub3 = fig.add_subplot(gs[2, 1])

# 路径
base_dir = "/home/sun/phdproject/dqn/DQN8/作图资料i/图片png/设备图"
main_img_path = os.path.join(base_dir, "大车整体图.png")
sub1_path = os.path.join(base_dir, "雷达mid360.png")
sub2_path = os.path.join(base_dir, "工控机.png")
sub3_path = os.path.join(base_dir, "无人小车底盘.png")

# 主图处理
img_main = Image.open(main_img_path)
CROP_BOX = (0, 70, 405, 545)
img_main = img_main.crop(CROP_BOX)
ax_main.imshow(img_main)
ax_main.axis('off')

# 子图处理函数
def process_subimg(img_path, ax, label):
    img = Image.open(img_path)
    # 转换为统一的方形，并带有白背景
    size = max(img.size)
    # 放大一点画布
    bg_size = int(size * 1.1)
    img.thumbnail((size, size), Image.Resampling.LANCZOS)
    bg = Image.new('RGB', (bg_size, bg_size), (255, 255, 255))
    offset = ((bg_size - img.size[0]) // 2, (bg_size - img.size[1]) // 2)
    
    if img.mode == 'RGBA':
        bg.paste(img, offset, img)
    else:
        bg.paste(img, offset)
        
    ax.imshow(bg)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # 细边框 0.8pt 灰黑色
    for spine in ax.spines.values():
        spine.set_edgecolor('#333333')
        spine.set_linewidth(0.8)
    
    # 添加标签
    ax.set_xlabel(label, fontsize=8, fontweight='bold', labelpad=4)

process_subimg(sub1_path, ax_sub1, "(a) Livox Mid-360 LiDAR")
process_subimg(sub2_path, ax_sub2, "(b) Jetson AGX Orin")
process_subimg(sub3_path, ax_sub3, "(c) Ackermann chassis")

plt.tight_layout()
fig.canvas.draw()

# 添加箭头函数
def add_arrow(ax_src, xy_data, ax_dst, rad=0.0):
    trans_data = ax_src.transData
    trans_fig = fig.transFigure.inverted()
    
    pt_src_fig = trans_fig.transform(trans_data.transform(xy_data))
    
    bbox_dst = ax_dst.get_position()
    # 箭头指向子图左边缘垂直中心，可以稍微向左偏一点
    pt_dst_fig = (bbox_dst.x0 - 0.015, bbox_dst.y0 + bbox_dst.height / 2.0)
    
    arrow = FancyArrowPatch(
        posA=pt_src_fig, posB=pt_dst_fig,
        transform=fig.transFigure,
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle="-|>,head_length=3.5,head_width=1.5",
        linewidth=1,
        color='black',
        shrinkA=2, shrinkB=0,
        zorder=10
    )
    fig.patches.append(arrow)

# 绘制引导箭头 (直一点，避免交叉)
add_arrow(ax_main, (200, 160), ax_sub1, rad=0.2)
add_arrow(ax_main, (155, 255), ax_sub2, rad=0.1)
add_arrow(ax_main, (200, 400), ax_sub3, rad=-0.2)

# 保存
out_dir = "/home/sun/phdproject/dqn/DQN8/paperdqn8.3/media"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "fig_platform.pdf")
fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.05)
print(f"Figure saved to {out_path}")
