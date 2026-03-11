# SCI 级作图指南 — DQN8 论文

## 通用标准
- **格式**：矢量 PDF（`matplotlib: plt.savefig('fig.pdf', bbox_inches='tight')`）
- **字号**：≥ 8pt，与正文协调（`plt.rcParams['font.size'] = 10`）
- **线宽**：≥ 1pt（`plt.rcParams['lines.linewidth'] = 1.5`）
- **颜色+线型双编码**：确保黑白打印可区分
  - 使用不同的 linestyle (`-`, `--`, `-.`, `:`) + marker (`o`, `s`, `^`, `D`, `v`, `*`)
- **图注**：完整可独立理解
- **子图**：用 (a)(b)(c) 标注

## 各图改进方案

### Fig 1: 框架总图 (fig_framework)
- 当前：PNG 位图
- 改进：用 TikZ 或 draw.io 重绘为矢量 PDF
- 要求：模块标注清晰，数据流向用箭头标注

### Fig 2: 平台照片 (fig_platform)
- 当前：JPG
- 改进：保持照片格式但转为高分辨率 (≥300 DPI)
- 添加标注：传感器位置、坐标系

### Fig 5: CNN-PDDQN 框架图 (fig_cnn_pddqn)
- 当前：PNG，含 "SCI DIAGRAM" 等内部标注
- 改进：重绘为矢量图，去掉内部标注
- 更名为"DRL 运动规划框架"

### Fig 6: 伪代码 (fig_algorithm)
- 已替换为 LaTeX algorithm 环境 ✅

### 路径对比图 (fig_sim_long, fig_sim_short, fig_real_paths)
- 当前：轨迹线太细，legend 太小
- 改进：
  - 加粗轨迹线 (linewidth ≥ 2pt)
  - 增大 legend 字号 (fontsize ≥ 8pt)
  - 不同算法使用不同颜色+线型
  - 起终点用明显标记 (★/●)

### 柱状图 (fig_bar_long, fig_bar_short, fig_bar_real)
- 当前：x 轴标签倾斜难读
- 改进：
  - 水平排列标签或使用缩写
  - 添加数值标注在柱顶
  - 分组颜色 + 填充纹理双编码

### 训练曲线图 [TODO: 需要训练日志]
- 绘制脚本：`dqn8_plots/plot_training.py`
- 内容：reward, success rate, loss vs episodes
- 6 种变体分别绘制 + 同图对比
- 添加滑动平均线 (window=20)

### 消融对比柱状图 [TODO: 待数据]
- 模板：grouped bar chart
- x 轴：消融变体 (A/B/C/D)
- y 轴：成功率
- 分组：Short / Long

## 绘图脚本位置
- 主脚本：`dqn8_plots/run_all.py`
- 输出目录：`paperdqn8.1/figs/`
- matplotlib 配置：使用 `plt.style.use('seaborn-v0_8-whitegrid')`

## 配色方案
```python
COLORS = {
    'CNN-PDDQN': '#1f77b4',  # 蓝
    'CNN-DDQN':  '#ff7f0e',  # 橙
    'CNN-DQN':   '#2ca02c',  # 绿
    'MLP-PDDQN': '#d62728',  # 红
    'MLP-DDQN':  '#9467bd',  # 紫
    'MLP-DQN':   '#8c564b',  # 棕
    'Hybrid A*': '#7f7f7f',  # 灰
    'RRT*':      '#bcbd22',  # 黄绿
    'LO-HA*':    '#17becf',  # 青
}
```
