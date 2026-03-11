# 作图资料索引（DQN4 → DQN8 迁移）

> 整理日期：2026-03-10
> 来源：DQN4 论文作图素材，已核实并导出为科研级 PDF（≥300dpi）

## exports/ — 导出的高清 PDF（可直接用于论文）

| 文件 | 内容 | DQN8 对应图 | 状态 |
|---|---|---|---|
| `vehicle_topview_lineart.pdf` | 车辆俯视线稿 + **双圆碰撞检测** | `fig_dual_circle` | ✅ 直接可用（矢量） |
| `vehicle_dimensions_visio.pdf` | 运动学模型 + 车辆尺寸标注图（含 mm 级参数） | 论文参数表 | ✅ 直接可用（矢量） |
| `vehicle_dimensions_annotated.pdf` | 车辆尺寸标注原始照片（带红笔标注 600mm） | 参考 | ✅ 参考用 |
| `dqn4_framework_diagram.pdf` | DQN4 框架总图（位图 300dpi） | `fig_framework` 参考 | ⚠️ 需重绘为 DQN8 版本 |
| `dqn4_framework_slide.pdf` | DQN4 框架幻灯片（矢量） | `fig_framework` 参考 | ⚠️ 需重绘为 DQN8 版本 |
| `forest_field_photo.pdf` | 森林实车测试照片（300dpi） | `fig_realmap_site` 候选 | ✅ 直接可用 |
| `robot_front_photo.pdf` | 机器人正面照（300dpi） | `fig_platform` 备选 | ✅ 直接可用 |
| `pointcloud_map.pdf` | 点云地图（300dpi） | `fig_pointcloud` 候选 | ✅ 直接可用 |
| `cnn_inference_pipeline.pdf` | CNN 推理管线框图（300dpi） | `fig_cnn_pddqn` 参考 | ⚠️ 分辨率偏低，建议重绘 |
| `kinematic_model_visio.pdf` | Visio 版运动学模型图 | `fig_kinematics` 备选 | ✅ 可用 |

### 车辆关键尺寸（来自 VSDX 尺寸图）

| 参数 | 值 (mm) |
|---|---|
| 总长 | 924 |
| 轴距 | 700 |
| 总宽（含轮） | 740 |
| 前轮距 | 600 |
| 轮宽 | 180 |

## 图片png/ — 原始素材与生成脚本

### 设备图/
- `Image.py` — **设备标注图生成脚本**（PIL，主图+缩略图+箭头标注）→ `fig_platform`
- `anchors.json` — 5 个标注锚点坐标
- `大车整体图.png` — 主图素材
- `显示器.png / 雷达mid360.png / 工控机.png / 电源.png / 无人小车底盘.png` — 5 个组件缩略图
- `robot_ai_style.png / robot_ai_style_no_text.png` — 已生成的成品

### 流程图/
- `page.py` — **系统总流程图生成脚本**（matplotlib，5 步流程框图）
  - ⚠️ 第 4 步算法列表需更新为 DQN8 的 8 算法
  - ⚠️ 第 5 步结果截图需替换
  - ⚠️ SCI 投稿需英文化
- `workflow_overview_cn_sci_v2.png/pdf` — 生成的 2×3 布局版本
- `workflow_overview_cn_sci_v3.png/pdf` — 生成的横排版本
- 素材图：`设备图.jpg / 点云图.jpg / 格栅地图.jpg / 可通行分割后的点云.jpg / 路径规划结果.png / 强化学习路径规划的设计.jpg`
- `测试场景.jpg` — 森林场景照片

### 运动学模型/
- `png.py` — **运动学示意图生成脚本**（matplotlib，四轮+自行车模型叠加，智能标签避让）→ `fig_kinematics_a`
- `vehicle_kinematic_model.png` — 已生成的成品
- `two_dof_kinematic_diagram.png` — 彩色风格版本（风格不统一，不推荐）
- `Untitled Diagram.drawio` — draw.io 矢量源文件

## 根目录 PPT/VSDX 源文件

| 文件 | 内容 |
|---|---|
| `1111.pptx` | DQN4 框架总图（1 页，嵌入高清位图） |
| `1.21(2).pptx` | DQN4 汇报 PPT（6 页，含森林照片/点云/CNN 框图/设备照） |
| `double_circle_vehicle_shapes.pptx` | 车辆俯视线稿 + 双圆矢量图层（1 页） |
| `闲鱼tb34su[5].vsdx` | Visio：运动学模型 + 车辆尺寸标注（1 页） |

## DQN8 论文图清单 vs 素材对照

| 论文图 | 素材来源 | 状态 |
|---|---|---|
| `fig_framework` | `exports/dqn4_framework_slide.pdf` + `流程图/page.py` | ⚠️ 需重绘 |
| `fig_platform` | `设备图/Image.py` → 成品已有 | ✅ |
| `fig_kinematics_a` | `运动学模型/png.py` → 成品已有 | ✅ |
| `fig_kinematics_b` | `运动学模型/Untitled Diagram.drawio` 或重绘 | ⚠️ 建议统一风格 |
| `fig_dual_circle` | `exports/vehicle_topview_lineart.pdf` | ✅ |
| `fig_cnn_pddqn` | `exports/cnn_inference_pipeline.pdf`（参考） | ⚠️ 需重绘 |
| `fig_realmap_site` | `exports/forest_field_photo.pdf` | ✅ |
| `fig_pointcloud` | `exports/pointcloud_map.pdf` | ✅ |
| 实验结果图 | **无** — 需从 `runs/paperstoryV2.1/` 生成 | ❌ 需新建 |
