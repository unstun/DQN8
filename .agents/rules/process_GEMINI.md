# DQN8/AGENTS.md（Ubuntu 24.04 + ros2py310）

> 作用域：`/home/sun/phdproject/dqn/DQN8/**`。通用环境参考上层 `../AGENTS.md`。

## 0. 硬约束

1) 每次回复以"帅哥，"开头。
2) 改文件前输出3–7步计划+文件清单+风险+验证，等"开始"后再动手。
3) 默认中文回复。
4) DQN/DDQN等术语遵循原论文；不确定先查证。
5) 代码改动后在`configs/`新增`repro_YYYYMMDD_<topic>.json`；纯文档豁免。
6) `AGENTS.md`、`CLAUDE.md`、`GEMINI.md`三文件逐行一致；改后用`diff`两两验证。
7) 训练/推理远程优先执行，失败后切换本地。
8) **消融实验必须留档**：每次消融实验结束后，在`runs/ablation_logs/`写入 Markdown 记录，包含变体矩阵、各变体 success_rate 对比表、结论。文件命名`ablation_YYYYMMDD_<topic>.md`。

## 1. 常用命令

- 自检/训练/推理：`conda run --cwd /home/sun/phdproject/dqn/DQN8 -n ros2py310 python {train,infer}.py {--self-check | --profile <name>}`
- 输出目录：`runs/`；文档：`README.md`、`runtxt.md`。

## 2. 踩坑

### 2.1 SSH远端执行

- **必须**加`--cwd`：`conda run --cwd /home/sun/phdproject/dqn/DQN8 -n ros2py310 python train.py ...`
- `~/.bashrc`的conda init须在`case $- in`前（2026-02-21已修，复发查`~/.bashrc.bak.*`）。

### 2.2 Write/Edit限制

- 单次≤50行，超过须拆分调用。

### 2.3 联网调研

- WebFetch/WebSearch不混批；每批同类≤2；优先arXiv/GitHub HTML版。
- 付费墙403：Playwright（`browser_navigate`→`browser_wait_for`5s→`browser_snapshot`）。

### 2.4 本地执行工作流

`PROJ=/home/sun/phdproject/dqn/DQN8` / `ENV=ros2py310`

0. **训练**：`nohup conda run --cwd $PROJ -n $ENV python train.py --profile $PROFILE > runs/${PROFILE}_$(date +%Y%m%d_%H%M%S).log 2>&1 &`
1. **推理**：`conda run --cwd $PROJ -n $ENV python infer.py --profile $PROFILE`
2. **检查完成**：`ls $PROJ/runs/$EXP/train_*/infer/*/table2_kpis.csv 2>/dev/null && echo DONE || echo RUNNING`

## 3. 模块索引

### 3.1 核心包 `amr_dqn/`

| 文件 | 行数 | 职责 |
|------|------|------|
| `__init__.py` | 22 | 包说明 + 模块布局总览 |
| `agents.py` | 580 | DQN/DDQN/PDDQN 智能体（TD学习、目标网络、DQfD margin+CE loss） |
| `env.py` | 2080 | Gymnasium 环境：AMRGridEnv（8方向网格）+ AMRBicycleEnv（Ackermann自行车模型） |
| `networks.py` | 150 | Q网络：MLPQNetwork（全连接）、CNNQNetwork（卷积+标量拼接） |
| `replay_buffer.py` | 100 | 均匀经验回放缓冲区（DQfD demo 保护机制） |
| `reward_norm.py` | 50 | Welford 在线奖励归一化器（均值/方差 + clip） |
| `forest_policy.py` | 117 | 统一动作选择管线（训练/推理共用：Q贪心→可行性→top-k→fallback） |
| `schedules.py` | 27 | epsilon 衰减调度（线性、自适应sigmoid） |
| `smoothing.py` | 32 | Chaikin 角切割路径平滑 |
| `metrics.py` | 88 | 路径KPI：长度、曲率、转角数、最大转角 |
| `config_io.py` | 169 | JSON配置加载 + argparse 集成 |
| `runtime.py` | 94 | PyTorch/CUDA/Matplotlib 运行时设置 |
| `runs.py` | 159 | 时间戳实验目录管理（runs/<exp>/train_YYYYMMDD_HHMMSS/） |

### 3.2 CLI 入口 `amr_dqn/cli/`

| 文件 | 行数 | 职责 |
|------|------|------|
| `train.py` | 1600 | 训练主循环：多环境×多算法、DQfD demo预填充、周期性eval、checkpoint保存 |
| `infer.py` | 2800 | 推理评估：加载模型→rollout→KPI表→路径/控制图→CSV/Markdown输出 |
| `benchmark.py` | 244 | 训练+推理编排器（子进程调用 + KPI验证） |
| `config.py` | 56 | 生成 train+infer 合并JSON配置模板 |
| `precompute_forest_paths.py` | 70 | 离线 Hybrid A* 专家路径缓存 |

### 3.3 地图 `amr_dqn/maps/`

| 文件 | 职责 |
|------|------|
| `__init__.py` | MapSpec 协议、forest/realmap 加载入口、环境名注册 |
| `forest.py` | 程序化森林地图生成（ForestParams、种子控制、间距校验） |
| `pgm.py` | ROS PGM 栅格地图加载器（realmap_a） |
| `precomputed/` | Hybrid A* 预计算专家路径 JSON（forest_a/b/c/d） |

### 3.4 基线 `amr_dqn/baselines/`

| 文件 | 职责 |
|------|------|
| `pathplan.py` | Hybrid A* / RRT* 规划封装（车辆footprint + Ackermann参数默认值） |

### 3.5 第三方 `amr_dqn/third_party/pathplan/`

| 文件 | 职责 |
|------|------|
| `hybrid_a_star/planner.py` | Hybrid A* 核心（Ackermann运动学 + Reeds-Shepp启发式） |
| `hybrid_a_star/reeds_shepp.py` | Dubins/Reeds-Shepp 路径原语 |
| `rrt/rrt_star.py` | Spline-based RRT* 实现 |
| `geometry.py` | 2D几何：OBB碰撞、双圆近似、向量运算 |
| `robot.py` | Ackermann 运动学模型（AckermannParams/State） |
| `map_utils.py` | GridMap：栅格地图膨胀、碰撞查询 |

### 3.6 绘图 `dqn8_plots/`

| 文件 | 职责 |
|------|------|
| `run_all.py` | 一键绘图入口（串联所有 plot 脚本） |
| `common.py` | 共享配置：算法颜色、显示名、样式 |
| `plot_training.py` | 训练曲线（奖励 + 成功率） |
| `plot_paths.py` | 路径对比图（障碍物网格 + 轨迹叠加） |
| `plot_kpi_bars.py` | KPI 分组柱状图 |
| `plot_kpi_radar.py` | KPI 雷达图（归一化0-1轴） |
| `plot_kpi_table.py` | KPI 表格渲染为图片（最优值高亮） |
| `plot_map_overview.py` | 地图总览图（起终点标注） |

### 3.7 顶层入口

| 文件 | 职责 |
|------|------|
| `train.py` | 薄包装器 → `amr_dqn.cli.train.main()` |
| `infer.py` | 薄包装器 → `amr_dqn.cli.infer.main()` |
| `benchmark.py` | 薄包装器 → `amr_dqn.cli.benchmark.main()` |
| `config.py` | 薄包装器 → `amr_dqn.cli.config.main()` |
| `visualize_forest.py` | 森林地图交互式可视化脚本 |
