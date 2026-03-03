# DQN8 — DQN/DDQN Path Planning for Autonomous Mobile Robots

基于深度强化学习（DQN/DDQN/PDDQN）的移动机器人路径规划，支持森林环境（程序化生成）和真实地图环境（PGM栅格）。

## 目录结构

```text
DQN8/
├── train.py / infer.py          # 顶层入口（薄包装器）
├── benchmark.py / config.py     # 辅助入口
├── visualize_forest.py          # 森林地图可视化
│
├── amr_dqn/                     # 核心 Python 包
│   ├── agents.py                #   DQN/DDQN/PDDQN 智能体
│   ├── env.py                   #   Gymnasium 环境（AMRGridEnv + AMRBicycleEnv）
│   ├── networks.py              #   Q 网络（MLP + CNN）
│   ├── replay_buffer.py         #   经验回放（含 DQfD demo 保护）
│   ├── reward_norm.py           #   Welford 在线奖励归一化
│   ├── forest_policy.py         #   统一动作选择管线
│   ├── schedules.py             #   epsilon 衰减调度
│   ├── smoothing.py             #   Chaikin 路径平滑
│   ├── metrics.py               #   路径 KPI 计算
│   ├── config_io.py             #   JSON 配置加载
│   ├── runtime.py               #   PyTorch/CUDA 设置
│   ├── runs.py                  #   实验目录管理
│   │
│   ├── cli/                     #   命令行入口
│   │   ├── train.py             #     训练主循环
│   │   ├── infer.py             #     推理 + KPI 评估 + 出图
│   │   ├── benchmark.py         #     训练→推理编排
│   │   ├── config.py            #     配置模板生成
│   │   └── precompute_forest_paths.py  # 专家路径缓存
│   │
│   ├── maps/                    #   地图定义
│   │   ├── __init__.py          #     MapSpec 协议 + 环境注册
│   │   ├── forest.py            #     程序化森林生成
│   │   ├── pgm.py               #     ROS PGM 加载器
│   │   └── precomputed/         #     Hybrid A* 预计算路径
│   │
│   ├── baselines/               #   经典规划基线
│   │   └── pathplan.py          #     Hybrid A* / RRT* 封装
│   │
│   └── third_party/pathplan/    #   内置路径规划库
│       ├── hybrid_a_star/       #     Hybrid A*（Reeds-Shepp）
│       ├── rrt/                 #     RRT*
│       ├── geometry.py          #     2D 碰撞检测
│       ├── robot.py             #     Ackermann 运动学
│       └── map_utils.py         #     GridMap 工具
│
├── configs/                     # 实验配置 JSON
│   ├── paper_forest_a.json      #   论文用 forest_a 配置
│   ├── paper_realmap_a.json     #   论文用 realmap_a 配置
│   └── _archive/                #   历史配置存档
│
├── dqn8_plots/                  # 独立绘图套件
│   ├── run_all.py               #   一键绘图入口
│   ├── plot_training.py         #   训练曲线
│   ├── plot_paths.py            #   路径对比图
│   ├── plot_kpi_bars.py         #   KPI 柱状图
│   ├── plot_kpi_radar.py        #   KPI 雷达图
│   ├── plot_kpi_table.py        #   KPI 表格图
│   └── plot_map_overview.py     #   地图总览
│
├── realmap/                     # 真实地图数据
│   ├── map_a.pgm                #   栅格占据图
│   └── map_a.yaml               #   地图元数据
│
├── paper/                       # 论文 LaTeX 源码
├── runs/                        # 实验输出（.gitignore）
│   └── ablation_logs/           #   消融实验记录
│
└── docs/                        # 补充文档
```

## 环境

- Conda 环境：`ros2py310`
- 默认 CUDA（强制 CPU：`--device cpu`）

自检（确认 PyTorch/CUDA 可用）：

```bash
conda run -n ros2py310 python train.py --self-check
conda run -n ros2py310 python infer.py --self-check
```

## Quick Start

### 训练

```bash
# 使用 profile（推荐，读取 configs/<name>.json）
conda run -n ros2py310 python train.py --profile paper_forest_a

# 后台训练（带日志）
nohup conda run -n ros2py310 python train.py --profile paper_forest_a \
  > runs/paper_forest_a_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### 推理

```bash
conda run -n ros2py310 python infer.py --profile paper_forest_a
```

### 检查完成

```bash
ls runs/paper_forest_a/train_*/infer/*/table2_kpis.csv 2>/dev/null && echo DONE || echo RUNNING
```

### 绘图

```bash
python dqn8_plots/run_all.py \
  --base-dir runs/paper_forest_a/train_XXXXXXXX_XXXXXX/infer/XXXXXXXX_XXXXXX \
  --train-dir runs/paper_forest_a/train_XXXXXXXX_XXXXXX \
  --out-dir figures/
```

## 支持的环境

| 环境名 | 类型 | 说明 |
| --- | --- | --- |
| `forest_a` | 程序化森林 | 360x360 大图，85棵树，宽间距 |
| `forest_b` | 程序化森林 | 96x96 小图，28棵树，紧间距 |
| `forest_c` | 程序化森林 | 160x160 中图，85棵树，密间距 |
| `forest_d` | 程序化森林 | 96x96 小图，28棵树，宽间距 |
| `realmap_a` | PGM 真实地图 | 来自 ROS 栅格占据图 |

## 支持的算法

| 算法标识 | 说明 |
| --- | --- |
| `mlp-dqn` | MLP Q网络 + 标准 DQN |
| `mlp-ddqn` | MLP Q网络 + Double DQN |
| `mlp-pddqn` | MLP Q网络 + Polyak 目标更新 DDQN |
| `cnn-dqn` | CNN Q网络 + 标准 DQN |
| `cnn-ddqn` | CNN Q网络 + Double DQN |
| `cnn-pddqn` | CNN Q网络 + Polyak 目标更新 DDQN |

基线：Hybrid A*、RRT*（自动在推理时对比）。

## 配置

所有训练/推理参数通过 `configs/*.json` 管理。结构：

```json
{
  "train": { "envs": ["forest_a"], "rl_algos": ["mlp-dqn", "cnn-ddqn"], "episodes": 1000, ... },
  "infer": { "envs": ["forest_a"], "baselines": ["all"], "runs": 10, ... }
}
```

生成完整参数模板：`conda run -n ros2py310 python config.py --stdout`

更完整的命令示例与参数说明见：`runtxt.md`。
