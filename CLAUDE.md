# DQN8 项目规则（论文写作 + 工程开发）

> 作用域：`/home/sun/phdproject/dqn/DQN8/**`

## 0. 硬约束

1) 每次回复以"帅哥，"开头。
2) 改文件前输出3–7步计划+文件清单+风险+验证，等"开始"后再动手。
3) 默认中文回复；论文正文先中文写作，定稿后统一英文润色。README 等项目文档也用中文撰写。
4) **严禁凭记忆生成 BibTeX**——未核实引用标 `[CITATION NEEDED]`。
5) `AGENTS.md`、`CLAUDE.md`、`.agents/rules/GEMINI.md` 三文件逐行一致；改后 `diff` 验证。
6) 纯文档改动豁免 `configs/` 新增规则；代码改动须在 `configs/` 新增 `repro_YYYYMMDD_<topic>.json`。
7) **消融实验留档**：结束后在 `runs/ablation_logs/` 写 `ablation_YYYYMMDD_<topic>.md`。
8) 每次可以先尝试ssh再本地计算
9) **代码搜索策略**：语义理解/探索代码库时**始终优先使用 ACE**（`mcp__augment-context-engine__codebase-retrieval`）；精确匹配标识符/字符串时使用 `Grep`（rg）。禁止用 Bash 调 grep/rg。ACE 调用若报错/超时，立即回退到 `Grep` + `Glob` 继续工作，不阻塞流程。
10) Conda 环境：`ros2py310`
11) 所有训练/推理参数通过 `configs/*.json` 管理。
12) 自检/训练/推理：`conda run --cwd /home/sun/phdproject/dqn/DQN8 -n ros2py310 python {train,infer}.py {--self-check | --profile <name>}`输出目录：`runs/`；文档：`README.md`、`runtxt.md`。
13) 每次需要联网时使用 Playwright（`npx playwright`）。联网搜索默认使用 DuckDuckGo MCP。

## 1. 常用命令

```bash
PROJ=/home/sun/phdproject/dqn/DQN8; ENV=ros2py310

# 训练（后台）
nohup conda run --cwd $PROJ -n $ENV python train.py --profile $PROFILE \
  > runs/${PROFILE}_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# 推理
conda run --cwd $PROJ -n $ENV python infer.py --profile $PROFILE

# 检查完成
ls $PROJ/runs/$EXP/train_*/infer/*/table2_kpis.csv 2>/dev/null && echo DONE || echo RUNNING
```

### 术语规范

| 中文                              | 英文                | 禁用                 | 说明                                                                                       |
| --------------------------------- | ------------------- | -------------------- | ------------------------------------------------------------------------------------------ |
| Dijkstra代价场 / 到达目标的代价场 | goal distance field | cost-to-go field/map | Dijkstra 预计算的绕障最短路径距离场；"cost-to-go"在 DRL 语境下易与 RL 值函数混淆，禁止使用 |

- 观测通道描述：occupancy, **goal distance**, EDT clearance
- 奖励塑形描述：potential function defined as the geodesic goal distance
- 代码变量名（`cost_to_go` 等）暂不改，仅论文正文执行此规范

### 引用核查（强制）

1. `search_web` / Semantic Scholar 定位 → 2. DOI 2+ 数据源确认 → 3. `curl -LH "Accept: application/x-bibtex" https://doi.org/<DOI>` → 4. 确认 claim 存在 → 失败标 `[CITATION NEEDED]`

## 3. 论文实验评估框架

### 3.1 双环境×三结果评估体系

**环境**：Realmap（复杂）

**每环境 3 类结果**：

| 结果          | 筛选         | 路径距离                  | Runs    | 汇报 KPI                            |
| ------------- | ------------ | ------------------------- | ------- | ----------------------------------- |
| SR            | BK 可达      | Long ≥18m / Short 6–14m | 100     | **仅成功率**                  |
| Quality Long  | 8 算法全成功 | ≥18m                     | ~5–53  | 路径长度、曲率、计算时间、Composite |
| Quality Short | 8 算法全成功 | 6–14m                    | ~43–65 | 路径长度、曲率、计算时间、Composite |

- **8-algo filter**：排除 Hybrid A*（SR 过低），保留 6 DRL + RRT* + LO-HA*。
- **SR 结果**只报成功率，不看路径质量（失败路径无有效指标）。
- **Quality 结果**只报路径质量，成功率恒 100%（消除混杂变量）。
- 归一化：逐 pair 在 8 算法内 minmax，然后取均值。

1. gs + 结果存入 `runs/snapshot_YYYYMMDD_<desc>/`。

### 3.2.1 快照归档 & 论文表述规范

> **详细 SOP 见 `docs/snapshot_sop.md`**（归档目录结构、README 模板、checklist、论文表述禁忌）。

**铁律**：论文中禁止出现具体 checkpoint epoch 编号；归档存入 `runs/snapshot_YYYYMMDD_<desc>/`。

### 3.4 论文叙事约束

- **核心论点**：受约束 DRL 路径规划框架（非单一算法）优于经典规划器。
- **尽量满足**（非严格约束，太严格会导致无法筛选 checkpoint）：
  CNN 优于 MLP（同变体），PDDQN 优于 DDQN 优于 DQN（同架构）；
  DRL 整体计算效率显著优于经典方法。
  个别指标或模式不满足可接受，整体趋势成立即可。
- **系统级创新**：动作掩码、Dijkstra 奖励塑形、DQfD 预训练、自行车运动学约束、双圆碰撞检测。
- **若结果不支持叙事**：换 checkpoint 或调训练参数重跑，不可篡改数据。

### 3.5 Composite Score 与指标优先级

```text
n_pl = minmax_norm(path_length_m)     # 路径长度（最核心质量指标）
n_k  = minmax_norm(avg_curvature_1_m) # 平均曲率（平滑性）
n_ct = minmax_norm(planning_time_s)   # 规划计算时间

base_score = (w_pl × n_pl + w_k × n_k + w_ct × n_ct) / (w_pl + w_k + w_ct)
composite_score = base_score / success_rate
```

**权重（2026-03-08 确定）**：

| 优先级 | 分量         | 参数名                          | 权重          | 占比  |
| ------ | ------------ | ------------------------------- | ------------- | ----- |
| 1      | 路径长度     | `--composite-w-path-length`   | **1.0** | 55.6% |
| 2      | 平均曲率     | `--composite-w-avg-curvature` | **0.6** | 33.3% |
| 3      | 规划计算时间 | `--composite-w-planning-time` | **0.2** | 11.1% |

- minmax_norm 在每个 Environment 分组内独立归一化。
- 越低越好；success_rate=0 时设为 +inf。
- 所有推理/筛选命令统一使用此权重，不再使用旧的等权(1:1:1)。
- **SR 结果**：仅评价成功率，不看路径质量指标。
- **Quality 结果**：按上表优先级排序决策，路径长度 > 曲率 > 计算时间。

### 3.6 关键训练目录

| 目录                                                                  | 说明                          | 算法      | Epochs              | 备注                                                                                            |
| --------------------------------------------------------------------- | ----------------------------- | --------- | ------------------- | ----------------------------------------------------------------------------------------------- |
| `runs/algo5_10k_realmap/train_20260309_161800/`                     | **Realmap 6-algo 10k**  | 6 DRL     | 10000               | 5 algo 新训练 + cnn-pddqn 从 pddqn10k 合并（配置已验证一致）；6000 checkpoints 在远端 ubuntu-zt |
| `runs/screen_6algo_10k_realmap/`                                    | **10k Screen 推理结果** | 6 DRL     | 100-10000 (每100ep) | sr_long + sr_short × 100 epochs = 200 推理；本地已同步                                         |
| `runs/pddqn10k_realmap/train_20260308_073353/`                      | CNN-PDDQN 万轮                | cnn-pddqn | 10000               | 已合并入 algo5_10k                                                                              |
| `runs/v14b_realmap/train_20260307_062153/`                          | Realmap 6-algo 3k             | 6 DRL     | 3000                | paperstoryV1-V3 基础                                                                            |
| `runs/home/sun/phdproject/dqn/DQN8/runs/repro_20260226_v14b_1000ep` | Forest 基线                   | 6 DRL     | 1000                | Forest 环境                                                                                     |
| `runs/abl_md_*/train_*/`                                            | **消融实验 9 变体训练** | 9 DRL     | 5000                | MHA/Dueling 消融，详见 §3.7                                                                    |
| `runs/abl_md_infer_*/`                                              | **消融实验推理结果**    | 9 DRL     | —                  | 每变体 sr_long + sr_short 各 50 runs                                                            |
| `paperruns/ablation/`                                               | **消融实验汇总入口**    | —        | —                  | 符号链接，按变体分 train/infer 子目录                                                           |

### 3.7 消融实验（MHA / Dueling）

> 数据位置：`paperruns/ablation/<variant>/{train,infer}`

**消融矩阵**（2 base × 4 模块配置 + MLP 基线 = 9 变体）：

| 变体          | base algo          | MHA (4 heads) | Dueling | cnn_drop_edt |
| ------------- | ------------------ | ------------- | ------- | ------------ |
| mlp           | mlp-dqn + mlp-ddqn | ✗            | ✗      | false        |
| cnn_dqn       | cnn-dqn            | ✗            | ✗      | true         |
| cnn_ddqn      | cnn-ddqn           | ✗            | ✗      | true         |
| cnn_dqn_mha   | cnn-dqn            | ✓            | ✗      | true         |
| cnn_ddqn_mha  | cnn-ddqn           | ✓            | ✗      | true         |
| cnn_dqn_duel  | cnn-dqn            | ✗            | ✓      | true         |
| cnn_ddqn_duel | cnn-ddqn           | ✗            | ✓      | true         |
| cnn_dqn_md    | cnn-dqn            | ✓            | ✓      | true         |
| cnn_ddqn_md   | cnn-ddqn           | ✓            | ✓      | true         |

**训练参数**（统一）：

- 环境：realmap_a，5000 episodes，seed=0
- save_every=50（100 个 checkpoint），Checkpoint 自动选择（3 候选 greedy 评估）
- 其他超参与主线一致（sensor_range=6, obs_map_size=12, n_sectors=36 等）

**推理参数**（统一）：

- 每变体 × 2 场景：sr_long（≥18m）、sr_short（6–14m）
- runs=50，seed=42（所有变体共享相同起终点对）
- baselines=[]（纯 DRL 互比，无经典规划器）
- 直接比较原始指标（SR、路径长度、曲率、规划时间），不做 minmax 归一化

## 5. 远程服务器

| 优先级 | 名称               | Host           | 用户   | 密码             | GPU             | 说明                                    |
| ------ | ------------------ | -------------- | ------ | ---------------- | --------------- | --------------------------------------- |
| 1      | uhost-1nwalbarw6ki | 117.50.216.203 | ubuntu | g7TXK26Q85Jp493f | RTX 4090 (24GB) | 租用 GPU 服务器，Conda ros2py310 已部署 |
| 2      | ubuntu-zt          | (ZeroTier)     | sun    | —               | —              | 长期训练服务器，存放 6000+ checkpoints  |

- 连接方式：优先用 paramiko（本地无 sshpass）
- 远端项目路径：`$HOME/DQN8/`
- 远端 Conda：`$HOME/miniconda3/bin/conda`，环境 `ros2py310`

## 4. 踩坑

- **SSH 执行必须** `--cwd`：`conda run --cwd $PROJ -n $ENV python ...`
- 联网调研：WebFetch/WebSearch 不混批，每批≤2；付费墙用浏览器工具。
- LaTeX：`xelatex` 支持中文注释；提交版用 `pdflatex`；缺包 `sudo tlmgr install <pkg>`。
