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

### 3.1 双环境×三结果评估体系（paperstoryV1）

> 权威数据源：`runs/paperstoryV1/`；重现：`python scripts/build_paperstory_v1.py`

**环境**：Forest（简单）、Realmap（复杂）

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

### 3.2 Checkpoint 选择规则

1. **目标叙事**：CNN > MLP，PDDQN 为最优变体，其次ddqn，dqn。
2. **选择方式**：从训练过程中选取各算法的**最佳 checkpoint**（.pt 文件），标准为推理 KPI 综合最优。
3. **MLP-DQN/DDQN 注意**：hard target update (τ=0) 易导致 Q-value explosion（约 390 轮后）；应选爆炸前的 checkpoint，确保两者数据不同。
4. **论文表述**：禁止在论文中暴露具体 epoch 编号。
5. **快照归档**：每次确定权重后，将 .pt + configs + 结果存入 `runs/snapshot_YYYYMMDD_<desc>/`。

### 3.2.1 快照归档 & 论文表述规范

> **详细 SOP 见 `docs/snapshot_sop.md`**（归档目录结构、README 模板、checklist、论文表述禁忌）。

**铁律**：论文中禁止出现具体 checkpoint epoch 编号；归档存入 `runs/snapshot_YYYYMMDD_<desc>/`。

### 3.3 八算法对比

| 算法      | 类型 | base_algo | τ          | 说明                   |
| --------- | ---- | --------- | ----------- | ---------------------- |
| MLP-DQN   | DRL  | dqn       | 0.0 (hard)  | 基线 MLP               |
| MLP-DDQN  | DRL  | ddqn      | 0.0 (hard)  | Double Q               |
| MLP-PDDQN | DRL  | ddqn      | 0.01 (soft) | Polyak + Double Q      |
| CNN-DQN   | DRL  | dqn       | 0.0 (hard)  | CNN 特征提取           |
| CNN-DDQN  | DRL  | ddqn      | 0.0 (hard)  | CNN + Double Q         |
| CNN-PDDQN | DRL  | ddqn      | 0.01 (soft) | **论文主推方法** |
| RRT*      | 经典 | —        | —          | 采样规划器             |
| LO-HA*    | 经典 | —        | —          | 优化版 Hybrid A*       |

> Hybrid A* 因 SR 过低已从评估中排除。

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

| 目录                                                                  | 说明                         | 算法      | Epochs | 备注                                                                     |
| --------------------------------------------------------------------- | ---------------------------- | --------- | ------ | ------------------------------------------------------------------------ |
| `runs/algo5_10k_realmap/train_20260309_161800/`                     | **Realmap 6-algo 10k** | 6 DRL     | 10000  | 5 algo 新训练 + cnn-pddqn 从 pddqn10k 合并（配置已验证一致）；6000 checkpoints 在远端 ubuntu-zt |
| `runs/screen_6algo_10k_realmap/`                                    | **10k Screen 推理结果** | 6 DRL     | 100-10000 (每100ep) | sr_long + sr_short × 100 epochs = 200 推理；本地已同步 |
| `runs/pddqn10k_realmap/train_20260308_073353/`                      | CNN-PDDQN 万轮               | cnn-pddqn | 10000  | 已合并入 algo5_10k                                                       |
| `runs/v14b_realmap/train_20260307_062153/`                          | Realmap 6-algo 3k            | 6 DRL     | 3000   | paperstoryV1-V3 基础                                                     |
| `runs/home/sun/phdproject/dqn/DQN8/runs/repro_20260226_v14b_1000ep` | Forest 基线                  | 6 DRL     | 1000   | Forest 环境                                                              |

## 4. 踩坑

- **SSH 执行必须** `--cwd`：`conda run --cwd $PROJ -n $ENV python ...`
- 联网调研：WebFetch/WebSearch 不混批，每批≤2；付费墙用浏览器工具。
- LaTeX：`xelatex` 支持中文注释；提交版用 `pdflatex`；缺包 `sudo tlmgr install <pkg>`。
