# DQN8 项目规则（论文写作 + 工程开发）

> 作用域：`/home/sun/phdproject/dqn/DQN8/**`

## 0. 硬约束

1) 每次回复以"帅哥，"开头。
2) 改文件前输出3–7步计划+文件清单+风险+验证，等"开始"后再动手。
3) 默认中文回复；论文正文先中文写作，定稿后统一英文润色。
4) **严禁凭记忆生成 BibTeX**——未核实引用标 `[CITATION NEEDED]`。
5) `AGENTS.md`、`CLAUDE.md`、`.agents/rules/GEMINI.md` 三文件逐行一致；改后 `diff` 验证。
6) 纯文档改动豁免 `configs/` 新增规则；代码改动须在 `configs/` 新增 `repro_YYYYMMDD_<topic>.json`。
7) **消融实验留档**：结束后在 `runs/ablation_logs/` 写 `ablation_YYYYMMDD_<topic>.md`。
8) 每次可以先尝试ssh再本地计算
12) **代码搜索策略**：语义理解/探索代码库时**始终优先使用 ACE**（`mcp__augment-context-engine__codebase-retrieval`）；精确匹配标识符/字符串时使用 `Grep`（rg）。禁止用 Bash 调 grep/rg。ACE 调用若报错/超时，立即回退到 `Grep` + `Glob` 继续工作，不阻塞流程。
9) Conda 环境：`ros2py310`
10) 所有训练/推理参数通过 `configs/*.json` 管理。
11) 自检/训练/推理：`conda run --cwd /home/sun/phdproject/dqn/DQN8 -n ros2py310 python {train,infer}.py {--self-check | --profile <name>}`输出目录：`runs/`；文档：`README.md`、`runtxt.md`。

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

## 2. 论文（Elsevier Measurement）

| 项目 | 要求 |
|------|------|
| 文档类 | `elsarticle` review / twocolumn |
| 引用 | `natbib` 数字式，`\cite{}` |
| 字数 | 正文 6000–8000 词 |
| 摘要 | 150–250 词，结构化 |
| Highlights | 3–5 条，每条≤85字符 |

- **编译**：`cd paperdqn8 && latexmk -pdf -xelatex main.tex`
- **论文写作前必读** `paperdqn8/paper_todo.md`（优先级）和 `paperdqn8/paper_review.md`（审稿意见）。
- 绘图：`dqn8_plots/run_all.py`，存至 `paperdqn8/figs/`，向量格式优先，确保黑白可读。

### 引用核查（强制）

1. `search_web` / Semantic Scholar 定位 → 2. DOI 2+ 数据源确认 → 3. `curl -LH "Accept: application/x-bibtex" https://doi.org/<DOI>` → 4. 确认 claim 存在 → 失败标 `[CITATION NEEDED]`

## 3. 论文实验评估框架

### 3.1 四模式评估体系

| 模式 | 筛选条件 | 路径距离 | Runs | 汇报 KPI |
|------|----------|----------|------|----------|
| Mode 1 BK-Long | Bicycle-kinematic 可达 | ≥18m | 100 | **仅成功率** |
| Mode 2 BK-Short | Bicycle-kinematic 可达 | 6–14m | 100 | **仅成功率** |
| Mode 3 AllSuc-Long | 全算法成功 | ≥18m | ~49 | 路径长度、曲率、计算时间、综合评分 |
| Mode 4 AllSuc-Short | 全算法成功 | 6–14m | ~67 | 路径长度、曲率、计算时间、综合评分 |

- Mode 1/2 只报成功率：失败路径无有效 path length，混入平均会污染数据。
- Mode 3/4 只报路径质量：成功率恒 100%，消除成功率混杂变量。
- 推理命令中 `--filter-all-succeed` 控制 Mode 3/4 的后过滤。

### 3.2 Checkpoint 选择规则

1. **目标叙事**：CNN > MLP，PDDQN 为最优变体。
2. **选择方式**：从训练过程中选取各算法的**最佳 checkpoint**（.pt 文件），标准为推理 KPI 综合最优。
3. **MLP-DQN/DDQN 注意**：hard target update (τ=0) 易导致 Q-value explosion（约 390 轮后）；应选爆炸前的 checkpoint，确保两者数据不同。
4. **论文中说明**：checkpoint 选择写为"best validation checkpoint"或"early-stopped at peak performance"。
5. **快照归档**：每次确定权重后，将 .pt + configs + 结果存入 `runs/snapshot_YYYYMMDD_<desc>/`。

### 3.2.1 快照归档标准（参考 snapshot_20260304_4modes_v3）

**触发条件**：每次完成全模式推理、确认 checkpoint 后立即归档。

**目录结构**：

```
runs/snapshot_YYYYMMDD_<desc>/
├── README.md                        # 必须，见下方模板
├── models/                          # 必须：6 个 DRL 算法的 .pt 文件
│   └── <algo>.pt                    # 命名：mlp-dqn / mlp-ddqn / mlp-pddqn
│                                    #       cnn-dqn / cnn-ddqn / cnn-pddqn
├── infer_configs/                   # 必须：每个评估模式一个 JSON
│   └── repro_YYYYMMDD_<mode>.json
├── pairs/                           # Quality 模式必须：allsuc pairs JSON
│   └── <env>_<mode>_allsuc_pairs.json
└── results/
    └── <mode_name>/                 # 每个评估模式一个子目录
        ├── configs/run.json         # 运行时配置快照
        ├── fig12_paths.png          # 路径可视化
        ├── fig13_controls.png       # 控制量可视化
        ├── table2_kpis.csv / .md
        ├── table2_kpis_mean.csv / .md / _raw.csv
        └── （AllSuc 模式另有 *_filtered.csv / .md）
```

**README.md 必须含**：
1. 日期 + 本版与上版变更说明
2. 训练源路径（`runs/<exp>/train_<datetime>/`）
3. Checkpoint 选择表：`| 算法 | τ | 规则 | Checkpoint(ep) | 理由 |`
4. 各模式结果汇总（成功率表 + Composite Score 表）
5. 叙事验证清单（✅/❌，全 ✅ 才可用于论文）
6. 推理输出目录映射：`| 模式 | Run ID | 完整路径 |`

**归档 checklist**：
- [ ] 创建目录结构（`mkdir -p .../snapshot/{models,infer_configs,pairs,results}`）
- [ ] 复制 `.pt` 文件，按 `<algo>.pt` 规范重命名
- [ ] 复制推理配置 JSON 到 `infer_configs/`
- [ ] 复制 `allsuc_pairs.json` 到 `pairs/`（Quality 模式）
- [ ] 从推理输出目录完整复制 results（含 `configs/`、`fig*.png`、全部 CSV/MD）
- [ ] 编写 README.md（含上述 6 节）
- [ ] 叙事验证清单全 ✅，否则换 checkpoint 重跑

### 3.3 九算法对比

| 算法 | 类型 | base_algo | τ | 说明 |
|------|------|-----------|---|------|
| MLP-DQN | DRL | dqn | 0.0 (hard) | 基线 MLP |
| MLP-DDQN | DRL | ddqn | 0.0 (hard) | Double Q |
| MLP-PDDQN | DRL | ddqn | 0.01 (soft) | Polyak + Double Q |
| CNN-DQN | DRL | dqn | 0.0 (hard) | CNN 特征提取 |
| CNN-DDQN | DRL | ddqn | 0.0 (hard) | CNN + Double Q |
| CNN-PDDQN | DRL | ddqn | 0.01 (soft) | **论文主推方法** |
| Hybrid A* | 经典 | — | — | 基线规划器 |
| RRT* | 经典 | — | — | 采样规划器 |
| LO-HA* | 经典 | — | — | 优化版 Hybrid A* |

### 3.4 论文叙事约束

- **核心论点**：受约束 DRL 路径规划框架（非单一算法）优于经典规划器。
- **必须成立**：CNN-PDDQN 综合最优 > CNN-DQN/DDQN > MLP-PDDQN > MLP-DQN/DDQN；DRL 整体计算效率 10–20× 优于经典方法。
- **系统级创新**：动作掩码、Dijkstra 奖励塑形、DQfD 预训练、自行车运动学约束、双圆碰撞检测。
- **若结果不支持叙事**：换 checkpoint 或调训练参数重跑，不可篡改数据。

### 3.5 Composite Score 计算公式

```
n_pt = minmax_norm(path_time_s)       # 路径行驶时间（反映路径长度+质量）
n_k  = minmax_norm(avg_curvature_1_m) # 平均曲率（平滑性）
n_pl = minmax_norm(planning_time_s)   # 规划计算时间

base_score = (w_pt × n_pt + w_k × n_k + w_pl × n_pl) / (w_pt + w_k + w_pl)
composite_score = base_score / success_rate
```

**权重（2026-03-07 确定）**：

| 分量 | 参数名 | 权重 | 占比 |
|------|--------|------|------|
| 路径行驶时间 | `--composite-w-path-time` | **1.0** | 66.7% |
| 平均曲率 | `--composite-w-avg-curvature` | **0.3** | 20.0% |
| 规划计算时间 | `--composite-w-planning-time` | **0.2** | 13.3% |

- minmax_norm 在每个 Environment 分组内独立归一化。
- 越低越好；success_rate=0 时设为 +inf。
- 所有推理/筛选命令统一使用此权重，不再使用旧的等权(1:1:1)。

## 4. 踩坑

- **SSH 执行必须** `--cwd`：`conda run --cwd $PROJ -n $ENV python ...`
- 联网调研：WebFetch/WebSearch 不混批，每批≤2；付费墙用浏览器工具。
- LaTeX：`xelatex` 支持中文注释；提交版用 `pdflatex`；缺包 `sudo tlmgr install <pkg>`。
