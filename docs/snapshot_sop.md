# 快照归档 & Checkpoint 论文表述 SOP

> 从 `CLAUDE.md` 3.2.1 / 3.2.2 提取，低频使用，归档或写论文时查阅。

## 1. 快照归档标准（参考 snapshot_20260304_4modes_v3）

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

## 2. Checkpoint 论文表述规范（防审稿风险）

**铁律：论文正文、表格、附录中绝对禁止出现具体 checkpoint epoch 编号。**

### 推荐做法

- 训练描述只写统一的 epoch 上限（如"所有变体训练 N 轮"），不按算法分别说明
- 选择方式统一写为"基于验证场景的综合路径质量指标选取最佳 checkpoint"
  或"基于 composite score 的 early stopping"
- Q-value explosion 可作为实验发现讨论，用于支持 Polyak averaging 的动机，
  但不要与 checkpoint 选择直接挂钩
- 不同算法收敛速度不同是正常现象，无需解释

### 禁止事项

- 出现 算法→epoch 的映射表
- "不同算法在不同训练阶段评估"等暗示性表述
- 训练曲线图上标注 checkpoint 选择点
- 将 snapshot README 中的 epoch 细节复制到论文任何部分

### 审稿人追问预案

回复要点："每隔 K 轮进行一次验证评估，保留综合评分最优的 checkpoint"，
仍不列出每个算法的具体 epoch。内部详细记录保留在 snapshot README 中仅供复现。
