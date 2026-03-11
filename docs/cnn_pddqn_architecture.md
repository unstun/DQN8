# CNN-PDDQN 完整架构分析

## 1. 观测空间（Observation）

观测向量维度：**11 + 3×N²**（N=12 → 总维度 443）

### 11 个标量特征（全部归一化到 [-1, 1]）

| # | 特征 | 含义 |
|---|------|------|
| 0 | `ax_n` | 智能体 x 坐标 |
| 1 | `ay_n` | 智能体 y 坐标 |
| 2 | `gx_n` | 目标 x 坐标 |
| 3 | `gy_n` | 目标 y 坐标 |
| 4 | `sin_psi` | 航向角正弦 |
| 5 | `cos_psi` | 航向角余弦 |
| 6 | `v_n` | 当前速度 / 最大速度 |
| 7 | `delta_n` | 当前转向角 / 最大转向角 |
| 8 | `cost_n` | Dijkstra cost-to-goal（归一化） |
| 9 | `alpha_n` | 目标相对角 / π |
| 10 | `od_n` | 最近障碍物距离（EDT 截断归一化） |

### 3 个地图通道（各 12×12 降采样，值域 [-1, 1]）

| 通道 | 含义 | 降采样方式 |
|------|------|-----------|
| Occupancy | 占据栅格 | `INTER_NEAREST`，padding=1（占据） |
| Cost-to-go | Dijkstra 到达代价场 | 双线性，padding=1（最大代价） |
| EDT | 欧氏距离变换（清除距离） | 双线性，padding=0（零清除） |

**注意**：MLP 架构会剥离第 3 通道（EDT），只使用 `11 + 2×N²` 维。CNN 使用全部 3 通道。

---

## 2. 神经网络结构（CNNQNetwork）

```
输入: flat vector [443]
  ├─ scalars [11] ──────────────────────────────────────┐
  └─ maps [432] → reshape → [3, 12, 12]                │
       │                                                │
       ▼                                                │
  Conv2d(3→32, 3×3, s=1, p=1) + ReLU   → [32, 12, 12] │
  Conv2d(32→64, 3×3, s=2, p=1) + ReLU  → [64, 6, 6]   │
  Conv2d(64→64, 3×3, s=2, p=1) + ReLU  → [64, 3, 3]   │
       │                                                │
       ▼ flatten → [576]                                │
       │                                                │
       └──────── concat ◄──────────────────────────────┘
                   │
                   ▼ [587]
         Linear(587→256) + ReLU
         Linear(256→256) + ReLU
         Linear(256→256) + ReLU
         Linear(256→35) → Q-values
```

- CNN 分支产出 64×3×3 = 576 维特征
- 与 11 维标量拼接后进入 3 层 256 单元的 MLP 头（`hidden_layers=3`）
- 输出 35 个动作的 Q 值
- 源码位置：`amr_dqn/networks.py:92`

---

## 3. 动作空间（35 离散动作）

7 个转向角速率 × 5 个加速度的网格组合（`amr_dqn/env.py:327`）：

- **转向角速率**: `[-δ̇_max, -⅔δ̇_max, -⅓δ̇_max, 0, ⅓δ̇_max, ⅔δ̇_max, δ̇_max]`（δ̇_max = 60°/s）
- **加速度**: `[-a_max, -0.5a_max, 0, 0.5a_max, a_max]`（a_max = 1.5 m/s²）

---

## 4. 自行车运动学模型（BicycleModelParams）

| 参数 | 值 |
|------|-----|
| 轴距 | 0.6 m |
| 最大速度 | 2.0 m/s |
| 最大转向角 | 27° |
| 积分步长 | dt = 0.05 s |
| 碰撞检测 | 双圆轮廓（r=0.436m） |

后轴中心自行车模型，单步欧拉积分。源码位置：`amr_dqn/env.py:314`

---

## 5. PDDQN = Polyak Double DQN

### 与 DQN/DDQN 的关键区别

| 特性 | DQN | DDQN | **PDDQN** |
|------|-----|------|-----------|
| base_algo | dqn | ddqn | **ddqn** |
| TD 目标选 action | target net | **online net** | **online net** |
| TD 目标评估 Q | target net | target net | target net |
| τ (target update) | 0 (hard) | 0 (hard) | **0.01 (soft)** |
| 目标网络更新 | 每 1000 步整体复制 | 每 1000 步整体复制 | **每步 Polyak 平滑** |

### Double DQN 逻辑（`amr_dqn/agents.py:439`）

```
a* = argmax Q_online(s', a')       # online 网络选动作
target = r + γ^n · Q_target(s', a*) # target 网络评估
```

### Polyak 软更新（`amr_dqn/agents.py:496`）

```
θ_target ← (1 - 0.01) · θ_target + 0.01 · θ_online  （每个训练步）
```

---

## 6. 训练流水线（三阶段）

```
┌───────────────────────────────────────────────────────────────┐
│  阶段 1: DQfD 专家示范预填充                                    │
│                                                                 │
│  专家策略: Cost-to-Go 贪心短视野 Rollout（非 Hybrid A*）          │
│  (config: forest_expert="auto" + forest_random_start_goal=true  │
│   → 解析为 "cost_to_go")                                        │
│                                                                 │
│  原理: 对全部 35 个动作做 horizon=15 步恒定控制量前向模拟，       │
│        在预计算的 Dijkstra cost-to-goal 代价场上评估终态代价，    │
│        过滤碰撞动作后选择 cost-to-goal 最小的动作。              │
│        不涉及图搜索，是纯贪心前向 rollout。                      │
│  实现: env.expert_action_cost_to_go → _fallback_action_short_   │
│        rollout (env.py:1598)                                     │
│                                                                 │
│  专家跑 episode → Replay（target = learning_starts × 40，       │
│  configs 中 learning_starts=5000 → ~200k target transitions）    │
│  只保存成功到达终点的 episode, demo=True                          │
├───────────────────────────────────────────────────────────────┤
│  阶段 2: 监督预训练（40k 步）                                    │
│  L = L_CE(softmax(Q), a_expert) + λ·L_margin                   │
│  每 2k 步快速评估，能到终点则提前停止                              │
│  结束后 Q_target ← Q_online                                     │
├───────────────────────────────────────────────────────────────┤
│  阶段 3: 主训练循环（N episodes）                                │
│  每步:                                                           │
│   1. ε-greedy + 动作掩码选动作                                   │
│   2. 环境 step（自行车运动学）                                    │
│   3. 存入 Replay（含 next_action_mask, demo, n_steps）           │
│   4. TD 更新:                                                    │
│      L = L_TD + λ_m·L_margin + λ_ce·L_CE                        │
│   5. 梯度裁剪（max_norm=10）                                     │
│   6. Polyak 软更新 target net (τ=0.01)                           │
│                                                                   │
│  附加机制:                                                        │
│   - 专家混入: 70%→0%（前85%轮次逐步衰减）                         │
│     (同样使用 cost_to_go 专家，非 Hybrid A*)                      │
│   - 课程学习: 起点由近到远                                        │
│   - ε 衰减: 0.9→0.01（线性，2000轮）                             │
│   - 周期性 checkpoint 保存                                        │
└───────────────────────────────────────────────────────────────┘
```

### 专家策略选择逻辑说明

代码中存在两种专家策略，由 `forest_expert` + `forest_random_start_goal` 共同决定：

| forest_expert | forest_random_start_goal | 实际专家 | 说明 |
|---------------|--------------------------|----------|------|
| `"auto"` | `true` | **`cost_to_go`** | 所有实际训练配置均走此路径 |
| `"auto"` | `false` | `hybrid_astar` | 仅存在于早期 example 配置 |
| `"hybrid_astar"` | — | `hybrid_astar` | 可手动指定但未使用 |
| `"cost_to_go"` | — | `cost_to_go` | 可手动指定但未使用 |

**Cost-to-Go 专家**（实际使用）：
- 35 个动作 × 15 步恒定控制量前向 rollout
- 在 Dijkstra 代价场上评估终态，选 cost 最小的安全动作
- 不涉及任何图搜索，计算量远小于 Hybrid A*

**Hybrid A* 引导专家**（代码存在但未实际用于训练）：
- 先计算 Hybrid A* 参考路径（每起点缓存一次）
- 然后用 pure-pursuit 风格的评分函数选择跟踪动作
- 仍非直接执行 A* 步骤，而是评分选最优离散动作

### 奖励函数（`amr_dqn/env.py:1291`）

| 分量 | 公式 | 默认系数 | 说明 |
|------|------|---------|------|
| 进展塑形 | `k_p × (cost_before - cost_after)` | 12.0 | Dijkstra cost-to-go 差值，车辆轮廓感知（取前后双圆心 max cost） |
| 时间惩罚 | `-k_t`（每步） | 0.1 | 鼓励快速到达 |
| 转向平滑 | `-k_δ × (Δδ)²` | 1.5 | 惩罚急转 |
| 加速平滑 | `-k_a × (Δa)² × v_scale` | 0.2 | 高速时惩罚更强 |
| 曲率惩罚 | `-k_κ × tan²(δ)` | 0.2 | 鼓励直线行驶 |
| 近障惩罚 | `-k_o × (1/od - 1/safe_d)` | 1.5 | 当 od < 0.20m 时触发 |
| 速度-间距耦合 | `-k_v × clearance_ratio × v_ratio²` | 2.0 | 低间距时惩罚高速 |
| 碰撞终端 | `-200` | — | |
| 到达终端 | `+350 ~ +400` | — | 含目标接近度加分 |
| 卡住终端 | `-300` | — | 窗口化 stuck 检测（20步内位移<0.02m 且速度<0.05m/s） |

**进展塑形的 cost-to-go 计算**：`_cost_to_goal_pose_m()` 对双圆轮廓的前后圆心分别查询 Dijkstra 代价场，取 `max(cost_front, cost_rear)`，确保整车安全进展（`env.py:2062`）。

### 损失函数

```
L = L_TD(Huber) + λ_margin · L_margin(demo) + λ_CE · L_CE(demo)
```

- `L_TD`：Smooth L1 (Huber) 损失
- `L_margin`：`max(0, Q(s,a_other) + 0.8 - Q(s,a_expert))`，仅作用于 demo 转移
- `L_CE`：交叉熵行为克隆损失，仅作用于 demo 转移

**注意**：DDQN/PDDQN 的 TD target 计算中，online 网络和 target 网络均对 `next_action_mask` 做 `masked_fill`（`agents.py:442-446`），即无效动作被排除在 argmax 和 Q 评估之外。这是与标准 DDQN 的一个重要区别。

### Replay Buffer

- 均匀采样，容量 100k
- demo 转移受保护（DQfD safeguard：demo 槽位不被非 demo 数据覆盖）
- 存储字段：`(obs, action, reward, next_obs, done, next_action_mask, demo, n_steps)`
- 支持 n-step returns（默认 `n_step=1`，即标准 1-step TD；n>1 时在 episode 内累积折扣回报）
- 源码位置：`amr_dqn/replay_buffer.py`

### 超参数默认值

| 参数 | 值 |
|------|-----|
| γ (discount) | 0.995 |
| learning_rate | 5e-4 |
| batch_size | 128 |
| replay_capacity | 100,000 |
| hidden_dim | 256 |
| hidden_layers | 3 |
| grad_clip_norm | 10.0 |
| ε_start / ε_final / ε_decay | 0.9 / 0.01 / 2000 |
| optimizer | Adam |

---

## 7. 推理流水线（两阶段）

```
┌──────────────────────────────────────────────────┐
│  Phase 1: DQN 全局路径规划                         │
│                                                    │
│  加载 checkpoint → 贪心 rollout                    │
│  动作选择管线 (forest_select_action):               │
│   1. Q = Q_online(obs)                             │
│   2. a0 = argmax Q                                 │
│   3. a0 可行？→ 返回                               │
│   4. 否 → top-k 候选搜索                           │
│   5. 否 → admissible_mask 上 argmax Q              │
│   6. 末策 → 启发式短视野 rollout                    │
│                                                    │
│  输出：全局路径 waypoints                           │
├──────────────────────────────────────────────────┤
│  Phase 2: MPC 路径跟踪                             │
│                                                    │
│  采样式 MPC (256 候选) 跟踪 DQN 规划路径            │
│  连续控制 (δ̇, a) → step_continuous                 │
│  输出：实际执行轨迹 + KPI                           │
└──────────────────────────────────────────────────┘
```

### 动作选择管线详解（`amr_dqn/forest_policy.py`）

训练与推理共享同一管线，唯一区别：

| | 训练 | 推理 |
|--|------|------|
| explore | True（ε-greedy） | False（纯贪心） |
| min_progress | 0（只检碰撞） | 1e-4（碰撞+进展） |

管线步骤：
1. ε 探索触发时 → 从 admissible mask 中随机选
2. 贪心 `a0 = argmax Q`
3. `a0` 通过 `is_action_admissible()` → 返回
4. 失败 → top-k（k=10）候选逐个检查
5. 全失败 → 全动作 admissible_mask + masked argmax Q
6. 仍无 → `_fallback_action_short_rollout()` 启发式

### 可行性检查（`admissible_action_mask`，`amr_dqn/env.py:1953`）

- 对每个动作做 horizon=15 步恒定控制量 rollout
- 检查：(1) 无碰撞 (2) 障碍物距离 ≥ min_od_m (3) cost-to-goal 减少 ≥ min_progress_m
- Fallback 链：
  1. 无前进可行动作 → 允许倒车动作（`v_end < -0.10 m/s`）（`env.py:1986`）
  2. 仍无 → 回退到仅碰撞安全的动作子集（`fallback_to_safe=True`）

---

## 8. 系统级创新总结

| 创新 | 作用 | 源码位置 |
|------|------|---------|
| **动作掩码** | 短视野 rollout 滤除碰撞动作 | `env.py:1934` |
| **Dijkstra 奖励塑形** | 障碍物感知的 cost-to-goal 引导进展 | `env.py:1238` |
| **DQfD 预训练** | 专家示范 bootstrap Q 网络 | `agents.py:361` |
| **自行车运动学** | 后轴自行车模型保证运动学可行性 | `env.py:351` |
| **双圆碰撞检测** | 保守但高效的碰撞近似 | `env.py:416` |
| **两阶段推理** | DQN 全局规划 + MPC 精确跟踪 | `cli/infer.py:302` |

---

## 9. 九算法对比矩阵

| 算法 | 架构 | base_algo | τ | 特点 |
|------|------|-----------|---|------|
| MLP-DQN | MLP | dqn | 0.0 | 基线，hard target update |
| MLP-DDQN | MLP | ddqn | 0.0 | Double Q，hard update |
| MLP-PDDQN | MLP | ddqn | 0.01 | Polyak + Double Q |
| CNN-DQN | CNN | dqn | 0.0 | CNN 特征提取 |
| CNN-DDQN | CNN | ddqn | 0.0 | CNN + Double Q |
| **CNN-PDDQN** | **CNN** | **ddqn** | **0.01** | **论文主推方法** |
| Hybrid A* | 经典 | — | — | 基线规划器 |
| RRT* | 经典 | — | — | 采样规划器 |
| LO-HA* | 经典 | — | — | 优化版 Hybrid A* |
