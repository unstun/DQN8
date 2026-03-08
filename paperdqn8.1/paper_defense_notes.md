# 论文写作备忘：Dijkstra Cost Field 的定位与审稿人应对

> 来源：2026-03-07 对话梳理。写 Method / Experiment / Rebuttal 时参考。

---

## 1. 核心定位：RL 是全局规划器，Dijkstra 是输入特征

**错误说法**：Dijkstra 做全局规划，RL 做局部跟踪。
**正确说法**：RL 本身就是全局规划器，与 Hybrid A*、RRT* 同级对比。Dijkstra cost field 是对已知地图的预处理特征，类比 Hybrid A* 内部的启发函数 h(n)。

论文中应明确写：
> The Dijkstra cost-to-goal field serves as a **heuristic input feature** to the RL agent, analogous to the heuristic function in Hybrid A*. The RL agent itself acts as the **global path planner**, directly compared against Hybrid A*, RRT*, and LO-HA*.

## 2. 本文范围：仅全局规划

本文只做全局路径规划，不涉及局部规划/跟踪控制（MPC 等）。RL 输出的是运动学可行的完整路径，与 Hybrid A*、RRT*、LO-HA* 在同一层级直接对比。

## 3. 预见的审稿人质疑 & 应对

### Q1: "给 RL 喂 Dijkstra cost map 是不是作弊？RL 只是在跟 Dijkstra 走？"

**应对**：
- Hybrid A* 也用启发函数（non-holonomic-without-obstacles heuristic）引导搜索，本质一样
- Dijkstra 在 **无运动学约束的栅格** 上算最短路，完全不考虑：
  - 自行车运动学（最小转弯半径、转向速率限制）
  - 车辆体积（双圆碰撞足迹 vs 点占 1 格）
  - 路径平滑性（转向抖动、曲率、加速度连续性）
  - 速度规划（窄通道减速、终点刹车）
- RL 在 Dijkstra 梯度场的引导下，解决的是 **运动学可行、安全、平滑** 的全局路径生成问题
- 全局规划器使用全局地图信息（包括预处理特征）天经地义

### Q2: "为什么不直接 Dijkstra + Pure Pursuit / 跟踪控制器？"

**应对**：
- 经典跟踪器在密集障碍+窄通道中容易失败（无法处理复杂避障）
- RL 学到的策略能同时优化安全性、平滑性、效率，不只是跟路径
- 推理速度远快于 Hybrid A*（实验数据支撑）

### Q3: "Dijkstra 预计算的开销？"

**应对**：
- 最大地图 forest_a = 360×360 = 13 万格，Python heapq 约几十 ms
- 仅在 reset() 时算一次（episode 内目标/地图不变），不是每步算
- 相比 RL 推理（整个 episode 几百步 × ~1ms/步）可忽略
- 论文中 computation time 对比中可以把 Dijkstra 预处理时间包含进去，仍然远快于 Hybrid A*

## 4. 论文各章节落笔建议

| 章节 | 要点 |
|------|------|
| Method - Observation | 明确写 "global planning observation"，列出 11 标量 + 3 map channels，强调 cost-to-goal 是 heuristic feature |
| Method - Reward | 说明 cost-based progress shaping 替代 Euclidean distance，优势是 obstacle-aware |
| Experiment - Setup | 说明所有算法共享相同地图信息，公平对比 |
| Experiment - Computation | 可以把 Dijkstra 预处理时间计入 RL 的总时间，仍有优势 |
| Discussion | 可以主动讨论 "RL as global planner with heuristic features" 的定位，展示对方法的深入理解 |

---

## 5. MLP-DQN 的定位：消融基线，非"原版 DQN"

> 来源：2026-03-07 MLP vs CNN 讨论 + Codex 审核反馈。

### 5.1 核心结论

- **保留 MLP 组是合理的**，但理由是"去掉空间归纳偏置后的消融基线"，而非"原版 DQN 的另一种正统实现"。
- 经典 DQN（Mnih et al., Nature 2015 & arXiv 2013）从像素学习时用的是**卷积网络**，MLP-DQN 不是"原版 DQN"。

### 5.2 论文中的正确表述

| 错误写法 | 正确写法 |
|---------|---------|
| "the original DQN" / "standard DQN" 指代 MLP 版 | "MLP-based DQN baseline" 或 "DQN with MLP encoder" |
| "Atari 下 CNN 是必须的" | "CNN 是像素/栅格输入下的标准且更合适选择" |
| "MLP-DQN 有文献正统性" | "MLP-DQN 是同一观测下扁平编码 vs 空间编码的消融对照" |

### 5.3 实际比较的是什么

本项目观测 = 标量特征 + 地图通道（3 × N²）。两种编码方式：

- **MLP（扁平编码器）**：将标量和地图统一展平为一维向量，直接全连接，**丢失空间结构**。
- **CNN（空间编码器）**：将地图通道拆出来做卷积提取空间特征，再与标量拼接，**保留空间归纳偏置**。

所以 MLP vs CNN 的消融本质是：**扁平编码器 vs 空间编码器**，验证空间归纳偏置对路径规划任务的价值。

### 5.4 MLP-DQN 在文献中的位置

- MLP 作为 Q 网络骨干在**低维状态向量**场景（如 CartPole）中广泛使用（Stable Baselines3 的 `MlpPolicy`）。
- 在路径规划领域，输入为激光雷达距离值等向量时用 MLP 很常见；输入为栅格地图时文献一律用 CNN。
- 本项目输入含栅格地图，MLP 组的意义在于消融验证，而非声称这是标准做法。

### 5.5 已知代码/文档不一致（需修复）

- `RL系统架构文档.md:120` 写地图通道数为 2 × obs_map_size²，但代码按 3 × N² 处理（见 `amr_dqn/networks.py:15`）。论文应以代码为准。
- 代码中 `dqn` 这个 legacy alias 实际映射到 `mlp-dqn`（见 `amr_dqn/agents.py:71`），论文中应统一使用 "MLP-DQN" 命名。

### 5.6 审稿人应对

**Q: "为什么要做 MLP-DQN？原版 DQN 不就是 CNN 吗？"**

> 我们的 MLP-DQN 并非声称还原 Mnih et al. (2015) 的原始架构。它作为消融基线，用于验证空间编码器（CNN）相对于扁平编码器（MLP）在栅格地图输入下的优势。实验结果表明，CNN 变体在所有指标上均优于对应的 MLP 变体，证实了空间归纳偏置对路径规划任务的关键作用。
