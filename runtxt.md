# Run commands + parameter guide

This repo exposes two CLIs (wrappers for `amr_dqn/cli/*`):

- `conda run -n ros2py310 python train.py ...`
- `conda run -n ros2py310 python infer.py ...`

By default both CLIs use CUDA (`--device cpu` to force CPU).

Below are example commands, followed by a parameter-by-parameter reference explaining what each flag does and how to tune it.

## 0) Quick self-check (device/CUDA)

Use this to verify CUDA/PyTorch setup before a long run:

```bash
conda run -n ros2py310 python train.py --self-check
conda run -n ros2py310 python infer.py --self-check
```

## 1) Train (forest env: bicycle dynamics + DQfD-style stabilizers)

Notes for `forest_*` envs:

- Forest uses a fixed internal map resolution of **0.1 m/cell** (so `--cell-size` is ignored for forest).
- Default forest training already enables:
  - action masking ("shield") for non-expert actions
  - Hybrid A* expert mixing (creates `demo=True` transitions)
  - DQfD-style demo losses on `demo=True` transitions (large-margin + behavior cloning)
  - optional start-state curriculum (`--forest-curriculum`, on by default)

Baseline multi-map run (Fig.13-style curves; saves `models/<env>/{mlp-dqn.pt,mlp-ddqn.pt}`):

```bash
conda run -n ros2py310 python train.py --envs forest_a forest_b forest_c forest_d --out outputs_forest --rl-algos mlp-dqn mlp-ddqn --episodes 1000
```

Single-map DQfD-style run (fast iteration; similar to `runs/outputs_forest_dqfd{1,2}` configs):

```bash
conda run -n ros2py310 python train.py --envs forest_b --out outputs_forest_dqfd2 --episodes 600 --no-progress --no-timestamp-runs --no-forest-curriculum --forest-expert-prob-start 1.0 --forest-expert-prob-final 0.0 --forest-expert-prob-decay 0.7
```

Optional: add a short expert replay prefill before learning starts (closer to original "DQfD"):

```bash
python train.py ... --forest-demo-prefill
```

## 2) Inference + plots/KPIs

`--models` accepts: experiment name/dir, train run dir, or a `models/` dir.

Important: keep `--sensor-range` and `--n-sectors` consistent with training (or checkpoint loading can fail due to observation-dimension mismatch).

Use an experiment name (auto-picks latest train run under `runs/<name>/...`):

```bash
conda run -n ros2py310 python infer.py --envs forest_a forest_b forest_c forest_d --models outputs_forest --out outputs_forest --runs 1
```

Forest DQfD-style inference (picks the latest models under `runs/outputs_forest_dqfd2/...`):

```bash
conda run -n ros2py310 python infer.py --envs forest_b --models outputs_forest_dqfd2 --out outputs_forest_dqfd2_infer --runs 1
```

Or point directly at a specific models dir (useful if you trained with `--no-timestamp-runs`):

```bash
conda run -n ros2py310 python infer.py --envs forest_b --models runs/outputs_forest_dqfd2/models --out outputs_forest_dqfd2_infer --runs 1 --device cpu --no-timestamp-runs
```

Include classical baselines (Hybrid A* + RRT*) in the same KPI table + path plots:

```bash
conda run -n ros2py310 python infer.py --envs forest_a forest_b forest_c forest_d --models outputs_forest --out outputs_forest --baselines all --baseline-timeout 5 --hybrid-max-nodes 200000 --rrt-max-iter 5000
```

Baseline-only (no checkpoints required):

```bash
conda run -n ros2py310 python infer.py --envs forest_b --out outputs_forest_baselines --baselines all --skip-rl --baseline-timeout 5 --hybrid-max-nodes 200000 --rrt-max-iter 5000 --device cpu
```

---

# Training Pipeline (V9)

本节描述 V9（`repro_20260223_v9_raw_replay_soft_shield`）的完整训练流程。
代码入口：`amr_dqn/cli/train.py` → `train_one()`。

## 总览

```
阶段 0: 初始化           → Agent + Replay Buffer + Normalizer
阶段 1: Demo 预填充      → 专家轨迹填入 Replay（~5000 条 transition）
阶段 2: 监督预训练        → DQfD 风格 CE+Margin 损失（40000 步）
阶段 3: 主训练循环        → 300 轮 episode，混合专家+ε-greedy + TD 学习
阶段 4: 模型选择与保存    → final / best / pretrain 三候选 PK
```

## 阶段 0：初始化

```python
# train.py:359-376
agent = DQNFamilyAgent(algo, obs_dim, n_actions, config, seed, device)
rew_normalizer = RunningRewardNormalizer(clip=5.0)   # Welford 在线均值/方差
```

| 组件 | 说明 |
|------|------|
| Q 网络 + Q_target | MLP (3×256) 或 CNN；target 初始与 online 同步 |
| Replay Buffer | 容量 100K，存 (obs, action, reward, next_obs, done, demo, next_mask) |
| Reward Normalizer | V9: 只跟踪统计量，normalize 在采样时做 |
| Eval 场景池 | 预采样 5 组固定 (start, goal)，保证评估可复现 |

## 阶段 1：Demo 预填充

**目的**：冷启动——Q 网络一开始什么都不会，纯随机探索几乎到不了终点。

```python
# train.py:436-514
# 用 Hybrid A* / cost-to-go 专家跑若干轮，目标填 ~5000 条 transition
while demo_added < demo_target:
    obs, _ = env.reset(random_start_goal)      # 随机起终点
    while not done:
        action = expert_action()                # 专家控制
        next_obs, reward, done, ... = env.step(action)
        ep.append((obs, action, reward, ...))
    if reached:                                 # 只保留成功 episode
        for transition in ep:
            rew_normalizer.update(reward)        # 更新统计量
            reward = clip(reward, -5, 5)         # 存 raw（安全裁剪）
            agent.observe(..., demo=True)         # 标记为 demo
```

关键点：
- 只保存成功到达终点的 episode（失败的专家轨迹丢弃）
- `demo=True` 标记使后续 DQfD 损失只作用于这些 transition
- Reward 存原始值，normalizer 只更新统计量（V9 改进）

## 阶段 2：监督预训练（DQfD 风格）

**目的**：在 TD 学习开始前让 Q 网络具备基本导航能力。

```python
# train.py:516-553
# 40000 步监督学习
agent.pretrain_on_demos(steps=40000)
```

每步做什么：
1. 从 Replay 采样 batch=128，取 `demo=True` 的 transition
2. **Cross-Entropy 损失**：`CE(softmax(Q(s)), a_expert)` — 让 Q 值分布对准专家动作
3. **Margin 损失**：`max(0, Q(s,a_other) + 0.8 - Q(s,a_expert))` — 确保专家动作 Q 值最高

每 2000 步做快速评估：如果 greedy 策略已能到达终点则提前停止。
完成后同步 `Q_target = Q`，并保存 pretrain 快照作为 checkpoint 候选。

## 阶段 3：主训练循环（300 轮）

每轮 episode 的完整流程：

### 3.1 环境 Reset

```python
# train.py:668-683
env.reset(seed=seed+ep, options={"random_start_goal": True, "rand_min_cost_m": 6.0})
```

每轮随机采样 (start, goal)，最小 cost 距离 6m，全随机（`fixed_prob=0`）。

### 3.2 逐步交互（最多 600 步）

```
每步执行三层动作选择门控：
│
├─ 第一层：专家探索？
│   p_exp = 0.4 × linear_decay(ep, decay=0.5)
│   随机数 < p_exp → 调用 Hybrid A*/cost-to-go 专家
│
└─ 第二层：forest_select_action(training_mode=True)
    │
    ├─ ε-greedy 触发？(ε: 0.9→0.01，线性衰减 250 轮)
    │   是 → 从 admissible_mask（碰撞检查，无前进检查）随机选
    │
    └─ 否 → Q 网络贪心
        ├─ argmax Q 是 admissible？→ 用它
        ├─ top-10 候选逐个检查 → 用第一个 admissible
        ├─ full mask fallback → masked argmax Q
        └─ 最终兜底 → 短距 rollout 启发式
```

`training_mode=True`（V9 关键改进）：
- 只检查碰撞安全（`min_progress_m=0`）
- **不**检查"是否前进"，让 Q 网络从 reward 信号自己学前进
- 推理时 `training_mode=False`，恢复前进检查

### 3.3 Reward 处理（V9）

```python
# train.py:733-738
rew_normalizer.update(float(reward))          # 更新 Welford 统计量
reward = clip(reward, -5, 5)                  # 存入 replay 的是 raw 值
```

V9 改进：**存 raw，采样时归一化**。
- V8 在此处做 `reward = normalizer.normalize(reward)` 导致 replay 中 reward 随统计量漂移
- V9 只更新统计量，normalize 推迟到 `agent.update()` 采样时

### 3.4 Episode 结束后：存 Replay + TD 更新

```python
# train.py:758-774
# 1. 批量存入 replay buffer
for transition in ep_buffer:
    agent.observe(..., demo=(used_expert and reached))  # 只有成功的专家步标 demo

# 2. 执行累积的 TD 更新（每 4 步一次，需 global_step > 5000）
for _ in range(pending_updates):
    loss_info = agent.update(rew_normalizer=rew_normalizer)
```

`agent.update()` 内部（`agents.py:399-492`）：

```
采样 batch=128 from Replay
    │
    ├─ rewards = normalize_tensor(rewards)   ← V9: 用当前统计量归一化
    │
    ├─ TD target:
    │   DQN:  r + γ^n × max_a Q_target(s', a)
    │   DDQN: r + γ^n × Q_target(s', argmax_a Q_online(s', a))
    │   (next_action_mask 过滤不安全动作)
    │
    ├─ TD 损失:    SmoothL1(Q(s,a), target)
    ├─ Margin 损失: max(0, Q(s,a_other)+0.8 - Q(s,a_demo))  [仅 demo 样本]
    ├─ CE 损失:    CrossEntropy(Q(s), a_demo)                [仅 demo 样本]
    ├─ Total = TD + 1.0×Margin + 1.0×CE
    │
    ├─ 梯度裁剪 max_norm=10.0
    ├─ Adam 更新 lr=5e-4
    │
    └─ Target 网络更新:
        DQN/DDQN: 硬替换（每 1000 步）
        PDDQN:    Polyak 软更新（τ=0.005，每步）
```

### 3.5 评估与 Checkpoint

- **每 10 轮**在 5 组固定场景做 greedy eval（`explore=False, training_mode=False`）
- 记录 success_rate / avg_return / planning_cost
- 维护 `best_q`：按 reached > timeout > collision 排序的最佳 episode checkpoint

## 阶段 4：模型选择与保存

```python
# train.py:903-924
# 三候选 greedy eval PK:
#   final_q:    第 300 轮结束时的网络
#   best_q:     训练中 episode_score 最高的快照
#   pretrain_q: 监督预训练结束时的快照
# 选 greedy eval 得分最高的 → 保存为 .pt
```

## Reward 构成

每步 reward 由多个分量叠加（`env.py:1175-1220`）：

| 分量 | 系数 | 含义 |
|------|------|------|
| `k_p × (cost_before − cost_after)` | 12.0 | **前进奖励**（接近终点=正值） |
| `−k_t` | −0.1 | 时间惩罚（每步固定） |
| `−k_δ × (Δδ)²` | −1.5 | 转向平滑性 |
| `−k_a × (Δa)² × v²` | −0.2 | 加速平滑性 |
| `−k_κ × tan(δ)²` | −0.2 | 曲率惩罚 |
| `−k_o × 1/od` | −1.5 | 靠近障碍物惩罚（OD < 安全距离时） |
| `−k_v × speed_coupling` | −2.0 | 近障碍物高速惩罚 |
| 碰撞 | **−200** | 终止 |
| 到达终点 | **+400** | 终止 |
| 卡住 | −stuck_penalty | 终止 |

典型成功 episode 的 return ≈ 150~350（取决于路径长度和平滑度）。

## 关键时间线（V9 / 300 轮）

```
Episode:  0 ─────── 50 ────── 100 ────── 150 ────── 200 ────── 250 ────── 300
              │          │          │          │          │          │
Expert:    40% →     ~28% →     ~16% →      ~8% →      ~3% →      ~1% →    ~0%
Epsilon:   0.9 →     0.72 →     0.54 →     0.37 →     0.19 →     0.01 →   0.01
              │                                                       │
              └── Q 网络逐渐接管 ─────────────────────────────────→ 完全自主
```

## 六种算法的区别

| 算法 | 架构 | TD Target | Target 更新 |
|------|------|-----------|-------------|
| MLP-DQN | MLP 3×256 (2ch) | max_a Q_target(s',a) | 硬替换/1000步 |
| MLP-DDQN | MLP 3×256 (2ch) | Q_target(s', argmax Q_online) | 硬替换/1000步 |
| MLP-PDDQN | MLP 3×256 (2ch) | Q_target(s', argmax Q_online) | Polyak τ=0.005 |
| CNN-DQN | CNN+FC (3ch) | max_a Q_target(s',a) | 硬替换/1000步 |
| CNN-DDQN | CNN+FC (3ch) | Q_target(s', argmax Q_online) | 硬替换/1000步 |
| CNN-PDDQN | CNN+FC (3ch) | Q_target(s', argmax Q_online) | Polyak τ=0.005 |

MLP 用 2 通道（occupancy + cost-to-go），CNN 用 3 通道（+ EDT clearance）。
PDDQN = Polyak Double DQN（在 DDQN 基础上把硬替换换成软更新）。

---

# Parameter reference

Run `conda run -n ros2py310 python train.py --help` / `conda run -n ros2py310 python infer.py --help` to see the same flags from the CLI help output.

## A) Training (`conda run -n ros2py310 python train.py ...`)

Training always runs **both** algorithms (DQN and DDQN) for every env in `--envs`, so runtime scales roughly with:

`2 * len(envs) * episodes * max_steps`.

### A.1 Experiment / output paths

- `--out` (default: `outputs`): Experiment name or directory.
  - If you pass a *bare name* (e.g. `outputs_forest`), it resolves to `runs/<out>/` (see `--runs-root`).
  - If you pass a path (e.g. `runs/myexp` or `D:\runs\myexp`), it is used as-is.
- `--runs-root` (default: `runs`): Base directory for bare experiment names in `--out`.
  - Adjust when you want all results under a different root folder.
- `--timestamp-runs` / `--no-timestamp-runs` (default: enabled): Controls whether each run gets its own timestamped subfolder.
  - Enabled: writes to `runs/<out>/train_<timestamp>/...` and updates `runs/<out>/latest.txt`.
  - Disabled: writes directly into `runs/<out>/...` (risk of overwriting mixed outputs).

### A.2 Environment selection + episode length

- `--envs` (default: `forest_a forest_b forest_c forest_d`): Which maps to train on (space-separated).
  - Options: `forest_a forest_b forest_c forest_d`.
  - Adjust for faster iteration by training on a single env, e.g. `--envs forest_b`.
- `--episodes` (default: `1000`): Episodes per env per algorithm.
  - Increase for more learning; decrease for quick smoke tests.
- `--max-steps` (default: `600`): Max steps per episode before time-limit truncation.

### A.3 Observation / geometry parameters

These affect the environment observation vector and therefore model checkpoint compatibility.

- `--sensor-range` (default: `6`): Forest lidar range in **meters**.
  - Adjust to change how far obstacles are "seen". If you change it, retrain and keep inference consistent.
- `--n-sectors` (default: `36`): Forest-only lidar sectors (36=10°, 72=5°). Ignored for non-forest envs.
  - Larger values increase observation dimension (`obs_dim = 11 + n_sectors`) and compute cost.
  - Must match between training and inference.
- `--cell-size` (default: `1.0`): Grid env cell size in meters (affects physical scaling of distance/OD terms). Ignored for forest envs.
  - Adjust only if you intentionally rescale the grid map physics/reward units.

### A.4 Reproducibility

- `--seed` (default: `0`): Base seed for RNGs (the CLI derives per-episode/per-algorithm seeds from this).
  - Change it to get a different stochastic run; keep fixed for repeatability.

### A.5 Runtime / device

- `--device` (default: `cuda`): Torch device selection.
  - `auto`: use CUDA if available else CPU
  - `cuda`: require CUDA (error if not available)
  - `cpu`: force CPU
- `--cuda-device` (default: `0`): CUDA device index when using CUDA (`--device=cuda` or `--device=auto` with CUDA available).
- `--self-check` (default: off): Print torch/CUDA runtime info and exit (use to verify CUDA setup).
- `--progress` / `--no-progress` (default: auto): Show a training progress bar.
  - Default is on when running in a TTY (interactive terminal), otherwise off.

### A.6 Training schedule (affects learning speed + stability)

- `--train-freq` (default: `4`): Perform one gradient update every N environment steps (after `--learning-starts`).
  - Smaller = more updates (slower, often more stable); larger = fewer updates (faster, may under-train).
- `--learning-starts` (default: `2000`): Number of environment steps to collect before starting updates.
  - If you run very short jobs (few episodes / small `--max-steps`), reduce this too, otherwise training may do zero updates.
  - Forest `--forest-demo-prefill` uses this value as the demo prefill target size.

### A.7 Plot smoothing (does not affect training)

- `--ma-window` (default: `20`): Moving-average window for reward curve plotting (`1` = raw).
- `--eval-every` (default: `0`): Run a **greedy** evaluation rollout every N episodes and write `training_eval.csv` (useful for smoother “policy gets stronger” curves).
  - `0` disables evaluation logging (fastest).
  - Evaluation uses `explore=False` and the same forest admissible-action logic as inference (mask + fallback).
- `--eval-runs` (default: `5`): Number of evaluation rollouts per eval point.
  - For `--forest-random-start-goal`, this is the fixed `(start, goal)` batch size sampled once and reused across training.
- `--eval-score-time-weight` (default: `0.5`): Time weight (m/s) for `planning_cost` in `training_eval.{csv,xlsx}`:
  - `planning_cost = (avg_path_length + w * inference_time_s) / max(success_rate, eps)`
  - When `--eval-every > 0`, training also writes `training_eval_metrics.png` (success_rate/avg_steps/avg_return/planning_cost vs episode).

### A.8 Forest-only stabilizers (recommended to keep enabled for forest)

- `--forest-curriculum` / `--no-forest-curriculum` (default: enabled): Start-state curriculum.
  - Early training starts closer to the goal; later training shifts back toward the canonical start.
  - Disable if you want all episodes to start from the canonical start state (harder early training).
- `--curriculum-band-m` (default: `2.0`): Start-state sampling band width (meters) for curriculum starts.
  - Smaller = narrower difficulty band; larger = more variety (can include easier starts even later).
- `--curriculum-ramp` (default: `0.35`): Fraction of total episodes needed to ramp the curriculum back to the canonical start.
  - Smaller = reaches canonical start sooner (harder sooner, less train/test mismatch).
  - Larger = stays in "easy-start" regime longer (easier early learning, but can create mismatch).
- `--forest-demo-prefill` / `--no-forest-demo-prefill` (default: enabled): Prefill replay buffer with Hybrid A* expert rollouts before learning starts.
  - Enable when forest training collapses to degenerate behaviors (e.g., stop/jitter) early in training.
- `--forest-demo-horizon` (default: `15`): Expert horizon steps (constant action) used by the Hybrid A* guided expert (prefill + exploration).
  - Larger = more lookahead (slower, often safer); smaller = faster but more myopic.
- `--forest-demo-w-clearance` (default: `0.8`): Clearance weight in the expert score (`-cost_to_go + w_clearance * min_clearance`).
  - Increase to bias expert toward safer paths; decrease to bias toward shortest cost-to-go.
- `--forest-expert-exploration` / `--no-forest-expert-exploration` (default: enabled): Mix Hybrid A* expert into the behavior policy early in training.
  - When enabled, non-expert actions also use action masking (safety/progress shield).
  - Disable only if you want pure epsilon-greedy behavior (usually much less stable in forest).
- `--forest-expert-prob-start` (default: `0.7`): Probability of taking the expert action at the start of training.
  - Increase for more expert guidance early; decrease for more on-policy exploration.
- `--forest-expert-prob-final` (default: `0.0`): Expert probability at the end of training.
  - Keep near `0.0` if you want the final policy to be purely learned (no expert dependence).
- `--forest-expert-prob-decay` (default: `0.6`): Fraction of episodes over which expert probability decays from start -> final.
  - Smaller = decays faster; larger = uses expert longer.

## B) Inference (`conda run -n ros2py310 python infer.py ...`)

### B.1 Model source + output paths

- `--models` (default: `outputs`): Where to load models from.
  - Can be: an experiment name (bare), an experiment directory, a specific training run directory, or a `models/` directory.
  - For a bare experiment name `X`, inference loads from the latest run under `runs/X/.../models/`.
- `--out` (default: `outputs`): Output experiment name/dir for inference results.
  - If `--out` points to the same experiment as `--models`, inference outputs are stored under that training run: `<train_run>/infer/<timestamp>/...`.
- `--runs-root` (default: `runs`): Base directory for bare experiment names in `--models` / `--out`.
- `--timestamp-runs` / `--no-timestamp-runs` (default: enabled): Timestamp inference outputs to avoid mixing results.

### B.2 Environment / observation parameters (must match training)

- `--envs` (default: `forest_a forest_b forest_c forest_d`): Which envs to evaluate.
- `--max-steps` (default: `600`): Rollout horizon.
- `--sensor-range` (default: `6`): Same meaning as training. Must match the training setting for the checkpoints you are loading.
- `--n-sectors` (default: `36`): Forest-only; must match training.
- `--cell-size` (default: `1.0`): Ignored for forest envs (kept for backwards compatibility).

### B.3 KPI averaging / randomness

- `--runs` (default: `5`): Number of rollout repeats per env per algorithm for averaging KPIs.
  - Increase for less variance; decrease for faster evaluation.
- `--seed` (default: `0`): Base seed for rollouts (the CLI offsets it per algorithm/run internally).

### B.4 Runtime / device

- `--device` (default: `cuda`): Same as training.
- `--cuda-device` (default: `0`): Same as training.
- `--self-check` (default: off): Same as training.
- `--progress` / `--no-progress` (default: auto): Show an inference progress bar.
  - Default is on when running in a TTY (interactive terminal), otherwise off.
- `--kpi-time-mode` (default: `rollout`): How `inference_time_s` is measured for RL rollouts.
  - `rollout`: full rollout wall-clock time (includes `env.step`)
  - `policy`: action-selection compute time only (Q forward + admissibility checks)

### B.5 Baselines (Hybrid A* + RRT*)

- `--baselines` (default: none): Add classical planners to the KPI table/plots.
  - Options: `hybrid_astar`, `rrt_star`, or `all`.
- `--skip-rl` (default: off): Run baselines without loading DQN/DDQN checkpoints.
- `--baseline-timeout` (default: `5.0`): Per-planner time budget (seconds).
- `--hybrid-max-nodes` (default: `200000`): Hybrid A* node budget.
- `--rrt-max-iter` (default: `5000`): RRT* iteration budget.
- `--forest-baseline-rollout` (default: enabled): Forest-only; roll out a baseline tracking controller on planned paths and include tracking compute in `inference_time_s` (disable with `--no-forest-baseline-rollout`).
  - `--forest-baseline-mpc-candidates` (default: `256`): MPC samples per control step.
  - `--forest-baseline-save-traces` (default: disabled): Save per-run executed baseline trajectories to `<run_dir>/traces/*.csv` (forest-only).

---

## C) Code-level hyperparameters (not CLI flags yet)

The DQN/DDQN hyperparameters are defined in `amr_dqn/agents.py` as `AgentConfig`. To change them:

1) Edit `amr_dqn/agents.py` (`AgentConfig` defaults), then retrain.
2) Keep training + inference consistent (a checkpoint trained with one config may not load if you later change network sizes).

Key fields (defaults shown):

- `gamma=0.995`: Discount factor (higher = longer-horizon credit assignment; forest needs high gamma).
- `learning_rate=5e-4`: Adam learning rate (higher = faster/less stable; lower = slower/more stable).
- `replay_capacity=100_000`: Replay buffer size (larger = more diverse replay, more memory).
- `batch_size=128`: Minibatch size per update (larger = smoother gradients, more compute).
- `target_update_steps=1000`: Hard target update period (in gradient updates).
- `grad_clip_norm=10.0`: Gradient clipping threshold (reduce if you see unstable spikes).
- `eps_start=0.9`, `eps_final=0.01`, `eps_decay=2000`: Exploration schedule (start, end, and decay rate/episodes).
- `hidden_layers=2`, `hidden_dim=128`: MLP size (bigger = more capacity, slower).
- `demo_margin=0.8`, `demo_lambda=1.0`, `demo_ce_lambda=1.0`: DQfD-style demo losses (used only on `demo=True` transitions from forest expert mixing/prefill).




forest_a

Train: python train.py --envs forest_a --out forest_a_rand --episodes 1000 --max-steps 600 --forest-random-start-goal --forest-rand-fixed-prob 0 --forest-expert auto --device cuda
Infer: python infer.py --envs forest_a --models forest_a_rand --out forest_a_rand_eval --baselines all --random-start-goal --runs 20 --rand-fixed-prob 0 --kpi-time-mode policy --max-steps 600 --device cuda
forest_b

Train: python train.py --envs forest_b --out forest_b_rand --episodes 1000 --max-steps 600 --forest-random-start-goal --forest-rand-fixed-prob 0 --forest-expert auto --device cuda
Infer: python infer.py --envs forest_b --models forest_b_rand --out forest_b_rand_eval --baselines all --random-start-goal --runs 20 --rand-fixed-prob 0 --kpi-time-mode policy --max-steps 600 --device cuda
forest_c

Train: python train.py --envs forest_c --out forest_c_rand --episodes 1000 --max-steps 600 --forest-random-start-goal --forest-rand-fixed-prob 0 --forest-expert auto --device cuda
Infer: python infer.py --envs forest_c --models forest_c_rand --out forest_c_rand_eval --baselines all --random-start-goal --runs 20 --rand-fixed-prob 0 --kpi-time-mode policy --max-steps 600 --device cuda
forest_d

Train: python train.py --envs forest_d --out forest_d_rand --episodes 1000 --max-steps 600 --forest-random-start-goal --forest-rand-fixed-prob 0 --forest-expert auto --device cuda
Infer: python infer.py --envs forest_d --models forest_d_rand --out forest_d_rand_eval --baselines all --random-start-goal --runs 20 --rand-fixed-prob 0 --kpi-time-mode policy --max-steps 600 --device cuda



python train.py --envs forest_a --out forest_a_gap_rand --episodes 1000 --max-steps 1000 --forest-random-start-goal --forest-rand-fixed-prob 0 --forest-expert cost_to_go --device cuda


python infer.py --envs forest_a --models runs/forest_a_gap_rand/train_20260121_071731 --out forest_a_gap_rand --baselines all --random-start-goal --runs 4 --rand-fixed-prob 0 --kpi-time-mode policy --max-steps 1000 --device cuda
python infer.py --envs forest_a --models runs\forest_a_gap_rand\train_20260121_075305 --out forest_a_gap_rand --baselines all --random-start-goal --runs 4 --rand-fixed-prob 0 --kpi-time-mode policy --max-steps 1000 --device cuda
runs\forest_a_gap_rand\train_20260121_075305
runs\forest_a_gap25_rev\train_20260121_095505

python infer.py --envs forest_a --models runs\forest_a_gap25_rev\train_20260121_095505 --out forest_a_gap_rand --baselines all --random-start-goal --runs 4 --rand-fixed-prob 0 --kpi-time-mode policy --max-steps 1000 --device cuda --seed 30

python train.py --envs forest_a --out forest_a_gap375_masked --episodes 2000 --max-steps 1000 --obs-map-size 24 --forest-random-start-goal --forest-rand-fixed-prob 0 --forest-expert cost_to_go --forest-demo-pretrain-steps 80000 --device cuda

python infer.py --envs forest_a --models runs/forest_a_gap375_masked/train_20260121_105536 --out forest_a_gap375_masked_eval --baselines all --random-start-goal --runs 4 --rand-fixed-prob 0 --kpi-time-mode policy --max-steps 1000 --obs-map-size 24 --device cuda --seed 30


cd d:\BaiduSyncdisk\study\phdprojec\dqn
python train.py --envs forest_a --out forest_a_gap30_rev_12_23am --episodes 300 --max-steps 1000 --forest-random-start-goal --forest-rand-min-cost-m 6 --forest-rand-max-cost-m 0 --forest-rand-fixed-prob 0 --forest-rand-tries 200 --forest-expert cost_to_go --no-forest-expert-exploration --forest-demo-pretrain-steps 80000 --learning-starts 5000 --device cuda --cuda-device 0 --seed 0 --eval-every 10 --eval-runs 5 --eval-score-time-weight 0.5
runs\forest_a_gap30_rev_12_23am\train_20260122_002417
conda run -n ros2py310 python infer.py --envs forest_a --models runs\forest_a_gap30_rev_12_23am\train_20260122_002417 --out runs\forest_a_gap30_rev_12_23am\train_20260122_002417--random-start-goal --runs 4 --rand-min-cost-m 6 --rand-max-cost-m 0 --rand-fixed-prob 0 --rand-tries 200 --rand-reject-unreachable --kpi-time-mode policy --max-steps 1000 --device cuda --cuda-device 0 --seed 21

python infer.py --envs forest_a --models "runs\forest_a_gap30_rev_12_23am\train_20260122_002417" --out "runs\forest_a_gap30_rev_12_23am\train_20260122_002417" --random-start-goal --runs 4 --rand-min-cost-m 6 --rand-max-cost-m 0 --rand-fixed-prob 0 --rand-tries 200 --rand-reject-unreachable --kpi-time-mode policy --max-steps 1000 --device cuda --cuda-device 0 --seed 23
runs\forest_a_gap30_rev_4_03am\train_20260122_040328
python infer.py --envs forest_a --models "runs\forest_a_gap30_rev_4_03am\train_20260122_040328" --out "runs\forest_a_gap30_rev_4_03am\train_20260122_040328" --random-start-goal --runs 4 --rand-min-cost-m 6 --rand-max-cost-m 0 --rand-fixed-prob 0 --rand-tries 200 --rand-reject-unreachable --kpi-time-mode policy --max-steps 1000 --device cuda --cuda-device 0 --seed 23
python train.py --envs forest_a --out forest_a_gap30_rev_4_03am --episodes 10000 --max-steps 1000 --forest-random-start-goal --forest-rand-min-cost-m 6 --forest-rand-max-cost-m 0 --forest-rand-fixed-prob 0 --forest-rand-tries 200 --forest-expert cost_to_go --no-forest-expert-exploration --forest-demo-pretrain-steps 80000 --learning-starts 5000 --device cuda --cuda-device 0 --seed 0 --eval-every 10 --eval-runs 5 --eval-score-time-weight 0.5
