# GitHub 版本变更档案（可持续追加）

更新日期：2026-02-23
仓库路径：`/home/sun/phdproject/dqn/DQN6`
用途：统一记录版本标签/提交改动，后续直接追加 `V4/V5/V6...`。

## 1) 使用约定

- 推荐每次发布都打 tag（例如 `V4`），避免只记短哈希。
- 文档统一记录两类引用：
  - 标签对象（若是 annotated tag）
  - 实际提交（`<tag>^{}` 解引用结果）
- 若版本没有 tag，可先用提交哈希记录，后续补 tag 再回填。

## 2) 快速追加流程（新增 Vx）

1. 解析版本引用（拿到真实提交）：
   - `git rev-parse Vx^{}`
2. 拉取版本提交信息和改动文件：
   - `git show -s --format='COMMIT=%H%nABBREV=%h%nAUTHOR=%an <%ae>%nDATE=%ad%nSUBJECT=%s' Vx^{}`
   - `git show --name-status --stat Vx^{}`
3. 拉取与上一个版本的区间统计：
   - `git diff --shortstat V(prev)^{}..Vx^{}`
   - `git diff --name-status V(prev)^{}..Vx^{}`
4. 在本文档追加两处：
   - `版本索引` 表新增一行
   - `版本详情` 和 `区间对比` 新增对应条目

## 3) 版本索引（当前）

| 版本 | Git 引用 | 类型 | 提交哈希 |
|---|---|---|---|
| V1 | `v1` | 轻量标签（lightweight tag） | `58195597cd4231360ff5f7da15512ef4e23916c7` |
| V2 | `V2` | 注解标签（annotated tag） | `a95ec45db39ed7293faa731f651ec181a3306e79` |
| V3 | `8bfc10d` | 普通提交（当前 `main`） | `8bfc10df0b870f361e79bcdada5567559e73a33f` |

备注：`V2` 标签对象哈希为 `4383a666d90f82001865ad8b0dba20ad92f95a5b`，`V2^{}` 对应提交为 `a95ec45...`。

## 4) 版本详情（当前）

### V1 (`5819559`)
- 提交信息：`Infer: add tqdm progress bars`（2026-01-27 01:06:03 -0900）
- 文件改动：
  - `M amr_dqn/cli/infer.py`
  - `M runtxt.md`

### V2 (`a95ec45`)
- 提交信息：`V2`（2026-02-22 19:27:34 -1000）
- 文件改动：
  - `A AGENTS.md`
  - `A CLAUDE.md`
  - `A RL系统架构文档.md`
  - `M amr_dqn/env.py`（碰撞判定由 `r` 调整为 `r + half_cell_m`）

### V3 (`8bfc10d`)
- 提交信息：`V3: EDT clearance channel for CNN + split obs (MLP keeps 2ch)`（2026-02-22 21:46:04 -1000）
- 文件改动：
  - `M amr_dqn/agents.py`（新增 `_prep_obs()`，MLP 自动裁剪观测维度）
  - `M amr_dqn/env.py`（观测新增 EDT 第 3 通道，`obs_dim` 变为 `11+3*N^2`）
  - `M amr_dqn/networks.py`（CNN 布局推断支持 `(11,3)`）
  - `A configs/repro_20260222_cnn_edt_channel.json`

## 5) 区间对比（当前）

- `v1..V2^{}`：1 个提交，4 个文件改动，`611 insertions(+), 6 deletions(-)`
- `V2^{}..8bfc10d`：1 个提交，4 个文件改动，`68 insertions(+), 7 deletions(-)`

## 6) 追加模板（复制即可）

### VX (`<tag-or-commit>`)
- 提交信息：`<subject>`（`<date>`）
- 文件改动：
  - `M/A/D <path1>`
  - `M/A/D <path2>`

区间对比新增一行：
- `V(prev)^{}..VX^{}`：`<N> commits, <files> files changed, <ins> insertions(+), <del> deletions(-)`

## 7) 复核命令

- `git rev-parse v1 V2 8bfc10d`
- `git show --name-status --stat 5819559 a95ec45 8bfc10d`
- `git diff --shortstat v1..V2^{}`
- `git diff --shortstat V2^{}..8bfc10d`
