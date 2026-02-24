# DQN6/AGENTS.md（Ubuntu 24.04 + ros2py310）

> 作用域：`/home/sun/phdproject/dqn/DQN6/**`。通用环境参考上层 `../AGENTS.md`。

## 0. 硬约束

1) 每次回复以"帅哥，"开头。
2) 改文件前输出3–7步计划+文件清单+风险+验证，等"开始"后再动手。
3) 默认中文回复。
4) DQN/DDQN等术语遵循原论文；不确定先查证。
5) 代码改动后在`configs/`新增`repro_YYYYMMDD_<topic>.json`；纯文档豁免。
6) `AGENTS.md`与`CLAUDE.md`逐行一致；改后`diff -u AGENTS.md CLAUDE.md`验证。
7) 训练/推理优先SSH远端，先同步代码再训练。

## 1. 常用命令

- 自检/训练/推理：`conda run --cwd /home/sun/phdproject/dqn/DQN6 -n ros2py310 python {train,infer}.py {--self-check | --profile <name>}`
- 输出目录：`runs/`；文档：`README.md`、`runtxt.md`。

## 2. 踩坑

### 2.1 SSH远端执行

- **必须**加`--cwd`：`conda run --cwd /home/sun/phdproject/dqn/DQN6 -n ros2py310 python train.py ...`
- `~/.bashrc`的conda init须在`case $- in`前（2026-02-21已修，复发查`~/.bashrc.bak.*`）。

### 2.2 Write/Edit限制

- 单次≤50行，超过须拆分调用。

### 2.3 联网调研

- WebFetch/WebSearch不混批；每批同类≤2；优先arXiv/GitHub HTML版。
- 付费墙403：Playwright（`browser_navigate`→`browser_wait_for`5s→`browser_snapshot`）。

### 2.4 SSH远端工作流

`REMOTE=ubuntu-zt` / `PROJ=/home/sun/phdproject/dqn/DQN6` / `ENV=ros2py310`

0. **同步代码**：`rsync -av --exclude='runs/' --exclude='__pycache__/' $PROJ/ $REMOTE:$PROJ/`
1. **启动**：`ssh $REMOTE 'cd $PROJ; nohup bash -lc "conda run --cwd $PROJ -n $ENV python train.py --profile $PROFILE && conda run --cwd $PROJ -n $ENV python infer.py --profile $PROFILE" > runs/${PROFILE}_$(date +%Y%m%d_%H%M%S).log 2>&1 & echo PID=$!'`
2. **按需检查**：`ssh $REMOTE "ls $PROJ/runs/$EXP/train_*/infer/*/table2_kpis.csv 2>/dev/null && echo DONE || echo RUNNING"`
3. **回填**：`rsync -av --partial $REMOTE:$PROJ/runs/$EXP/train_<ts>/ $PROJ/runs/$EXP/train_<ts>/`；更新`latest.txt`
4. **校验**：远端/本地文件数一致 + `table2_kpis.csv`存在 + `latest.txt`正确
