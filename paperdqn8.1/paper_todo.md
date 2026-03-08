# 论文当前待解决问题清单

> 本文件跟踪 CNN-PDDQN 论文（paperdqn8/main.tex）的改进任务。
> 完整审稿意见见 `paperdqn8/paper_review.md`。
> **每次新窗口开始论文写作前必须先读此文件。**

---

## 当前阶段：大改 — 重写论文故事线

### P0 — 正在进行

- [X] **切换到英文 + Elsevier Measurement 模板**

  - 当前状态：中文 + IEEEtran，需改为英文 + `elsarticle`
  - 包括：标题、摘要、正文、图注、表格全部英文化
  - 模板：`\documentclass[review]{elsarticle}` + `natbib`
- [ ] **重写论文 story / narrative**

  - 核心矛盾：标题方法 CNN-PDDQN 在真实场景成功率最低（70-75%），MLP-DDQN 反而最优
  - 需要重新定位 contribution：
    - 方案A：改标题为通用框架名（如"DRL-based motion planning framework"），CNN-PDDQN 只是变体之一
    - 方案B：保留 CNN-PDDQN 标题，但深入分析 sim-to-real gap 并提出改进方向
    - **待用户决定方案**
- [ ] **重写 Introduction — 从堆砌到 gap analysis**

  - 当前问题：纯罗列"A做了X，B做了Y"，无批判性分析
  - 需要：明确指出现有方法的3个具体 gap → 本文如何填补
  - 必须讨论：为什么选 DQN 而非 PPO/SAC

### P1 — 大改完成后处理

- [ ] 增加 training curve（仓库有 `dqn8_plots/plot_training.py`）
- [ ] 扩充实验样本量 + 统计显著性检验
- [ ] 写完整的 Discussion + Limitations 章节
- [ ] 超参数敏感性分析
- [ ] 消融实验补充分析（动作掩码消融、CNN vs MLP 特征分析）

### P2 — 投稿前 polish

- [ ] 图表矢量化（PNG → PDF）
- [ ] 修复符号不一致（θ vs α 等）
- [ ] 增加 DRL baseline 对比（PPO/SAC）
- [ ] 真实场景措辞修正（非实车部署，是基于真实地图的仿真）
- [ ] 填写作者信息、Acknowledgments

---

## 关键决策记录

| 日期       | 决策                         | 备注               |
| ---------- | ---------------------------- | ------------------ |
| 2026-03-02 | 论文审稿意见完成             | 见 paper_review.md |
| 2026-03-02 | P0 优先：切换英文 + 重写故事 | 标题方法定位待定   |
