# 模拟审稿意见：CNN-PDDQN 论文

> **角色**：以 Elsevier *Measurement* / IEEE 级别 Reviewer 的视角审阅
> **审阅对象**：[main.tex](file:///home/sun/phdproject/dqn/DQN8/paperdqn8/main.tex)（IEEEtran 双栏，7 页）

---

## 总体印象

论文提出了一套"先验点云建图 → 可通行性评估 → CNN-PDDQN 路径规划"的完整 pipeline，idea 本身有一定实际意义（森林狭窄通道 + 阿克曼约束）。但**当前稿件在写作深度、实验设计、学术规范等方面存在严重不足，离可投稿状态还有较大距离。**

> [!CAUTION]
> 以下按"致命问题 → 重大问题 → 中等问题 → 小问题"分级，建议优先处理致命和重大问题。

---

## 🔴 致命问题（Major — 必须解决否则直接拒稿）

### 1. 论文语言：全中文，但 GEMINI.md 要求投 Elsevier Measurement（英文期刊）

- 整篇论文正文、标题、摘要、图注全是中文
- 如果目标期刊是 *Measurement* 或任何英文 SCI 期刊，**必须全英文撰写**
- 当前文档类用的还是 `IEEEtran`，如果投 Elsevier 应该用 `elsarticle`

### 2. 标题方法 CNN-PDDQN 在真实场景中并非最优，自我矛盾

这是**最致命的学术问题**：

| 场景 | CNN-PDDQN 成功率 | 实际最优 |
|------|----------------|--------|
| 真实短路径 | **0.75** | MLP-PDDQN / MLP-DDQN (0.90) |
| 真实长路径 | **0.70** | MLP-DDQN (0.95) |

- 论文标题叫 "CNN-PDDQN"，但真实场景中 CNN-PDDQN 是 **所有 DRL 变体中成功率最低的**
- 摘要却写"真实林地长路径中 MLP-DDQN 成功率达 95%"——等于承认标题方法不是最好的
- 审稿人会直接质疑：**你的 contribution 到底是 CNN-PDDQN 还是 MLP-DDQN？为什么标题方法反而最差？**

> [!IMPORTANT]
> 必须重新定位 contribution：要么改标题为通用框架名（而非具体 CNN-PDDQN），要么增加分析解释为什么 CNN 在 sim 中好但迁移到 real 时退化，并提出解决方案。

### 3. Introduction 文献综述堆砌严重，缺乏批判性分析

- 当前 Introduction 的结构是：列出 A 做了什么、B 做了什么、C 做了什么……纯罗列
- **完全没有**对现有方法的局限性进行深入分析
- 没有回答关键问题：为什么现有 DRL 方法不能直接用于森林场景？具体困难是什么？
- "传统方法优点是原理成熟，缺点是路径易不连续、易陷局部最优"——这种一句话概括过于笼统，审稿人会要求具体化

### 4. 缺少 Related Work 专节或深入对比

- 只有 Introduction 中的简单罗列，没有深入比较 "我和他们的本质区别是什么"
- 对 DRL-based 路径规划的 survey（如 PPO/SAC/TD3 等 actor-critic 方法）**完全没有讨论**
- 没有讨论为什么选 DQN 而不是 PPO/SAC，这是审稿人必问的问题

### 5. 没有 Training Curve / Learning Dynamics 展示

- 整篇论文没有任何训练曲线图（loss curve, reward curve, success rate vs. episodes）
- 代码仓库中有 `plot_training.py`，显然可以生成，但论文没有使用
- 审稿人会质疑：训练是否收敛？需要多少 episodes？不同算法收敛速度如何？

---

## 🟠 重大问题（Major — 需要显著修改）

### 6. 实验样本量太小、缺少统计显著性分析

- 仿真实验仅用 **10 组**起终点，真实场景实验也只有 **20 组**（短 + 长各 10）
- 没有标准差、置信区间、显著性检验（t-test, Wilcoxon）
- 表格中数字精度到小数点后三位，但 sample size 只有 10，统计意义存疑
- 如 CNN-PDDQN 在真实短路径成功率 0.75 对比 MLP-PDDQN 的 0.90，10 次测试中只差 1.5 次——可能完全是随机波动

### 7. Discussion 几乎不存在

- 当前 "结果与讨论" 部分实际上 **只有结果描述，没有深入讨论**
- 缺少以下关键讨论：
  - 为什么 CNN 在仿真中优于 MLP，但在真实地图中明显退化？
  - 动作掩码和可行域过滤到底贡献了多少？（没有消融）
  - 方法的 limitations 有哪些？（完全没提）
  - 计算复杂度分析（网络参数量、推理 FLOPs）
  - 与 SOTA DRL 导航方法（如 DRLnav, BARN Challenge 等）的定位关系

### 8. 消融实验设计反直觉且分析不充分

- **奖励消融**：基线 A（无 goal shaping + 无速度约束）反而最好？这意味着你论文中大篇幅描述的奖励设计可能是无效的
  - 审稿人会问：你最终系统用的到底是哪个奖励配置？和论文 Methods 中描述的一致吗？
- **DQN+MPC 消融**：纯 DQN 比 DQN+MPC 好，这很有趣但分析太浅
  - "DQN 路径不保证可行性" 是结论而非原因分析——需要展示失败 case 的轨迹图

### 9. 真实场景验证仅为仿真复放，非实车部署

- 论文声称"真实林地实验"，但实际上是**在真实地图上跑仿真**，而非实际 UGV 部署
- 这是一个严重的 claim 不匹配问题——介绍了硬件平台（Fig.2），但实验没有任何实车数据
- 需要明确标注这是 "基于真实林地地图的仿真验证" 而非 "real-world deployment"

---

## 🟡 中等问题（Minor — 建议修改以提升质量）

### 10. 图表质量

- **Fig.5 (CNN-PDDQN 框架图)**：标题写 "SCI DIAGRAM"、包含 "Forest (Bicycle)" 等内部标注，不适合发表
- **Fig.6 (伪代码)**：以截图形式嵌入，而非用 LaTeX `algorithm` 环境——当前已有 algorithm 环境但输出为 PNG？
- **路径对比图（Fig.7, 9, 14）**：轨迹线太细，legend 太小，建议增大 marker 和字号
- **柱状图（Fig.8, 10, 15）**：x 轴标签倾斜后较难读，建议改为水平排列或缩写
- 所有图均为 PNG 位图——发表级论文应用 **PDF/EPS 矢量图**

### 11. 公式和符号不一致

- 式(3) 中用 $\alpha_{ij}$，但前文定义的是 $\theta_{ij}$，符号不一致
- MDP 四元组写 $M=(S,A,P,R)$ 但全文没有定义转移概率 $P$
- "综合得分" 公式中 $\tilde{L}, \tilde{\bar{\kappa}}, \tilde{T}$ 三者等权，但为什么选等权没有说明

### 12. 缺少超参数敏感性分析

- 奖励函数中有 $k_p, k_t, k_\delta, k_a, k_\kappa, k_o, k_{obs,max}, d_s, \varepsilon$ 等 **至少 9 个超参数**
- 这些参数如何选择的？敏感性如何？完全没有讨论
- 这些参数跨场景迁移时是否需要重新调整？

### 13. 对比算法选择偏弱

- Hybrid A* 和 RRT* 是经典算法但不够 SOTA
- 没有对比近年 DRL 导航方法（如 PPO-based, SAC-based, Actor-Critic）
- DQN 家族内部对比（DQN/DDQN/PDDQN × MLP/CNN = 6 种）反而给自己挖坑——结果显示差异不大

### 14. 缺少表述中文 → 英文投稿的学术规范

- Acknowledgments 和 Conflicts of Interest 还是中文模板占位符
- 作者信息是 "Firstname Lastname"，显然未填写
- 无 Introduction 末段的 "paper organization" 说明

---

## 🔵 小问题（建议改进）

### 15. BibTeX 可疑条目

- `xu2026rl` 发表年份写 2026，需核实（可能是 online first）
- 部分引用缺少页码或 DOI

### 16. 杂项

- `\usepackage{xeCJK}` 如果最终投英文期刊则不需要
- "在 PyCharm 平台实现了所提算法"——审稿人不关心你用什么 IDE
- 双栏 IEEEtran 格式下 `longtable` 宏包加载了但从未使用

---

## 改进建议优先级排序

| 优先级 | 改进项 | 预估工作量 |
|--------|--------|----------|
| **P0** | 确定目标期刊，切换到英文 + 对应模板 | 3-5 天 |
| **P0** | 重新定位 contribution——解释 or 改标题 | 1 天 |
| **P0** | 重写 Introduction，加入深入的 gap analysis | 2-3 天 |
| **P1** | 增加 training curve 和收敛性分析 | 1 天 |
| **P1** | 扩充实验 sample size + 统计检验 | 2-3 天 |
| **P1** | 写完整的 Discussion + Limitations | 2-3 天 |
| **P1** | 增加超参数敏感性分析 | 1-2 天 |
| **P2** | 图表矢量化 + 质量提升 | 1-2 天 |
| **P2** | 修复符号不一致、公式细节 | 0.5 天 |
| **P2** | 增加 PPO/SAC 等 DRL baseline 对比 | 3-5 天 |
| **P3** | 真实场景说明措辞修正 | 0.5 天 |

---

## 仓库层面可利用但论文中未展示的资源

从代码仓库来看，以下资源可以直接补充到论文中：

1. **`dqn8_plots/plot_training.py`** → 训练曲线图（reward, success rate vs. episode）
2. **`amr_dqn/forest_policy.py`** → 动作掩码逻辑的具体实现，可用于 Methods 补充说明
3. **`amr_dqn/networks.py`** → 网络结构细节（参数量），可用于计算复杂度分析
4. **`configs/`** → 超参数配置文件，可作为 supplementary material 的 hyperparameter table
5. **`amr_dqn/metrics.py`** → KPI 计算逻辑，可验证论文中指标定义的正确性
