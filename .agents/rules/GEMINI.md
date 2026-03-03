# DQN8/AGENTS.md（论文写作模式 — Elsevier Measurement）

> 作用域：`/home/sun/phdproject/dqn/DQN8/**`。工程开发规则见 `process_AGENTS.md`。

## 0. 硬约束

1) 每次回复以"帅哥，"开头。
2) 改文件前输出3–7步计划+文件清单+风险+验证，等"开始"后再动手。
3) 默认中文回复；论文正文英文写作。
4) **严禁凭记忆生成 BibTeX**——未经程序化核实的引用一律标 `[CITATION NEEDED]` 并告知用户。
5) `AGENTS.md`、`CLAUDE.md`、`GEMINI.md` 三文件逐行一致；改后用 `diff` 两两验证。
6) 纯文档改动豁免 `configs/` 新增规则。
7) **写作以 Elsevier Measurement 格式为准**（双栏 elsarticle，页面上限约12–14页正文）。

## 1. 论文路径与文件结构

```
paperdqn8/
├── main.tex          # 主文件（elsarticle 双栏）
├── references.bib    # 文献库（程序化获取，勿手填）
├── figs/             # 所有图片（PDF 向量格式优先）
└── media/            # 补充材料
```

- **编译**：`cd paperdqn8 && latexmk -pdf -xelatex main.tex`
- **快速检查**：`latexmk -pdf -xelatex -interaction=nonstopmode main.tex 2>&1 | tail -20`

## 2. Elsevier Measurement 期刊规则

| 项目 | 要求 |
|------|------|
| 文档类 | `\documentclass[review]{elsarticle}` (投稿) / `twocolumn`（最终） |
| 引用格式 | `natbib` + 数字式 `[1]`，用 `\cite{}` |
| 图表 | 每图附独立 caption；双栏图用 `figure*` |
| 字数 | 正文约 6000–8000 词（不含摘要/参考文献） |
| 摘要 | 150–250 词，结构化（Background / Methods / Results / Conclusions） |
| 关键词 | 5–8 个，分号分隔 |
| Highlights | 3–5 条，每条≤85字符 |
| 必须章节 | Introduction / Methods / Results / Discussion / Conclusion |
| Limitations | 在 Discussion 末段或独立段落说明 |

## 3. 写作工作流

```
写作进度跟踪：
- [ ] Step 1: 锁定一句话贡献（与用户确认）
- [ ] Step 2: 起草 Highlights + 摘要 → 反馈 → 修改
- [ ] Step 3: 起草 Introduction → 反馈 → 修改
- [ ] Step 4: 起草 Methods → 反馈 → 修改
- [ ] Step 5: 起草 Results → 反馈 → 修改
- [ ] Step 6: 起草 Discussion + Limitations → 反馈 → 修改
- [ ] Step 7: 起草 Conclusion → 反馈 → 修改
- [ ] Step 8: Related Work / 文献核查 → 反馈 → 修改
- [ ] Step 9: 提交前 Checklist
```

## 4. 引用核查流程（强制）

1. **搜索**：用 `search_web` 或 Semantic Scholar API 定位文献
2. **核实**：DOI 在 2+ 数据源（Semantic Scholar + arXiv/CrossRef）中确认存在
3. **获取 BibTeX**：`curl -LH "Accept: application/x-bibtex" https://doi.org/<DOI>`
4. **核实声明**：确认所引 claim 确实出现于该文献中
5. 任何步骤失败 → 标 `[CITATION NEEDED]` 并告知用户

## 5. 绘图规范

- 格式：PDF 或 EPS（向量）；位图至少 300 DPI
- 调用 `dqn8_plots/run_all.py` 一键生成所有图
- 图片存至 `paperdqn8/figs/`，文件名与 `\includegraphics{}` 保持一致
- 所有图表确保黑白可读（色盲友好）

## 6. 踩坑

### 6.1 LaTeX 编译

- `xelatex` 支持中文注释；提交版用标准 `pdflatex`
- 首次编译前确认 `elsarticle.cls` 已安装：`kpsewhich elsarticle.cls`
- 缺包：`sudo tlmgr install <package>`

### 6.2 联网调研

- WebFetch/WebSearch 不混批；每批同类≤2；优先 arXiv / GitHub HTML 版
- 付费墙 403：用浏览器工具代替

### 6.3 写作语言

- 论文正文全英文
- 代码注释、提交说明、对话全中文
