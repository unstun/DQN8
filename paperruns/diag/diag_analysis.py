#!/usr/bin/env python3
"""diag 消融实验推理结果分析脚本

与 ablation 版本相同逻辑，但指向 abl_diag_* 目录。
edt_collision_margin = "diag" (√2/2 补偿)

输出：
  paperruns/diag/results/
    sr_long.csv, sr_short.csv          — 成功率表
    quality_long.csv, quality_short.csv — 路径质量表（仅全成功 pair）
    diag_summary.md                    — Markdown 汇总报告
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

# ── Composite Score 权重 ─────────────────────────────────────────────
W_PL = 1.0   # 路径长度
W_K  = 0.6   # 平均曲率
W_PT = 0.2   # 规划计算时间

# ── 路径配置 ──────────────────────────────────────────────────────────
PROJ = Path("/home/sun/phdproject/dqn/DQN8")
RUNS = PROJ / "runs"
OUT  = PROJ / "paperruns" / "diag" / "results"
OUT.mkdir(parents=True, exist_ok=True)

# 9 变体目录名 (diag 版)
VARIANTS = [
    "cnn_dqn", "cnn_ddqn",
    "cnn_dqn_mha", "cnn_ddqn_mha",
    "cnn_dqn_duel", "cnn_ddqn_duel",
    "cnn_dqn_md", "cnn_ddqn_md",
    "mlp",
]

# variant → 展示名
VARIANT_DISPLAY = {
    "cnn_dqn":       "CNN-DQN",
    "cnn_ddqn":      "CNN-DDQN",
    "cnn_dqn_mha":   "CNN-DQN+MHA",
    "cnn_ddqn_mha":  "CNN-DDQN+MHA",
    "cnn_dqn_duel":  "CNN-DQN+Duel",
    "cnn_ddqn_duel": "CNN-DDQN+Duel",
    "cnn_dqn_md":    "CNN-DQN+MD",
    "cnn_ddqn_md":   "CNN-DDQN+MD",
}


def find_run_dir(variant: str, mode: str) -> Path | None:
    """找到变体在指定 mode (sr_long/sr_short) 下的推理目录."""
    infer_dir = RUNS / f"abl_diag_infer_{variant}"
    if not infer_dir.exists():
        return None
    for sub in sorted(infer_dir.iterdir()):
        cfg_path = sub / "configs" / "run.json"
        if not cfg_path.exists():
            continue
        with open(cfg_path) as f:
            cfg = json.load(f)
        profile = cfg["args"].get("profile", "")
        if mode in profile:
            return sub
    return None


def load_raw(variant: str, mode: str) -> pd.DataFrame | None:
    """加载 table2_kpis_raw.csv，添加 variant 列."""
    run_dir = find_run_dir(variant, mode)
    if run_dir is None:
        print(f"  [WARN] {variant} / {mode} 未找到")
        return None
    csv_path = run_dir / "table2_kpis_raw.csv"
    if not csv_path.exists():
        print(f"  [WARN] {csv_path} 不存在")
        return None
    df = pd.read_csv(csv_path)
    df["variant"] = variant
    return df


def make_algo_label(variant: str, csv_algo: str) -> str:
    """生成唯一的算法展示标签."""
    if variant == "mlp":
        return csv_algo  # MLP-DQN / MLP-DDQN
    return VARIANT_DISPLAY[variant]


# ── 加载全部数据 ─────────────────────────────────────────────────────
print("=" * 60)
print("加载 diag 消融实验推理数据")
print("=" * 60)

data = {}
for mode in ["sr_long", "sr_short"]:
    for v in VARIANTS:
        df = load_raw(v, mode)
        if df is not None:
            data[(mode, v)] = df
            algos = df["Algorithm"].unique()
            print(f"  {v:20s} {mode:10s}  {len(df):3d} rows  algos={list(algos)}")

# ══════════════════════════════════════════════════════════════════════
# Mode 1: 成功率对比 (SR)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Mode 1: 成功率 (SR) 对比")
print("=" * 60)

sr_tables = {}
for mode in ["sr_long", "sr_short"]:
    rows = []
    for v in VARIANTS:
        key = (mode, v)
        if key not in data:
            continue
        df = data[key]
        for csv_algo, grp in df.groupby("Algorithm"):
            label = make_algo_label(v, csv_algo)
            n_total = len(grp)
            n_succ  = (grp["success_rate"] == 1.0).sum()
            sr = n_succ / n_total if n_total > 0 else 0.0
            rows.append({
                "Algorithm":     label,
                "Total_Runs":    n_total,
                "Successes":     n_succ,
                "Success_Rate":  round(sr, 4),
                "SR_pct":        f"{sr*100:.1f}%",
            })
    sr_df = pd.DataFrame(rows)
    sr_df = sr_df.sort_values("Success_Rate", ascending=False).reset_index(drop=True)
    sr_tables[mode] = sr_df
    out_path = OUT / f"{mode}.csv"
    sr_df.to_csv(out_path, index=False)
    print(f"\n── {mode} ──")
    print(sr_df[["Algorithm", "Total_Runs", "Successes", "SR_pct"]].to_string(index=False))

# ══════════════════════════════════════════════════════════════════════
# Mode 2: 路径质量对比 (Quality) — 仅全成功 pair
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Mode 2: 路径质量对比 (Quality) — 全成功 pair")
print("=" * 60)

quality_tables = {}
for mode in ["sr_long", "sr_short"]:
    algo_dfs = {}
    for v in VARIANTS:
        key = (mode, v)
        if key not in data:
            continue
        df = data[key]
        for csv_algo, grp in df.groupby("Algorithm"):
            label = make_algo_label(v, csv_algo)
            algo_dfs[label] = grp.set_index("run_idx")

    all_run_indices = None
    for label, adf in algo_dfs.items():
        succ_idx = set(adf[adf["success_rate"] == 1.0].index)
        if all_run_indices is None:
            all_run_indices = succ_idx
        else:
            all_run_indices &= succ_idx

    if all_run_indices is None:
        all_run_indices = set()

    common_pairs = sorted(all_run_indices)
    n_algos = len(algo_dfs)
    print(f"\n── {mode} ──")
    print(f"  算法数: {n_algos}, 总 run_idx: 50, 全成功 pair 数: {len(common_pairs)}")

    per_algo_succ = {}
    for label, adf in sorted(algo_dfs.items()):
        n = (adf["success_rate"] == 1.0).sum()
        per_algo_succ[label] = n
        print(f"    {label:20s} 成功: {n}/50")

    if len(common_pairs) == 0:
        print("  [WARN] 没有全成功 pair，跳过")
        continue

    print(f"  全成功 run_idx: {common_pairs}")

    rows = []
    for label in sorted(algo_dfs.keys()):
        adf = algo_dfs[label]
        subset = adf.loc[common_pairs]
        avg_pl  = subset["avg_path_length"].mean()
        std_pl  = subset["avg_path_length"].std()
        avg_k   = subset["avg_curvature_1_m"].mean()
        std_k   = subset["avg_curvature_1_m"].std()
        avg_pt  = subset["planning_time_s"].mean()
        std_pt  = subset["planning_time_s"].std()
        rows.append({
            "Algorithm":       label,
            "N_pairs":         len(common_pairs),
            "Path_Length_mean": round(avg_pl, 4),
            "Path_Length_std":  round(std_pl, 4),
            "Curvature_mean":  round(avg_k, 6),
            "Curvature_std":   round(std_k, 6),
            "Plan_Time_mean":  round(avg_pt, 5),
            "Plan_Time_std":   round(std_pt, 5),
        })

    q_df = pd.DataFrame(rows)

    def minmax_norm(s: pd.Series) -> pd.Series:
        smin, smax = s.min(), s.max()
        if smax - smin < 1e-12:
            return pd.Series(0.0, index=s.index)
        return (s - smin) / (smax - smin)

    n_pl = minmax_norm(q_df["Path_Length_mean"])
    n_k  = minmax_norm(q_df["Curvature_mean"])
    n_pt = minmax_norm(q_df["Plan_Time_mean"])
    q_df["Composite"] = ((W_PL * n_pl + W_K * n_k + W_PT * n_pt)
                         / (W_PL + W_K + W_PT))
    q_df["Composite"] = q_df["Composite"].round(4)
    q_df = q_df.sort_values("Composite").reset_index(drop=True)

    quality_tables[mode] = q_df
    out_path = OUT / f"quality_{mode.replace('sr_', '')}.csv"
    q_df.to_csv(out_path, index=False)
    print()
    print(q_df.to_string(index=False))

# ══════════════════════════════════════════════════════════════════════
# 生成 Markdown 汇总报告
# ══════════════════════════════════════════════════════════════════════
all_algo_labels = set()
for v in VARIANTS:
    if (("sr_long", v)) in data:
        df = data[("sr_long", v)]
        for csv_algo in df["Algorithm"].unique():
            all_algo_labels.add(make_algo_label(v, csv_algo))

md_lines = []
md_lines.append("# diag 消融实验推理结果汇总\n")
md_lines.append(f"edt_collision_margin = **diag** (√2/2 补偿)\n")
md_lines.append(f"变体数: {len(VARIANTS)}, 算法标签数: {len(all_algo_labels)}\n")
md_lines.append(f"算法列表: {', '.join(sorted(all_algo_labels))}\n")

md_lines.append("\n## Mode 1: 成功率对比\n")
for mode in ["sr_long", "sr_short"]:
    label = "Long (≥18m)" if "long" in mode else "Short (6-14m)"
    md_lines.append(f"\n### {label}\n")
    if mode in sr_tables:
        df = sr_tables[mode]
        md_lines.append(df[["Algorithm", "Total_Runs", "Successes", "SR_pct"]].to_markdown(index=False))
        md_lines.append("\n")
        md_lines.append(f"\n**排名 (高→低):** {' > '.join(df['Algorithm'].tolist())}\n")

md_lines.append("\n## Mode 2: 路径质量对比 (全成功 pair)\n")
for mode in ["sr_long", "sr_short"]:
    label = "Long (≥18m)" if "long" in mode else "Short (6-14m)"
    md_lines.append(f"\n### Quality {label}\n")
    if mode in quality_tables:
        q_df = quality_tables[mode]
        md_lines.append(f"全成功 pair 数: **{q_df['N_pairs'].iloc[0]}**\n")
        md_lines.append(f"Composite 权重: path_length={W_PL}, curvature={W_K}, plan_time={W_PT}\n\n")
        md_lines.append(q_df.to_markdown(index=False))
        md_lines.append("\n")
        md_lines.append(f"\n**Composite 排名 (低→高，越低越好):** {' < '.join(q_df['Algorithm'].tolist())}\n")
        q_sorted_pl = q_df.sort_values("Path_Length_mean")
        md_lines.append(f"\n**路径长度排名 (短→长):** {' < '.join(q_sorted_pl['Algorithm'].tolist())}\n")
        q_sorted_k = q_df.sort_values("Curvature_mean")
        md_lines.append(f"\n**曲率排名 (小→大):** {' < '.join(q_sorted_k['Algorithm'].tolist())}\n")
    else:
        md_lines.append("无全成功 pair，无法对比。\n")

md_report = "\n".join(md_lines)
md_path = OUT / "diag_summary.md"
md_path.write_text(md_report, encoding="utf-8")
print(f"\n{'='*60}")
print(f"报告已保存: {md_path}")
print(f"CSV 文件: {[p.name for p in sorted(OUT.glob('*.csv'))]}")
