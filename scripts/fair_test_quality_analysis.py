#!/usr/bin/env python3
"""Fair test quality analysis: all-succeed filter + per-pair minmax normalization."""
import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path("/home/sun/phdproject/dqn/DQN8/runs/fair_test_md_vs_classical")
OUT = BASE / "quality_analysis"
OUT.mkdir(exist_ok=True)

# Composite weights
W_PL, W_K, W_PT = 1.0, 0.6, 0.2

suites = {
    "sr_long":  BASE / "20260313_132138" / "table2_kpis_raw.csv",
    "sr_short": BASE / "20260313_134029" / "table2_kpis_raw.csv",
}

for suite_name, csv_path in suites.items():
    print(f"\n{'='*60}")
    print(f"  {suite_name.upper()}")
    print(f"{'='*60}")

    df = pd.read_csv(csv_path)
    algos = df["Algorithm"].unique().tolist()
    n_runs = df["run_idx"].nunique()
    print(f"Algorithms: {algos}, Runs: {n_runs}")

    # --- SR summary ---
    sr = df.groupby("Algorithm")["success_rate"].mean()
    print(f"\nSuccess Rate:")
    for a in algos:
        print(f"  {a}: {sr[a]:.0%}")

    # --- All-succeed filter ---
    # For each run_idx, check all 3 algos succeeded
    success_by_run = df.pivot(index="run_idx", columns="Algorithm", values="success_rate")
    all_succeed_mask = (success_by_run == 1.0).all(axis=1)
    good_runs = all_succeed_mask[all_succeed_mask].index.tolist()
    print(f"\nAll-succeed pairs: {len(good_runs)} / {n_runs}")

    if len(good_runs) == 0:
        print("  No all-succeed pairs, skipping quality analysis.")
        continue

    df_q = df[df["run_idx"].isin(good_runs)].copy()

    # --- Per-pair minmax normalization ---
    metrics = ["avg_path_length", "avg_curvature_1_m", "planning_time_s"]
    metric_labels = {"avg_path_length": "path_length", "avg_curvature_1_m": "curvature", "planning_time_s": "planning_time"}

    norm_rows = []
    for rid in good_runs:
        pair = df_q[df_q["run_idx"] == rid]
        row_data = {"run_idx": rid}
        for m in metrics:
            vals = pair[m].values
            mn, mx = vals.min(), vals.max()
            for _, r in pair.iterrows():
                algo = r["Algorithm"]
                raw = r[m]
                normed = (raw - mn) / (mx - mn) if mx > mn else 0.0
                row_data[f"{metric_labels[m]}_raw_{algo}"] = raw
                row_data[f"{metric_labels[m]}_norm_{algo}"] = normed
        # Composite per algo
        for _, r in pair.iterrows():
            algo = r["Algorithm"]
            n_pl = row_data[f"path_length_norm_{algo}"]
            n_k  = row_data[f"curvature_norm_{algo}"]
            n_pt = row_data[f"planning_time_norm_{algo}"]
            comp = (W_PL * n_pl + W_K * n_k + W_PT * n_pt) / (W_PL + W_K + W_PT)
            row_data[f"composite_{algo}"] = comp
        norm_rows.append(row_data)

    df_norm = pd.DataFrame(norm_rows)

    # --- Mean normalized scores ---
    print(f"\nQuality Analysis (per-pair minmax, {len(good_runs)} pairs):")
    print(f"{'Algorithm':<12} {'PathLen':>8} {'Curv':>8} {'PlanTime':>8} {'Composite':>10}")
    print("-" * 50)
    results = {}
    for algo in algos:
        pl = df_norm[f"path_length_norm_{algo}"].mean()
        k  = df_norm[f"curvature_norm_{algo}"].mean()
        pt = df_norm[f"planning_time_norm_{algo}"].mean()
        comp = df_norm[f"composite_{algo}"].mean()
        results[algo] = {"path_length": pl, "curvature": k, "planning_time": pt, "composite": comp}
        print(f"{algo:<12} {pl:>8.4f} {k:>8.4f} {pt:>8.4f} {comp:>10.4f}")

    # Mark best (lowest)
    best_comp = min(results, key=lambda a: results[a]["composite"])
    print(f"\nBest composite: {best_comp} ({results[best_comp]['composite']:.4f})")

    # --- Raw mean (only all-succeed pairs) ---
    print(f"\nRaw means (all-succeed pairs only):")
    print(f"{'Algorithm':<12} {'PathLen(m)':>10} {'Curv(1/m)':>10} {'PlanTime(s)':>12} {'InferTime(s)':>12}")
    print("-" * 60)
    for algo in algos:
        sub = df_q[df_q["Algorithm"] == algo]
        print(f"{algo:<12} {sub['avg_path_length'].mean():>10.4f} {sub['avg_curvature_1_m'].mean():>10.6f} {sub['planning_time_s'].mean():>12.5f} {sub['inference_time_s'].mean():>12.5f}")

    # --- Head-to-head on quality pairs ---
    print(f"\nHead-to-head wins (composite, per pair):")
    for a1 in algos:
        for a2 in algos:
            if a1 >= a2:
                continue
            wins_a1 = (df_norm[f"composite_{a1}"] < df_norm[f"composite_{a2}"]).sum()
            wins_a2 = (df_norm[f"composite_{a2}"] < df_norm[f"composite_{a1}"]).sum()
            ties = len(good_runs) - wins_a1 - wins_a2
            print(f"  {a1} vs {a2}: {wins_a1}-{wins_a2} (ties: {ties})")

    # --- Save detailed CSV ---
    out_csv = OUT / f"{suite_name}_quality_pairs.csv"
    df_norm.to_csv(out_csv, index=False)

    # --- Save summary ---
    summary_rows = []
    for algo in algos:
        summary_rows.append({
            "Algorithm": algo,
            "n_pairs": len(good_runs),
            "norm_path_length": results[algo]["path_length"],
            "norm_curvature": results[algo]["curvature"],
            "norm_planning_time": results[algo]["planning_time"],
            "composite": results[algo]["composite"],
        })
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(OUT / f"{suite_name}_quality_summary.csv", index=False)

print(f"\n\nOutput saved to: {OUT}")
