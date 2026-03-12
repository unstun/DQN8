#!/usr/bin/env python3
"""Quality-mode analysis: all-succeed pairs only, per-pair minmax normalization."""
import pandas as pd
import numpy as np

W_PL = 1.0  # path length
W_K  = 0.6  # curvature
W_CT = 0.2  # planning time

datasets = [
    ("Forest", "runs/eval_algo6_10k_forest_half_40runs/20260312_223315/table2_kpis_raw.csv"),
    ("Realmap", "runs/eval_algo6_10k_realmap_half_40runs/20260312_223410/table2_kpis_raw.csv"),
]

for env_label, csv_path in datasets:
    df = pd.read_csv(csv_path)
    algos = df["Algorithm"].unique().tolist()
    n_algos = len(algos)

    all_runs = df["run_idx"].unique()
    good_runs = []
    for r in all_runs:
        sub = df[df["run_idx"] == r]
        if (sub["success_rate"] == 1.0).all() and len(sub) == n_algos:
            good_runs.append(r)

    print(f"\n{'='*80}")
    print(f"  {env_label}: {len(good_runs)}/{len(all_runs)} all-succeed pairs")
    print(f"{'='*80}")

    if not good_runs:
        print("  No all-succeed pairs!")
        continue

    qdf = df[df["run_idx"].isin(good_runs)].copy()

    # Per-pair minmax normalization
    norm_rows = []
    for r in good_runs:
        p = qdf[qdf["run_idx"] == r].copy()
        for c in ["avg_path_length", "avg_curvature_1_m", "planning_time_s"]:
            mn, mx = p[c].min(), p[c].max()
            p[f"n_{c}"] = (p[c] - mn) / (mx - mn) if mx > mn else 0.0
        p["composite"] = (
            W_PL * p["n_avg_path_length"]
            + W_K * p["n_avg_curvature_1_m"]
            + W_CT * p["n_planning_time_s"]
        ) / (W_PL + W_K + W_CT)
        norm_rows.append(p)

    ndf = pd.concat(norm_rows)

    # --- Success Rate summary first ---
    print(f"\n--- Success Rate (all 40 runs) ---")
    sr = df.groupby("Algorithm")["success_rate"].mean().round(4)
    sr = sr.sort_values(ascending=False)
    for algo, val in sr.items():
        print(f"  {algo}: {val:.1%}")

    # --- Mean RAW Quality ---
    print(f"\n--- Mean RAW Quality (all-succeed pairs only) ---")
    mean_raw = qdf.groupby("Algorithm")[
        ["avg_path_length", "avg_curvature_1_m", "planning_time_s"]
    ].mean().round(4)
    print(mean_raw.to_string())

    # --- Mean Normalized (lower=better) ---
    print(f"\n--- Mean Normalized Quality (lower=better) ---")
    mn = ndf.groupby("Algorithm")[
        ["n_avg_path_length", "n_avg_curvature_1_m", "n_planning_time_s", "composite"]
    ].mean().round(4)
    mn.columns = ["n_path_len", "n_curv", "n_plan_t", "composite"]
    mn = mn.sort_values("composite")
    print(mn.to_string())

    # --- Win count ---
    print(f"\n--- Win Count (lowest composite per pair) ---")
    wins = {a: 0 for a in algos}
    for r in good_runs:
        p = ndf[ndf["run_idx"] == r]
        best = p.loc[p["composite"].idxmin(), "Algorithm"]
        wins[best] += 1
    for a in sorted(wins, key=lambda x: -wins[x]):
        print(f"  {a}: {wins[a]}/{len(good_runs)} wins")

    # --- Per-pair composite pivot ---
    print(f"\n--- Per-Pair Composite Pivot ---")
    pivot = ndf.pivot_table(index="run_idx", columns="Algorithm", values="composite")
    pivot = pivot.round(4)
    print(pivot.to_string())
