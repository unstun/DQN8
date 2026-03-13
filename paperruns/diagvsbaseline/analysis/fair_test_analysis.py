#!/usr/bin/env python3
"""Fair test analysis: CNN-DDQN+MD (rollout timing) vs RRT* vs LO-HA*, diag EDT.

Compares full-pipeline DRL timing (including grid map + cost field construction)
against classical baselines. Raw metrics only, no normalization.
"""
import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path("/home/sun/phdproject/dqn/DQN8/runs/fair_diag_ddqn_md_vs_baseline")
OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)


def find_csv(base: Path) -> dict[str, Path]:
    """Find sr_long and sr_short CSVs based on avg path length."""
    csvs = sorted(base.glob("*/table2_kpis_raw.csv"))
    suites = {}
    for csv_path in csvs:
        df = pd.read_csv(csv_path, nrows=5)
        avg_pl = df["avg_path_length"].mean() if "avg_path_length" in df.columns else 0
        if avg_pl > 15:
            suites["sr_long"] = csv_path
        else:
            suites["sr_short"] = csv_path
    return suites


suites = find_csv(BASE)
if not suites:
    print(f"No CSVs found in {BASE}")
    exit(1)

print(f"Found suites: {list(suites.keys())}")

all_summaries = []

for suite_name, csv_path in sorted(suites.items()):
    print(f"\n{'='*70}")
    print(f"  {suite_name.upper()} — Fair Comparison (rollout timing)")
    print(f"{'='*70}")

    df = pd.read_csv(csv_path)
    algos = df["Algorithm"].unique().tolist()
    n_runs = df["run_idx"].nunique()
    print(f"Algorithms: {algos}, Runs: {n_runs}")

    # --- SR summary ---
    print(f"\n--- Success Rate ---")
    sr_rows = []
    for a in algos:
        sub = df[df["Algorithm"] == a]
        n_total = len(sub)
        n_succ = int(sub["success_rate"].sum())
        sr = n_succ / n_total if n_total > 0 else 0
        sr_rows.append({"Algorithm": a, "Total": n_total, "Success": n_succ, "SR": f"{sr:.0%}"})
        print(f"  {a:<20s} {n_succ}/{n_total} = {sr:.0%}")
    pd.DataFrame(sr_rows).to_csv(OUT / f"{suite_name}_sr.csv", index=False)

    # --- All-succeed filter ---
    success_by_run = df.pivot(index="run_idx", columns="Algorithm", values="success_rate")
    all_succeed_mask = (success_by_run == 1.0).all(axis=1)
    good_runs = all_succeed_mask[all_succeed_mask].index.tolist()
    print(f"\nAll-succeed pairs: {len(good_runs)} / {n_runs}")

    if len(good_runs) == 0:
        print("  No all-succeed pairs, skipping quality analysis.")
        continue

    df_q = df[df["run_idx"].isin(good_runs)].copy()

    # --- Raw means (all-succeed pairs) ---
    print(f"\n--- Raw Means ({len(good_runs)} all-succeed pairs) ---")
    print(f"{'Algorithm':<20s} {'PathLen(m)':>10} {'Curv(1/m)':>10} {'PlanTime(s)':>12}")
    print("-" * 55)
    summary_rows = []
    for algo in algos:
        sub = df_q[df_q["Algorithm"] == algo]
        pl = sub["avg_path_length"].mean()
        k = sub["avg_curvature_1_m"].mean()
        pt = sub["planning_time_s"].mean()
        pl_std = sub["avg_path_length"].std()
        k_std = sub["avg_curvature_1_m"].std()
        pt_std = sub["planning_time_s"].std()
        print(f"{algo:<20s} {pl:>10.3f} {k:>10.6f} {pt:>12.5f}")
        summary_rows.append({
            "Algorithm": algo,
            "n_pairs": len(good_runs),
            "path_length_mean": round(pl, 4),
            "path_length_std": round(pl_std, 4),
            "curvature_mean": round(k, 6),
            "curvature_std": round(k_std, 6),
            "planning_time_mean": round(pt, 5),
            "planning_time_std": round(pt_std, 5),
        })

    # --- Planning time speedup ---
    print(f"\n--- Planning Time Speedup (DRL vs baselines) ---")
    drl_algos = [a for a in algos if "RRT" not in a and "HA*" not in a and "Hybrid" not in a]
    baseline_algos = [a for a in algos if a not in drl_algos]
    for drl in drl_algos:
        drl_time = df_q[df_q["Algorithm"] == drl]["planning_time_s"].mean()
        for bl in baseline_algos:
            bl_time = df_q[df_q["Algorithm"] == bl]["planning_time_s"].mean()
            ratio = bl_time / drl_time if drl_time > 0 else float("inf")
            print(f"  {bl} / {drl} = {bl_time:.4f}s / {drl_time:.4f}s = {ratio:.1f}x")

    # --- Head-to-head (raw path length, per pair) ---
    print(f"\n--- Head-to-Head (path length, per pair) ---")
    for i, a1 in enumerate(algos):
        for a2 in algos[i + 1:]:
            v1 = df_q[df_q["Algorithm"] == a1].set_index("run_idx")["avg_path_length"]
            v2 = df_q[df_q["Algorithm"] == a2].set_index("run_idx")["avg_path_length"]
            common = v1.index.intersection(v2.index)
            w1 = (v1[common] < v2[common]).sum()
            w2 = (v2[common] < v1[common]).sum()
            ties = len(common) - w1 - w2
            print(f"  {a1} vs {a2}: {w1}-{w2} (ties: {ties})")

    # --- Save ---
    all_summaries.extend([{"suite": suite_name, **r} for r in summary_rows])
    pd.DataFrame(summary_rows).to_csv(OUT / f"{suite_name}_quality_summary.csv", index=False)

# --- Combined summary ---
if all_summaries:
    df_all = pd.DataFrame(all_summaries)
    df_all.to_csv(OUT / "combined_summary.csv", index=False)

print(f"\n\nAll output saved to: {OUT}")
