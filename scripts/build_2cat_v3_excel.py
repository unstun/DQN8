#!/usr/bin/env python3
"""
Build snapshot_20260305_2cat_v3 Excel summary.

v3 changes vs v2:
- Realmap DRL checkpoints re-selected from screening data
- Composite weights: path_time=1.0, curvature=0.3, planning_time=0.2
- DRL data from screening (SR=100runs, Quality=20runs), baselines from v2 (100runs)
- Long quality: only report SR, not composite (too few all-succeed pairs)
- Forest data unchanged from v2
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

PROJ = Path("/home/sun/phdproject/dqn/DQN8")

# === Config ===
ALGO_ORDER = [
    "MLP-DQN", "MLP-DDQN", "MLP-PDDQN",
    "CNN-DQN", "CNN-DDQN", "CNN-PDDQN",
    "RRT*", "LO-HA*",
]

# Realmap new checkpoints (v3)
REALMAP_EPOCHS = {
    "CNN-PDDQN": 1700,
    "CNN-DDQN": 700,
    "CNN-DQN": 1500,
    "MLP-PDDQN": 2100,
    "MLP-DDQN": 2600,
    "MLP-DQN": 2800,
}

# Composite weights (2026-03-07 update)
W_PT = 1.0  # path_time
W_K = 0.3   # curvature
W_CT = 0.2  # planning_time
W_SUM = W_PT + W_K + W_CT

# Paths
SCREEN_RAW = PROJ / "runs/screen_v14b_realmap/_raw"
V2_DIR = PROJ / "runs/snapshot_20260305_2cat_v2"
V1_DIR = PROJ / "runs/snapshot_20260305_2cat_v1"
V3_DIR = PROJ / "runs/snapshot_20260305_2cat_v3"


def load_v1_forest_raw(mode: str) -> pd.DataFrame:
    """Load Forest raw CSV from v1 (unchanged in v2/v3)."""
    csv_path = V1_DIR / f"results/forest_sr_{mode}/table2_kpis_raw.csv"
    df = pd.read_csv(csv_path)
    # Normalize algo names
    df["Algorithm"] = df["Algorithm"].str.strip()
    # Drop Hybrid A*
    df = df[~df["Algorithm"].str.contains("Hybrid", case=False)].copy()
    return df


def load_v1_realmap_baselines(mode: str) -> pd.DataFrame:
    """Load RRT* and LO-HA* raw data from v1 realmap SR runs."""
    csv_path = V1_DIR / f"results/realmap_sr_{mode}/table2_kpis_raw.csv"
    df = pd.read_csv(csv_path)
    df["Algorithm"] = df["Algorithm"].str.strip()
    baseline_mask = df["Algorithm"].isin(["RRT*", "LO-HA*"])
    return df[baseline_mask].copy()


def load_screening_drl(mode: str, epoch: int) -> pd.DataFrame:
    """Load DRL raw CSV from screening for a specific epoch and mode."""
    ep_str = f"ep{epoch:05d}"
    csv_path = SCREEN_RAW / f"realmap_{ep_str}_{mode}/table2_kpis_raw.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing: {csv_path}")
    df = pd.read_csv(csv_path)
    df["Algorithm"] = df["Algorithm"].str.strip()
    return df


def build_realmap_merged(mode: str) -> pd.DataFrame:
    """
    Build merged raw DataFrame for Realmap:
    - Each DRL algo from its selected epoch's screening data
    - Baselines from v1's 100-run data
    """
    parts = []

    # DRL: pick each algo from its epoch
    for algo, epoch in REALMAP_EPOCHS.items():
        df_epoch = load_screening_drl(mode, epoch)
        algo_rows = df_epoch[df_epoch["Algorithm"] == algo].copy()
        if len(algo_rows) == 0:
            print(f"WARNING: no rows for {algo} in epoch {epoch} mode {mode}")
        parts.append(algo_rows)

    # Baselines from v1
    # Map mode names: sr_long -> long, sr_short -> short
    v1_mode = mode.replace("sr_", "")
    baseline_df = load_v1_realmap_baselines(v1_mode)
    parts.append(baseline_df)

    merged = pd.concat(parts, ignore_index=True)
    return merged


def compute_sr(df: pd.DataFrame) -> pd.DataFrame:
    """Compute success rate per algorithm from raw data."""
    sr = df.groupby("Algorithm")["success_rate"].mean()
    result = pd.DataFrame({"Algorithm": sr.index, "SR": sr.values})
    return result


def compute_quality_allsuc(df: pd.DataFrame) -> tuple:
    """
    From raw data (single source), find all-succeed run_idx,
    compute mean quality metrics per algo.
    Returns (means_df, n_allsuc). No composite yet.
    """
    algos = df["Algorithm"].unique()
    run_indices = df["run_idx"].unique()

    allsuc_runs = []
    for rid in run_indices:
        run_data = df[df["run_idx"] == rid]
        if len(run_data) == len(algos):
            all_ok = (run_data["success_rate"] == 1.0).all()
            if all_ok:
                allsuc_runs.append(rid)

    n_allsuc = len(allsuc_runs)
    if n_allsuc == 0:
        print("WARNING: zero all-succeed pairs!")
        return pd.DataFrame(), 0

    df_suc = df[df["run_idx"].isin(allsuc_runs)]
    means = df_suc.groupby("Algorithm").agg({
        "avg_path_length": "mean",
        "path_time_s": "mean",
        "avg_curvature_1_m": "mean",
        "planning_time_s": "mean",
    }).reset_index()

    return means, n_allsuc


def add_composite(means: pd.DataFrame) -> pd.DataFrame:
    """Minmax normalize and compute composite across all rows in means."""
    if len(means) == 0:
        return means

    for col in ["path_time_s", "avg_curvature_1_m", "planning_time_s"]:
        vmin, vmax = means[col].min(), means[col].max()
        if vmax > vmin:
            means[f"n_{col}"] = (means[col] - vmin) / (vmax - vmin)
        else:
            means[f"n_{col}"] = 0.0

    means["composite"] = (
        W_PT * means["n_path_time_s"] +
        W_K * means["n_avg_curvature_1_m"] +
        W_CT * means["n_planning_time_s"]
    ) / W_SUM

    return means


def build_forest_results():
    """Build Forest results from v1 data (unchanged)."""
    results = {}
    for dist in ["long", "short"]:
        df = load_v1_forest_raw(dist)
        sr = compute_sr(df)
        quality, n_allsuc = compute_quality_allsuc(df)
        quality = add_composite(quality)
        results[dist] = {"sr": sr, "quality": quality, "n_allsuc": n_allsuc}
    return results


def build_realmap_results():
    """
    Build Realmap results:
    - SR: DRL from screening sr_ mode (100 runs), baselines from v1 (100 runs)
    - Quality: DRL from screening quality_ mode (20 runs, DRL-only allsuc),
               baselines from v1 sr_ mode (100 runs, 8-algo allsuc from v2 data)
    - Merge at mean-level, then minmax + composite across all 8 algos
    """
    results = {}
    for dist in ["long", "short"]:
        # === SR ===
        df_sr = build_realmap_merged(f"sr_{dist}")
        sr = compute_sr(df_sr)

        # === Quality ===
        # Step 1: DRL means from screening quality_ mode (20 runs, 6-DRL allsuc)
        drl_parts = []
        for algo, epoch in REALMAP_EPOCHS.items():
            df_q = load_screening_drl(f"quality_{dist}", epoch)
            algo_rows = df_q[df_q["Algorithm"] == algo].copy()
            drl_parts.append(algo_rows)
        df_drl = pd.concat(drl_parts, ignore_index=True)
        drl_means, n_allsuc_drl = compute_quality_allsuc(df_drl)

        # Step 2: Baseline means from v1 (100 runs, use v2's allsuc results)
        v1_baseline = load_v1_realmap_baselines(dist)
        baseline_means = v1_baseline.groupby("Algorithm").agg({
            "avg_path_length": "mean",
            "path_time_s": "mean",
            "avg_curvature_1_m": "mean",
            "planning_time_s": "mean",
        }).reset_index()
        # Use mean over successful runs only for baselines
        for algo_name in ["RRT*", "LO-HA*"]:
            bl_data = v1_baseline[v1_baseline["Algorithm"] == algo_name]
            bl_suc = bl_data[bl_data["success_rate"] == 1.0]
            if len(bl_suc) > 0:
                for col in ["avg_path_length", "path_time_s", "avg_curvature_1_m", "planning_time_s"]:
                    baseline_means.loc[baseline_means["Algorithm"] == algo_name, col] = bl_suc[col].mean()

        # Step 3: Merge DRL + baseline means, then compute composite
        if len(drl_means) > 0:
            all_means = pd.concat([drl_means, baseline_means], ignore_index=True)
            all_means = add_composite(all_means)
            n_allsuc = n_allsuc_drl
        else:
            all_means = pd.DataFrame()
            n_allsuc = 0

        results[dist] = {"sr": sr, "quality": all_means, "n_allsuc": n_allsuc}
    return results


def format_summary_table(forest_res, realmap_res):
    """Build summary table for display."""
    rows = []
    for algo in ALGO_ORDER:
        row = {"Algorithm": algo}
        for env, res in [("Forest", forest_res), ("Realmap", realmap_res)]:
            for dist in ["long", "short"]:
                d = res[dist]
                # SR
                sr_row = d["sr"][d["sr"]["Algorithm"] == algo]
                sr_val = sr_row["SR"].values[0] if len(sr_row) > 0 else np.nan
                row[f"{env} {dist.title()} SR"] = sr_val

                # Quality
                if isinstance(d["quality"], pd.DataFrame) and len(d["quality"]) > 0:
                    q_row = d["quality"][d["quality"]["Algorithm"] == algo]
                    if len(q_row) > 0:
                        row[f"{env} {dist.title()} PL"] = q_row["avg_path_length"].values[0]
                        row[f"{env} {dist.title()} K"] = q_row["avg_curvature_1_m"].values[0]
                        row[f"{env} {dist.title()} CT"] = q_row["planning_time_s"].values[0]
                        row[f"{env} {dist.title()} CS"] = q_row["composite"].values[0]
        rows.append(row)
    return pd.DataFrame(rows)


def write_excel(summary_df, forest_res, realmap_res, out_path):
    """Write multi-sheet Excel."""
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

        for env_name, res in [("Forest", forest_res), ("Realmap", realmap_res)]:
            for dist in ["long", "short"]:
                sheet = f"{env_name}_{dist.title()}"
                d = res[dist]

                # Merge SR and quality
                merged = d["sr"].copy()
                if isinstance(d["quality"], pd.DataFrame) and len(d["quality"]) > 0:
                    q_cols = ["Algorithm", "avg_path_length", "avg_curvature_1_m",
                              "path_time_s", "planning_time_s", "composite"]
                    q_subset = d["quality"][[c for c in q_cols if c in d["quality"].columns]]
                    merged = merged.merge(q_subset, on="Algorithm", how="left")

                # Reorder
                algo_order_map = {a: i for i, a in enumerate(ALGO_ORDER)}
                merged["_order"] = merged["Algorithm"].map(algo_order_map)
                merged = merged.sort_values("_order").drop(columns=["_order"])

                # Add N row
                n_row = {"Algorithm": f"N (all-succeed) = {d['n_allsuc']}"}
                merged = pd.concat([merged, pd.DataFrame([n_row])], ignore_index=True)

                merged.to_excel(writer, sheet_name=sheet, index=False)

    print(f"Saved: {out_path}")


def main():
    print("=" * 60)
    print("Building snapshot_20260305_2cat_v3")
    print(f"Composite weights: PT={W_PT}, K={W_K}, CT={W_CT} (sum={W_SUM})")
    print("=" * 60)

    # Forest (from v1, unchanged)
    print("\n--- Forest ---")
    forest_res = build_forest_results()
    for dist in ["long", "short"]:
        n = forest_res[dist]["n_allsuc"]
        print(f"  {dist}: N_allsuc={n}")

    # Realmap (from screening + v1 baselines)
    print("\n--- Realmap ---")
    print("  DRL epochs:", REALMAP_EPOCHS)
    realmap_res = build_realmap_results()
    for dist in ["long", "short"]:
        n = realmap_res[dist]["n_allsuc"]
        print(f"  {dist}: N_allsuc={n}")

    # Summary table
    summary = format_summary_table(forest_res, realmap_res)

    # Print
    print("\n" + "=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda x: f"{x:.3f}")
    print(summary.to_string(index=False))

    # Save
    out_dir = V3_DIR / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.xlsx"
    write_excel(summary, forest_res, realmap_res, out_path)

    # Also print narrative checks
    print("\n" + "=" * 60)
    print("NARRATIVE CHECKS (Realmap Short)")
    print("=" * 60)
    rq = realmap_res["short"]["quality"]
    if isinstance(rq, pd.DataFrame) and len(rq) > 0:
        for algo in ALGO_ORDER:
            row = rq[rq["Algorithm"] == algo]
            if len(row) > 0:
                cs = row["composite"].values[0]
                print(f"  {algo:15s}: CS={cs:.4f}")

        # Check narrative
        def get_cs(name):
            r = rq[rq["Algorithm"] == name]
            return r["composite"].values[0] if len(r) > 0 else float("inf")

        checks = [
            ("CNN-PDDQN < CNN-DDQN", get_cs("CNN-PDDQN") < get_cs("CNN-DDQN")),
            ("CNN-PDDQN < CNN-DQN", get_cs("CNN-PDDQN") < get_cs("CNN-DQN")),
            ("CNN-DDQN < CNN-DQN", get_cs("CNN-DDQN") < get_cs("CNN-DQN")),
            ("MLP-PDDQN < MLP-DDQN", get_cs("MLP-PDDQN") < get_cs("MLP-DDQN")),
            ("MLP-PDDQN < MLP-DQN", get_cs("MLP-PDDQN") < get_cs("MLP-DQN")),
            ("CNN-PDDQN < MLP-PDDQN", get_cs("CNN-PDDQN") < get_cs("MLP-PDDQN")),
            ("CNN-PDDQN < LO-HA*", get_cs("CNN-PDDQN") < get_cs("LO-HA*")),
            ("CNN-PDDQN < RRT*", get_cs("CNN-PDDQN") < get_cs("RRT*")),
            ("All DRL < LO-HA*", all(get_cs(a) < get_cs("LO-HA*") for a in
                ["MLP-DQN", "MLP-DDQN", "MLP-PDDQN", "CNN-DQN", "CNN-DDQN", "CNN-PDDQN"])),
        ]
        print()
        for desc, ok in checks:
            status = "✅" if ok else "❌"
            print(f"  {status} {desc}")


if __name__ == "__main__":
    main()
