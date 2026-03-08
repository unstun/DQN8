#!/usr/bin/env python3
"""Build a single Excel summary for snapshot_20260305_2cat_v2.

Reads the 4 SR-mode raw CSVs from v1, drops Hybrid A*, and produces:
- SR: success rate from 100 runs per algorithm
- Quality: mean path_length / curvature / planning_time / composite
  over all-algorithm-succeed pairs (no fixed cap)
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ---------- paths ----------
V1 = Path("runs/snapshot_20260305_2cat_v1/results")
OUT_DIR = Path("runs/snapshot_20260305_2cat_v2/results")

MODES = {
    "Forest Long":  V1 / "forest_sr_long"  / "table2_kpis_raw.csv",
    "Forest Short": V1 / "forest_sr_short" / "table2_kpis_raw.csv",
    "Realmap Long": V1 / "realmap_sr_long"  / "table2_kpis_raw.csv",
    "Realmap Short": V1 / "realmap_sr_short" / "table2_kpis_raw.csv",
}

DROP_ALGO = "Hybrid A*"

# composite weights
W_PT, W_K, W_PL = 1.0, 0.8, 0.2
W_SUM = W_PT + W_K + W_PL


def minmax(s: pd.Series) -> pd.Series:
    lo, hi = s.min(), s.max()
    if hi - lo < 1e-12:
        return pd.Series(0.0, index=s.index)
    return (s - lo) / (hi - lo)


def process_mode(csv_path: Path):
    df = pd.read_csv(csv_path)
    df = df[df["Algorithm"] != DROP_ALGO].copy()

    # ---- SR (100 runs) ----
    sr = df.groupby("Algorithm")["success_rate"].mean()

    # ---- Quality: all-succeed pairs ----
    ok = df.groupby("run_idx")["success_rate"].min()
    keep_idx = set(ok[ok >= 1.0 - 1e-9].index)
    n_quality = len(keep_idx)

    qdf = df[df["run_idx"].isin(keep_idx)]
    qmean = qdf.groupby("Algorithm")[
        ["avg_path_length", "avg_curvature_1_m", "planning_time_s"]
    ].mean()

    # composite on means (minmax across algos)
    n_pt = minmax(qmean["avg_path_length"])
    n_k  = minmax(qmean["avg_curvature_1_m"])
    n_pl = minmax(qmean["planning_time_s"])
    qmean["composite"] = (W_PT * n_pt + W_K * n_k + W_PL * n_pl) / W_SUM

    return sr, qmean, n_quality


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # algo display order
    algo_order = [
        "MLP-DQN", "MLP-DDQN", "MLP-PDDQN",
        "CNN-DQN", "CNN-DDQN", "CNN-PDDQN",
        "RRT*", "LO-HA*",
    ]

    rows = []
    n_quality_row = {}

    for mode_name, csv_path in MODES.items():
        sr, qmean, n_q = process_mode(csv_path)
        n_quality_row[mode_name] = n_q
        for algo in algo_order:
            row = {"Algorithm": algo}
            row[f"{mode_name} SR"] = sr.get(algo, np.nan)
            row[f"{mode_name} PL(m)"] = qmean.loc[algo, "avg_path_length"] if algo in qmean.index else np.nan
            row[f"{mode_name} K(1/m)"] = qmean.loc[algo, "avg_curvature_1_m"] if algo in qmean.index else np.nan
            row[f"{mode_name} CT(s)"] = qmean.loc[algo, "planning_time_s"] if algo in qmean.index else np.nan
            row[f"{mode_name} CS"] = qmean.loc[algo, "composite"] if algo in qmean.index else np.nan
            # find or update existing row
            existing = [r for r in rows if r["Algorithm"] == algo]
            if existing:
                existing[0].update(row)
            else:
                rows.append(row)

    big = pd.DataFrame(rows)
    big = big.set_index("Algorithm").loc[algo_order].reset_index()

    # add N row
    n_row = {"Algorithm": "N (quality pairs)"}
    for mode_name in MODES:
        n_row[f"{mode_name} SR"] = 100
        n_row[f"{mode_name} PL(m)"] = n_quality_row[mode_name]
        n_row[f"{mode_name} K(1/m)"] = ""
        n_row[f"{mode_name} CT(s)"] = ""
        n_row[f"{mode_name} CS"] = ""
    big = pd.concat([big, pd.DataFrame([n_row])], ignore_index=True)

    # write Excel
    xlsx_path = OUT_DIR / "summary.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        big.to_excel(writer, sheet_name="Summary", index=False)

        # also write per-mode detail sheets with per-run data
        for mode_name, csv_path in MODES.items():
            df = pd.read_csv(csv_path)
            df = df[df["Algorithm"] != DROP_ALGO]
            sheet = mode_name.replace(" ", "_")
            df.to_excel(writer, sheet_name=sheet, index=False)

    print(f"Wrote {xlsx_path}")
    print()
    print(big.to_string(index=False))


if __name__ == "__main__":
    main()
