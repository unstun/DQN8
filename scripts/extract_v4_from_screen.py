#!/usr/bin/env python3
"""Extract V4 realmap results from screen raw data.

Uses per-pair raw data from screen_v14b_realmap to construct
mixed-epoch checkpoint combo results without re-running inference.

Key insight: all screen epochs use seed=0, same pairs for same mode.
So per-pair results can be cross-referenced across epochs.
"""

import csv
import os
import sys
from pathlib import Path
from collections import defaultdict

# V4 checkpoint combo (algo -> best epoch)
COMBO = {
    "CNN-PDDQN": 3000,
    "CNN-DDQN": 2000,
    "CNN-DQN": 2800,
    "MLP-PDDQN": 2400,
    "MLP-DDQN": 2200,
    "MLP-DQN": 700,
}

RAW_DIR = Path("runs/screen_v14b_realmap/_raw")
OUT_DIR = Path("runs/snapshot_20260308_realmap_v4/results")

MODES = ["sr_long", "sr_short"]  # quality derived from SR 100-run data

# Composite score weights (from CLAUDE.md)
W_PT = 1.0   # path_time_s
W_K = 0.3    # avg_curvature_1_m
W_PL = 0.2   # planning_time_s


def load_raw(mode: str, algo: str, epoch: int) -> list[dict]:
    """Load per-pair raw data for one algo at one epoch."""
    fname = RAW_DIR / f"realmap_ep{epoch:05d}_{mode}" / "table2_kpis_raw.csv"
    if not fname.exists():
        print(f"  WARNING: {fname} not found")
        return []
    rows = []
    with open(fname) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["Algorithm"] == algo:
                rows.append(row)
    return rows


def compute_sr(rows: list[dict]) -> float:
    """Compute success rate from per-pair data."""
    if not rows:
        return 0.0
    n_success = sum(1 for r in rows if float(r["success_rate"]) == 1.0)
    return n_success / len(rows)


def minmax_norm(values: list[float]) -> list[float]:
    """Min-max normalize values (lower is better)."""
    if not values:
        return []
    vmin, vmax = min(values), max(values)
    if vmax == vmin:
        return [0.0] * len(values)
    return [(v - vmin) / (vmax - vmin) for v in values]


def compute_quality_metrics(algo_rows: dict[str, list[dict]]) -> dict:
    """Compute quality metrics on all-succeed intersection.

    algo_rows: {algo_name: [per-pair rows from raw CSV]}
    Returns dict with per-algo quality metrics and intersection info.
    """
    # Find all-succeed intersection by run_idx
    all_run_idxs = None
    for algo, rows in algo_rows.items():
        success_idxs = {int(r["run_idx"]) for r in rows if float(r["success_rate"]) == 1.0}
        if all_run_idxs is None:
            all_run_idxs = success_idxs
        else:
            all_run_idxs &= success_idxs

    if not all_run_idxs:
        return {"n_filtered": 0}

    all_run_idxs = sorted(all_run_idxs)
    n_filtered = len(all_run_idxs)

    # Extract metrics for each algo on filtered pairs
    results = {"n_filtered": n_filtered, "filtered_pairs": all_run_idxs, "algos": {}}

    # Collect raw values for minmax normalization
    all_pt = []  # path_time_s
    all_k = []   # avg_curvature_1_m
    all_pl = []  # planning_time_s

    algo_metrics = {}
    for algo, rows in algo_rows.items():
        idx_to_row = {int(r["run_idx"]): r for r in rows}
        filtered = [idx_to_row[i] for i in all_run_idxs if i in idx_to_row]

        pts = [float(r["path_time_s"]) for r in filtered]
        ks = [float(r["avg_curvature_1_m"]) for r in filtered]
        pls = [float(r["planning_time_s"]) for r in filtered]
        path_lens = [float(r["avg_path_length"]) for r in filtered]

        algo_metrics[algo] = {
            "path_time_s": pts,
            "avg_curvature_1_m": ks,
            "planning_time_s": pls,
            "avg_path_length": path_lens,
        }
        all_pt.extend(pts)
        all_k.extend(ks)
        all_pl.extend(pls)

    # Minmax normalize across all algos
    pt_min, pt_max = min(all_pt), max(all_pt)
    k_min, k_max = min(all_k), max(all_k)
    pl_min, pl_max = min(all_pl), max(all_pl)

    def norm(v, vmin, vmax):
        if vmax == vmin:
            return 0.0
        return (v - vmin) / (vmax - vmin)

    for algo, metrics in algo_metrics.items():
        n = len(metrics["path_time_s"])
        mean_pt = sum(metrics["path_time_s"]) / n
        mean_k = sum(metrics["avg_curvature_1_m"]) / n
        mean_pl = sum(metrics["planning_time_s"]) / n
        mean_path_len = sum(metrics["avg_path_length"]) / n

        # Composite score per pair, then average
        composites = []
        for i in range(n):
            n_pt = norm(metrics["path_time_s"][i], pt_min, pt_max)
            n_k = norm(metrics["avg_curvature_1_m"][i], k_min, k_max)
            n_pl = norm(metrics["planning_time_s"][i], pl_min, pl_max)
            cs = (W_PT * n_pt + W_K * n_k + W_PL * n_pl) / (W_PT + W_K + W_PL)
            composites.append(cs)

        mean_composite = sum(composites) / n

        results["algos"][algo] = {
            "avg_path_length": mean_path_len,
            "path_time_s": mean_pt,
            "avg_curvature_1_m": mean_k,
            "planning_time_s": mean_pl,
            "composite_score": mean_composite,
        }

    return results


def write_sr_table(mode: str, algo_data: dict[str, list[dict]], out_dir: Path):
    """Write SR results table."""
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "table2_kpis.csv"
    md_path = out_dir / "table2_kpis.md"
    raw_path = out_dir / "table2_kpis_raw.csv"

    # Aggregated table
    rows = []
    for algo in ["MLP-DQN", "MLP-DDQN", "MLP-PDDQN", "CNN-DQN", "CNN-DDQN", "CNN-PDDQN"]:
        pair_rows = algo_data[algo]
        sr = compute_sr(pair_rows)
        # Compute mean metrics for successful runs only
        success_rows = [r for r in pair_rows if float(r["success_rate"]) == 1.0]
        if success_rows:
            mean_pl = sum(float(r["avg_path_length"]) for r in success_rows) / len(success_rows)
            mean_pt = sum(float(r["path_time_s"]) for r in success_rows) / len(success_rows)
            mean_k = sum(float(r["avg_curvature_1_m"]) for r in success_rows) / len(success_rows)
            mean_ct = sum(float(r["planning_time_s"]) for r in success_rows) / len(success_rows)
        else:
            mean_pl = mean_pt = mean_k = mean_ct = 0.0

        rows.append({
            "Environment": "Env. (realmap_a)",
            "Algorithm": algo,
            "success_rate": f"{sr:.2f}",
            "avg_path_length": f"{mean_pl:.4f}",
            "path_time_s": f"{mean_pt:.4f}",
            "avg_curvature_1_m": f"{mean_k:.6f}",
            "planning_time_s": f"{mean_ct:.5f}",
        })

    # Write CSV
    fields = ["Environment", "Algorithm", "success_rate", "avg_path_length",
              "path_time_s", "avg_curvature_1_m", "planning_time_s"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # Write MD
    with open(md_path, "w") as f:
        f.write(f"# SR Results - {mode}\n\n")
        f.write(f"| Algorithm | SR | Avg Path Len | Path Time (s) | Avg Curvature | Planning Time (s) |\n")
        f.write(f"|-----------|-----|-------------|---------------|---------------|-------------------|\n")
        for r in rows:
            f.write(f"| {r['Algorithm']} | {r['success_rate']} | {r['avg_path_length']} | "
                    f"{r['path_time_s']} | {r['avg_curvature_1_m']} | {r['planning_time_s']} |\n")

    # Write raw per-pair CSV
    all_raw = []
    for algo in ["MLP-DQN", "MLP-DDQN", "MLP-PDDQN", "CNN-DQN", "CNN-DDQN", "CNN-PDDQN"]:
        all_raw.extend(algo_data[algo])

    if all_raw:
        raw_fields = list(all_raw[0].keys())
        with open(raw_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=raw_fields)
            w.writeheader()
            w.writerows(all_raw)

    print(f"  Written: {csv_path}")
    print(f"  Written: {md_path}")
    print(f"  Written: {raw_path}")


def write_quality_table(mode: str, quality_results: dict, out_dir: Path):
    """Write quality (filtered) results table."""
    out_dir.mkdir(parents=True, exist_ok=True)

    n_filtered = quality_results["n_filtered"]
    algos = quality_results["algos"]

    csv_path = out_dir / "table2_kpis_filtered.csv"
    md_path = out_dir / "table2_kpis_filtered.md"

    rows = []
    for algo in ["MLP-DQN", "MLP-DDQN", "MLP-PDDQN", "CNN-DQN", "CNN-DDQN", "CNN-PDDQN"]:
        m = algos[algo]
        rows.append({
            "Environment": "Env. (realmap_a)",
            "Algorithm": algo,
            "n_filtered_pairs": n_filtered,
            "avg_path_length": f"{m['avg_path_length']:.4f}",
            "path_time_s": f"{m['path_time_s']:.4f}",
            "avg_curvature_1_m": f"{m['avg_curvature_1_m']:.6f}",
            "planning_time_s": f"{m['planning_time_s']:.5f}",
            "composite_score": f"{m['composite_score']:.4f}",
        })

    # Write CSV
    fields = ["Environment", "Algorithm", "n_filtered_pairs", "avg_path_length",
              "path_time_s", "avg_curvature_1_m", "planning_time_s", "composite_score"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # Write MD
    with open(md_path, "w") as f:
        f.write(f"# Quality Results (filter_all_succeed) - {mode}\n")
        f.write(f"# Filtered pairs: {n_filtered} / 100\n\n")
        f.write(f"| Algorithm | Path Len | Path Time (s) | Curvature | Planning Time (s) | Composite |\n")
        f.write(f"|-----------|----------|---------------|-----------|-------------------|-----------|\n")
        for r in rows:
            f.write(f"| {r['Algorithm']} | {r['avg_path_length']} | {r['path_time_s']} | "
                    f"{r['avg_curvature_1_m']} | {r['planning_time_s']} | {r['composite_score']} |\n")

    print(f"  Written: {csv_path}")
    print(f"  Written: {md_path}")


def main():
    os.chdir(Path(__file__).resolve().parent.parent)

    print("=" * 60)
    print("V4 Realmap: Extract results from screen raw data")
    print("=" * 60)
    print(f"\nCheckpoint combo:")
    for algo, ep in COMBO.items():
        print(f"  {algo}: ep{ep}")

    for mode in MODES:
        print(f"\n{'=' * 60}")
        print(f"Processing: {mode}")
        print(f"{'=' * 60}")

        # Load per-pair data for each algo at its best epoch
        algo_data = {}
        for algo, epoch in COMBO.items():
            rows = load_raw(mode, algo, epoch)
            algo_data[algo] = rows
            sr = compute_sr(rows)
            print(f"  {algo} @ ep{epoch}: {len(rows)} pairs, SR={sr:.2f}")

        # Determine output mode name
        if "long" in mode:
            dist = "long"
        else:
            dist = "short"

        # SR output
        sr_dir = OUT_DIR / f"sr_{dist}"
        write_sr_table(f"sr_{dist}", algo_data, sr_dir)

        # Quality (filter_all_succeed) output
        quality_dir = OUT_DIR / f"quality_{dist}"
        quality_results = compute_quality_metrics(algo_data)
        n = quality_results["n_filtered"]
        print(f"\n  filter_all_succeed: {n}/100 pairs passed")

        if n > 0:
            write_quality_table(f"quality_{dist}", quality_results, quality_dir)
            print(f"\n  Quality metrics (filtered):")
            for algo in ["MLP-DQN", "MLP-DDQN", "MLP-PDDQN", "CNN-DQN", "CNN-DDQN", "CNN-PDDQN"]:
                m = quality_results["algos"][algo]
                print(f"    {algo}: path={m['avg_path_length']:.2f}m, "
                      f"curv={m['avg_curvature_1_m']:.4f}, "
                      f"time={m['planning_time_s']:.4f}s, "
                      f"composite={m['composite_score']:.4f}")
        else:
            print("  WARNING: No pairs passed filter_all_succeed!")

    print(f"\n{'=' * 60}")
    print("Done! Results in:", OUT_DIR)
    print("NOTE: Baselines not included (not in screen data).")
    print("      Run baselines separately if needed.")


if __name__ == "__main__":
    main()
