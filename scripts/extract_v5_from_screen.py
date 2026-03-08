#!/usr/bin/env python3
"""Extract V5 realmap results: 9 algorithms from existing data.

DRL (6 algos): from screen_v14b_realmap raw per-pair data, mixed-epoch combo.
Baselines (3 algos): from existing repro inference runs (same seed=0, same pairs).

No new inference needed.
"""

import csv
import os
from pathlib import Path

# ── V5 checkpoint combo (algo -> best epoch) ──
COMBO = {
    "CNN-PDDQN": 3000,
    "CNN-DDQN": 2000,
    "CNN-DQN": 2800,
    "MLP-PDDQN": 2400,
    "MLP-DDQN": 2200,
    "MLP-DQN": 700,
}

ALGO_ORDER = ["MLP-DQN", "MLP-DDQN", "MLP-PDDQN",
              "CNN-DQN", "CNN-DDQN", "CNN-PDDQN",
              "Hybrid A*", "RRT*", "LO-HA*"]

DRL_ALGOS = [a for a in ALGO_ORDER if a not in ("Hybrid A*", "RRT*", "LO-HA*")]
BASELINE_ALGOS = ["Hybrid A*", "RRT*", "LO-HA*"]

# 8-algo filter: exclude Hybrid A* (SR too low for quality long)
FILTER8_ALGOS = [a for a in ALGO_ORDER if a != "Hybrid A*"]

# ── Data sources ──
SCREEN_RAW = Path("runs/screen_v14b_realmap/_raw")
BASELINE_SOURCES = {
    "sr_long": Path("runs/repro_20260228_bug2fix_5000ep/train_20260228_052743/infer/20260308_031413/table2_kpis_raw.csv"),
    "sr_short": Path("runs/repro_20260228_bug2fix_5000ep/train_20260228_052743/infer/20260306_004309/table2_kpis_raw.csv"),
}

OUT_DIR = Path("runs/snapshot_20260308_realmap_v5/results")

# Composite score weights (from CLAUDE.md)
W_PT = 1.0   # path_time_s
W_K = 0.3    # avg_curvature_1_m
W_PL = 0.2   # planning_time_s


def load_screen_raw(mode: str, algo: str, epoch: int) -> list[dict]:
    """Load per-pair raw data for one DRL algo at one epoch from screen."""
    fname = SCREEN_RAW / f"realmap_ep{epoch:05d}_{mode}" / "table2_kpis_raw.csv"
    rows = []
    with open(fname) as f:
        for row in csv.DictReader(f):
            if row["Algorithm"] == algo:
                rows.append(row)
    return rows


def load_baseline_raw(mode: str, algo: str) -> list[dict]:
    """Load per-pair raw data for one baseline algo."""
    fname = BASELINE_SOURCES[mode]
    rows = []
    with open(fname) as f:
        for row in csv.DictReader(f):
            if row["Algorithm"] == algo:
                rows.append(row)
    return rows


def compute_sr(rows: list[dict]) -> float:
    if not rows:
        return 0.0
    return sum(1 for r in rows if float(r["success_rate"]) == 1.0) / len(rows)


def mean_of(rows: list[dict], key: str) -> float:
    vals = [float(r[key]) for r in rows]
    return sum(vals) / len(vals) if vals else 0.0


def norm(v, vmin, vmax):
    if vmax == vmin:
        return 0.0
    return (v - vmin) / (vmax - vmin)


def compute_quality(algo_rows: dict[str, list[dict]],
                    filter_algos: list[str] | None = None,
                    report_algos: list[str] | None = None) -> dict:
    """Compute quality metrics on all-succeed intersection.

    filter_algos: which algos must ALL succeed for a pair to pass (default: all)
    report_algos: which algos to report metrics for (default: all in algo_rows)
    """
    if filter_algos is None:
        filter_algos = list(algo_rows.keys())
    if report_algos is None:
        report_algos = list(algo_rows.keys())

    all_run_idxs = None
    for algo in filter_algos:
        rows = algo_rows[algo]
        success_idxs = {int(r["run_idx"]) for r in rows if float(r["success_rate"]) == 1.0}
        if all_run_idxs is None:
            all_run_idxs = success_idxs
        else:
            all_run_idxs &= success_idxs

    if not all_run_idxs:
        return {"n_filtered": 0}

    all_run_idxs = sorted(all_run_idxs)
    n_filtered = len(all_run_idxs)

    # Collect all values for minmax normalization
    all_pt, all_k, all_pl = [], [], []
    algo_filtered = {}
    for algo in report_algos:
        rows = algo_rows[algo]
        idx_map = {int(r["run_idx"]): r for r in rows}
        filtered = [idx_map[i] for i in all_run_idxs]
        algo_filtered[algo] = filtered
        all_pt.extend(float(r["path_time_s"]) for r in filtered)
        all_k.extend(float(r["avg_curvature_1_m"]) for r in filtered)
        all_pl.extend(float(r["planning_time_s"]) for r in filtered)

    pt_min, pt_max = min(all_pt), max(all_pt)
    k_min, k_max = min(all_k), max(all_k)
    pl_min, pl_max = min(all_pl), max(all_pl)

    results = {"n_filtered": n_filtered, "filtered_run_idxs": all_run_idxs, "algos": {}}
    for algo in report_algos:
        filtered = algo_filtered[algo]
        n = len(filtered)
        mean_pt = sum(float(r["path_time_s"]) for r in filtered) / n
        mean_k = sum(float(r["avg_curvature_1_m"]) for r in filtered) / n
        mean_pl = sum(float(r["planning_time_s"]) for r in filtered) / n
        mean_path_len = sum(float(r["avg_path_length"]) for r in filtered) / n

        composites = []
        for r in filtered:
            n_pt = norm(float(r["path_time_s"]), pt_min, pt_max)
            n_k = norm(float(r["avg_curvature_1_m"]), k_min, k_max)
            n_pl = norm(float(r["planning_time_s"]), pl_min, pl_max)
            cs = (W_PT * n_pt + W_K * n_k + W_PL * n_pl) / (W_PT + W_K + W_PL)
            composites.append(cs)

        results["algos"][algo] = {
            "avg_path_length": mean_path_len,
            "path_time_s": mean_pt,
            "avg_curvature_1_m": mean_k,
            "planning_time_s": mean_pl,
            "composite_score": sum(composites) / n,
        }
    return results


def write_sr_table(mode: str, algo_rows: dict[str, list[dict]], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_out = []
    for algo in ALGO_ORDER:
        pair_rows = algo_rows[algo]
        sr = compute_sr(pair_rows)
        succ = [r for r in pair_rows if float(r["success_rate"]) == 1.0]
        rows_out.append({
            "Algorithm": algo,
            "success_rate": f"{sr:.2f}",
            "avg_path_length": f"{mean_of(succ, 'avg_path_length'):.4f}" if succ else "N/A",
            "path_time_s": f"{mean_of(succ, 'path_time_s'):.4f}" if succ else "N/A",
            "avg_curvature_1_m": f"{mean_of(succ, 'avg_curvature_1_m'):.6f}" if succ else "N/A",
            "planning_time_s": f"{mean_of(succ, 'planning_time_s'):.5f}" if succ else "N/A",
        })

    fields = ["Algorithm", "success_rate", "avg_path_length", "path_time_s",
              "avg_curvature_1_m", "planning_time_s"]
    with open(out_dir / "table2_kpis.csv", "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()
        csv.DictWriter(f, fieldnames=fields).writerows(rows_out)

    with open(out_dir / "table2_kpis.md", "w") as f:
        f.write(f"# SR Results - {mode} (100 runs, 9 algorithms)\n\n")
        f.write("| Algorithm | SR | Path Len (m) | Path Time (s) | Curvature (1/m) | Planning Time (s) |\n")
        f.write("|-----------|-----|-------------|---------------|-----------------|-------------------|\n")
        for r in rows_out:
            f.write(f"| {r['Algorithm']} | {r['success_rate']} | {r['avg_path_length']} | "
                    f"{r['path_time_s']} | {r['avg_curvature_1_m']} | {r['planning_time_s']} |\n")

    # Raw per-pair CSV
    all_raw = []
    for algo in ALGO_ORDER:
        all_raw.extend(algo_rows[algo])
    if all_raw:
        raw_fields = list(all_raw[0].keys())
        with open(out_dir / "table2_kpis_raw.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=raw_fields)
            w.writeheader()
            w.writerows(all_raw)

    print(f"  Written: {out_dir}/table2_kpis.csv, .md, _raw.csv")


def write_quality_table(mode: str, quality: dict, out_dir: Path,
                        algo_order: list[str] | None = None, label: str = "9 algos"):
    out_dir.mkdir(parents=True, exist_ok=True)
    if algo_order is None:
        algo_order = ALGO_ORDER
    n_f = quality["n_filtered"]
    rows_out = []
    for algo in algo_order:
        m = quality["algos"][algo]
        rows_out.append({
            "Algorithm": algo,
            "n_filtered_pairs": n_f,
            "avg_path_length": f"{m['avg_path_length']:.4f}",
            "path_time_s": f"{m['path_time_s']:.4f}",
            "avg_curvature_1_m": f"{m['avg_curvature_1_m']:.6f}",
            "planning_time_s": f"{m['planning_time_s']:.5f}",
            "composite_score": f"{m['composite_score']:.4f}",
        })

    fields = ["Algorithm", "n_filtered_pairs", "avg_path_length", "path_time_s",
              "avg_curvature_1_m", "planning_time_s", "composite_score"]
    with open(out_dir / "table2_kpis_filtered.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows_out)

    with open(out_dir / "table2_kpis_filtered.md", "w") as f:
        f.write(f"# Quality Results (filter_all_succeed, {label}) - {mode}\n")
        f.write(f"# Filtered pairs: {n_f} / 100\n\n")
        f.write("| Algorithm | Path Len (m) | Path Time (s) | Curvature (1/m) | Planning Time (s) | Composite |\n")
        f.write("|-----------|-------------|---------------|-----------------|-------------------|-----------|\n")
        for r in rows_out:
            f.write(f"| {r['Algorithm']} | {r['avg_path_length']} | {r['path_time_s']} | "
                    f"{r['avg_curvature_1_m']} | {r['planning_time_s']} | {r['composite_score']} |\n")

    print(f"  Written: {out_dir}/table2_kpis_filtered.csv, .md")


def main():
    os.chdir(Path(__file__).resolve().parent.parent)

    print("=" * 60)
    print("V5 Realmap: 9-algorithm extraction from existing data")
    print("=" * 60)
    print("\nDRL checkpoint combo:")
    for algo, ep in COMBO.items():
        print(f"  {algo}: ep{ep}")
    print("\nBaseline sources:")
    for mode, src in BASELINE_SOURCES.items():
        print(f"  {mode}: {src}")

    for mode in ["sr_long", "sr_short"]:
        dist = "long" if "long" in mode else "short"
        print(f"\n{'=' * 60}")
        print(f"Processing: {mode}")
        print(f"{'=' * 60}")

        # Load DRL per-pair data
        algo_rows = {}
        for algo, epoch in COMBO.items():
            rows = load_screen_raw(mode, algo, epoch)
            algo_rows[algo] = rows
            sr = compute_sr(rows)
            print(f"  [DRL]  {algo:12s} @ ep{epoch}: {len(rows):3d} pairs, SR={sr:.2f}")

        # Load baseline per-pair data
        for algo in BASELINE_ALGOS:
            rows = load_baseline_raw(mode, algo)
            algo_rows[algo] = rows
            sr = compute_sr(rows)
            print(f"  [BASE] {algo:12s}         : {len(rows):3d} pairs, SR={sr:.2f}")

        # Verify pair alignment
        ref_pairs = [(int(r["run_idx"]), r["start_x"], r["start_y"], r["goal_x"], r["goal_y"])
                     for r in algo_rows["CNN-PDDQN"]]
        for algo in BASELINE_ALGOS:
            base_pairs = [(int(r["run_idx"]), r["start_x"], r["start_y"], r["goal_x"], r["goal_y"])
                          for r in algo_rows[algo]]
            if ref_pairs != base_pairs:
                print(f"  WARNING: {algo} pairs don't match DRL pairs!")
                return
        print(f"  ✓ All 9 algorithms on same 100 pairs")

        # Write SR results
        sr_dir = OUT_DIR / f"sr_{dist}"
        write_sr_table(f"sr_{dist}", algo_rows, sr_dir)

        # Compute & write quality (filter_all_succeed across all 9)
        quality = compute_quality(algo_rows)
        n_f = quality["n_filtered"]
        print(f"\n  filter_all_succeed (9 algos): {n_f}/100 pairs")

        if n_f > 0:
            quality_dir = OUT_DIR / f"quality_{dist}"
            write_quality_table(f"quality_{dist}", quality, quality_dir,
                                algo_order=ALGO_ORDER, label="9 algos")
            print(f"\n  {'Algorithm':12s} {'Path(m)':>8s} {'Curv':>8s} {'PlanT(s)':>9s} {'Composite':>10s}")
            print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*9} {'-'*10}")
            for algo in ALGO_ORDER:
                m = quality["algos"][algo]
                print(f"  {algo:12s} {m['avg_path_length']:8.2f} {m['avg_curvature_1_m']:8.4f} "
                      f"{m['planning_time_s']:9.4f} {m['composite_score']:10.4f}")

        # 8-algo filter: exclude Hybrid A* (too low SR for quality long)
        quality8 = compute_quality(algo_rows,
                                   filter_algos=FILTER8_ALGOS,
                                   report_algos=FILTER8_ALGOS)
        n_f8 = quality8["n_filtered"]
        print(f"\n  filter_all_succeed (8 algos, excl. Hybrid A*): {n_f8}/100 pairs")

        if n_f8 > 0:
            quality8_dir = OUT_DIR / f"quality_{dist}_8algo"
            write_quality_table(f"quality_{dist}", quality8, quality8_dir,
                                algo_order=FILTER8_ALGOS, label="8 algos, excl. Hybrid A*")
            print(f"\n  {'Algorithm':12s} {'Path(m)':>8s} {'Curv':>8s} {'PlanT(s)':>9s} {'Composite':>10s}")
            print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*9} {'-'*10}")
            for algo in FILTER8_ALGOS:
                m = quality8["algos"][algo]
                print(f"  {algo:12s} {m['avg_path_length']:8.2f} {m['avg_curvature_1_m']:8.4f} "
                      f"{m['planning_time_s']:9.4f} {m['composite_score']:10.4f}")

    print(f"\n{'=' * 60}")
    print(f"Done! Results in: {OUT_DIR}")


if __name__ == "__main__":
    main()
