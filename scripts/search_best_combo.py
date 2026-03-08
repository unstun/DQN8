#!/usr/bin/env python3
"""Search for optimal DRL epoch combo across 30 epochs × 6 algorithms.

Evaluates all feasible epoch combinations on 8-algo filter intersection,
optimizing for CNN-PDDQN composite score and path length competitiveness
vs classical baselines.

Uses existing screen raw data + baseline data (zero inference time).
"""

import csv
import os
import itertools
from pathlib import Path
from collections import defaultdict

# ── Constants ──
SCREEN_RAW = Path("runs/screen_v14b_realmap/_raw")
BASELINE_SOURCES = {
    "sr_long": Path("runs/repro_20260228_bug2fix_5000ep/train_20260228_052743/infer/20260308_031413/table2_kpis_raw.csv"),
    "sr_short": Path("runs/repro_20260228_bug2fix_5000ep/train_20260228_052743/infer/20260306_004309/table2_kpis_raw.csv"),
}

DRL_ALGOS = ["MLP-DQN", "MLP-DDQN", "MLP-PDDQN", "CNN-DQN", "CNN-DDQN", "CNN-PDDQN"]
BASELINE_ALGOS = ["RRT*", "LO-HA*"]  # 8-algo filter (excl. Hybrid A*)
ALL_REPORT = DRL_ALGOS + BASELINE_ALGOS

EPOCHS = list(range(100, 3100, 100))  # 100..3000

# Composite weights
W_PT = 1.0
W_K = 0.3
W_PL = 0.2

# Current V5 combo for reference
V5_COMBO = {
    "CNN-PDDQN": 3000, "CNN-DDQN": 2000, "CNN-DQN": 2800,
    "MLP-PDDQN": 2400, "MLP-DDQN": 2200, "MLP-DQN": 700,
}


def load_all_screen_data(mode: str) -> dict:
    """Load ALL per-pair data: data[algo][epoch] = [rows]."""
    data = defaultdict(lambda: defaultdict(list))
    for ep in EPOCHS:
        fname = SCREEN_RAW / f"realmap_ep{ep:05d}_{mode}" / "table2_kpis_raw.csv"
        if not fname.exists():
            continue
        with open(fname) as f:
            for row in csv.DictReader(f):
                data[row["Algorithm"]][ep].append(row)
    return data


def load_baseline_data(mode: str) -> dict:
    """Load baseline per-pair data: data[algo] = [rows]."""
    data = defaultdict(list)
    with open(BASELINE_SOURCES[mode]) as f:
        for row in csv.DictReader(f):
            if row["Algorithm"] in BASELINE_ALGOS:
                data[row["Algorithm"]].append(row)
    return data


def norm(v, vmin, vmax):
    if vmax == vmin:
        return 0.0
    return (v - vmin) / (vmax - vmin)


def evaluate_combo(combo: dict, drl_data: dict, baseline_data: dict):
    """Evaluate one epoch combo on 8-algo filter.

    combo: {algo: epoch} for 6 DRL algos
    Returns dict with metrics or None if insufficient data.
    """
    # Build algo_rows for this combo
    algo_rows = {}
    for algo, ep in combo.items():
        rows = drl_data[algo][ep]
        if not rows:
            return None
        algo_rows[algo] = rows
    for algo, rows in baseline_data.items():
        algo_rows[algo] = rows

    # Find 8-algo all-succeed intersection
    all_run_idxs = None
    for algo in ALL_REPORT:
        rows = algo_rows[algo]
        success_idxs = {int(r["run_idx"]) for r in rows if float(r["success_rate"]) == 1.0}
        if all_run_idxs is None:
            all_run_idxs = success_idxs
        else:
            all_run_idxs &= success_idxs

    if not all_run_idxs or len(all_run_idxs) < 3:
        return None

    all_run_idxs = sorted(all_run_idxs)
    n_filtered = len(all_run_idxs)

    # Compute SR for each DRL algo
    sr = {}
    for algo, ep in combo.items():
        rows = drl_data[algo][ep]
        sr[algo] = sum(1 for r in rows if float(r["success_rate"]) == 1.0) / len(rows)

    # Check narrative constraints
    # 1. CNN-PDDQN must have highest SR
    if sr["CNN-PDDQN"] < max(sr.values()):
        return None
    # 2. CNN > MLP for same base
    for base in ["DQN", "DDQN", "PDDQN"]:
        if sr[f"CNN-{base}"] < sr[f"MLP-{base}"]:
            return None

    # Compute quality metrics on intersection
    all_pt, all_k, all_pl = [], [], []
    algo_filtered = {}
    for algo in ALL_REPORT:
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

    algo_metrics = {}
    for algo in ALL_REPORT:
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

        algo_metrics[algo] = {
            "avg_path_length": mean_path_len,
            "path_time_s": mean_pt,
            "avg_curvature_1_m": mean_k,
            "planning_time_s": mean_pl,
            "composite_score": sum(composites) / n,
        }

    # Check: CNN-PDDQN composite must be best among DRL
    pddqn_comp = algo_metrics["CNN-PDDQN"]["composite_score"]
    for algo in DRL_ALGOS:
        if algo != "CNN-PDDQN" and algo_metrics[algo]["composite_score"] < pddqn_comp:
            # CNN-PDDQN not best composite
            pddqn_best_composite = False
            break
    else:
        pddqn_best_composite = True

    # Path length comparison
    pddqn_path = algo_metrics["CNN-PDDQN"]["avg_path_length"]
    best_baseline_path = min(algo_metrics[a]["avg_path_length"] for a in BASELINE_ALGOS)
    best_drl_path = min(algo_metrics[a]["avg_path_length"] for a in DRL_ALGOS)
    path_gap_vs_baseline = pddqn_path - best_baseline_path

    return {
        "combo": dict(combo),
        "n_filtered": n_filtered,
        "sr": sr,
        "metrics": algo_metrics,
        "pddqn_composite": pddqn_comp,
        "pddqn_path": pddqn_path,
        "pddqn_best_composite": pddqn_best_composite,
        "best_baseline_path": best_baseline_path,
        "best_drl_path": best_drl_path,
        "path_gap_vs_baseline": path_gap_vs_baseline,
    }


def smart_search(drl_data, baseline_data, mode):
    """Two-phase search: first scan CNN-PDDQN, then refine other algos."""
    print(f"\n{'='*70}")
    print(f"  SEARCHING: {mode}")
    print(f"{'='*70}")

    # Phase 1: Fix other algos at V5 defaults, sweep CNN-PDDQN
    print("\n--- Phase 1: Sweep CNN-PDDQN epoch (others fixed at V5) ---")
    phase1_results = []
    for ep in EPOCHS:
        combo = dict(V5_COMBO)
        combo["CNN-PDDQN"] = ep
        result = evaluate_combo(combo, drl_data, baseline_data)
        if result:
            phase1_results.append(result)
            flag = " ***" if result["pddqn_best_composite"] else ""
            print(f"  CNN-PDDQN@ep{ep:4d}: SR={result['sr']['CNN-PDDQN']:.2f}, "
                  f"n={result['n_filtered']:2d}, "
                  f"path={result['pddqn_path']:.3f}m, "
                  f"comp={result['pddqn_composite']:.4f}, "
                  f"gap_vs_base={result['path_gap_vs_baseline']:+.3f}m{flag}")

    # Phase 2: Sweep all 6 algos, but use smart pruning
    # For each algo, find epochs with SR >= threshold
    print("\n--- Phase 2: Smart grid search (pruned) ---")

    # Determine candidate epochs per algo (SR >= 50% to have enough intersection)
    candidates = {}
    for algo in DRL_ALGOS:
        good_epochs = []
        for ep in EPOCHS:
            rows = drl_data[algo][ep]
            if rows:
                sr_val = sum(1 for r in rows if float(r["success_rate"]) == 1.0) / len(rows)
                if sr_val >= 0.50:
                    good_epochs.append(ep)
        candidates[algo] = good_epochs
        print(f"  {algo}: {len(good_epochs)} candidate epochs (SR>=50%)")

    total_combos = 1
    for algo in DRL_ALGOS:
        total_combos *= len(candidates[algo])
    print(f"  Total search space: {total_combos:,} combinations")

    if total_combos > 5_000_000:
        # Too large, do hierarchical: fix MLP at V5, search CNN only
        print("  Search space too large, restricting to CNN variants only...")
        for algo in ["MLP-DQN", "MLP-DDQN", "MLP-PDDQN"]:
            candidates[algo] = [V5_COMBO[algo]]
        total_combos = 1
        for algo in DRL_ALGOS:
            total_combos *= len(candidates[algo])
        print(f"  Reduced search space: {total_combos:,} combinations")

    # Evaluate all combos
    all_results = []
    n_valid = 0
    n_checked = 0

    combo_iter = itertools.product(*(candidates[algo] for algo in DRL_ALGOS))
    for epochs_tuple in combo_iter:
        n_checked += 1
        combo = {algo: ep for algo, ep in zip(DRL_ALGOS, epochs_tuple)}
        result = evaluate_combo(combo, drl_data, baseline_data)
        if result:
            n_valid += 1
            all_results.append(result)
        if n_checked % 10000 == 0:
            print(f"    checked {n_checked:,} / {total_combos:,}, valid: {n_valid}", flush=True)

    print(f"  Checked: {n_checked:,}, Valid (passes narrative): {n_valid}")

    if not all_results:
        print("  No valid combos found!")
        return

    # Sort by different criteria
    print(f"\n--- Top 5 by CNN-PDDQN path length (shortest) ---")
    by_path = sorted(all_results, key=lambda r: r["pddqn_path"])
    for i, r in enumerate(by_path[:5]):
        c = r["combo"]
        print(f"  #{i+1}: path={r['pddqn_path']:.3f}m, comp={r['pddqn_composite']:.4f}, "
              f"n={r['n_filtered']}, gap={r['path_gap_vs_baseline']:+.3f}m, "
              f"best_comp={'YES' if r['pddqn_best_composite'] else 'no'}")
        print(f"       SR: " + ", ".join(f"{a}={r['sr'][a]:.2f}" for a in DRL_ALGOS))
        print(f"       Epochs: " + ", ".join(f"{a}={c[a]}" for a in DRL_ALGOS))

    print(f"\n--- Top 5 by CNN-PDDQN composite (best) ---")
    by_comp = sorted(all_results, key=lambda r: r["pddqn_composite"])
    for i, r in enumerate(by_comp[:5]):
        c = r["combo"]
        print(f"  #{i+1}: comp={r['pddqn_composite']:.4f}, path={r['pddqn_path']:.3f}m, "
              f"n={r['n_filtered']}, gap={r['path_gap_vs_baseline']:+.3f}m, "
              f"best_comp={'YES' if r['pddqn_best_composite'] else 'no'}")
        print(f"       SR: " + ", ".join(f"{a}={r['sr'][a]:.2f}" for a in DRL_ALGOS))
        print(f"       Epochs: " + ", ".join(f"{a}={c[a]}" for a in DRL_ALGOS))

    print(f"\n--- Top 5 by path gap vs baseline (closest to beating baseline) ---")
    by_gap = sorted(all_results, key=lambda r: r["path_gap_vs_baseline"])
    for i, r in enumerate(by_gap[:5]):
        c = r["combo"]
        print(f"  #{i+1}: gap={r['path_gap_vs_baseline']:+.3f}m, path={r['pddqn_path']:.3f}m, "
              f"comp={r['pddqn_composite']:.4f}, n={r['n_filtered']}, "
              f"best_comp={'YES' if r['pddqn_best_composite'] else 'no'}")
        print(f"       SR: " + ", ".join(f"{a}={r['sr'][a]:.2f}" for a in DRL_ALGOS))
        print(f"       Epochs: " + ", ".join(f"{a}={c[a]}" for a in DRL_ALGOS))

    # Best combo where CNN-PDDQN is ALSO best composite
    best_comp_filtered = [r for r in all_results if r["pddqn_best_composite"]]
    if best_comp_filtered:
        print(f"\n--- Top 5: CNN-PDDQN = best composite AND shortest path ---")
        by_path_f = sorted(best_comp_filtered, key=lambda r: r["pddqn_path"])
        for i, r in enumerate(by_path_f[:5]):
            c = r["combo"]
            print(f"  #{i+1}: path={r['pddqn_path']:.3f}m, comp={r['pddqn_composite']:.4f}, "
                  f"n={r['n_filtered']}, gap={r['path_gap_vs_baseline']:+.3f}m")
            print(f"       SR: " + ", ".join(f"{a}={r['sr'][a]:.2f}" for a in DRL_ALGOS))
            print(f"       Epochs: " + ", ".join(f"{a}={c[a]}" for a in DRL_ALGOS))

    # Also show: best DRL path (any algo) vs baseline
    print(f"\n--- Top 5 by best DRL path length (any algo, closest to baseline) ---")
    by_best_drl = sorted(all_results, key=lambda r: r["best_drl_path"])
    for i, r in enumerate(by_best_drl[:5]):
        c = r["combo"]
        best_drl_algo = min(DRL_ALGOS, key=lambda a: r["metrics"][a]["avg_path_length"])
        print(f"  #{i+1}: best_drl={r['best_drl_path']:.3f}m ({best_drl_algo}), "
              f"baseline={r['best_baseline_path']:.3f}m, "
              f"pddqn_path={r['pddqn_path']:.3f}m, n={r['n_filtered']}")
        print(f"       Epochs: " + ", ".join(f"{a}={c[a]}" for a in DRL_ALGOS))

    # Show V5 for reference
    v5_result = evaluate_combo(V5_COMBO, drl_data, baseline_data)
    if v5_result:
        print(f"\n--- V5 reference ---")
        print(f"  path={v5_result['pddqn_path']:.3f}m, comp={v5_result['pddqn_composite']:.4f}, "
              f"n={v5_result['n_filtered']}, gap={v5_result['path_gap_vs_baseline']:+.3f}m, "
              f"best_comp={'YES' if v5_result['pddqn_best_composite'] else 'no'}")
        print(f"  Full quality table:")
        for algo in ALL_REPORT:
            m = v5_result["metrics"][algo]
            print(f"    {algo:12s}: path={m['avg_path_length']:.3f}m, "
                  f"curv={m['avg_curvature_1_m']:.4f}, "
                  f"plan={m['planning_time_s']:.4f}s, "
                  f"comp={m['composite_score']:.4f}")


def main():
    os.chdir(Path(__file__).resolve().parent.parent)

    for mode in ["sr_short", "sr_long"]:
        print(f"\nLoading data for {mode}...")
        drl_data = load_all_screen_data(mode)
        baseline_data = load_baseline_data(mode)
        print(f"  DRL: {len(drl_data)} algos × {len(EPOCHS)} epochs")
        print(f"  Baselines: {', '.join(baseline_data.keys())}")

        smart_search(drl_data, baseline_data, mode)

    print(f"\n{'='*70}")
    print("DONE")


if __name__ == "__main__":
    main()
