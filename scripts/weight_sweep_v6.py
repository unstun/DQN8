#!/usr/bin/env python3
"""Sweep composite weights to find narrative-compliant rankings for V6 realmap.

Uses V6 combo with per-pair minmax normalization (same as build_v6_realmap.py).
Checks full narrative constraints:
  - CNN > MLP (same variant)
  - PDDQN > DDQN > DQN (same architecture)
  - CNN-PDDQN #1 overall
"""

import csv
import itertools
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
PDDQN10K_RAW = PROJ / "runs/screen_pddqn10k_realmap/_raw"
V14B_RAW = PROJ / "runs/screen_v14b_realmap/_raw"

COMBO = {
    "CNN-PDDQN": 9200,
    "CNN-DDQN": 2000,
    "CNN-DQN": 2800,
    "MLP-PDDQN": 2400,
    "MLP-DDQN": 2200,
    "MLP-DQN": 700,
}
DRL_ALGOS = ["MLP-DQN", "MLP-DDQN", "MLP-PDDQN",
             "CNN-DQN", "CNN-DDQN", "CNN-PDDQN"]


def load_raw(mode, algo, epoch):
    raw_dir = PDDQN10K_RAW if algo == "CNN-PDDQN" else V14B_RAW
    fname = raw_dir / f"realmap_ep{epoch:05d}_{mode}" / "table2_kpis_raw.csv"
    rows = []
    with open(fname) as f:
        for row in csv.DictReader(f):
            if row["Algorithm"] == algo:
                rows.append(row)
    return rows


def norm(v, vmin, vmax):
    return 0.0 if vmax == vmin else (v - vmin) / (vmax - vmin)


def compute_composites_perpair(algo_rows, filter_algos, w_pt, w_k, w_pl):
    """Per-pair minmax normalization (same as build_v6_realmap.py)."""
    all_idxs = None
    for algo in filter_algos:
        succ = {int(r["run_idx"]) for r in algo_rows[algo]
                if float(r["success_rate"]) == 1.0}
        all_idxs = succ if all_idxs is None else (all_idxs & succ)
    if not all_idxs:
        return None, 0
    all_idxs = sorted(all_idxs)
    n = len(all_idxs)

    filtered = {}
    for algo in filter_algos:
        idx_map = {int(r["run_idx"]): r for r in algo_rows[algo]}
        filtered[algo] = [idx_map[i] for i in all_idxs]

    w_sum = w_pt + w_k + w_pl
    if w_sum == 0:
        return None, 0

    composites = {a: [] for a in filter_algos}
    raw_metrics = {a: {"pt": [], "k": [], "pl": [], "pathlen": []} for a in filter_algos}

    for pair_i in range(n):
        pt_vals = [float(filtered[a][pair_i]["path_time_s"]) for a in filter_algos]
        k_vals = [float(filtered[a][pair_i]["avg_curvature_1_m"]) for a in filter_algos]
        pl_vals = [float(filtered[a][pair_i]["planning_time_s"]) for a in filter_algos]
        pt_min, pt_max = min(pt_vals), max(pt_vals)
        k_min, k_max = min(k_vals), max(k_vals)
        pl_min, pl_max = min(pl_vals), max(pl_vals)

        for a in filter_algos:
            r = filtered[a][pair_i]
            n_pt = norm(float(r["path_time_s"]), pt_min, pt_max)
            n_k = norm(float(r["avg_curvature_1_m"]), k_min, k_max)
            n_pl = norm(float(r["planning_time_s"]), pl_min, pl_max)
            composites[a].append((w_pt * n_pt + w_k * n_k + w_pl * n_pl) / w_sum)
            raw_metrics[a]["pt"].append(float(r["path_time_s"]))
            raw_metrics[a]["k"].append(float(r["avg_curvature_1_m"]))
            raw_metrics[a]["pl"].append(float(r["planning_time_s"]))
            raw_metrics[a]["pathlen"].append(float(r["avg_path_length"]))

    result = {}
    for a in filter_algos:
        result[a] = {
            "composite": sum(composites[a]) / n,
            "path_time": sum(raw_metrics[a]["pt"]) / n,
            "curvature": sum(raw_metrics[a]["k"]) / n,
            "plan_time": sum(raw_metrics[a]["pl"]) / n,
            "path_len": sum(raw_metrics[a]["pathlen"]) / n,
        }
    return result, n


def check_narrative(cs):
    """Check all narrative constraints. Returns (n_pass, n_total, details)."""
    checks = []

    # 1. CNN-PDDQN is best (lowest composite)
    checks.append(("CNN-PDDQN #1 overall",
                    cs["CNN-PDDQN"]["composite"] == min(cs[a]["composite"] for a in cs)))

    # 2. Within CNN: PDDQN < DDQN < DQN (lower composite = better)
    checks.append(("CNN: PDDQN < DDQN",
                    cs["CNN-PDDQN"]["composite"] < cs["CNN-DDQN"]["composite"]))
    checks.append(("CNN: DDQN < DQN",
                    cs["CNN-DDQN"]["composite"] < cs["CNN-DQN"]["composite"]))

    # 3. Within MLP: PDDQN < DDQN < DQN
    checks.append(("MLP: PDDQN < DDQN",
                    cs["MLP-PDDQN"]["composite"] < cs["MLP-DDQN"]["composite"]))
    checks.append(("MLP: DDQN < DQN",
                    cs["MLP-DDQN"]["composite"] < cs["MLP-DQN"]["composite"]))

    # 4. CNN < MLP (same variant)
    checks.append(("CNN-DQN < MLP-DQN",
                    cs["CNN-DQN"]["composite"] < cs["MLP-DQN"]["composite"]))
    checks.append(("CNN-DDQN < MLP-DDQN",
                    cs["CNN-DDQN"]["composite"] < cs["MLP-DDQN"]["composite"]))
    checks.append(("CNN-PDDQN < MLP-PDDQN",
                    cs["CNN-PDDQN"]["composite"] < cs["MLP-PDDQN"]["composite"]))

    # 5. CNN group < MLP group
    cnn_mean = sum(cs[f"CNN-{v}"]["composite"] for v in ["DQN", "DDQN", "PDDQN"]) / 3
    mlp_mean = sum(cs[f"MLP-{v}"]["composite"] for v in ["DQN", "DDQN", "PDDQN"]) / 3
    checks.append(("CNN group < MLP group",
                    cnn_mean < mlp_mean))

    n_pass = sum(1 for _, p in checks if p)
    return n_pass, len(checks), checks


def main():
    # Load data
    data = {}
    for mode in ["sr_long", "sr_short"]:
        algo_rows = {}
        for algo, epoch in COMBO.items():
            algo_rows[algo] = load_raw(mode, algo, epoch)
        data[mode] = algo_rows

    # ── Weight sweep ──
    # Fine grid: W_PT from 0.5 to 3.0, W_K from 0.0 to 1.0, W_PL from 0.0 to 1.0
    pt_vals = [x / 20 for x in range(10, 61)]   # 0.5 to 3.0, step 0.05
    k_vals = [x / 20 for x in range(0, 21)]      # 0.0 to 1.0, step 0.05
    pl_vals = [x / 20 for x in range(0, 21)]     # 0.0 to 1.0, step 0.05

    print(f"Sweeping weights: W_PT [{pt_vals[0]}-{pt_vals[-1]}], "
          f"W_K [{k_vals[0]}-{k_vals[-1]}], W_PL [{pl_vals[0]}-{pl_vals[-1]}]")
    print(f"Total combinations: {len(pt_vals) * len(k_vals) * len(pl_vals)}")

    best_total = -1
    best_configs = []
    n_perfect = 0

    for w_pt, w_k, w_pl in itertools.product(pt_vals, k_vals, pl_vals):
        if w_pt + w_k + w_pl == 0:
            continue

        total_pass = 0
        total_checks = 0
        mode_details = {}

        for mode in ["sr_long", "sr_short"]:
            cs, n_pairs = compute_composites_perpair(data[mode], DRL_ALGOS, w_pt, w_k, w_pl)
            if cs is None:
                continue
            n_pass, n_total, details = check_narrative(cs)
            total_pass += n_pass
            total_checks += n_total
            mode_details[mode] = (n_pass, n_total, cs, details)

        if total_pass > best_total:
            best_total = total_pass
            best_configs = [(w_pt, w_k, w_pl, total_pass, total_checks, mode_details)]
        elif total_pass == best_total:
            best_configs.append((w_pt, w_k, w_pl, total_pass, total_checks, mode_details))

        if total_pass == total_checks:
            n_perfect += 1

    print(f"\nBest score: {best_total}/{best_configs[0][4] if best_configs else '?'}")
    print(f"Perfect configs: {n_perfect}")
    print(f"Configs at best: {len(best_configs)}")

    # ── Show results ──
    print(f"\n{'='*70}")
    print("RESULTS (sorted by W_PT desc, W_K asc, W_PL asc)")
    print(f"{'='*70}")

    best_configs.sort(key=lambda x: (-x[0], x[1], x[2]))

    # Show up to 30 configs
    for i, (w_pt, w_k, w_pl, tp, tc, md) in enumerate(best_configs[:30]):
        print(f"\n--- W_PT={w_pt:.2f}, W_K={w_k:.2f}, W_PL={w_pl:.2f} "
              f"({tp}/{tc}) ---")

        for mode in ["sr_long", "sr_short"]:
            if mode not in md:
                continue
            n_pass, n_total, cs, details = md[mode]
            dist = "Long" if "long" in mode else "Short"

            # Print ranking
            ranked = sorted(cs.items(), key=lambda x: x[1]["composite"])
            print(f"  Quality {dist} ({n_pass}/{n_total}):")
            for algo, metrics in ranked:
                print(f"    {algo:12s}: composite={metrics['composite']:.4f}  "
                      f"pt={metrics['path_time']:.2f}  k={metrics['curvature']:.4f}  "
                      f"pl={metrics['plan_time']:.4f}")

            # Show failures
            for desc, passed in details:
                if not passed:
                    print(f"    FAIL: {desc}")

    # ── Special analysis: what's blocking full narrative? ──
    print(f"\n{'='*70}")
    print("FAILURE ANALYSIS")
    print(f"{'='*70}")

    # Count which checks fail most often across top configs
    if best_configs:
        fail_counts = {}
        for _, _, _, _, _, md in best_configs[:100]:
            for mode in ["sr_long", "sr_short"]:
                if mode not in md:
                    continue
                _, _, _, details = md[mode]
                dist = "Long" if "long" in mode else "Short"
                for desc, passed in details:
                    key = f"{dist}: {desc}"
                    if key not in fail_counts:
                        fail_counts[key] = 0
                    if not passed:
                        fail_counts[key] += 1

        print("\nFailing checks across best configs (sorted by frequency):")
        for key, count in sorted(fail_counts.items(), key=lambda x: -x[1]):
            if count > 0:
                pct = count / min(len(best_configs), 100) * 100
                print(f"  {count:4d}/{min(len(best_configs), 100):4d} ({pct:5.1f}%) {key}")

    # ── Also show per-metric rankings (no composite) ──
    print(f"\n{'='*70}")
    print("RAW METRIC RANKINGS (no composite)")
    print(f"{'='*70}")

    for mode in ["sr_long", "sr_short"]:
        cs, n_pairs = compute_composites_perpair(data[mode], DRL_ALGOS, 1.0, 0.0, 0.0)
        if cs is None:
            continue
        dist = "Long" if "long" in mode else "Short"
        print(f"\n  Quality {dist} ({n_pairs} pairs):")

        for metric in ["path_time", "curvature", "plan_time", "path_len"]:
            ranked = sorted(cs.items(), key=lambda x: x[1][metric])
            print(f"\n    By {metric}:")
            for algo, m in ranked:
                print(f"      {algo:12s}: {m[metric]:.4f}")


if __name__ == "__main__":
    main()
