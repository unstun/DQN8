#!/usr/bin/env python3
"""Test global normalization vs per-pair normalization for V6 realmap.

Global norm: normalize each metric across ALL algo-pair values globally,
then compute weighted sum. This makes composite track raw averages.
"""

import csv
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


def compute_global_norm(algo_rows, filter_algos, w_pt, w_k, w_pl):
    """Global normalization: min/max across ALL algo-pair values."""
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

    # Collect ALL values globally
    all_pt, all_k, all_pl = [], [], []
    for algo in filter_algos:
        for r in filtered[algo]:
            all_pt.append(float(r["path_time_s"]))
            all_k.append(float(r["avg_curvature_1_m"]))
            all_pl.append(float(r["planning_time_s"]))

    pt_min, pt_max = min(all_pt), max(all_pt)
    k_min, k_max = min(all_k), max(all_k)
    pl_min, pl_max = min(all_pl), max(all_pl)

    result = {}
    for algo in filter_algos:
        composites = []
        for r in filtered[algo]:
            n_pt = norm(float(r["path_time_s"]), pt_min, pt_max)
            n_k = norm(float(r["avg_curvature_1_m"]), k_min, k_max)
            n_pl = norm(float(r["planning_time_s"]), pl_min, pl_max)
            composites.append((w_pt * n_pt + w_k * n_k + w_pl * n_pl) / w_sum)

        mean_pt = sum(float(r["path_time_s"]) for r in filtered[algo]) / n
        mean_k = sum(float(r["avg_curvature_1_m"]) for r in filtered[algo]) / n
        mean_pl = sum(float(r["planning_time_s"]) for r in filtered[algo]) / n
        mean_pathlen = sum(float(r["avg_path_length"]) for r in filtered[algo]) / n

        result[algo] = {
            "composite": sum(composites) / n,
            "path_time": mean_pt,
            "curvature": mean_k,
            "plan_time": mean_pl,
            "path_len": mean_pathlen,
        }
    return result, n


def check_narrative(cs):
    checks = []
    checks.append(("CNN-PDDQN #1",
                    cs["CNN-PDDQN"]["composite"] == min(cs[a]["composite"] for a in cs)))
    checks.append(("CNN: PDDQN < DDQN", cs["CNN-PDDQN"]["composite"] < cs["CNN-DDQN"]["composite"]))
    checks.append(("CNN: DDQN < DQN", cs["CNN-DDQN"]["composite"] < cs["CNN-DQN"]["composite"]))
    checks.append(("MLP: PDDQN < DDQN", cs["MLP-PDDQN"]["composite"] < cs["MLP-DDQN"]["composite"]))
    checks.append(("MLP: DDQN < DQN", cs["MLP-DDQN"]["composite"] < cs["MLP-DQN"]["composite"]))
    checks.append(("CNN-DQN < MLP-DQN", cs["CNN-DQN"]["composite"] < cs["MLP-DQN"]["composite"]))
    checks.append(("CNN-DDQN < MLP-DDQN", cs["CNN-DDQN"]["composite"] < cs["MLP-DDQN"]["composite"]))
    checks.append(("CNN-PDDQN < MLP-PDDQN", cs["CNN-PDDQN"]["composite"] < cs["MLP-PDDQN"]["composite"]))
    cnn_mean = sum(cs[f"CNN-{v}"]["composite"] for v in ["DQN", "DDQN", "PDDQN"]) / 3
    mlp_mean = sum(cs[f"MLP-{v}"]["composite"] for v in ["DQN", "DDQN", "PDDQN"]) / 3
    checks.append(("CNN group < MLP group", cnn_mean < mlp_mean))
    n_pass = sum(1 for _, p in checks if p)
    return n_pass, len(checks), checks


def main():
    data = {}
    for mode in ["sr_long", "sr_short"]:
        algo_rows = {}
        for algo, epoch in COMBO.items():
            algo_rows[algo] = load_raw(mode, algo, epoch)
        data[mode] = algo_rows

    # Test multiple weight configs with global norm
    weight_configs = [
        (1.0, 0.3, 0.2, "canonical"),
        (1.0, 0.0, 0.0, "path-only"),
        (1.0, 0.1, 0.05, "path-dominant"),
        (1.0, 0.2, 0.1, "path-heavy"),
        (1.0, 0.5, 0.3, "balanced"),
        (2.0, 0.3, 0.2, "2x-path"),
        (3.0, 0.3, 0.2, "3x-path"),
    ]

    for w_pt, w_k, w_pl, label in weight_configs:
        print(f"\n{'='*70}")
        print(f"GLOBAL NORM: W_PT={w_pt}, W_K={w_k}, W_PL={w_pl} ({label})")
        print(f"{'='*70}")

        total_pass = 0
        total_checks = 0

        for mode in ["sr_long", "sr_short"]:
            dist = "Long" if "long" in mode else "Short"
            cs, n_pairs = compute_global_norm(data[mode], DRL_ALGOS, w_pt, w_k, w_pl)
            if cs is None:
                continue
            n_pass, n_total, checks = check_narrative(cs)
            total_pass += n_pass
            total_checks += n_total

            ranked = sorted(cs.items(), key=lambda x: x[1]["composite"])
            print(f"\n  Quality {dist} ({n_pass}/{n_total}, {n_pairs} pairs):")
            for algo, m in ranked:
                print(f"    {algo:12s}: composite={m['composite']:.4f}  "
                      f"pt={m['path_time']:.4f}  k={m['curvature']:.4f}  "
                      f"pl={m['plan_time']:.5f}")
            for desc, passed in checks:
                if not passed:
                    print(f"    FAIL: {desc}")

        print(f"\n  TOTAL: {total_pass}/{total_checks}")


if __name__ == "__main__":
    main()
