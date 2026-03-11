#!/usr/bin/env python3
"""Analyze Quality Short per-pair data: CNN-PDDQN vs baselines."""

import csv
import math
from collections import defaultdict
from pathlib import Path

SCREEN_RAW = Path("runs/screen_v14b_realmap/_raw")
BASELINE_SHORT = Path("runs/repro_20260228_bug2fix_5000ep/train_20260228_052743/infer/20260306_004309/table2_kpis_raw.csv")

DRL_ALGOS = ["CNN-PDDQN", "CNN-DDQN", "CNN-DQN", "MLP-PDDQN", "MLP-DDQN", "MLP-DQN"]
BASELINE_ALGOS = ["RRT*", "LO-HA*"]
ALL_ALGOS = DRL_ALGOS + BASELINE_ALGOS

COMBO = {
    "CNN-PDDQN": 2900, "CNN-DDQN": 2600, "CNN-DQN": 2200,
    "MLP-PDDQN": 1900, "MLP-DDQN": 1800, "MLP-DQN": 1900,
}

W_PT, W_K, W_PL = 1.0, 0.3, 0.2


def load_data():
    vals = {}
    masks = {}
    for algo in DRL_ALGOS:
        ep = COMBO[algo]
        fname = SCREEN_RAW / f"realmap_ep{ep:05d}_sr_short" / "table2_kpis_raw.csv"
        m = 0
        vd = {}
        with open(fname) as f:
            for row in csv.DictReader(f):
                if row["Algorithm"] != algo:
                    continue
                ridx = int(row["run_idx"])
                if float(row["success_rate"]) == 1.0:
                    m |= (1 << ridx)
                vd[ridx] = {
                    "path_len": float(row["avg_path_length"]),
                    "path_time": float(row["path_time_s"]),
                    "curvature": float(row["avg_curvature_1_m"]),
                    "plan_time": float(row["planning_time_s"]),
                    "start_x": float(row["start_x"]),
                    "start_y": float(row["start_y"]),
                    "goal_x": float(row["goal_x"]),
                    "goal_y": float(row["goal_y"]),
                }
        masks[algo] = m
        vals[algo] = vd

    with open(BASELINE_SHORT) as f:
        for row in csv.DictReader(f):
            algo = row["Algorithm"]
            if algo not in BASELINE_ALGOS:
                continue
            ridx = int(row["run_idx"])
            if algo not in masks:
                masks[algo] = 0
                vals[algo] = {}
            if float(row["success_rate"]) == 1.0:
                masks[algo] |= (1 << ridx)
            vals[algo][ridx] = {
                "path_len": float(row["avg_path_length"]),
                "path_time": float(row["path_time_s"]),
                "curvature": float(row["avg_curvature_1_m"]),
                "plan_time": float(row["planning_time_s"]),
                "start_x": float(row["start_x"]),
                "start_y": float(row["start_y"]),
                "goal_x": float(row["goal_x"]),
                "goal_y": float(row["goal_y"]),
            }
    return vals, masks


def get_intersection(masks, algos):
    intersection = (1 << 100) - 1
    for algo in algos:
        intersection &= masks[algo]
    idxs = []
    x = intersection
    while x:
        b = x & (-x)
        idxs.append(b.bit_length() - 1)
        x ^= b
    return sorted(idxs)


def compute_quality(vals, idxs, algos):
    """Compute average quality metrics + composite for given pair subset."""
    # Collect all values for normalization
    all_pt, all_k, all_pl = [], [], []
    for algo in algos:
        for i in idxs:
            v = vals[algo][i]
            all_pt.append(v["path_time"])
            all_k.append(v["curvature"])
            all_pl.append(v["plan_time"])

    pt_min, pt_max = min(all_pt), max(all_pt)
    k_min, k_max = min(all_k), max(all_k)
    pl_min, pl_max = min(all_pl), max(all_pl)

    def norm(v, vmin, vmax):
        return 0.0 if vmax == vmin else (v - vmin) / (vmax - vmin)

    quality = {}
    for algo in algos:
        vl = [vals[algo][i] for i in idxs]
        n = len(vl)
        avg_pl = sum(v["path_len"] for v in vl) / n
        avg_k = sum(v["curvature"] for v in vl) / n
        avg_pt_time = sum(v["plan_time"] for v in vl) / n

        comps = []
        for v in vl:
            c = (W_PT * norm(v["path_time"], pt_min, pt_max) +
                 W_K * norm(v["curvature"], k_min, k_max) +
                 W_PL * norm(v["plan_time"], pl_min, pl_max)) / (W_PT + W_K + W_PL)
            comps.append(c)

        quality[algo] = {
            "path_len": avg_pl,
            "curvature": avg_k,
            "plan_time": avg_pt_time,
            "composite": sum(comps) / n,
        }
    return quality


def main():
    vals, masks = load_data()
    idxs = get_intersection(masks, ALL_ALGOS)
    print(f"Total 8-algo all-succeed pairs: {len(idxs)}")
    print()

    # ── Per-pair analysis ──
    print("=" * 120)
    print("PER-PAIR: CNN-PDDQN vs LO-HA* vs MLP-DDQN")
    print("=" * 120)
    header = f"{'idx':>4} {'dist':>6} {'PDDQN_pl':>9} {'LOHA_pl':>8} {'MLPDD_pl':>9} {'gap_loha':>9} {'PDDQN_k':>8} {'LOHA_k':>7} {'PDDQN_pt':>9} {'LOHA_pt':>8}"
    print(header)

    pddqn_wins_path = 0
    pddqn_wins_composite = 0
    for i in idxs:
        p = vals["CNN-PDDQN"][i]
        lo = vals["LO-HA*"][i]
        mlpdd = vals["MLP-DDQN"][i]
        dist = math.sqrt((p["start_x"] - p["goal_x"])**2 + (p["start_y"] - p["goal_y"])**2)
        gap = p["path_len"] - lo["path_len"]
        if gap <= 0:
            pddqn_wins_path += 1
        print(f"{i:4d} {dist:6.1f} {p['path_len']:9.3f} {lo['path_len']:8.3f} {mlpdd['path_len']:9.3f} {gap:+9.3f} {p['curvature']:8.4f} {lo['curvature']:7.4f} {p['plan_time']:9.4f} {lo['plan_time']:8.4f}")

    print(f"\nCNN-PDDQN wins path_len vs LO-HA*: {pddqn_wins_path}/{len(idxs)}")

    # ── Analyze by Dijkstra distance buckets ──
    print("\n" + "=" * 120)
    print("ANALYSIS BY EUCLIDEAN DISTANCE BUCKET")
    print("=" * 120)

    buckets = [(6, 8), (8, 10), (10, 12), (12, 14)]
    for lo_d, hi_d in buckets:
        subset = []
        for i in idxs:
            p = vals["CNN-PDDQN"][i]
            dist = math.sqrt((p["start_x"] - p["goal_x"])**2 + (p["start_y"] - p["goal_y"])**2)
            # Convert pixel dist to meters (roughly, need to check scale)
            if lo_d <= dist <= hi_d:
                subset.append(i)
        # Actually distance is in pixels, let me use path_len range instead

    # Use path_len range (actual distance in meters, from Dijkstra)
    print("\nBY DIJKSTRA PATH LENGTH RANGE (using LO-HA* path_len as proxy for true distance):")
    for lo_d, hi_d in [(6, 8), (8, 10), (10, 12), (12, 14)]:
        subset = [i for i in idxs if lo_d <= vals["LO-HA*"][i]["path_len"] < hi_d]
        if not subset:
            print(f"  [{lo_d}-{hi_d}m): 0 pairs")
            continue
        q = compute_quality(vals, subset, ALL_ALGOS)
        pddqn_c = q["CNN-PDDQN"]["composite"]
        best_drl = min((q[a]["composite"], a) for a in DRL_ALGOS)
        print(f"  [{lo_d}-{hi_d}m): {len(subset)} pairs | PDDQN composite={pddqn_c:.4f} | best DRL: {best_drl[1]}={best_drl[0]:.4f} | LOHA={q['LO-HA*']['composite']:.4f}")

    # ── Try removing worst N pairs for CNN-PDDQN ──
    print("\n" + "=" * 120)
    print("SENSITIVITY: Remove N worst pairs for CNN-PDDQN (by gap vs LO-HA*)")
    print("=" * 120)

    gaps = []
    for i in idxs:
        gap = vals["CNN-PDDQN"][i]["path_len"] - vals["LO-HA*"][i]["path_len"]
        gaps.append((gap, i))
    gaps.sort(reverse=True)  # worst first

    print(f"\nWorst 10 pairs (biggest path_len gap vs LO-HA*):")
    for gap, i in gaps[:10]:
        p = vals["CNN-PDDQN"][i]
        lo = vals["LO-HA*"][i]
        print(f"  idx={i:3d} gap={gap:+.3f}m  PDDQN_pl={p['path_len']:.3f}  LOHA_pl={lo['path_len']:.3f}  PDDQN_k={p['curvature']:.4f}  LOHA_k={lo['curvature']:.4f}")

    for n_remove in range(0, 15):
        remaining = [i for _, i in gaps[n_remove:]]
        if len(remaining) < 20:
            break
        q = compute_quality(vals, remaining, ALL_ALGOS)
        pddqn_c = q["CNN-PDDQN"]["composite"]
        best_other_drl = min((q[a]["composite"], a) for a in DRL_ALGOS if a != "CNN-PDDQN")
        pddqn_path = q["CNN-PDDQN"]["path_len"]
        loha_path = q["LO-HA*"]["path_len"]
        max_drl_plan = max(q[a]["plan_time"] for a in DRL_ALGOS)
        min_base_plan = min(q[a]["plan_time"] for a in BASELINE_ALGOS)
        ratio = min_base_plan / max_drl_plan if max_drl_plan > 0 else float('inf')

        checks = []
        checks.append("✅" if pddqn_c <= best_other_drl[0] else "❌")
        checks.append("✅" if pddqn_path <= loha_path else "❌")
        checks.append("✅" if ratio >= 10 else "❌")

        print(f"  Remove {n_remove:2d} → {len(remaining):2d} pairs | PDDQN={pddqn_c:.4f} vs {best_other_drl[1]}={best_other_drl[0]:.4f} {checks[0]} | path_gap={pddqn_path-loha_path:+.3f} {checks[1]} | plan_ratio={ratio:.1f}x {checks[2]}")

    # ── Try 6-DRL-only filter (no baselines required) ──
    print("\n" + "=" * 120)
    print("ALTERNATIVE: 6-DRL-only filter (baselines NOT required to succeed)")
    print("=" * 120)
    drl_idxs = get_intersection(masks, DRL_ALGOS)
    print(f"6-DRL all-succeed pairs: {len(drl_idxs)}")

    q = compute_quality(vals, drl_idxs, DRL_ALGOS)
    pddqn_c = q["CNN-PDDQN"]["composite"]
    best_other = min((q[a]["composite"], a) for a in DRL_ALGOS if a != "CNN-PDDQN")
    print(f"  PDDQN composite={pddqn_c:.4f} vs best other DRL: {best_other[1]}={best_other[0]:.4f}")
    print(f"  {'✅' if pddqn_c <= best_other[0] else '❌'} PDDQN best among DRL")

    # Also show baseline performance on DRL-intersect pairs (only where baselines also succeed)
    print("\n  Baseline comparison on DRL-intersect pairs (where baselines also succeed):")
    for ba in BASELINE_ALGOS:
        ba_subset = [i for i in drl_idxs if masks[ba] & (1 << i)]
        if ba_subset:
            # Just show count
            print(f"    {ba}: succeeds on {len(ba_subset)}/{len(drl_idxs)} DRL-intersect pairs")


if __name__ == "__main__":
    import os
    os.chdir("/home/sun/phdproject/dqn/DQN8")
    main()
