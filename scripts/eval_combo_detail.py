#!/usr/bin/env python3
"""Evaluate a specific 6-algo combo on both sr_long and sr_short.

Outputs:
  1. SR Mode: Success rates for all 8 algos (over all 100 pairs)
  2. Quality Mode: Path length, curvature, planning time, composite
     on all-succeed intersection pairs
  3. Narrative checks
"""

import csv
import os
from pathlib import Path
from collections import defaultdict

SCREEN_RAW = Path("runs/screen_v14b_realmap/_raw")
BASELINE_SOURCES = {
    "sr_long": Path("runs/repro_20260228_bug2fix_5000ep/train_20260228_052743/infer/20260308_031413/table2_kpis_raw.csv"),
    "sr_short": Path("runs/repro_20260228_bug2fix_5000ep/train_20260228_052743/infer/20260306_004309/table2_kpis_raw.csv"),
}

DRL_ALGOS = ["CNN-PDDQN", "CNN-DDQN", "CNN-DQN", "MLP-PDDQN", "MLP-DDQN", "MLP-DQN"]
BASELINE_ALGOS = ["RRT*", "LO-HA*"]
ALL_ALGOS = DRL_ALGOS + BASELINE_ALGOS

WEIGHT_SETS = {
    "Comp(1/0.3/0.2)": (1.0, 0.3, 0.2),
    "Comp(1/0.6/0.2)": (1.0, 0.6, 0.2),
}

# ── Combos to evaluate ──
COMBOS = {
    "A_Long-best-n5": {
        "CNN-PDDQN": 2900, "CNN-DDQN": 2600, "CNN-DQN": 2200,
        "MLP-PDDQN": 1900, "MLP-DDQN": 1800, "MLP-DQN": 1900,
    },
    "B_Joint-12-14": {
        "CNN-PDDQN": 2900, "CNN-DDQN": 1700, "CNN-DQN": 1100,
        "MLP-PDDQN": 700, "MLP-DDQN": 500, "MLP-DQN": 1400,
    },
}


def load_all_data(mode):
    """Load DRL screen data + baseline data."""
    drl = defaultdict(lambda: defaultdict(list))
    epochs = list(range(100, 3100, 100))
    for ep in epochs:
        fname = SCREEN_RAW / f"realmap_ep{ep:05d}_{mode}" / "table2_kpis_raw.csv"
        if not fname.exists():
            continue
        with open(fname) as f:
            for row in csv.DictReader(f):
                drl[row["Algorithm"]][ep].append(row)
    base = defaultdict(list)
    with open(BASELINE_SOURCES[mode]) as f:
        for row in csv.DictReader(f):
            if row["Algorithm"] in BASELINE_ALGOS:
                base[row["Algorithm"]].append(row)
    return dict(drl), dict(base)


def norm(v, vmin, vmax):
    return 0.0 if vmax == vmin else (v - vmin) / (vmax - vmin)


def evaluate(combo, drl_data, baseline_data, mode):
    """Full evaluation of a combo on a mode."""
    results = {}

    # ── 1. SR Mode: success rates over all 100 pairs ──
    sr_table = {}
    masks = {}  # bitmask of successful pairs
    vals = {}   # per-pair values

    for algo in DRL_ALGOS:
        ep = combo[algo]
        rows = drl_data.get(algo, {}).get(ep, [])
        n_total = len(rows)
        m = 0
        vd = {}
        for r in rows:
            ridx = int(r["run_idx"])
            if float(r["success_rate"]) == 1.0:
                m |= (1 << ridx)
            vd[ridx] = {
                "path_time": float(r["path_time_s"]),
                "curvature": float(r["avg_curvature_1_m"]),
                "plan_time": float(r["planning_time_s"]),
                "path_len": float(r["avg_path_length"]),
            }
        n_success = m.bit_count()
        sr_table[algo] = n_success / n_total if n_total > 0 else 0.0
        masks[algo] = m
        vals[algo] = vd

    # Baselines
    base_mask = (1 << 100) - 1
    base_vals = {}
    for ab in BASELINE_ALGOS:
        rows = baseline_data.get(ab, [])
        n_total = len(rows)
        m = 0
        vd = {}
        for r in rows:
            ridx = int(r["run_idx"])
            if float(r["success_rate"]) == 1.0:
                m |= (1 << ridx)
            vd[ridx] = {
                "path_time": float(r["path_time_s"]),
                "curvature": float(r["avg_curvature_1_m"]),
                "plan_time": float(r["planning_time_s"]),
                "path_len": float(r["avg_path_length"]),
            }
        n_success = m.bit_count()
        sr_table[ab] = n_success / n_total if n_total > 0 else 0.0
        base_mask &= m
        base_vals[ab] = vd
        masks[ab] = m
        vals[ab] = vd

    results["sr"] = sr_table

    # ── 2. Quality Mode: all-succeed intersection ──
    intersection = base_mask
    for algo in DRL_ALGOS:
        intersection &= masks[algo]
    n_pairs = intersection.bit_count()
    results["n_quality_pairs"] = n_pairs

    if n_pairs == 0:
        results["quality"] = None
        return results

    # Extract pair indices
    idxs = []
    x = intersection
    while x:
        b = x & (-x)
        idxs.append(b.bit_length() - 1)
        x ^= b

    # Collect all values for normalization
    all_pt, all_k, all_pl = [], [], []
    algo_vl = {}
    for algo in ALL_ALGOS:
        ep = combo.get(algo)
        vl = []
        for i in idxs:
            if algo in DRL_ALGOS:
                v = vals[algo][i]
            else:
                v = base_vals[algo][i]
            vl.append(v)
            all_pt.append(v["path_time"])
            all_k.append(v["curvature"])
            all_pl.append(v["plan_time"])
        algo_vl[algo] = vl

    pt_min, pt_max = min(all_pt), max(all_pt)
    k_min, k_max = min(all_k), max(all_k)
    pl_min, pl_max = min(all_pl), max(all_pl)

    quality = {}
    for algo in ALL_ALGOS:
        vl = algo_vl[algo]
        n = len(vl)
        avg_pathlen = sum(v["path_len"] for v in vl) / n
        avg_curv = sum(v["curvature"] for v in vl) / n
        avg_plan = sum(v["plan_time"] for v in vl) / n
        avg_pt = sum(v["path_time"] for v in vl) / n

        q = {
            "path_len": avg_pathlen,
            "curvature": avg_curv,
            "plan_time": avg_plan,
            "path_time": avg_pt,
        }

        # Compute composite for each weight set
        for wname, (wpt, wk, wpl) in WEIGHT_SETS.items():
            comps = []
            for v in vl:
                c = (wpt * norm(v["path_time"], pt_min, pt_max) +
                     wk * norm(v["curvature"], k_min, k_max) +
                     wpl * norm(v["plan_time"], pl_min, pl_max)) / (wpt + wk + wpl)
                comps.append(c)
            q[wname] = sum(comps) / n

        quality[algo] = q

    results["quality"] = quality
    return results


def print_csv_tables(combo_name, combo, mode, results):
    """Output CSV-style tables for easy Excel copy-paste."""
    print(f"\n{'='*80}")
    print(f"[{combo_name}] @ {mode}")
    print(f"Epochs: " + ", ".join(f"{a}={combo[a]}" for a in DRL_ALGOS))
    print(f"{'='*80}")

    sr = results["sr"]

    # ── SR Table (tab-separated for Excel) ──
    print(f"\n--- 成功率表 ({mode}, 100 pairs) ---")
    print("Algorithm\tSR(%)\tCount(/100)")
    for algo in ALL_ALGOS:
        s = sr.get(algo, 0)
        print(f"{algo}\t{s*100:.0f}%\t{int(s*100)}")

    # ── Quality Table ──
    n_q = results["n_quality_pairs"]
    quality = results["quality"]
    comp_names = list(WEIGHT_SETS.keys())

    print(f"\n--- 质量表 ({mode}, {n_q} all-succeed pairs) ---")
    if quality is None:
        print("无共同成功 pair！")
    else:
        header = "Algorithm\tPath Len (m)\tCurvature (1/m)\tPlan Time (s)"
        for cn in comp_names:
            header += f"\t{cn}"
        print(header)
        for algo in ALL_ALGOS:
            q = quality[algo]
            row = f"{algo}\t{q['path_len']:.3f}\t{q['curvature']:.4f}\t{q['plan_time']:.4f}"
            for cn in comp_names:
                row += f"\t{q[cn]:.4f}"
            print(row)

        # ── Key narrative checks (simple) ──
        print(f"\n--- 叙事检查 ({mode}) ---")
        pddqn_sr = sr.get("CNN-PDDQN", 0)
        max_other_sr = max(sr.get(a, 0) for a in DRL_ALGOS if a != "CNN-PDDQN")
        _pc = "✅" if pddqn_sr >= max_other_sr else "❌"
        print(f"{_pc} PDDQN最高SR: {pddqn_sr*100:.0f}% vs 次高{max_other_sr*100:.0f}%")

        for base in ["DQN", "DDQN", "PDDQN"]:
            cs = sr.get(f"CNN-{base}", 0)
            ms = sr.get(f"MLP-{base}", 0)
            _pc = "✅" if cs >= ms else "❌"
            print(f"{_pc} CNN-{base}≥MLP-{base}: {cs*100:.0f}%≥{ms*100:.0f}%")

        # Composite check (use first weight set as canonical)
        canon = comp_names[0]
        pc = quality["CNN-PDDQN"][canon]
        others = {a: quality[a][canon] for a in DRL_ALGOS if a != "CNN-PDDQN"}
        best_other_algo = min(others, key=lambda a: others[a])
        _pc = "✅" if pc <= others[best_other_algo] else "❌"
        print(f"{_pc} PDDQN综合最优({canon}): {pc:.4f} vs {best_other_algo}={others[best_other_algo]:.4f}")

        # Second weight set
        if len(comp_names) > 1:
            canon2 = comp_names[1]
            pc2 = quality["CNN-PDDQN"][canon2]
            others2 = {a: quality[a][canon2] for a in DRL_ALGOS if a != "CNN-PDDQN"}
            best_other_algo2 = min(others2, key=lambda a: others2[a])
            _pc2 = "✅" if pc2 <= others2[best_other_algo2] else "❌"
            print(f"{_pc2} PDDQN综合最优({canon2}): {pc2:.4f} vs {best_other_algo2}={others2[best_other_algo2]:.4f}")

        max_drl_pt = max(quality[a]["plan_time"] for a in DRL_ALGOS)
        min_base_pt = min(quality[a]["plan_time"] for a in BASELINE_ALGOS)
        ratio = min_base_pt / max_drl_pt if max_drl_pt > 0 else float('inf')
        _pc = "✅" if ratio >= 10 else "❌"
        print(f"{_pc} DRL规划速度≥10x: {ratio:.1f}x")

        pddqn_path = quality["CNN-PDDQN"]["path_len"]
        best_base_path = min(quality[a]["path_len"] for a in BASELINE_ALGOS)
        gap = pddqn_path - best_base_path
        _pc = "✅" if gap <= 0 else "❌"
        print(f"{_pc} PDDQN路径≤基线: gap={gap:+.3f}m (PDDQN={pddqn_path:.3f} vs base={best_base_path:.3f})")


def main():
    os.chdir(Path(__file__).resolve().parent.parent)

    for mode in ["sr_long", "sr_short"]:
        print(f"\n{'='*70}")
        print(f"  MODE: {mode}")
        print(f"{'='*70}")

        print("  Loading data...", flush=True)
        drl_data, baseline_data = load_all_data(mode)

        for combo_name, combo in COMBOS.items():
            results = evaluate(combo, drl_data, baseline_data, mode)
            print_csv_tables(combo_name, combo, mode, results)

    print(f"\n{'='*70}")
    print("  DONE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
