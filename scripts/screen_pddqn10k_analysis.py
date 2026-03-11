#!/usr/bin/env python3
"""Sweep CNN-PDDQN epochs (100-10000 step 100) with V5 fixed checkpoints for other algos.

For each CNN-PDDQN epoch, compute:
  - SR Long, SR Short (from pddqn10k screen data)
  - Quality Long/Short composite score vs CNN-DQN (8-algo filter, exclude Hybrid A*)
  - Whether CNN-PDDQN beats CNN-DQN on composite

Other 5 DRL algos use V5 fixed checkpoints from screen_v14b_realmap:
  CNN-DDQN: 2000, CNN-DQN: 2800, MLP-PDDQN: 2400, MLP-DDQN: 2200, MLP-DQN: 700

Baselines (RRT*, LO-HA*) come from:
  Long:  runs/repro_20260228_bug2fix_5000ep/.../20260308_031413/table2_kpis_raw.csv
  Short: runs/repro_20260228_bug2fix_5000ep/.../20260306_004309/table2_kpis_raw.csv
"""

import csv, os, sys
from pathlib import Path
from collections import defaultdict

# ── Paths ──
PROJ = Path(__file__).resolve().parent.parent
PDDQN_RAW = PROJ / "runs/screen_pddqn10k_realmap/_raw"
V14B_RAW = PROJ / "runs/screen_v14b_realmap/_raw"
BASELINE_LONG = PROJ / "runs/repro_20260228_bug2fix_5000ep/train_20260228_052743/infer/20260308_031413/table2_kpis_raw.csv"
BASELINE_SHORT = PROJ / "runs/repro_20260228_bug2fix_5000ep/train_20260228_052743/infer/20260306_004309/table2_kpis_raw.csv"

# ── V5 fixed checkpoints (excluding CNN-PDDQN) ──
V5_EPOCHS = {
    "CNN-DDQN": 2000,
    "CNN-DQN": 2800,
    "MLP-PDDQN": 2400,
    "MLP-DDQN": 2200,
    "MLP-DQN": 700,
}

# ── CNN-PDDQN epochs to sweep ──
PDDQN_EPOCHS = list(range(100, 10100, 100))  # 100 to 10000

# ── Composite weights ──
W_PT, W_K, W_PL = 1.0, 0.3, 0.2
W_SUM = W_PT + W_K + W_PL  # 1.5

# ── 8-algo filter (exclude Hybrid A*) ──
DRL_ALGOS = ["MLP-DQN", "MLP-DDQN", "MLP-PDDQN", "CNN-DQN", "CNN-DDQN", "CNN-PDDQN"]
BASELINE_ALGOS = ["RRT*", "LO-HA*"]  # included in filter
ALL_8 = DRL_ALGOS + BASELINE_ALGOS


def load_csv(path):
    """Load CSV and return list of dicts."""
    with open(path) as f:
        return list(csv.DictReader(f))


def parse_rows(rows):
    """Parse rows into {run_idx: {field: value}} with success flag."""
    result = {}
    for r in rows:
        idx = int(r["run_idx"])
        result[idx] = {
            "success": float(r["success_rate"]) == 1.0,
            "path_time_s": float(r["path_time_s"]),
            "avg_curvature_1_m": float(r["avg_curvature_1_m"]),
            "planning_time_s": float(r["planning_time_s"]),
            "avg_path_length": float(r["avg_path_length"]),
        }
    return result


def load_algo_data(raw_dir, epoch, mode, algo_filter=None):
    """Load data for a specific epoch/mode from screen dir.
    Returns {algo: {run_idx: {...}}}
    """
    path = raw_dir / f"realmap_ep{epoch:05d}_{mode}" / "table2_kpis_raw.csv"
    if not path.exists():
        return {}
    rows = load_csv(path)
    by_algo = defaultdict(list)
    for r in rows:
        a = r["Algorithm"]
        if algo_filter and a not in algo_filter:
            continue
        by_algo[a].append(r)
    return {algo: parse_rows(rr) for algo, rr in by_algo.items()}


def load_baseline(path, algos):
    """Load baseline data for specific algos. Returns {algo: {run_idx: {...}}}"""
    rows = load_csv(path)
    by_algo = defaultdict(list)
    for r in rows:
        a = r["Algorithm"]
        if a in algos:
            by_algo[a].append(r)
    return {algo: parse_rows(rr) for algo, rr in by_algo.items()}


def norm(v, vmin, vmax):
    if vmax == vmin:
        return 0.0
    return (v - vmin) / (vmax - vmin)


def compute_composite_for_pairs(all_algo_data, pair_idxs):
    """Compute per-pair normalized composite, then return mean per algo.

    all_algo_data: {algo: {run_idx: {field: value}}}
    pair_idxs: list of run_idx where all 8 algos succeed

    Returns {algo: mean_composite} or None if no pairs.
    """
    if not pair_idxs:
        return None

    # For each pair, collect all 8 algos' values, do minmax norm within pair
    algo_composites = {a: [] for a in ALL_8}

    for idx in pair_idxs:
        # Collect raw values for all 8 algos
        pt_vals = []
        k_vals = []
        pl_vals = []
        for a in ALL_8:
            d = all_algo_data[a][idx]
            pt_vals.append(d["path_time_s"])
            k_vals.append(d["avg_curvature_1_m"])
            pl_vals.append(d["planning_time_s"])

        pt_min, pt_max = min(pt_vals), max(pt_vals)
        k_min, k_max = min(k_vals), max(k_vals)
        pl_min, pl_max = min(pl_vals), max(pl_vals)

        for i, a in enumerate(ALL_8):
            n_pt = norm(pt_vals[i], pt_min, pt_max)
            n_k = norm(k_vals[i], k_min, k_max)
            n_pl = norm(pl_vals[i], pl_min, pl_max)
            comp = (W_PT * n_pt + W_K * n_k + W_PL * n_pl) / W_SUM
            algo_composites[a].append(comp)

    return {a: sum(v) / len(v) for a, v in algo_composites.items()}


def main():
    os.chdir(PROJ)
    print("=" * 100)
    print("CNN-PDDQN Epoch Sweep: V5 fixed checkpoints + pddqn10k screen")
    print("=" * 100)

    # ── Load fixed data (V5 checkpoints for other 5 DRL algos) ──
    print("\nLoading V5 fixed checkpoint data...")
    v5_data = {"sr_long": {}, "sr_short": {}}
    for algo, ep in V5_EPOCHS.items():
        for mode in ["sr_long", "sr_short"]:
            d = load_algo_data(V14B_RAW, ep, mode, algo_filter={algo})
            if algo in d:
                v5_data[mode][algo] = d[algo]
                n_suc = sum(1 for v in d[algo].values() if v["success"])
                print(f"  {algo:12s} ep{ep:05d} {mode:8s}: {n_suc}/100 success")
            else:
                print(f"  WARNING: {algo} not found at ep{ep:05d} {mode}")

    # ── Load baselines ──
    print("\nLoading baseline data...")
    base_long = load_baseline(BASELINE_LONG, set(BASELINE_ALGOS) | {"Hybrid A*"})
    base_short = load_baseline(BASELINE_SHORT, set(BASELINE_ALGOS) | {"Hybrid A*"})
    for a in BASELINE_ALGOS:
        for label, bd in [("Long", base_long), ("Short", base_short)]:
            if a in bd:
                n_suc = sum(1 for v in bd[a].values() if v["success"])
                print(f"  {a:12s} {label:5s}: {n_suc}/100 success")

    # ── Find success masks for fixed algos ──
    # For 8-algo filter: all 8 algos must succeed on same run_idx
    # Fixed algos: 5 DRL + 2 baselines = 7 fixed
    # Variable: CNN-PDDQN

    # Build fixed success set (7 algos)
    fixed_success = {"sr_long": None, "sr_short": None}
    for mode, base_d in [("sr_long", base_long), ("sr_short", base_short)]:
        success_sets = []
        # 5 fixed DRL algos
        for algo in V5_EPOCHS:
            if algo in v5_data[mode]:
                s = {idx for idx, v in v5_data[mode][algo].items() if v["success"]}
                success_sets.append(s)
        # 2 baselines
        for ba in BASELINE_ALGOS:
            if ba in base_d:
                s = {idx for idx, v in base_d[ba].items() if v["success"]}
                success_sets.append(s)

        if success_sets:
            fixed_success[mode] = set.intersection(*success_sets)
        else:
            fixed_success[mode] = set()

        print(f"\n  Fixed 7-algo success pairs ({mode}): {len(fixed_success[mode])}")

    # ── Sweep CNN-PDDQN epochs ──
    print("\nSweeping CNN-PDDQN epochs...")
    results = []

    for ep in PDDQN_EPOCHS:
        # Load CNN-PDDQN data for this epoch
        pddqn_long = load_algo_data(PDDQN_RAW, ep, "sr_long", algo_filter={"CNN-PDDQN"})
        pddqn_short = load_algo_data(PDDQN_RAW, ep, "sr_short", algo_filter={"CNN-PDDQN"})

        if "CNN-PDDQN" not in pddqn_long or "CNN-PDDQN" not in pddqn_short:
            continue

        pl = pddqn_long["CNN-PDDQN"]
        ps = pddqn_short["CNN-PDDQN"]

        # SR
        sr_long = sum(1 for v in pl.values() if v["success"]) / len(pl)
        sr_short = sum(1 for v in ps.values() if v["success"]) / len(ps)

        # 8-algo success pairs
        pddqn_suc_long = {idx for idx, v in pl.items() if v["success"]}
        pddqn_suc_short = {idx for idx, v in ps.items() if v["success"]}

        allsuc_long = fixed_success["sr_long"] & pddqn_suc_long
        allsuc_short = fixed_success["sr_short"] & pddqn_suc_short

        # Build combined data dicts for quality computation
        ql_pddqn = ql_dqn = ql_margin = None
        ql_win = False
        n_ql = len(allsuc_long)

        qs_pddqn = qs_dqn = qs_margin = None
        qs_win = False
        n_qs = len(allsuc_short)

        # Quality Long
        if allsuc_long:
            all_data_long = {}
            # CNN-PDDQN from pddqn10k
            all_data_long["CNN-PDDQN"] = pl
            # Other 5 DRL from V14b
            for algo in V5_EPOCHS:
                all_data_long[algo] = v5_data["sr_long"][algo]
            # Baselines
            for ba in BASELINE_ALGOS:
                all_data_long[ba] = base_long[ba]

            composites = compute_composite_for_pairs(all_data_long, sorted(allsuc_long))
            if composites:
                ql_pddqn = composites["CNN-PDDQN"]
                ql_dqn = composites["CNN-DQN"]
                ql_margin = ql_dqn - ql_pddqn  # positive = PDDQN wins
                ql_win = ql_pddqn < ql_dqn

        # Quality Short
        if allsuc_short:
            all_data_short = {}
            all_data_short["CNN-PDDQN"] = ps
            for algo in V5_EPOCHS:
                all_data_short[algo] = v5_data["sr_short"][algo]
            for ba in BASELINE_ALGOS:
                all_data_short[ba] = base_short[ba]

            composites = compute_composite_for_pairs(all_data_short, sorted(allsuc_short))
            if composites:
                qs_pddqn = composites["CNN-PDDQN"]
                qs_dqn = composites["CNN-DQN"]
                qs_margin = qs_dqn - qs_pddqn
                qs_win = qs_pddqn < qs_dqn

        results.append({
            "epoch": ep,
            "sr_long": sr_long,
            "sr_short": sr_short,
            "n_ql": n_ql,
            "ql_win": ql_win,
            "ql_pddqn": ql_pddqn,
            "ql_dqn": ql_dqn,
            "ql_margin": ql_margin,
            "n_qs": n_qs,
            "qs_win": qs_win,
            "qs_pddqn": qs_pddqn,
            "qs_dqn": qs_dqn,
            "qs_margin": qs_margin,
        })

    # ── Sort: ql_win=True first, then sr_long+sr_short desc ──
    results.sort(key=lambda r: (
        -int(r["ql_win"]),
        -(r["sr_long"] + r["sr_short"]),
        -(r["ql_margin"] or -999),
    ))

    # ── Print table ──
    print(f"\n{'='*130}")
    print("FULL TABLE: CNN-PDDQN epoch sweep (sorted: QL_win first, then SR_long+SR_short desc)")
    print(f"{'='*130}")
    hdr = (f"{'Epoch':>6s} {'SR_L':>6s} {'SR_S':>6s} {'SR_sum':>6s} "
           f"{'QL_win':>6s} {'QL_PDDQN':>9s} {'QL_DQN':>9s} {'QL_mrg':>8s} {'#QL':>4s} "
           f"{'QS_win':>6s} {'QS_PDDQN':>9s} {'QS_DQN':>9s} {'QS_mrg':>8s} {'#QS':>4s}")
    print(hdr)
    print("-" * len(hdr))

    for r in results:
        ep = r["epoch"]
        sl = f"{r['sr_long']:.0%}"
        ss = f"{r['sr_short']:.0%}"
        ssum = f"{r['sr_long']+r['sr_short']:.0%}"

        qlw = "YES" if r["ql_win"] else "no"
        qlp = f"{r['ql_pddqn']:.4f}" if r["ql_pddqn"] is not None else "N/A"
        qld = f"{r['ql_dqn']:.4f}" if r["ql_dqn"] is not None else "N/A"
        qlm = f"{r['ql_margin']:+.4f}" if r["ql_margin"] is not None else "N/A"
        nql = str(r["n_ql"])

        qsw = "YES" if r["qs_win"] else "no"
        qsp = f"{r['qs_pddqn']:.4f}" if r["qs_pddqn"] is not None else "N/A"
        qsd = f"{r['qs_dqn']:.4f}" if r["qs_dqn"] is not None else "N/A"
        qsm = f"{r['qs_margin']:+.4f}" if r["qs_margin"] is not None else "N/A"
        nqs = str(r["n_qs"])

        # Highlight marker
        marker = ""
        if r["ql_win"] and r["sr_long"] >= 0.30 and r["sr_short"] >= 0.60:
            marker = " <<<"

        print(f"{ep:>6d} {sl:>6s} {ss:>6s} {ssum:>6s} "
              f"{qlw:>6s} {qlp:>9s} {qld:>9s} {qlm:>8s} {nql:>4s} "
              f"{qsw:>6s} {qsp:>9s} {qsd:>9s} {qsm:>8s} {nqs:>4s}{marker}")

    # ── Summary: Top candidates ──
    winners = [r for r in results if r["ql_win"]]
    print(f"\n{'='*100}")
    print(f"SUMMARY: {len(winners)} epochs win Quality Long (CNN-PDDQN < CNN-DQN)")
    print(f"{'='*100}")

    if winners:
        print("\nTop 20 winners by SR_long + SR_short:")
        print(f"{'Epoch':>6s} {'SR_L':>6s} {'SR_S':>6s} {'QL_mrg':>8s} {'#QL':>4s} "
              f"{'QS_win':>6s} {'QS_mrg':>8s} {'#QS':>4s}")
        print("-" * 60)
        for r in winners[:20]:
            qlm = f"{r['ql_margin']:+.4f}" if r["ql_margin"] is not None else "N/A"
            qsm = f"{r['qs_margin']:+.4f}" if r["qs_margin"] is not None else "N/A"
            qsw = "YES" if r["qs_win"] else "no"
            print(f"{r['epoch']:>6d} {r['sr_long']:>5.0%} {r['sr_short']:>5.0%} "
                  f"{qlm:>8s} {r['n_ql']:>4d} "
                  f"{qsw:>6s} {qsm:>8s} {r['n_qs']:>4d}")

        # Also find best QL margin
        best_margin = max(winners, key=lambda r: r["ql_margin"] or -999)
        print(f"\n  Best QL margin: ep{best_margin['epoch']} margin={best_margin['ql_margin']:+.4f} "
              f"SR_L={best_margin['sr_long']:.0%} SR_S={best_margin['sr_short']:.0%}")

    # Both-win candidates
    both_win = [r for r in results if r["ql_win"] and r["qs_win"]]
    if both_win:
        print(f"\n{len(both_win)} epochs win BOTH Quality Long AND Short:")
        for r in both_win[:10]:
            print(f"  ep{r['epoch']:05d} SR_L={r['sr_long']:.0%} SR_S={r['sr_short']:.0%} "
                  f"QL_mrg={r['ql_margin']:+.4f} QS_mrg={r['qs_margin']:+.4f}")
    else:
        print("\n  No epochs win both Quality Long and Short.")

    # V5 reference (CNN-PDDQN at ep3000)
    v5_ref = [r for r in results if r["epoch"] == 3000]
    if v5_ref:
        r = v5_ref[0]
        print(f"\n  V5 reference (CNN-PDDQN@3000): SR_L={r['sr_long']:.0%} SR_S={r['sr_short']:.0%} "
              f"QL_win={r['ql_win']} QL_mrg={r['ql_margin']} QS_win={r['qs_win']}")


if __name__ == "__main__":
    main()
