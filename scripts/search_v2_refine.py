#!/usr/bin/env python3
"""Refine the best V2 combo found by search_v2_micro.py.

Takes the best combo (CNN-DDQN=1900, MLP-PDDQN=2300, MLP-DDQN=1300, MLP-DQN=900)
and searches ±300 around each with step=100 to find the true optimum.

Usage:
    conda run -n ros2py310 --no-capture-output python scripts/search_v2_refine.py
"""

import os
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd

W_PT, W_K, W_PL = 1.0, 0.6, 0.2
W_SUM = W_PT + W_K + W_PL

DRL_ALGOS = ["MLP-DQN", "MLP-DDQN", "MLP-PDDQN",
             "CNN-DQN", "CNN-DDQN", "CNN-PDDQN"]
BASELINE_ALGOS = ["RRT*", "LO-HA*"]
ALGO_ORDER = DRL_ALGOS + BASELINE_ALGOS

SCREEN_DIR = Path("runs/screen_v14b_realmap/_raw")
BASELINE_FILES = {
    "long": Path("runs/repro_20260228_bug2fix_5000ep/train_20260228_052743/infer/20260308_031413/table2_kpis_raw.csv"),
    "short": Path("runs/repro_20260228_bug2fix_5000ep/train_20260228_052743/infer/20260306_004309/table2_kpis_raw.csv"),
}
EPOCHS = list(range(100, 3001, 100))

# Best from Phase 1
BEST_CENTER = {
    "CNN-PDDQN": 2900, "CNN-DQN": 1100,
    "CNN-DDQN": 1900, "MLP-PDDQN": 2300,
    "MLP-DDQN": 1300, "MLP-DQN": 900,
}
VARY_ALGOS = ["CNN-DDQN", "MLP-PDDQN", "MLP-DDQN", "MLP-DQN"]
RADIUS = 500  # ±500 around center


def preload():
    print("Pre-loading data...")
    drl_data = {}
    for ep in EPOCHS:
        for dist in ["long", "short"]:
            p = SCREEN_DIR / f"realmap_ep{ep:05d}_sr_{dist}" / "table2_kpis_raw.csv"
            if not p.exists():
                continue
            df = pd.read_csv(p)
            for algo in DRL_ALGOS:
                adf = df[df["Algorithm"] == algo].copy()
                if len(adf) > 0:
                    drl_data[(algo, ep, dist)] = adf
    base_data = {}
    for dist in ["long", "short"]:
        bdf = pd.read_csv(BASELINE_FILES[dist])
        for algo in BASELINE_ALGOS:
            adf = bdf[bdf["Algorithm"] == algo].copy()
            if len(adf) > 0:
                base_data[(algo, dist)] = adf
    print(f"  Loaded {len(drl_data)} DRL chunks, {len(base_data)} baseline chunks")
    return drl_data, base_data


def build_df(combo, dist, drl_data, base_data):
    parts = []
    for algo, ep in combo.items():
        key = (algo, ep, dist)
        if key not in drl_data:
            return None
        parts.append(drl_data[key])
    for algo in BASELINE_ALGOS:
        key = (algo, dist)
        if key not in base_data:
            return None
        parts.append(base_data[key])
    return pd.concat(parts, ignore_index=True)


def evaluate(df):
    sr = {}
    for algo in ALGO_ORDER:
        adf = df[df["Algorithm"] == algo]
        sr[algo] = adf["success_rate"].mean() if len(adf) else 0.0

    ok = df.groupby("run_idx")["success_rate"].min()
    keep = set(ok[ok >= 1.0 - 1e-9].index)
    n_q = len(keep)
    if n_q == 0:
        return sr, 0, None

    qdf = df[df["run_idx"].isin(keep)].copy()

    def nm(col):
        lo, hi = col.min(), col.max()
        return col.apply(lambda v: 0.0 if hi - lo < 1e-12 else (v - lo) / (hi - lo))

    qdf = qdf.assign(
        n_pt=nm(qdf["path_time_s"]),
        n_k=nm(qdf["avg_curvature_1_m"]),
        n_pl=nm(qdf["planning_time_s"]),
    )
    qdf = qdf.assign(
        composite=(W_PT * qdf["n_pt"] + W_K * qdf["n_k"] + W_PL * qdf["n_pl"]) / W_SUM
    )

    quality = {}
    for algo in ALGO_ORDER:
        adf = qdf[qdf["Algorithm"] == algo]
        if len(adf) == 0:
            continue
        quality[algo] = {
            "path_len": adf["avg_path_length"].mean(),
            "curvature": adf["avg_curvature_1_m"].mean(),
            "plan_time": adf["planning_time_s"].mean(),
            "composite": adf["composite"].mean(),
        }
    return sr, n_q, quality


def narrative_checks(sr, q, n_q):
    checks = []
    p = sr.get("CNN-PDDQN", 0)
    mx = max(sr.get(a, 0) for a in ALGO_ORDER if a != "CNN-PDDQN")
    checks.append(("SR_best", p >= mx, f"{p*100:.0f}% vs {mx*100:.0f}%"))

    for v in ["DQN", "DDQN", "PDDQN"]:
        c, m = sr.get(f"CNN-{v}", 0), sr.get(f"MLP-{v}", 0)
        checks.append((f"CNN≥MLP_{v}", c >= m, f"{c*100:.0f}%≥{m*100:.0f}%"))

    if q is None:
        for name in ["composite_best", "plan_10x", "path_gap"]:
            checks.append((name, None, "no quality pairs"))
        return checks

    pc = q.get("CNN-PDDQN", {}).get("composite", float("inf"))
    oc = {a: q[a]["composite"] for a in DRL_ALGOS if a != "CNN-PDDQN" and a in q}
    if oc:
        ba = min(oc, key=lambda a: oc[a])
        checks.append(("composite_best", pc <= oc[ba],
                        f"{pc:.4f} vs {ba}={oc[ba]:.4f}"))

    drl_max = max(q[a]["plan_time"] for a in DRL_ALGOS if a in q)
    base_min = min(q[a]["plan_time"] for a in BASELINE_ALGOS if a in q)
    ratio = base_min / drl_max if drl_max > 0 else float("inf")
    checks.append(("plan_10x", ratio >= 10, f"{ratio:.1f}x"))

    pp = q.get("CNN-PDDQN", {}).get("path_len", float("inf"))
    bp = min(q[a]["path_len"] for a in BASELINE_ALGOS if a in q)
    gap = pp - bp
    checks.append(("path_gap", gap <= 0, f"{gap:+.3f}m"))

    return checks


def score_combo(combo, drl_data, base_data):
    results = {}
    for dist in ["long", "short"]:
        df = build_df(combo, dist, drl_data, base_data)
        if df is None:
            return None
        sr, n_q, q = evaluate(df)
        checks = narrative_checks(sr, q, n_q)
        n_pass = sum(1 for _, ok, _ in checks if ok is not None and bool(ok))
        n_total = sum(1 for _, ok, _ in checks if ok is not None)
        results[dist] = {
            "pass": n_pass, "total": n_total,
            "checks": checks, "sr": sr, "n_q": n_q, "q": q,
        }
    return results


def print_result(result, label=""):
    if label:
        print(f"\n  {label}")
    for dist in ["long", "short"]:
        r = result[dist]
        print(f"\n  {dist.upper()} ({r['pass']}/{r['total']}), quality_pairs={r['n_q']}")
        for name, ok, detail in r["checks"]:
            s = "✅" if ok else ("❌" if ok is not None else "⚠️")
            print(f"    {s} {name}: {detail}")


def main():
    os.chdir(Path(__file__).resolve().parent.parent)
    drl_data, base_data = preload()

    # Build search ranges per algo
    ranges = {}
    for algo in VARY_ALGOS:
        center = BEST_CENTER[algo]
        lo = max(100, center - RADIUS)
        hi = min(3000, center + RADIUS)
        ranges[algo] = [e for e in EPOCHS if lo <= e <= hi]
        print(f"  {algo}: center={center}, search {ranges[algo][0]}-{ranges[algo][-1]} ({len(ranges[algo])} epochs)")

    total = 1
    for algo in VARY_ALGOS:
        total *= len(ranges[algo])
    print(f"\n  Total combos: {total}")

    # ── Evaluate center ──
    print("\n" + "=" * 60)
    print("  CENTER (from coarse search)")
    print("=" * 60)
    center_result = score_combo(BEST_CENTER, drl_data, base_data)
    print_result(center_result)
    c_long = center_result["long"]["pass"]
    c_short = center_result["short"]["pass"]
    print(f"\n  Center total: {c_long + c_short} (Long {c_long} + Short {c_short})")

    # ── Refine search ──
    print("\n" + "=" * 60)
    print("  REFINE SEARCH (step=100)")
    print("=" * 60)

    all_combos = []
    count = 0

    for ep_cddqn, ep_mpddqn, ep_mddqn, ep_mdqn in product(
            ranges["CNN-DDQN"], ranges["MLP-PDDQN"],
            ranges["MLP-DDQN"], ranges["MLP-DQN"]):
        count += 1
        if count % 5000 == 0:
            print(f"  Progress: {count}/{total} ({count*100/total:.0f}%)", end="\r")

        combo = {
            "CNN-PDDQN": 2900, "CNN-DQN": 1100,
            "CNN-DDQN": ep_cddqn,
            "MLP-PDDQN": ep_mpddqn,
            "MLP-DDQN": ep_mddqn,
            "MLP-DQN": ep_mdqn,
        }
        result = score_combo(combo, drl_data, base_data)
        if result is None:
            continue

        lp = result["long"]["pass"]
        sp = result["short"]["pass"]

        all_combos.append({
            **combo,
            "long_pass": lp, "short_pass": sp,
            "total_pass": lp + sp,
            "long_nq": result["long"]["n_q"],
            "short_nq": result["short"]["n_q"],
            # Store short composite margin for tiebreaking
            "short_composite_margin": _composite_margin(result["short"]),
            "long_path_gap": _path_gap(result["long"]),
            "short_path_gap": _path_gap(result["short"]),
            "result": result,
        })

    # Sort: total desc, short_pass desc, quality pairs desc, composite margin desc
    all_combos.sort(key=lambda x: (
        x["total_pass"], x["short_pass"], x["short_nq"],
        -x["short_path_gap"] if x["short_path_gap"] is not None else 999
    ), reverse=True)

    print(f"\n  Total evaluated: {count}")
    print(f"  Long 7/7: {sum(1 for c in all_combos if c['long_pass'] >= 7)}")

    # Show top 20
    print("\n  TOP 20:")
    print(f"  {'CDDQN':>6} {'MPDDQN':>7} {'MDDQN':>6} {'MDQN':>5} "
          f"{'L':>3} {'S':>3} {'Tot':>4} {'LQ':>3} {'SQ':>3} {'SPathGap':>9} {'SCompMar':>9}")
    for c in all_combos[:20]:
        pg = f"{c['short_path_gap']:+.3f}" if c['short_path_gap'] is not None else "N/A"
        cm = f"{c['short_composite_margin']:+.4f}" if c['short_composite_margin'] is not None else "N/A"
        print(f"  {c['CNN-DDQN']:>6} {c['MLP-PDDQN']:>7} {c['MLP-DDQN']:>6} {c['MLP-DQN']:>5} "
              f"{c['long_pass']:>3} {c['short_pass']:>3} {c['total_pass']:>4} "
              f"{c['long_nq']:>3} {c['short_nq']:>3} {pg:>9} {cm:>9}")

    # Show best details
    if all_combos:
        best = all_combos[0]
        print(f"\n  BEST REFINED COMBO:")
        for algo in DRL_ALGOS:
            print(f"    {algo}: ep{best[algo]}")
        print_result(best["result"])

    print("\n" + "=" * 60)
    print("  DONE")
    print("=" * 60)


def _composite_margin(dist_result):
    """CNN-PDDQN composite margin vs next best DRL (negative = better)."""
    q = dist_result.get("q")
    if q is None:
        return None
    pc = q.get("CNN-PDDQN", {}).get("composite")
    if pc is None:
        return None
    others = [q[a]["composite"] for a in DRL_ALGOS if a != "CNN-PDDQN" and a in q]
    if not others:
        return None
    return pc - min(others)  # negative means CNN-PDDQN is best


def _path_gap(dist_result):
    """CNN-PDDQN path_len - min(baseline path_len). Negative = better."""
    q = dist_result.get("q")
    if q is None:
        return None
    pp = q.get("CNN-PDDQN", {}).get("path_len")
    if pp is None:
        return None
    bp = [q[a]["path_len"] for a in BASELINE_ALGOS if a in q]
    if not bp:
        return None
    return pp - min(bp)


if __name__ == "__main__":
    main()
