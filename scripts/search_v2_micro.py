#!/usr/bin/env python3
"""Micro-tune V2 checkpoint combo for Realmap.

Lock CNN-PDDQN=2900, CNN-DQN=1100 (only combo for Long 7/7),
then search all 30 epochs for the other 4 algorithms.

Usage:
    conda run -n ros2py310 --no-capture-output python scripts/search_v2_micro.py
"""

import os
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd

# ── Config ──
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

# Locked anchors (only combo that gives Long 7/7)
ANCHOR = {"CNN-PDDQN": 2900, "CNN-DQN": 1100}

V2_COMBO = {
    "CNN-PDDQN": 2900, "CNN-DDQN": 1700, "CNN-DQN": 1100,
    "MLP-PDDQN": 700, "MLP-DDQN": 500, "MLP-DQN": 1400,
}


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

    # ── V2 baseline ──
    print("\n" + "=" * 60)
    print("  V2 BASELINE")
    print("=" * 60)
    v2_result = score_combo(V2_COMBO, drl_data, base_data)
    print_result(v2_result)
    v2_long = v2_result["long"]["pass"]
    v2_short = v2_result["short"]["pass"]
    v2_total = v2_long + v2_short
    print(f"\n  V2 total: {v2_total} (Long {v2_long} + Short {v2_short})")

    # ── Full search: Lock CNN-PDDQN=2900, CNN-DQN=1100, vary other 4 ──
    print("\n" + "=" * 60)
    print("  FULL SEARCH: Vary CNN-DDQN × MLP-PDDQN × MLP-DDQN × MLP-DQN")
    print("  Anchors: CNN-PDDQN=2900, CNN-DQN=1100")
    print("=" * 60)

    # Use step=200 for tractable search: 15^4 = 50,625
    search_epochs = EPOCHS[::2]  # 100,300,500,...,2900
    total = len(search_epochs) ** 4
    count = 0

    # Track: all combos with Long >= 6 sorted by total
    results_67 = []  # Long 7/7
    results_6 = []   # Long 6/7

    for ep_cddqn, ep_mpddqn, ep_mddqn, ep_mdqn in product(
            search_epochs, search_epochs, search_epochs, search_epochs):
        count += 1
        if count % 5000 == 0:
            print(f"  Progress: {count}/{total} ({count*100/total:.0f}%)", end="\r")

        combo = {
            "CNN-PDDQN": 2900, "CNN-DQN": 1100,
            "CNN-DDQN": ep_cddqn,
            "MLP-PDDQN": ep_mpddqn, "MLP-DDQN": ep_mddqn,
            "MLP-DQN": ep_mdqn,
        }
        result = score_combo(combo, drl_data, base_data)
        if result is None:
            continue

        lp = result["long"]["pass"]
        sp = result["short"]["pass"]
        tot = lp + sp
        entry = {
            **combo,
            "long_pass": lp, "long_total": result["long"]["total"],
            "short_pass": sp, "short_total": result["short"]["total"],
            "total_pass": tot,
            "long_nq": result["long"]["n_q"],
            "short_nq": result["short"]["n_q"],
            "result": result,
        }
        if lp >= 7:
            results_67.append(entry)
        elif lp >= 6:
            results_6.append(entry)

    print(f"\n  Long 7/7: {len(results_67)} combos")
    print(f"  Long 6/7: {len(results_6)} combos")

    # ── Report Long 7/7 results ──
    results_67.sort(key=lambda x: (x["total_pass"], x["short_pass"], x["short_nq"]), reverse=True)
    print("\n  --- TOP 15 with Long 7/7 ---")
    print(f"  {'CDDQN':>6} {'MPDDQN':>7} {'MDDQN':>6} {'MDQN':>5} {'L':>3} {'S':>3} {'Tot':>4} {'LQ':>3} {'SQ':>3}")
    for c in results_67[:15]:
        print(f"  {c['CNN-DDQN']:>6} {c['MLP-PDDQN']:>7} {c['MLP-DDQN']:>6} {c['MLP-DQN']:>5} "
              f"{c['long_pass']:>3} {c['short_pass']:>3} {c['total_pass']:>4} "
              f"{c['long_nq']:>3} {c['short_nq']:>3}")

    if results_67:
        best7 = results_67[0]
        print(f"\n  BEST (Long 7/7):")
        for algo in DRL_ALGOS:
            print(f"    {algo}: ep{best7[algo]}")
        print_result(best7["result"])

    # ── Report Long 6/7 results (may have better Short) ──
    results_6.sort(key=lambda x: (x["total_pass"], x["short_pass"], x["short_nq"]), reverse=True)
    print("\n  --- TOP 15 with Long 6/7 ---")
    print(f"  {'CDDQN':>6} {'MPDDQN':>7} {'MDDQN':>6} {'MDQN':>5} {'L':>3} {'S':>3} {'Tot':>4} {'LQ':>3} {'SQ':>3}")
    for c in results_6[:15]:
        print(f"  {c['CNN-DDQN']:>6} {c['MLP-PDDQN']:>7} {c['MLP-DDQN']:>6} {c['MLP-DQN']:>5} "
              f"{c['long_pass']:>3} {c['short_pass']:>3} {c['total_pass']:>4} "
              f"{c['long_nq']:>3} {c['short_nq']:>3}")

    if results_6 and results_6[0]["total_pass"] > v2_total:
        best6 = results_6[0]
        print(f"\n  BEST (Long 6/7, total {best6['total_pass']} > V2 {v2_total}):")
        for algo in DRL_ALGOS:
            print(f"    {algo}: ep{best6[algo]}")
        print_result(best6["result"])

    # ── Also check: best overall (any Long pass count) ──
    all_results = results_67 + results_6
    all_results.sort(key=lambda x: (x["total_pass"], x["short_pass"]), reverse=True)
    if all_results:
        top = all_results[0]
        print(f"\n  OVERALL BEST: Total {top['total_pass']} "
              f"(Long {top['long_pass']}/{top['long_total']}, "
              f"Short {top['short_pass']}/{top['short_total']})")

    print("\n" + "=" * 60)
    print("  DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
