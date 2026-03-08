#!/usr/bin/env python3
"""Full 6-algo epoch combo search with decomposed pair enumeration.

Extends search_best_combo.py by also varying MLP epochs (not just CNN).

Key optimization:
- Group algos into 3 pairs: (CNN-X, MLP-X) for X in {DQN, DDQN, PDDQN}
- Pre-filter valid pairs (CNN SR >= MLP SR)
- CNN-PDDQN pair as outer loop (most constrained: highest SR)
- Bitmask intersection (Python ints) for ultra-fast set operations on 100 pairs
- Incremental intersection: outer→mid→final, prune early if < 3 pairs
"""

import csv
import os
import time
from pathlib import Path
from collections import defaultdict

# ── Constants ──
SCREEN_RAW = Path("runs/screen_v14b_realmap/_raw")
BASELINE_SOURCES = {
    "sr_long": Path("runs/repro_20260228_bug2fix_5000ep/train_20260228_052743/infer/20260308_031413/table2_kpis_raw.csv"),
    "sr_short": Path("runs/repro_20260228_bug2fix_5000ep/train_20260228_052743/infer/20260306_004309/table2_kpis_raw.csv"),
}

DRL_ALGOS = ["MLP-DQN", "MLP-DDQN", "MLP-PDDQN", "CNN-DQN", "CNN-DDQN", "CNN-PDDQN"]
BASELINE_ALGOS = ["RRT*", "LO-HA*"]
ALL_REPORT = DRL_ALGOS + BASELINE_ALGOS
EPOCHS = list(range(100, 3100, 100))  # 100..3000, 30 epochs

W_PT, W_K, W_PL = 1.0, 0.3, 0.2

V5_COMBO = {
    "CNN-PDDQN": 3000, "CNN-DDQN": 2000, "CNN-DQN": 2800,
    "MLP-PDDQN": 2400, "MLP-DDQN": 2200, "MLP-DQN": 700,
}


def load_all_data(mode):
    """Load all DRL screen data + baseline data."""
    drl = defaultdict(lambda: defaultdict(list))
    for ep in EPOCHS:
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


def precompute(drl_data, baseline_data):
    """Build bitmask and value lookup tables."""
    sr = {}       # sr[algo][ep] -> float
    masks = {}    # masks[algo][ep] -> int bitmask (bit i = run_idx i succeeded)
    vals = {}     # vals[algo][ep][run_idx] -> (pt, k, pl, pathlen)

    for algo in DRL_ALGOS:
        sr[algo], masks[algo], vals[algo] = {}, {}, {}
        for ep in EPOCHS:
            rows = drl_data.get(algo, {}).get(ep, [])
            if not rows:
                sr[algo][ep], masks[algo][ep], vals[algo][ep] = 0.0, 0, {}
                continue
            m, vd = 0, {}
            for r in rows:
                ridx = int(r["run_idx"])
                if float(r["success_rate"]) == 1.0:
                    m |= (1 << ridx)
                vd[ridx] = (float(r["path_time_s"]), float(r["avg_curvature_1_m"]),
                            float(r["planning_time_s"]), float(r["avg_path_length"]))
            sr[algo][ep] = m.bit_count() / len(rows)
            masks[algo][ep] = m
            vals[algo][ep] = vd

    # Baseline (fixed, no epoch choice)
    base_mask = (1 << 100) - 1
    base_vals = {}
    for ab in BASELINE_ALGOS:
        rows = baseline_data.get(ab, [])
        m, vd = 0, {}
        for r in rows:
            ridx = int(r["run_idx"])
            if float(r["success_rate"]) == 1.0:
                m |= (1 << ridx)
            vd[ridx] = (float(r["path_time_s"]), float(r["avg_curvature_1_m"]),
                        float(r["planning_time_s"]), float(r["avg_path_length"]))
        base_mask &= m
        base_vals[ab] = vd

    return sr, masks, vals, base_mask, base_vals


def norm(v, vmin, vmax):
    return 0.0 if vmax == vmin else (v - vmin) / (vmax - vmin)


def eval_quality(mask_int, combo, vals, base_vals):
    """Compute quality metrics for pairs in bitmask intersection."""
    # Extract run_idxs from bitmask
    idxs = []
    x = mask_int
    while x:
        b = x & (-x)
        idxs.append(b.bit_length() - 1)
        x ^= b
    n = len(idxs)

    # Gather per-pair values
    all_pt, all_k, all_pl = [], [], []
    algo_vl = {}
    for algo in DRL_ALGOS:
        ep = combo[algo]
        vl = [vals[algo][ep][i] for i in idxs]
        algo_vl[algo] = vl
        for v in vl:
            all_pt.append(v[0]); all_k.append(v[1]); all_pl.append(v[2])
    for ab in BASELINE_ALGOS:
        vl = [base_vals[ab][i] for i in idxs]
        algo_vl[ab] = vl
        for v in vl:
            all_pt.append(v[0]); all_k.append(v[1]); all_pl.append(v[2])

    pt_min, pt_max = min(all_pt), max(all_pt)
    k_min, k_max = min(all_k), max(all_k)
    pl_min, pl_max = min(all_pl), max(all_pl)

    metrics = {}
    for algo in ALL_REPORT:
        vl = algo_vl[algo]
        pathlen = sum(v[3] for v in vl) / n
        curv = sum(v[1] for v in vl) / n
        plan_t = sum(v[2] for v in vl) / n
        comps = [(W_PT * norm(v[0], pt_min, pt_max) +
                  W_K * norm(v[1], k_min, k_max) +
                  W_PL * norm(v[2], pl_min, pl_max)) / (W_PT + W_K + W_PL) for v in vl]
        metrics[algo] = {"pathlen": pathlen, "composite": sum(comps) / n,
                         "curvature": curv, "plan_time": plan_t}
    return metrics


def search_6algo(sr, masks, vals, base_mask, base_vals, mode):
    """Decomposed 6-algo search."""
    MIN_SR = 0.50
    print(f"\n  Building valid pairs (SR >= {MIN_SR})...")

    bases = {"DQN": ("CNN-DQN", "MLP-DQN"),
             "DDQN": ("CNN-DDQN", "MLP-DDQN"),
             "PDDQN": ("CNN-PDDQN", "MLP-PDDQN")}

    valid_pairs = {}
    for base, (ca, ma) in bases.items():
        pairs = []
        for ce in EPOCHS:
            sc = sr[ca][ce]
            if sc < MIN_SR: continue
            mc = masks[ca][ce]
            for me in EPOCHS:
                sm = sr[ma][me]
                if sm < MIN_SR or sc < sm: continue  # CNN >= MLP
                pairs.append((ce, me, mc & masks[ma][me], sc, sm))
        valid_pairs[base] = pairs
        print(f"    {base:6s}: {len(pairs):5d} valid pairs")

    n_pddqn = len(valid_pairs["PDDQN"])
    n_dqn = len(valid_pairs["DQN"])
    n_ddqn = len(valid_pairs["DDQN"])
    print(f"    Max combos (before global check): {n_pddqn * n_dqn * n_ddqn:,}")

    results = []
    n_checked = 0
    n_intersect_pass = 0
    MAX_RESULTS = 500_000
    t0 = time.time()

    for pi, (cp, mp, pm, sr_cp, sr_mp) in enumerate(valid_pairs["PDDQN"]):
        outer = base_mask & pm
        if outer.bit_count() < 3:
            continue

        # Pre-filter: CNN-PDDQN SR >= all other SRs in combo
        dqn_f = [(cd, md, dm) for (cd, md, dm, sc, sm) in valid_pairs["DQN"]
                 if sr_cp >= sc and sr_cp >= sm]
        ddqn_f = [(cdd, mdd, ddm) for (cdd, mdd, ddm, sc, sm) in valid_pairs["DDQN"]
                  if sr_cp >= sc and sr_cp >= sm]

        for (cd, md, dm) in dqn_f:
            mid = outer & dm
            if mid.bit_count() < 3:
                continue
            for (cdd, mdd, ddm) in ddqn_f:
                n_checked += 1
                final = mid & ddm
                nf = final.bit_count()
                if nf < 3:
                    continue
                n_intersect_pass += 1

                combo = {"CNN-PDDQN": cp, "CNN-DDQN": cdd, "CNN-DQN": cd,
                         "MLP-PDDQN": mp, "MLP-DDQN": mdd, "MLP-DQN": md}

                m = eval_quality(final, combo, vals, base_vals)
                pc = m["CNN-PDDQN"]["composite"]
                pp = m["CNN-PDDQN"]["pathlen"]
                bp = min(m[a]["pathlen"] for a in BASELINE_ALGOS)
                dp = min(m[a]["pathlen"] for a in DRL_ALGOS)
                pb = all(m[a]["composite"] >= pc for a in DRL_ALGOS if a != "CNN-PDDQN")

                sr_c = {a: sr[a][combo[a]] for a in DRL_ALGOS}

                results.append({
                    "combo": combo, "n": nf, "sr": sr_c,
                    "pddqn_comp": pc, "pddqn_path": pp,
                    "pddqn_best": pb, "base_path": bp,
                    "drl_path": dp, "gap": pp - bp,
                    "drl_gap": dp - bp,  # best DRL algo vs baseline
                    "metrics": m,
                })

                if len(results) >= MAX_RESULTS:
                    print(f"\n  ⚠ Hit {MAX_RESULTS:,} results cap, stopping search")
                    break
            if len(results) >= MAX_RESULTS:
                break
        if len(results) >= MAX_RESULTS:
            break

        # Progress every 50 PDDQN pairs
        if (pi + 1) % 50 == 0:
            el = time.time() - t0
            eta = el / (pi + 1) * (n_pddqn - pi - 1)
            print(f"    PDDQN {pi+1}/{n_pddqn}, checked={n_checked:,}, "
                  f"valid={len(results):,}, {el:.0f}s elapsed, ~{eta:.0f}s remaining",
                  flush=True)

    elapsed = time.time() - t0
    print(f"\n  Search done: {n_checked:,} checked, {n_intersect_pass:,} passed intersection, "
          f"{len(results):,} valid, {elapsed:.1f}s")
    return results


def report(results, v5_result, mode):
    """Print top results by various criteria."""
    if not results:
        print("  No valid combos found!")
        return

    print(f"\n  Total valid combos: {len(results):,}")
    n_with_mlp_change = sum(1 for r in results
                            if any(r["combo"][a] != V5_COMBO[a]
                                   for a in ["MLP-DQN", "MLP-DDQN", "MLP-PDDQN"]))
    print(f"  Combos with MLP changes from V5: {n_with_mlp_change:,}")

    def fmt_sr(r):
        return " ".join(f"{a[-5:]}={r['sr'][a]:.2f}" for a in DRL_ALGOS)

    def show_top(title, sorted_results, n=5):
        print(f"\n  --- {title} ---")
        for i, r in enumerate(sorted_results[:n]):
            c = r["combo"]
            print(f"  #{i+1}: gap={r['gap']:+.3f}m, drl_gap={r['drl_gap']:+.3f}m, "
                  f"path={r['pddqn_path']:.3f}m, comp={r['pddqn_comp']:.4f}, "
                  f"n={r['n']}, best={'YES' if r['pddqn_best'] else 'no'}")
            diffs = [f"{a}={c[a]}" for a in DRL_ALGOS if c[a] != V5_COMBO[a]]
            if diffs:
                print(f"       Δ V5: {', '.join(diffs)}")
            else:
                print(f"       Same as V5")

    show_top("Top 5 by CNN-PDDQN path gap vs baseline (smallest)",
             sorted(results, key=lambda r: r["gap"]))
    show_top("Top 5 by best-DRL path gap vs baseline (smallest)",
             sorted(results, key=lambda r: r["drl_gap"]))
    show_top("Top 5 by CNN-PDDQN composite (best)",
             sorted(results, key=lambda r: r["pddqn_comp"]))
    show_top("Top 5 by CNN-PDDQN path (shortest)",
             sorted(results, key=lambda r: r["pddqn_path"]))

    pddqn_best = [r for r in results if r["pddqn_best"]]
    if pddqn_best:
        show_top(f"Top 5: PDDQN=best composite ({len(pddqn_best)} combos), by gap",
                 sorted(pddqn_best, key=lambda r: r["gap"]))

    # Check if MLP changes improve best Long gap
    cnn_only = [r for r in results
                if all(r["combo"][a] == V5_COMBO[a] for a in ["MLP-DQN", "MLP-DDQN", "MLP-PDDQN"])]
    mlp_changed = [r for r in results
                   if any(r["combo"][a] != V5_COMBO[a] for a in ["MLP-DQN", "MLP-DDQN", "MLP-PDDQN"])]

    if cnn_only:
        best_cnn_only = min(cnn_only, key=lambda r: r["gap"])
        print(f"\n  --- Best gap (MLP=V5, CNN varies only) ---")
        print(f"    gap={best_cnn_only['gap']:+.3f}m, path={best_cnn_only['pddqn_path']:.3f}m, "
              f"comp={best_cnn_only['pddqn_comp']:.4f}, n={best_cnn_only['n']}")
        diffs = [f"{a}={best_cnn_only['combo'][a]}" for a in DRL_ALGOS if best_cnn_only['combo'][a] != V5_COMBO[a]]
        if diffs:
            print(f"    Δ V5: {', '.join(diffs)}")

    if mlp_changed:
        best_mlp = min(mlp_changed, key=lambda r: r["gap"])
        print(f"\n  --- Best gap (MLP also changed) ---")
        print(f"    gap={best_mlp['gap']:+.3f}m, path={best_mlp['pddqn_path']:.3f}m, "
              f"comp={best_mlp['pddqn_comp']:.4f}, n={best_mlp['n']}")
        diffs = [f"{a}={best_mlp['combo'][a]}" for a in DRL_ALGOS if best_mlp['combo'][a] != V5_COMBO[a]]
        if diffs:
            print(f"    Δ V5: {', '.join(diffs)}")

        if cnn_only:
            delta = best_mlp["gap"] - best_cnn_only["gap"]
            print(f"\n  >>> MLP tuning {'IMPROVES' if delta < -0.001 else 'does NOT improve'} "
                  f"gap by {delta:+.3f}m vs CNN-only search")

    # V5 reference
    if v5_result:
        print(f"\n  --- V5 reference ---")
        print(f"    gap={v5_result['gap']:+.3f}m, path={v5_result['pddqn_path']:.3f}m, "
              f"comp={v5_result['pddqn_comp']:.4f}, n={v5_result['n']}, "
              f"best={'YES' if v5_result['pddqn_best'] else 'no'}")


def eval_v5(sr, masks, vals, base_mask, base_vals):
    """Evaluate V5 combo."""
    v5_mask = base_mask
    for algo in DRL_ALGOS:
        v5_mask &= masks[algo][V5_COMBO[algo]]
    nf = v5_mask.bit_count()
    if nf < 3:
        return None
    m = eval_quality(v5_mask, V5_COMBO, vals, base_vals)
    pc = m["CNN-PDDQN"]["composite"]
    return {
        "combo": V5_COMBO, "n": nf,
        "sr": {a: sr[a][V5_COMBO[a]] for a in DRL_ALGOS},
        "pddqn_comp": pc, "pddqn_path": m["CNN-PDDQN"]["pathlen"],
        "pddqn_best": all(m[a]["composite"] >= pc for a in DRL_ALGOS if a != "CNN-PDDQN"),
        "base_path": min(m[a]["pathlen"] for a in BASELINE_ALGOS),
        "drl_path": min(m[a]["pathlen"] for a in DRL_ALGOS),
        "gap": m["CNN-PDDQN"]["pathlen"] - min(m[a]["pathlen"] for a in BASELINE_ALGOS),
        "drl_gap": min(m[a]["pathlen"] for a in DRL_ALGOS) - min(m[a]["pathlen"] for a in BASELINE_ALGOS),
        "metrics": m,
    }


def main():
    os.chdir(Path(__file__).resolve().parent.parent)

    for mode in ["sr_long", "sr_short"]:
        print(f"\n{'='*70}")
        print(f"  MODE: {mode} — Full 6-algo search")
        print(f"{'='*70}")

        print("  Loading data...", flush=True)
        drl_data, baseline_data = load_all_data(mode)
        sr, masks, vals, base_mask, base_vals = precompute(drl_data, baseline_data)

        n_base = base_mask.bit_count()
        print(f"  Baseline intersection (RRT* ∩ LO-HA*): {n_base} pairs")

        # Show per-algo SR at V5 epochs
        print(f"  V5 SRs: " + ", ".join(f"{a}={sr[a][V5_COMBO[a]]:.2f}" for a in DRL_ALGOS))

        results = search_6algo(sr, masks, vals, base_mask, base_vals, mode)
        v5_result = eval_v5(sr, masks, vals, base_mask, base_vals)
        report(results, v5_result, mode)

        # Full quality table for overall best gap
        if results:
            best = min(results, key=lambda r: r["gap"])
            print(f"\n  === Best gap combo — full quality table ===")
            print(f"  Combo: {best['combo']}")
            print(f"  {'Algorithm':12s} {'Path(m)':>8s} {'Curv':>8s} {'PlanT(s)':>9s} {'Composite':>10s}")
            print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*9} {'-'*10}")
            for algo in ALL_REPORT:
                m = best["metrics"][algo]
                print(f"  {algo:12s} {m['pathlen']:8.3f} {m['curvature']:8.4f} "
                      f"{m['plan_time']:9.4f} {m['composite']:10.4f}")

    print(f"\n{'='*70}")
    print("DONE")


if __name__ == "__main__":
    main()
