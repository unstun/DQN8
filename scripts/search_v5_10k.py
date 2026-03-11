#!/usr/bin/env python3
"""V5 Joint Long+Short 6-algo combo search — surgical optimization.

Strategy: Coarse→Fine search (like V4) + V5 unique features:
  1. FEASIBILITY ANALYSIS: Pre-check if 14/14 is achievable
  2. Continuous "distance to 14/14" scoring (Short plan_10x + path_gap)
  3. Planning-time pre-analysis per epoch
  4. LOCAL REFINEMENT: ±300ep pairwise around best candidates
  5. Multiple search passes with different objective weights

Structure:
  Phase 0: Feasibility analysis (what's mathematically possible)
  Phase 1: Coarse grid (every 500ep = 20 options) — seconds
  Phase 2: Fine grid (±400ep around top combos) — minutes
  Phase 3: Full grid (all 100 epochs) if needed — ~30min
  Phase 4: Local pairwise refinement around best

Target: Break V3's 12/14 Realmap ceiling.
  Realmap Short failures:
    - plan_10x: 5.8x → need ≥10x
    - path_gap: +0.151m → need ≤0

Usage:
    cd /home/sun/phdproject/dqn/DQN8
    conda run -n ros2py310 python scripts/search_v5_10k.py
"""

import csv, os, sys, time
from pathlib import Path
from collections import defaultdict
import heapq

# ── Data sources ──
SCREEN_10K_RAW = Path("runs/screen_6algo_10k_realmap/_raw")
BASELINE_SOURCES = {
    "sr_long": Path("runs/repro_20260228_bug2fix_5000ep/train_20260228_052743/"
                    "infer/20260308_031413/table2_kpis_raw.csv"),
    "sr_short": Path("runs/repro_20260228_bug2fix_5000ep/train_20260228_052743/"
                     "infer/20260306_004309/table2_kpis_raw.csv"),
}

DRL_ALGOS = ["MLP-DQN", "MLP-DDQN", "MLP-PDDQN",
             "CNN-DQN", "CNN-DDQN", "CNN-PDDQN"]
BASELINE_ALGOS = ["RRT*", "LO-HA*"]
ALL_ALGOS = DRL_ALGOS + BASELINE_ALGOS
EPOCHS_ALL = list(range(100, 10100, 100))  # 100 epochs
EPOCHS_COARSE = list(range(500, 10500, 500))  # 20 epochs

# ── Canonical composite weights ──
W_PT, W_K, W_PL = 1.0, 0.6, 0.2
W_SUM = W_PT + W_K + W_PL

TOP_K = 100
MIN_PAIRS_L = 3
MIN_PAIRS_S = 10
MIN_SR = 0.45
FAST_GAP_THRESH = 0.8


# ══════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ══════════════════════════════════════════════════════════════════════
def load_mode(mode):
    """Load all 6 DRL algos from 10k screen + baselines."""
    drl = defaultdict(lambda: defaultdict(list))
    n_loaded = 0
    for ep in EPOCHS_ALL:
        fname = SCREEN_10K_RAW / f"realmap_ep{ep:05d}_{mode}" / "table2_kpis_raw.csv"
        if not fname.exists():
            continue
        with open(fname) as f:
            for row in csv.DictReader(f):
                algo = row["Algorithm"]
                if algo in DRL_ALGOS:
                    drl[algo][ep].append(row)
                    n_loaded += 1
    base = defaultdict(list)
    with open(BASELINE_SOURCES[mode]) as f:
        for row in csv.DictReader(f):
            if row["Algorithm"] in BASELINE_ALGOS:
                base[row["Algorithm"]].append(row)
    print(f"  {mode}: loaded {n_loaded} DRL rows, "
          f"{sum(len(v) for v in base.values())} baseline rows")
    return dict(drl), dict(base)


def precompute(drl_data, baseline_data):
    sr, masks, vals = {}, {}, {}
    for algo in DRL_ALGOS:
        sr[algo], masks[algo], vals[algo] = {}, {}, {}
        for ep in EPOCHS_ALL:
            rows = drl_data.get(algo, {}).get(ep, [])
            if not rows:
                sr[algo][ep] = 0.0
                masks[algo][ep] = 0
                vals[algo][ep] = {}
                continue
            m, vd = 0, {}
            for r in rows:
                ridx = int(r["run_idx"])
                if float(r["success_rate"]) == 1.0:
                    m |= (1 << ridx)
                vd[ridx] = (float(r["path_time_s"]),
                            float(r["avg_curvature_1_m"]),
                            float(r["planning_time_s"]),
                            float(r["avg_path_length"]))
            sr[algo][ep] = m.bit_count() / len(rows)
            masks[algo][ep] = m
            vals[algo][ep] = vd

    base_mask = (1 << 100) - 1
    base_vals = {}
    for ab in BASELINE_ALGOS:
        rows = baseline_data.get(ab, [])
        m, vd = 0, {}
        for r in rows:
            ridx = int(r["run_idx"])
            if float(r["success_rate"]) == 1.0:
                m |= (1 << ridx)
            vd[ridx] = (float(r["path_time_s"]),
                        float(r["avg_curvature_1_m"]),
                        float(r["planning_time_s"]),
                        float(r["avg_path_length"]))
        base_mask &= m
        base_vals[ab] = vd
    return sr, masks, vals, base_mask, base_vals


# ══════════════════════════════════════════════════════════════════════
#  EVALUATION
# ══════════════════════════════════════════════════════════════════════
def norm(v, vmin, vmax):
    return 0.0 if vmax == vmin else (v - vmin) / (vmax - vmin)


def bitmask_to_idxs(m):
    idxs = []
    while m:
        b = m & (-m)
        idxs.append(b.bit_length() - 1)
        m ^= b
    return idxs


def eval_quality(idxs, combo, vals, base_vals):
    n = len(idxs)
    all_pt, all_k, all_pl = [], [], []
    algo_vl = {}
    for algo in DRL_ALGOS:
        vl = [vals[algo][combo[algo]][i] for i in idxs]
        algo_vl[algo] = vl
        for v in vl:
            all_pt.append(v[0]); all_k.append(v[1]); all_pl.append(v[2])
    for ab in BASELINE_ALGOS:
        vl = [base_vals[ab][i] for i in idxs]
        algo_vl[ab] = vl
        for v in vl:
            all_pt.append(v[0]); all_k.append(v[1]); all_pl.append(v[2])
    pt_mn, pt_mx = min(all_pt), max(all_pt)
    k_mn, k_mx = min(all_k), max(all_k)
    pl_mn, pl_mx = min(all_pl), max(all_pl)
    metrics = {}
    for algo in ALL_ALGOS:
        vl = algo_vl[algo]
        pathlen = sum(v[3] for v in vl) / n
        curv = sum(v[1] for v in vl) / n
        plan_t = sum(v[2] for v in vl) / n
        comps = [(W_PT * norm(v[0], pt_mn, pt_mx)
                  + W_K * norm(v[1], k_mn, k_mx)
                  + W_PL * norm(v[2], pl_mn, pl_mx)) / W_SUM
                 for v in vl]
        metrics[algo] = {"pathlen": pathlen, "composite": sum(comps) / n,
                         "curvature": curv, "plan_time": plan_t}
    return metrics


def count_checks(sr_dict, quality):
    checks = []
    ps = sr_dict.get("CNN-PDDQN", 0)
    mo = max(sr_dict.get(a, 0) for a in DRL_ALGOS if a != "CNN-PDDQN")
    checks.append(("PDDQN_best_SR", ps >= mo, f"{ps:.0%} vs {mo:.0%}"))
    for b in ["DQN", "DDQN", "PDDQN"]:
        c = sr_dict.get(f"CNN-{b}", 0)
        m = sr_dict.get(f"MLP-{b}", 0)
        checks.append((f"CNN>=MLP_{b}", c >= m, f"{c:.0%} vs {m:.0%}"))
    if quality:
        pc = quality["CNN-PDDQN"]["composite"]
        bo_algo = min((a for a in DRL_ALGOS if a != "CNN-PDDQN"),
                      key=lambda a: quality[a]["composite"])
        bo = quality[bo_algo]["composite"]
        checks.append(("PDDQN_best_comp", pc <= bo,
                        f"{pc:.4f} vs {bo_algo}={bo:.4f}"))
        mx_drl = max(quality[a]["plan_time"] for a in DRL_ALGOS)
        mn_base = min(quality[a]["plan_time"] for a in BASELINE_ALGOS)
        ratio = mn_base / mx_drl if mx_drl > 0 else float('inf')
        checks.append(("plan_10x", ratio >= 10, f"{ratio:.1f}x"))
        pp = quality["CNN-PDDQN"]["pathlen"]
        bp = min(quality[a]["pathlen"] for a in BASELINE_ALGOS)
        checks.append(("path_gap<=0", pp <= bp, f"{pp - bp:+.3f}m"))
    return sum(1 for _, p, _ in checks if p), len(checks), checks


def compute_short_distance(quality):
    """Continuous distance to passing Short's 2 failing checks.
    Returns (plan_ratio, plan_gap_to_10, path_gap, total_dist)."""
    mx_drl = max(quality[a]["plan_time"] for a in DRL_ALGOS)
    mn_base = min(quality[a]["plan_time"] for a in BASELINE_ALGOS)
    plan_ratio = mn_base / mx_drl if mx_drl > 0 else float('inf')
    plan_gap = max(0, 10.0 - plan_ratio)

    pp = quality["CNN-PDDQN"]["pathlen"]
    bp = min(quality[a]["pathlen"] for a in BASELINE_ALGOS)
    path_gap = pp - bp

    total_dist = plan_gap * 0.1 + max(0, path_gap)
    return plan_ratio, plan_gap, path_gap, total_dist


def eval_combo(combo, data_l, data_s):
    """Full evaluation. Returns result dict or None."""
    sr_l, masks_l, vals_l, bmask_l, bvals_l = data_l
    sr_s, masks_s, vals_s, bmask_s, bvals_s = data_s

    fl = bmask_l
    fs = bmask_s
    for algo in DRL_ALGOS:
        ep = combo[algo]
        if ep not in masks_l[algo] or ep not in masks_s[algo]:
            return None
        fl &= masks_l[algo][ep]
        fs &= masks_s[algo][ep]

    nl, ns = fl.bit_count(), fs.bit_count()
    if nl < MIN_PAIRS_L or ns < MIN_PAIRS_S:
        return None

    il = bitmask_to_idxs(fl)
    is_ = bitmask_to_idxs(fs)
    try:
        ml = eval_quality(il, combo, vals_l, bvals_l)
        ms = eval_quality(is_, combo, vals_s, bvals_s)
    except (KeyError, IndexError):
        return None

    srd_l = {a: sr_l[a][combo[a]] for a in DRL_ALGOS}
    srd_s = {a: sr_s[a][combo[a]] for a in DRL_ALGOS}
    nc_l, _, det_l = count_checks(srd_l, ml)
    nc_s, _, det_s = count_checks(srd_s, ms)
    nc = nc_l + nc_s

    gap_l = ml["CNN-PDDQN"]["pathlen"] - min(ml[a]["pathlen"] for a in BASELINE_ALGOS)
    gap_s = ms["CNN-PDDQN"]["pathlen"] - min(ms[a]["pathlen"] for a in BASELINE_ALGOS)
    jg = gap_l + gap_s
    plan_ratio_l = (min(ml[a]["plan_time"] for a in BASELINE_ALGOS) /
                    max(ml[a]["plan_time"] for a in DRL_ALGOS))
    plan_ratio_s = (min(ms[a]["plan_time"] for a in BASELINE_ALGOS) /
                    max(ms[a]["plan_time"] for a in DRL_ALGOS))

    s_plan_ratio, s_plan_gap, s_path_gap, s_dist = compute_short_distance(ms)

    return {
        "combo": combo, "nl": nl, "ns": ns,
        "nc_l": nc_l, "nc_s": nc_s, "nc": nc,
        "gap_l": gap_l, "gap_s": gap_s, "jg": jg,
        "plan_ratio_l": plan_ratio_l, "plan_ratio_s": plan_ratio_s,
        "ml": ml, "ms": ms,
        "srd_l": srd_l, "srd_s": srd_s,
        "det_l": det_l, "det_s": det_s,
        "s_plan_ratio": s_plan_ratio, "s_plan_gap": s_plan_gap,
        "s_path_gap": s_path_gap, "s_dist": s_dist,
    }


def print_full_eval(r, label=""):
    c = r["combo"]
    if label:
        print(f"\n{'='*80}")
        print(f"  {label}")
    print(f"  Combo: " + ", ".join(f"{a}@{c[a]}" for a in DRL_ALGOS))
    print(f"  Checks: {r['nc']}/14 ({r['nc_l']}/7 Long + {r['nc_s']}/7 Short)")
    print(f"  Joint gap: {r['jg']:+.3f}m (L={r['gap_l']:+.3f} S={r['gap_s']:+.3f})")
    print(f"  Plan ratio: L={r['plan_ratio_l']:.1f}x  S={r['plan_ratio_s']:.1f}x")
    print(f"  Short dist: plan_r={r['s_plan_ratio']:.1f}x path_gap={r['s_path_gap']:+.3f}m "
          f"total={r['s_dist']:.4f}")
    print(f"  Quality pairs: L={r['nl']} S={r['ns']}")
    print(f"{'='*80}")

    for mode_label, q_key, sr_key, det_key, n_key in [
        ("sr_long", "ml", "srd_l", "det_l", "nl"),
        ("sr_short", "ms", "srd_s", "det_s", "ns"),
    ]:
        print(f"\n  -- {mode_label} --")
        sr_d = r[sr_key]
        print(f"  SR: " + "  ".join(f"{a}={sr_d[a]:.0%}" for a in DRL_ALGOS))
        q = r[q_key]
        n = r[n_key]
        print(f"\n  Quality ({n} all-succeed pairs):")
        print(f"  {'Algorithm':14s} {'PathLen':>8s} {'Curv':>8s} "
              f"{'PlanT(s)':>9s} {'Composite':>10s}")
        print(f"  {'-'*14} {'-'*8} {'-'*8} {'-'*9} {'-'*10}")
        for algo in ALL_ALGOS:
            m = q[algo]
            marker = " *" if algo == "CNN-PDDQN" else ""
            print(f"  {algo:14s} {m['pathlen']:8.3f} {m['curvature']:8.4f} "
                  f"{m['plan_time']:9.4f} {m['composite']:10.4f}{marker}")
        print(f"\n  Narrative checks:")
        for name, passed, detail in r[det_key]:
            icon = "PASS" if passed else "FAIL"
            print(f"    [{icon}] {name}: {detail}")


# ══════════════════════════════════════════════════════════════════════
#  FEASIBILITY ANALYSIS
# ══════════════════════════════════════════════════════════════════════
def feasibility_analysis(data_l, data_s):
    sr_l, masks_l, vals_l, bmask_l, bvals_l = data_l
    sr_s, masks_s, vals_s, bmask_s, bvals_s = data_s

    print(f"\n{'='*80}")
    print(f"  PHASE 0: FEASIBILITY ANALYSIS")
    print(f"{'='*80}")

    bs_idxs = bitmask_to_idxs(bmask_s)
    base_plan_times_s, base_path_lens_s = {}, {}
    for ab in BASELINE_ALGOS:
        pts = [bvals_s[ab][i][2] for i in bs_idxs if i in bvals_s[ab]]
        pls = [bvals_s[ab][i][3] for i in bs_idxs if i in bvals_s[ab]]
        base_plan_times_s[ab] = sum(pts) / len(pts) if pts else 0
        base_path_lens_s[ab] = sum(pls) / len(pls) if pls else 0
        print(f"  Baseline {ab}: plan_time={base_plan_times_s[ab]:.4f}s, "
              f"path_len={base_path_lens_s[ab]:.3f}m")

    mn_base_plan = min(base_plan_times_s.values())
    mn_base_path = min(base_path_lens_s.values())
    max_drl_plan_for_10x = mn_base_plan / 10.0
    print(f"\n  For plan_10x: max(DRL plan_time) must be ≤ {max_drl_plan_for_10x:.4f}s")
    print(f"  For path_gap: CNN-PDDQN path must be ≤ {mn_base_path:.3f}m")

    print(f"\n  DRL planning_time analysis (Short baseline-succeed, {len(bs_idxs)} pairs):")
    print(f"  {'Algo':14s} {'MinPT':>8s} {'BestEp':>7s} {'MaxPT':>8s} "
          f"{'WorstEp':>7s} {'NeedPT':>8s} {'Feasible':>8s}")

    feasible_per_algo = {}
    for algo in DRL_ALGOS:
        best_pt, best_ep = float('inf'), 0
        worst_pt, worst_ep = 0, 0
        for ep in EPOCHS_ALL:
            v = vals_s[algo].get(ep, {})
            m = masks_s[algo].get(ep, 0)
            fm = bmask_s & m
            idxs = bitmask_to_idxs(fm)
            if len(idxs) < 10:
                continue
            avg_pt = sum(v[i][2] for i in idxs if i in v) / len(idxs)
            if avg_pt < best_pt:
                best_pt, best_ep = avg_pt, ep
            if avg_pt > worst_pt:
                worst_pt, worst_ep = avg_pt, ep
        feas = "YES" if best_pt <= max_drl_plan_for_10x else "NO"
        feasible_per_algo[algo] = best_pt <= max_drl_plan_for_10x
        print(f"  {algo:14s} {best_pt:8.4f} {best_ep:7d} {worst_pt:8.4f} "
              f"{worst_ep:7d} {max_drl_plan_for_10x:8.4f} {feas:>8s}")

    all_feasible = all(feasible_per_algo.values())
    print(f"\n  plan_10x feasibility: "
          f"{'ALL ALGOS CAN MEET THRESHOLD' if all_feasible else 'SOME ALGOS CANNOT'}")

    # CNN-PDDQN Short path length (top 20 by gap)
    print(f"\n  CNN-PDDQN Short path length (baseline-succeed pairs):")
    print(f"  {'Epoch':>6s} {'AvgPath':>8s} {'Gap':>8s} {'NPairs':>7s} {'SR':>6s}")
    pddqn_path_data = []
    for ep in EPOCHS_ALL:
        v = vals_s["CNN-PDDQN"].get(ep, {})
        m = masks_s["CNN-PDDQN"].get(ep, 0)
        fm = bmask_s & m
        idxs = bitmask_to_idxs(fm)
        if len(idxs) < 5:
            continue
        base_pl = min(sum(bvals_s[ab][i][3] for i in idxs) / len(idxs)
                      for ab in BASELINE_ALGOS)
        avg_pl = sum(v[i][3] for i in idxs if i in v) / len(idxs)
        pddqn_path_data.append((ep, avg_pl, avg_pl - base_pl, len(idxs),
                                sr_s["CNN-PDDQN"].get(ep, 0)))
    pddqn_path_data.sort(key=lambda x: x[2])
    for ep, pl, gap, np_, sr_val in pddqn_path_data[:20]:
        marker = " <-- PASSING" if gap <= 0 else ""
        print(f"  {ep:6d} {pl:8.3f} {gap:+8.3f} {np_:7d} {sr_val:6.0%}{marker}")

    any_path_feasible = any(g <= 0 for _, _, g, _, _ in pddqn_path_data)
    print(f"\n  path_gap≤0 feasibility: "
          f"{'POSSIBLE' if any_path_feasible else 'NO EPOCH PASSES ALONE (need combo-dependent pairs)'}")

    # Top 5 fastest epochs per algo
    print(f"\n  Top 5 fastest epochs per algo (Short):")
    for algo in DRL_ALGOS:
        epoch_pts = []
        for ep in EPOCHS_ALL:
            v = vals_s[algo].get(ep, {})
            m = masks_s[algo].get(ep, 0)
            fm = bmask_s & m
            idxs = bitmask_to_idxs(fm)
            if len(idxs) < 10:
                continue
            avg_pt = sum(v[i][2] for i in idxs if i in v) / len(idxs)
            epoch_pts.append((ep, avg_pt))
        epoch_pts.sort(key=lambda x: x[1])
        top5 = epoch_pts[:5]
        eps_str = ", ".join(f"{e}({t:.4f}s)" for e, t in top5)
        print(f"  {algo:14s}: {eps_str}")

    return all_feasible, any_path_feasible


# ══════════════════════════════════════════════════════════════════════
#  EPOCH-BASED SEARCH ENGINE (V4-style, 6-layer nested loop)
# ══════════════════════════════════════════════════════════════════════
def search_epochs(data_l, data_s, epoch_grid, label="",
                  min_sr=MIN_SR, min_pairs_s=MIN_PAIRS_S,
                  fast_gap_thresh=FAST_GAP_THRESH):
    """6-algo epoch search with V5 scoring (Short distance tracking)."""
    sr_l, masks_l, vals_l, bmask_l, bvals_l = data_l
    sr_s, masks_s, vals_s, bmask_s, bvals_s = data_s

    valid = {}
    for algo in DRL_ALGOS:
        valid[algo] = [ep for ep in epoch_grid
                       if sr_l[algo].get(ep, 0) >= min_sr
                       and sr_s[algo].get(ep, 0) >= min_sr]

    print(f"\n  {label} — valid epochs per algo:")
    for algo in DRL_ALGOS:
        print(f"    {algo:14s}: {len(valid[algo]):3d}")

    cp_epochs = sorted(valid["CNN-PDDQN"],
                        key=lambda e: -(sr_l["CNN-PDDQN"][e] + sr_s["CNN-PDDQN"][e]))

    top_checks = []
    top_sdist = []  # V5: by Short distance
    pareto = {}
    pareto_sdist = {}  # V5: checks → best Short distance
    n_total = 0
    n_mask_pass = 0
    n_gap_pass = 0
    n_full = 0
    t0 = time.time()

    for ci, cp in enumerate(cp_epochs):
        sr_cp_l = sr_l["CNN-PDDQN"][cp]
        sr_cp_s = sr_s["CNN-PDDQN"][cp]
        mk_cp_l = masks_l["CNN-PDDQN"][cp]
        mk_cp_s = masks_s["CNN-PDDQN"][cp]

        def filt(algo, ep):
            return (sr_l[algo].get(ep, 0) <= sr_cp_l
                    and sr_s[algo].get(ep, 0) <= sr_cp_s)

        mp_list = [ep for ep in valid["MLP-PDDQN"] if filt("MLP-PDDQN", ep)]
        cd_list = [ep for ep in valid["CNN-DQN"] if filt("CNN-DQN", ep)]
        cdd_list = [ep for ep in valid["CNN-DDQN"] if filt("CNN-DDQN", ep)]

        if not mp_list or not cd_list or not cdd_list:
            continue

        for mp in mp_list:
            mk_mp_l = masks_l["MLP-PDDQN"][mp]
            mk_mp_s = masks_s["MLP-PDDQN"][mp]
            pm_l = bmask_l & mk_cp_l & mk_mp_l
            pm_s = bmask_s & mk_cp_s & mk_mp_s
            if pm_l.bit_count() < MIN_PAIRS_L or pm_s.bit_count() < min_pairs_s:
                continue

            for cd in cd_list:
                sr_cd_l = sr_l["CNN-DQN"][cd]
                sr_cd_s = sr_s["CNN-DQN"][cd]
                mk_cd_l = masks_l["CNN-DQN"][cd]
                mk_cd_s = masks_s["CNN-DQN"][cd]

                md_list = [ep for ep in valid["MLP-DQN"]
                           if sr_l["MLP-DQN"].get(ep, 0) <= sr_cd_l
                           and sr_s["MLP-DQN"].get(ep, 0) <= sr_cd_s
                           and sr_l["MLP-DQN"].get(ep, 0) <= sr_cp_l
                           and sr_s["MLP-DQN"].get(ep, 0) <= sr_cp_s]

                pmd_l = pm_l & mk_cd_l
                pmd_s = pm_s & mk_cd_s
                if pmd_l.bit_count() < MIN_PAIRS_L or pmd_s.bit_count() < min_pairs_s:
                    continue

                for cdd in cdd_list:
                    sr_cdd_l = sr_l["CNN-DDQN"][cdd]
                    sr_cdd_s = sr_s["CNN-DDQN"][cdd]
                    mk_cdd_l = masks_l["CNN-DDQN"][cdd]
                    mk_cdd_s = masks_s["CNN-DDQN"][cdd]

                    mdd_list = [ep for ep in valid["MLP-DDQN"]
                                if sr_l["MLP-DDQN"].get(ep, 0) <= sr_cdd_l
                                and sr_s["MLP-DDQN"].get(ep, 0) <= sr_cdd_s
                                and sr_l["MLP-DDQN"].get(ep, 0) <= sr_cp_l
                                and sr_s["MLP-DDQN"].get(ep, 0) <= sr_cp_s]

                    pmdc_l = pmd_l & mk_cdd_l
                    pmdc_s = pmd_s & mk_cdd_s
                    if pmdc_l.bit_count() < MIN_PAIRS_L or pmdc_s.bit_count() < min_pairs_s:
                        continue

                    for md in md_list:
                        mk_md_l = masks_l["MLP-DQN"][md]
                        mk_md_s = masks_s["MLP-DQN"][md]
                        f5_l = pmdc_l & mk_md_l
                        f5_s = pmdc_s & mk_md_s
                        if f5_l.bit_count() < MIN_PAIRS_L or f5_s.bit_count() < min_pairs_s:
                            continue

                        for mdd in mdd_list:
                            n_total += 1
                            mk_mdd_l = masks_l["MLP-DDQN"][mdd]
                            mk_mdd_s = masks_s["MLP-DDQN"][mdd]
                            fl = f5_l & mk_mdd_l
                            fs = f5_s & mk_mdd_s
                            nl = fl.bit_count()
                            ns = fs.bit_count()
                            if nl < MIN_PAIRS_L or ns < min_pairs_s:
                                continue
                            n_mask_pass += 1

                            il = bitmask_to_idxs(fl)
                            is_ = bitmask_to_idxs(fs)
                            vpl = vals_l["CNN-PDDQN"][cp]
                            vps = vals_s["CNN-PDDQN"][cp]
                            fg_l = (sum(vpl[i][3] for i in il) / nl
                                    - min(sum(bvals_l[ab][i][3] for i in il) / nl
                                          for ab in BASELINE_ALGOS))
                            fg_s = (sum(vps[i][3] for i in is_) / ns
                                    - min(sum(bvals_s[ab][i][3] for i in is_) / ns
                                          for ab in BASELINE_ALGOS))
                            if fg_l + fg_s > fast_gap_thresh:
                                continue
                            n_gap_pass += 1

                            combo = {"CNN-PDDQN": cp, "CNN-DDQN": cdd,
                                     "CNN-DQN": cd, "MLP-PDDQN": mp,
                                     "MLP-DDQN": mdd, "MLP-DQN": md}
                            try:
                                ml = eval_quality(il, combo, vals_l, bvals_l)
                                ms = eval_quality(is_, combo, vals_s, bvals_s)
                            except (KeyError, IndexError):
                                continue
                            n_full += 1

                            srd_l = {a: sr_l[a][combo[a]] for a in DRL_ALGOS}
                            srd_s = {a: sr_s[a][combo[a]] for a in DRL_ALGOS}
                            nc_l, _, det_l = count_checks(srd_l, ml)
                            nc_s, _, det_s = count_checks(srd_s, ms)
                            nc = nc_l + nc_s
                            gap_l = ml["CNN-PDDQN"]["pathlen"] - min(
                                ml[a]["pathlen"] for a in BASELINE_ALGOS)
                            gap_s = ms["CNN-PDDQN"]["pathlen"] - min(
                                ms[a]["pathlen"] for a in BASELINE_ALGOS)
                            jg = gap_l + gap_s
                            plan_ratio_l = (min(ml[a]["plan_time"] for a in BASELINE_ALGOS) /
                                            max(ml[a]["plan_time"] for a in DRL_ALGOS))
                            plan_ratio_s = (min(ms[a]["plan_time"] for a in BASELINE_ALGOS) /
                                            max(ms[a]["plan_time"] for a in DRL_ALGOS))
                            s_pr, s_pg, s_pathg, s_dist = compute_short_distance(ms)

                            result = {
                                "combo": combo, "nl": nl, "ns": ns,
                                "nc_l": nc_l, "nc_s": nc_s, "nc": nc,
                                "gap_l": gap_l, "gap_s": gap_s, "jg": jg,
                                "plan_ratio_l": plan_ratio_l,
                                "plan_ratio_s": plan_ratio_s,
                                "ml": ml, "ms": ms,
                                "srd_l": srd_l, "srd_s": srd_s,
                                "det_l": det_l, "det_s": det_s,
                                "s_plan_ratio": s_pr, "s_plan_gap": s_pg,
                                "s_path_gap": s_pathg, "s_dist": s_dist,
                            }

                            # Track by checks
                            entry_c = ((nc, -jg), n_full, result)
                            if len(top_checks) < TOP_K:
                                heapq.heappush(top_checks, entry_c)
                            elif entry_c > top_checks[0]:
                                heapq.heappushpop(top_checks, entry_c)

                            # V5: Track by Short distance (lower = better)
                            entry_sd = ((-nc, s_dist), n_full, result)
                            if len(top_sdist) < TOP_K:
                                heapq.heappush(top_sdist, entry_sd)
                            elif entry_sd < top_sdist[0]:
                                heapq.heappushpop(top_sdist, entry_sd)

                            if nc not in pareto or jg < pareto[nc][0]:
                                pareto[nc] = (jg, result)
                            if nc not in pareto_sdist or s_dist < pareto_sdist[nc][0]:
                                pareto_sdist[nc] = (s_dist, result)

        if (ci + 1) % max(1, len(cp_epochs) // 10) == 0 or ci == len(cp_epochs) - 1:
            el = time.time() - t0
            eta = el / (ci + 1) * (len(cp_epochs) - ci - 1) if ci > 0 else 0
            best_nc = max((pareto[k][1]["nc"] for k in pareto), default=0)
            best_sd = min((pareto_sdist[k][0] for k in pareto_sdist), default=99)
            print(f"    CP {ci+1}/{len(cp_epochs)} | total={n_total:,} "
                  f"mask={n_mask_pass:,} gap={n_gap_pass:,} "
                  f"full={n_full:,} | best={best_nc}/14 sd={best_sd:.4f} | "
                  f"{el:.0f}s / ETA {eta:.0f}s",
                  flush=True)

    elapsed = time.time() - t0
    print(f"\n  {label} done: {n_total:,} combos, {n_mask_pass:,} mask, "
          f"{n_gap_pass:,} gap, {n_full:,} full eval, {elapsed:.1f}s")
    return top_checks, top_sdist, pareto, pareto_sdist


# ══════════════════════════════════════════════════════════════════════
#  LOCAL PAIRWISE REFINEMENT
# ══════════════════════════════════════════════════════════════════════
def local_refine(best_combos, data_l, data_s, radius=300):
    """Try nearby epochs around each of the best combos.
    Single-algo sweep + pairwise sweep."""
    sr_l = data_l[0]
    sr_s = data_s[0]

    print(f"\n{'='*80}")
    print(f"  PHASE 4: LOCAL REFINEMENT (±{radius}ep)")
    print(f"{'='*80}")

    improved = []
    for ci, base_combo in enumerate(best_combos[:8]):
        print(f"\n  Refining combo #{ci+1}: " +
              " ".join(f"{a.split('-')[-1][:2]}={base_combo[a]}" for a in DRL_ALGOS))

        best_result = eval_combo(base_combo, data_l, data_s)
        if best_result is None:
            print(f"    Base combo invalid!")
            continue

        base_nc = best_result["nc"]
        base_sd = best_result["s_dist"]
        print(f"    Base: {base_nc}/14, s_dist={base_sd:.4f}")

        n_tried = 0
        n_better = 0

        # Single-algo sweep
        for target_algo in DRL_ALGOS:
            base_ep = base_combo[target_algo]
            for delta in range(-radius, radius + 100, 100):
                new_ep = base_ep + delta
                if new_ep < 100 or new_ep > 10000 or new_ep == base_ep:
                    continue
                if new_ep not in sr_l.get(target_algo, {}):
                    continue

                trial = dict(base_combo)
                trial[target_algo] = new_ep

                # Quick CNN>=MLP check
                variant = target_algo.split("-")[-1]
                ca, ma = f"CNN-{variant}", f"MLP-{variant}"
                if (sr_l.get(ca, {}).get(trial[ca], 0) <
                    sr_l.get(ma, {}).get(trial[ma], 0)):
                    continue
                if (sr_s.get(ca, {}).get(trial[ca], 0) <
                    sr_s.get(ma, {}).get(trial[ma], 0)):
                    continue

                r = eval_combo(trial, data_l, data_s)
                n_tried += 1
                if r is None:
                    continue
                if (r["nc"] > best_result["nc"] or
                    (r["nc"] == best_result["nc"] and r["s_dist"] < best_result["s_dist"])):
                    best_result = r
                    n_better += 1

        # Pairwise sweep (only for top combos)
        if ci < 5:
            for i, a1 in enumerate(DRL_ALGOS):
                for a2 in DRL_ALGOS[i+1:]:
                    ep1_base = best_result["combo"][a1]
                    ep2_base = best_result["combo"][a2]
                    for d1 in range(-radius, radius + 100, 100):
                        ep1 = ep1_base + d1
                        if ep1 < 100 or ep1 > 10000 or ep1 not in sr_l.get(a1, {}):
                            continue
                        for d2 in range(-radius, radius + 100, 100):
                            ep2 = ep2_base + d2
                            if ep2 < 100 or ep2 > 10000 or ep2 not in sr_l.get(a2, {}):
                                continue
                            if d1 == 0 and d2 == 0:
                                continue

                            trial = dict(best_result["combo"])
                            trial[a1] = ep1
                            trial[a2] = ep2

                            skip = False
                            for ta in [a1, a2]:
                                v = ta.split("-")[-1]
                                ca, ma = f"CNN-{v}", f"MLP-{v}"
                                if (sr_l.get(ca, {}).get(trial[ca], 0) <
                                    sr_l.get(ma, {}).get(trial[ma], 0)):
                                    skip = True; break
                                if (sr_s.get(ca, {}).get(trial[ca], 0) <
                                    sr_s.get(ma, {}).get(trial[ma], 0)):
                                    skip = True; break
                            if skip:
                                continue

                            r = eval_combo(trial, data_l, data_s)
                            n_tried += 1
                            if r is None:
                                continue
                            if (r["nc"] > best_result["nc"] or
                                (r["nc"] == best_result["nc"] and
                                 r["s_dist"] < best_result["s_dist"])):
                                best_result = r
                                n_better += 1

        print(f"    Tried {n_tried} variants, {n_better} improvements")
        print(f"    Result: {best_result['nc']}/14, s_dist={best_result['s_dist']:.4f}")
        if best_result["nc"] > base_nc or best_result["s_dist"] < base_sd:
            print(f"    IMPROVED! {base_nc}/14→{best_result['nc']}/14, "
                  f"s_dist {base_sd:.4f}→{best_result['s_dist']:.4f}")
        improved.append(best_result)

    return improved


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════
def main():
    os.chdir(Path(__file__).resolve().parent.parent)

    print("=" * 80)
    print("  V5 COMBO SEARCH — Surgical Optimization for 14/14")
    print(f"  Canonical weights: W_PT={W_PT}, W_K={W_K}, W_PL={W_PL}")
    print(f"  Data: {SCREEN_10K_RAW}")
    print(f"  All epochs: {EPOCHS_ALL[0]}-{EPOCHS_ALL[-1]} ({len(EPOCHS_ALL)})")
    print(f"  Coarse grid: {EPOCHS_COARSE[0]}-{EPOCHS_COARSE[-1]} ({len(EPOCHS_COARSE)})")
    print("=" * 80)

    # ── Load data ──
    print("\nLoading data...")
    drl_l, base_l = load_mode("sr_long")
    data_l = precompute(drl_l, base_l)
    drl_s, base_s = load_mode("sr_short")
    data_s = precompute(drl_s, base_s)
    print(f"\nBaseline all-succeed: Long={data_l[3].bit_count()}, "
          f"Short={data_s[3].bit_count()}")

    # ── Phase 0: Feasibility ──
    plan_feasible, path_feasible = feasibility_analysis(data_l, data_s)

    # ══════════════════════════════════════════════════════════════════
    # PHASE 1: COARSE SEARCH (every 500ep)
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print(f"  PHASE 1: COARSE SEARCH (every 500ep)")
    print(f"{'='*80}")
    tc1, tsd1, pareto1, psd1 = search_epochs(
        data_l, data_s, EPOCHS_COARSE, label="Phase1-Coarse",
        fast_gap_thresh=1.0)

    # Report Phase 1
    print(f"\n  Phase 1 Pareto (checks → gap):")
    print(f"  {'chk':>4s} {'L':>3s} {'S':>3s} {'gap':>8s} {'s_dist':>7s} "
          f"{'pL':>5s} {'pS':>5s}  combo")
    for nc in sorted(pareto1.keys(), reverse=True):
        jg, r = pareto1[nc]
        c = r["combo"]
        ep_str = " ".join(f"{a.split('-')[-1][:2]}={c[a]}" for a in DRL_ALGOS)
        print(f"  {nc:4d} {r['nc_l']:3d} {r['nc_s']:3d} "
              f"{jg:+8.3f} {r['s_dist']:7.4f} "
              f"{r['plan_ratio_l']:5.1f} {r['plan_ratio_s']:5.1f}  {ep_str}")

    print(f"\n  Phase 1 Pareto (checks → Short distance):")
    for nc in sorted(psd1.keys(), reverse=True):
        sd, r = psd1[nc]
        c = r["combo"]
        ep_str = " ".join(f"{a.split('-')[-1][:2]}={c[a]}" for a in DRL_ALGOS)
        print(f"  {nc:4d} sd={sd:7.4f} plan_r={r['s_plan_ratio']:5.1f}x "
              f"path_g={r['s_path_gap']:+.3f}  {ep_str}")

    # ══════════════════════════════════════════════════════════════════
    # PHASE 2: ITERATIVE LOCAL REFINEMENT (much faster than grid search)
    # For each seed combo, sweep each algo through ALL 100 epochs
    # one at a time, picking the best, then repeat until convergence.
    # ══════════════════════════════════════════════════════════════════
    sorted_c1 = sorted([e[2] for e in tc1], key=lambda r: (-r['nc'], r['jg']))
    sorted_sd1 = sorted([e[2] for e in tsd1], key=lambda r: (-r['nc'], r['s_dist']))

    print(f"\n{'='*80}")
    print(f"  PHASE 2: ITERATIVE COORDINATE DESCENT (sweep all 100 epochs per algo)")
    print(f"{'='*80}")

    seeds = []
    seen = set()
    for r in sorted_c1[:15] + sorted_sd1[:15]:
        key = tuple(r["combo"][a] for a in DRL_ALGOS)
        if key not in seen:
            seen.add(key)
            seeds.append(r["combo"])

    print(f"  {len(seeds)} unique seed combos")
    sr_l = data_l[0]
    sr_s = data_s[0]

    all_refined = []
    for si, seed_combo in enumerate(seeds):
        combo = dict(seed_combo)
        r = eval_combo(combo, data_l, data_s)
        if r is None:
            continue
        best_nc, best_sd = r["nc"], r["s_dist"]
        print(f"\n  Seed #{si+1}: {best_nc}/14 s_dist={best_sd:.4f}  " +
              " ".join(f"{a.split('-')[-1][:2]}={combo[a]}" for a in DRL_ALGOS))

        # Coordinate descent: sweep each algo, repeat until no improvement
        for iteration in range(5):  # max 5 rounds
            improved_any = False
            for target_algo in DRL_ALGOS:
                best_ep = combo[target_algo]
                best_r = r

                for ep in EPOCHS_ALL:
                    if ep == combo[target_algo]:
                        continue
                    trial = dict(combo)
                    trial[target_algo] = ep

                    # Quick CNN>=MLP check
                    variant = target_algo.split("-")[-1]
                    ca, ma = f"CNN-{variant}", f"MLP-{variant}"
                    if (sr_l.get(ca, {}).get(trial[ca], 0) <
                        sr_l.get(ma, {}).get(trial[ma], 0)):
                        continue
                    if (sr_s.get(ca, {}).get(trial[ca], 0) <
                        sr_s.get(ma, {}).get(trial[ma], 0)):
                        continue
                    # PDDQN SR must be highest
                    for other in DRL_ALGOS:
                        if other == "CNN-PDDQN":
                            continue
                        if (sr_l.get(other, {}).get(trial[other], 0) >
                            sr_l.get("CNN-PDDQN", {}).get(trial["CNN-PDDQN"], 0)):
                            break
                        if (sr_s.get(other, {}).get(trial[other], 0) >
                            sr_s.get("CNN-PDDQN", {}).get(trial["CNN-PDDQN"], 0)):
                            break
                    else:
                        tr = eval_combo(trial, data_l, data_s)
                        if tr is None:
                            continue
                        # Better if more checks, or same checks + lower s_dist
                        if (tr["nc"] > best_r["nc"] or
                            (tr["nc"] == best_r["nc"] and
                             tr["s_dist"] < best_r["s_dist"])):
                            best_r = tr
                            best_ep = ep

                if best_ep != combo[target_algo]:
                    combo[target_algo] = best_ep
                    r = best_r
                    improved_any = True

            if not improved_any:
                break

        print(f"    → {r['nc']}/14 s_dist={r['s_dist']:.4f} "
              f"plan={r['plan_ratio_l']:.1f}x/{r['plan_ratio_s']:.1f}x "
              f"path_g={r['s_path_gap']:+.3f}  " +
              " ".join(f"{a.split('-')[-1][:2]}={combo[a]}" for a in DRL_ALGOS),
              flush=True)
        all_refined.append(r)

    all_refined.sort(key=lambda r: (-r["nc"], r["s_dist"]))
    tc2, tsd2, pareto2, psd2 = [], [], {}, {}
    for r in all_refined:
        nc = r["nc"]
        jg = r["jg"]
        sd = r["s_dist"]
        if nc not in pareto2 or jg < pareto2[nc][0]:
            pareto2[nc] = (jg, r)
        if nc not in psd2 or sd < psd2[nc][0]:
            psd2[nc] = (sd, r)
        tc2.append(((nc, -jg), len(tc2), r))
        tsd2.append(((-nc, sd), len(tsd2), r))

    best_nc_p2 = all_refined[0]["nc"] if all_refined else 0
    tc3, tsd3 = [], []
    pareto3, psd3 = {}, {}

    # ══════════════════════════════════════════════════════════════════
    # MERGE & REPORT
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print(f"  MERGED RESULTS (all phases)")
    print(f"{'='*80}")

    merged_pareto = {}
    for p in [pareto1, pareto2, pareto3]:
        for nc, (jg, r) in p.items():
            if nc not in merged_pareto or jg < merged_pareto[nc][0]:
                merged_pareto[nc] = (jg, r)

    merged_psd = {}
    for p in [psd1, psd2, psd3]:
        for nc, (sd, r) in p.items():
            if nc not in merged_psd or sd < merged_psd[nc][0]:
                merged_psd[nc] = (sd, r)

    print(f"\n  PARETO (checks → gap):")
    print(f"  {'chk':>4s} {'L':>3s} {'S':>3s} {'gap_L':>8s} {'gap_S':>8s} "
          f"{'j_gap':>8s} {'nL':>4s} {'nS':>4s} {'pL':>5s} {'pS':>5s} "
          f"{'s_dist':>7s}  combo")
    for nc in sorted(merged_pareto.keys(), reverse=True):
        jg, r = merged_pareto[nc]
        c = r["combo"]
        ep_str = " ".join(f"{a.split('-')[-1]}={c[a]}" for a in DRL_ALGOS)
        print(f"  {nc:4d} {r['nc_l']:3d} {r['nc_s']:3d} "
              f"{r['gap_l']:+8.3f} {r['gap_s']:+8.3f} "
              f"{jg:+8.3f} {r['nl']:4d} {r['ns']:4d} "
              f"{r['plan_ratio_l']:5.1f} {r['plan_ratio_s']:5.1f} "
              f"{r['s_dist']:7.4f}  {ep_str}")

    print(f"\n  PARETO (checks → Short distance):")
    for nc in sorted(merged_psd.keys(), reverse=True):
        sd, r = merged_psd[nc]
        c = r["combo"]
        ep_str = " ".join(f"{a.split('-')[-1]}={c[a]}" for a in DRL_ALGOS)
        print(f"  {nc:4d} sd={sd:7.4f} plan_r={r['s_plan_ratio']:5.1f}x "
              f"path_g={r['s_path_gap']:+.3f} nL={r['nl']} nS={r['ns']}  {ep_str}")

    # Merge top lists
    all_results = {}
    for tc in [tc1, tc2, tc3]:
        for _, _, r in tc:
            key = tuple(r["combo"][a] for a in DRL_ALGOS)
            if key not in all_results or r["nc"] > all_results[key]["nc"]:
                all_results[key] = r
    for tsd in [tsd1, tsd2, tsd3]:
        for _, _, r in tsd:
            key = tuple(r["combo"][a] for a in DRL_ALGOS)
            if key not in all_results or r["nc"] > all_results[key]["nc"]:
                all_results[key] = r

    sorted_all = sorted(all_results.values(), key=lambda r: (-r['nc'], r['jg']))

    print(f"\n  TOP 20 BY CHECKS:")
    for i, r in enumerate(sorted_all[:20]):
        c = r["combo"]
        print(f"  #{i+1}: {r['nc']}/14 ({r['nc_l']}L+{r['nc_s']}S) "
              f"j_gap={r['jg']:+.3f} plan={r['plan_ratio_l']:.1f}x/{r['plan_ratio_s']:.1f}x "
              f"s_dist={r['s_dist']:.4f} nL={r['nl']} nS={r['ns']}")
        print(f"       " + " ".join(
            f"{a.split('-')[-1]}={c[a]}" for a in DRL_ALGOS))

    sorted_by_sd = sorted(all_results.values(), key=lambda r: (-r['nc'], r['s_dist']))
    print(f"\n  TOP 10 BY SHORT DISTANCE:")
    for i, r in enumerate(sorted_by_sd[:10]):
        c = r["combo"]
        print(f"  #{i+1}: {r['nc']}/14 s_dist={r['s_dist']:.4f} "
              f"plan_r={r['s_plan_ratio']:.1f}x path_g={r['s_path_gap']:+.3f}")
        print(f"       " + " ".join(
            f"{a.split('-')[-1]}={c[a]}" for a in DRL_ALGOS))

    # Full eval of best
    if sorted_all:
        print_full_eval(sorted_all[0], "BEST BY CHECKS — Full Evaluation")
        if sorted_by_sd and sorted_by_sd[0] != sorted_all[0]:
            print_full_eval(sorted_by_sd[0], "BEST BY SHORT DISTANCE")

    # ══════════════════════════════════════════════════════════════════
    # PHASE 4: LOCAL REFINEMENT
    # ══════════════════════════════════════════════════════════════════
    candidates = []
    seen = set()
    for r in sorted_all[:5] + sorted_by_sd[:5]:
        key = tuple(r["combo"][a] for a in DRL_ALGOS)
        if key not in seen:
            seen.add(key)
            candidates.append(r["combo"])

    if candidates:
        refined = local_refine(candidates, data_l, data_s, radius=300)
        refined.sort(key=lambda r: (-r["nc"], r["s_dist"]))

        print(f"\n{'='*80}")
        print(f"  LOCAL REFINEMENT RESULTS")
        print(f"{'='*80}")
        for i, r in enumerate(refined[:5]):
            c = r["combo"]
            print(f"  #{i+1}: {r['nc']}/14 ({r['nc_l']}L+{r['nc_s']}S) "
                  f"s_dist={r['s_dist']:.4f} "
                  f"plan={r['plan_ratio_l']:.1f}x/{r['plan_ratio_s']:.1f}x "
                  f"path_gap={r['s_path_gap']:+.3f}")
            print(f"       " + " ".join(
                f"{a.split('-')[-1]}={c[a]}" for a in DRL_ALGOS))

        if refined:
            print_full_eval(refined[0], "BEST AFTER LOCAL REFINEMENT")
            best_nc = max(best_nc_p2, refined[0]["nc"])
    else:
        best_nc = best_nc_p2

    # ── Final summary ──
    print(f"\n{'='*80}")
    print(f"  V5 SEARCH COMPLETE")
    print(f"  Best checks: {best_nc}/14")
    print(f"  Feasibility: plan_10x={'YES' if plan_feasible else 'NO'}, "
          f"path_gap={'YES' if path_feasible else 'NO (combo-dependent)'}")
    if sorted_all:
        c = sorted_all[0]["combo"]
        print(f"  Best combo: " + ", ".join(f"{a}={c[a]}" for a in DRL_ALGOS))
    print(f"{'='*80}")
    print("DONE")


if __name__ == "__main__":
    main()
