#!/usr/bin/env python3
"""V4 Joint Long+Short 6-algo combo search — full 10k screen data.

Two-phase coarse→fine search over 6 DRL algos × 100 epochs each.

Phase 1: Coarse grid (every 500ep = 20 options/algo) — seconds
Phase 2: Fine grid (±400ep around top combos, every 100ep) — minutes

Usage:
    cd /home/sun/phdproject/dqn/DQN8
    conda run -n ros2py310 python scripts/search_v4_10k.py
"""

import csv, os, sys, time
from pathlib import Path
from collections import defaultdict
from itertools import product
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
MIN_SR = 0.50
FAST_GAP_THRESH = 0.8

# ── Reference combos ──
REF_COMBOS = {
    "V3_best": {
        "CNN-PDDQN": 2900, "CNN-DDQN": 1800, "CNN-DQN": 2200,
        "MLP-PDDQN": 700,  "MLP-DDQN": 500,  "MLP-DQN": 1400,
    },
    "V2.1_best": {
        "CNN-PDDQN": 2900, "CNN-DDQN": 1800, "CNN-DQN": 1100,
        "MLP-PDDQN": 2600, "MLP-DDQN": 1800, "MLP-DQN": 1400,
    },
}


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
    """Build SR, bitmasks, and per-run value dicts for fast evaluation."""
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


def eval_combo(combo, data_l, data_s):
    """Full evaluation of a 6-algo epoch combo. Returns result dict or None."""
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

    gap_l = ml["CNN-PDDQN"]["pathlen"] - min(
        ml[a]["pathlen"] for a in BASELINE_ALGOS)
    gap_s = ms["CNN-PDDQN"]["pathlen"] - min(
        ms[a]["pathlen"] for a in BASELINE_ALGOS)
    jg = gap_l + gap_s

    plan_ratio_l = (min(ml[a]["plan_time"] for a in BASELINE_ALGOS) /
                    max(ml[a]["plan_time"] for a in DRL_ALGOS))
    plan_ratio_s = (min(ms[a]["plan_time"] for a in BASELINE_ALGOS) /
                    max(ms[a]["plan_time"] for a in DRL_ALGOS))

    return {
        "combo": combo, "nl": nl, "ns": ns,
        "nc_l": nc_l, "nc_s": nc_s, "nc": nc,
        "gap_l": gap_l, "gap_s": gap_s, "jg": jg,
        "plan_ratio_l": plan_ratio_l, "plan_ratio_s": plan_ratio_s,
        "ml": ml, "ms": ms,
        "srd_l": srd_l, "srd_s": srd_s,
        "det_l": det_l, "det_s": det_s,
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
#  EPOCH-BASED SEARCH ENGINE
# ══════════════════════════════════════════════════════════════════════
def search_epochs(data_l, data_s, epoch_grid, label="",
                  min_sr=MIN_SR, min_pairs_s=MIN_PAIRS_S,
                  fast_gap_thresh=FAST_GAP_THRESH):
    """
    Iterate 6-algo epoch combos on a given grid.
    Outer loop: CNN-PDDQN → filter valid epochs for other 5 algos → iterate.
    """
    sr_l, masks_l, vals_l, bmask_l, bvals_l = data_l
    sr_s, masks_s, vals_s, bmask_s, bvals_s = data_s

    # ── Valid epochs per algo ──
    valid = {}
    for algo in DRL_ALGOS:
        valid[algo] = [ep for ep in epoch_grid
                       if sr_l[algo].get(ep, 0) >= min_sr
                       and sr_s[algo].get(ep, 0) >= min_sr]

    print(f"\n  {label} — valid epochs per algo:")
    for algo in DRL_ALGOS:
        print(f"    {algo:14s}: {len(valid[algo]):3d}")

    # Pre-sort CNN-PDDQN by descending SR (high SR → more pairs, better narrative)
    cp_epochs = sorted(valid["CNN-PDDQN"],
                        key=lambda e: -(sr_l["CNN-PDDQN"][e] + sr_s["CNN-PDDQN"][e]))

    top_checks = []
    top_gap = []
    pareto = {}
    n_total = 0
    n_sr_pass = 0
    n_mask_pass = 0
    n_gap_pass = 0
    n_full = 0
    t0 = time.time()

    for ci, cp in enumerate(cp_epochs):
        sr_cp_l = sr_l["CNN-PDDQN"][cp]
        sr_cp_s = sr_s["CNN-PDDQN"][cp]
        mk_cp_l = masks_l["CNN-PDDQN"][cp]
        mk_cp_s = masks_s["CNN-PDDQN"][cp]

        # Filter other algos: CNN >= MLP (same variant), PDDQN SR >= all
        def filt(algo, ep):
            return (sr_l[algo].get(ep, 0) <= sr_cp_l
                    and sr_s[algo].get(ep, 0) <= sr_cp_s)

        # MLP-PDDQN: CNN-PDDQN SR >= MLP-PDDQN SR
        mp_list = [ep for ep in valid["MLP-PDDQN"] if filt("MLP-PDDQN", ep)]

        # CNN-DQN: PDDQN SR >= CNN-DQN SR
        cd_list = [ep for ep in valid["CNN-DQN"] if filt("CNN-DQN", ep)]

        # CNN-DDQN: PDDQN SR >= CNN-DDQN SR
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

                # MLP-DQN: CNN-DQN SR >= MLP-DQN SR, PDDQN SR >= MLP-DQN SR
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

                    # MLP-DDQN: CNN-DDQN SR >= MLP-DDQN SR, PDDQN SR >= MLP-DDQN SR
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

                            # Fast gap check
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

                            # Full eval
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

                            result = {
                                "combo": combo, "nl": nl, "ns": ns,
                                "nc_l": nc_l, "nc_s": nc_s, "nc": nc,
                                "gap_l": gap_l, "gap_s": gap_s, "jg": jg,
                                "plan_ratio_l": plan_ratio_l,
                                "plan_ratio_s": plan_ratio_s,
                                "ml": ml, "ms": ms,
                                "srd_l": srd_l, "srd_s": srd_s,
                                "det_l": det_l, "det_s": det_s,
                            }

                            entry_c = ((nc, -jg), n_full, result)
                            if len(top_checks) < TOP_K:
                                heapq.heappush(top_checks, entry_c)
                            elif entry_c > top_checks[0]:
                                heapq.heappushpop(top_checks, entry_c)

                            entry_g = (-jg, n_full, result)
                            if len(top_gap) < TOP_K:
                                heapq.heappush(top_gap, entry_g)
                            elif entry_g > top_gap[0]:
                                heapq.heappushpop(top_gap, entry_g)

                            if nc not in pareto or jg < pareto[nc][0]:
                                pareto[nc] = (jg, result)

        # Progress
        if (ci + 1) % max(1, len(cp_epochs) // 10) == 0 or ci == len(cp_epochs) - 1:
            el = time.time() - t0
            eta = el / (ci + 1) * (len(cp_epochs) - ci - 1) if ci > 0 else 0
            best_nc = max((pareto[k][1]["nc"] for k in pareto), default=0)
            print(f"    CP {ci+1}/{len(cp_epochs)} | total={n_total:,} "
                  f"mask={n_mask_pass:,} gap={n_gap_pass:,} "
                  f"full={n_full:,} | best={best_nc}/14 | "
                  f"{el:.0f}s / ETA {eta:.0f}s",
                  flush=True)

    elapsed = time.time() - t0
    print(f"\n  {label} done: {n_total:,} combos, {n_mask_pass:,} mask, "
          f"{n_gap_pass:,} gap, {n_full:,} full eval, {elapsed:.1f}s")
    return top_checks, top_gap, pareto


# ══════════════════════════════════════════════════════════════════════
#  LANDSCAPE ANALYSIS
# ══════════════════════════════════════════════════════════════════════
def analyze_landscape(data_l, data_s):
    sr_l, masks_l, vals_l, bmask_l, bvals_l = data_l
    sr_s, masks_s, vals_s, bmask_s, bvals_s = data_s

    print(f"\n{'='*80}")
    print(f"  EPOCH LANDSCAPE — SR overview")
    print(f"{'='*80}")
    print(f"\n  {'Epoch':>6s}", end="")
    for algo in DRL_ALGOS:
        short = algo.replace("MLP-", "M").replace("CNN-", "C")
        print(f"  {short:>5s}L {short:>5s}S", end="")
    print()

    for ep in [100, 500, 1000, 1500, 2000, 2500, 3000, 4000, 5000,
               6000, 7000, 8000, 9000, 10000]:
        if ep not in EPOCHS_ALL:
            continue
        print(f"  {ep:6d}", end="")
        for algo in DRL_ALGOS:
            sl = sr_l[algo].get(ep, 0)
            ss = sr_s[algo].get(ep, 0)
            print(f"  {sl:5.0%}  {ss:5.0%}", end="")
        print()

    bl_count = bmask_l.bit_count()
    bs_count = bmask_s.bit_count()
    print(f"\n  Baseline all-succeed pairs: Long={bl_count}, Short={bs_count}")

    # CNN-PDDQN path gap trend
    print(f"\n  CNN-PDDQN path gap (on overlap with baselines):")
    print(f"  {'Epoch':>6s} {'Gap_L':>8s} {'Gap_S':>8s} {'SR_L':>6s} {'SR_S':>6s}")
    for ep in EPOCHS_ALL:
        ml = masks_l["CNN-PDDQN"].get(ep, 0)
        ms_ = masks_s["CNN-PDDQN"].get(ep, 0)
        fl = bmask_l & ml
        fs = bmask_s & ms_
        nl = fl.bit_count()
        ns = fs.bit_count()
        if nl < 3 or ns < 10:
            continue
        il = bitmask_to_idxs(fl)
        is_ = bitmask_to_idxs(fs)
        vl = vals_l["CNN-PDDQN"][ep]
        vs = vals_s["CNN-PDDQN"][ep]
        pl = sum(vl[i][3] for i in il if i in vl) / nl
        ps = sum(vs[i][3] for i in is_ if i in vs) / ns
        bpl = min(sum(bvals_l[ab][i][3] for i in il) / nl for ab in BASELINE_ALGOS)
        bps = min(sum(bvals_s[ab][i][3] for i in is_) / ns for ab in BASELINE_ALGOS)
        print(f"  {ep:6d} {pl-bpl:+8.3f} {ps-bps:+8.3f} "
              f"{sr_l['CNN-PDDQN'].get(ep,0):6.0%} {sr_s['CNN-PDDQN'].get(ep,0):6.0%}")


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════
def main():
    os.chdir(Path(__file__).resolve().parent.parent)

    print("=" * 80)
    print("  V4 COMBO SEARCH — Full 10k, Two-Phase Coarse→Fine")
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

    # ── Landscape ──
    analyze_landscape(data_l, data_s)

    # ── Reference combos ──
    print(f"\n{'='*80}")
    print(f"  REFERENCE COMBOS")
    print(f"{'='*80}")
    for name, combo in REF_COMBOS.items():
        r = eval_combo(combo, data_l, data_s)
        if r:
            print_full_eval(r, f"Reference: {name}")
        else:
            print(f"\n  {name}: evaluation failed")

    # ══════════════════════════════════════════════════════════════════
    # PHASE 1: COARSE SEARCH (every 500ep)
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print(f"  PHASE 1: COARSE SEARCH (every 500ep, {len(EPOCHS_COARSE)} options)")
    print(f"{'='*80}")
    tc1, tg1, pareto1 = search_epochs(
        data_l, data_s, EPOCHS_COARSE, label="Phase1-Coarse",
        fast_gap_thresh=1.0)

    # Report Phase 1
    sorted_c1 = sorted([e[2] for e in tc1], key=lambda r: (-r['nc'], r['jg']))
    print(f"\n  Phase 1 Pareto frontier:")
    print(f"  {'chk':>4s} {'L':>3s} {'S':>3s} {'gap_L':>8s} {'gap_S':>8s} "
          f"{'j_gap':>8s} {'nL':>4s} {'nS':>4s}  combo")
    for nc in sorted(pareto1.keys(), reverse=True):
        jg, r = pareto1[nc]
        c = r["combo"]
        ep_str = " ".join(f"{a.split('-')[-1][:2]}={c[a]}" for a in DRL_ALGOS)
        print(f"  {nc:4d} {r['nc_l']:3d} {r['nc_s']:3d} "
              f"{r['gap_l']:+8.3f} {r['gap_s']:+8.3f} "
              f"{jg:+8.3f} {r['nl']:4d} {r['ns']:4d}  {ep_str}")

    print(f"\n  Phase 1 Top 10 by checks:")
    for i, r in enumerate(sorted_c1[:10]):
        c = r["combo"]
        print(f"    #{i+1}: {r['nc']}/14 ({r['nc_l']}L+{r['nc_s']}S) "
              f"j_gap={r['jg']:+.3f} nL={r['nl']} nS={r['ns']}  "
              + " ".join(f"{a.split('-')[-1][:2]}={c[a]}" for a in DRL_ALGOS))

    # ══════════════════════════════════════════════════════════════════
    # PHASE 2: FINE SEARCH (±400ep around top combos)
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print(f"  PHASE 2: FINE SEARCH (±400ep around top combos)")
    print(f"{'='*80}")

    # Collect unique epoch neighborhoods from top combos
    # Use top 30 by checks + top 20 by gap for seeds
    seeds = set()
    for r in sorted_c1[:30]:
        seeds.add(tuple(r["combo"][a] for a in DRL_ALGOS))
    sorted_g1 = sorted([e[2] for e in tg1], key=lambda r: r['jg'])
    for r in sorted_g1[:20]:
        seeds.add(tuple(r["combo"][a] for a in DRL_ALGOS))

    # Build union of all fine-grid epochs for each algo
    fine_epochs_per_algo = {a: set() for a in DRL_ALGOS}
    RADIUS = 400
    for seed in seeds:
        for i, algo in enumerate(DRL_ALGOS):
            ep_center = seed[i]
            for ep in range(max(100, ep_center - RADIUS),
                            min(10000, ep_center + RADIUS) + 1, 100):
                fine_epochs_per_algo[algo].add(ep)

    fine_epochs = set()
    for algo, eps in fine_epochs_per_algo.items():
        fine_epochs |= eps
        print(f"  {algo:14s}: {len(eps):3d} fine epochs")
    fine_grid = sorted(fine_epochs)
    print(f"  Union fine grid: {len(fine_grid)} unique epochs")

    tc2, tg2, pareto2 = search_epochs(
        data_l, data_s, fine_grid, label="Phase2-Fine",
        fast_gap_thresh=FAST_GAP_THRESH)

    # ══════════════════════════════════════════════════════════════════
    # PHASE 3: FULL SEARCH (all 100 epochs) — if best < 14
    # ══════════════════════════════════════════════════════════════════
    best_nc_p2 = max((pareto2[k][1]["nc"] for k in pareto2), default=0) if pareto2 else 0
    pareto3 = {}
    tc3, tg3 = [], []

    if best_nc_p2 < 14:
        print(f"\n{'='*80}")
        print(f"  PHASE 3: FULL GRID SEARCH (all {len(EPOCHS_ALL)} epochs)")
        print(f"  Best so far: {best_nc_p2}/14, trying full grid...")
        print(f"{'='*80}")
        tc3, tg3, pareto3 = search_epochs(
            data_l, data_s, EPOCHS_ALL, label="Phase3-Full",
            min_sr=0.45, min_pairs_s=MIN_PAIRS_S,
            fast_gap_thresh=1.2)

    # ══════════════════════════════════════════════════════════════════
    # MERGE & REPORT
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print(f"  FINAL RESULTS (merged all phases)")
    print(f"{'='*80}")

    # Merge pareto frontiers
    merged_pareto = {}
    for p in [pareto1, pareto2, pareto3]:
        for nc, (jg, r) in p.items():
            if nc not in merged_pareto or jg < merged_pareto[nc][0]:
                merged_pareto[nc] = (jg, r)

    print(f"\n  PARETO FRONTIER:")
    print(f"  {'chk':>4s} {'L':>3s} {'S':>3s} {'gap_L':>8s} {'gap_S':>8s} "
          f"{'j_gap':>8s} {'nL':>4s} {'nS':>4s} {'pL':>5s} {'pS':>5s}  combo")
    for nc in sorted(merged_pareto.keys(), reverse=True):
        jg, r = merged_pareto[nc]
        c = r["combo"]
        ep_str = " ".join(f"{a.split('-')[-1]}={c[a]}" for a in DRL_ALGOS)
        print(f"  {nc:4d} {r['nc_l']:3d} {r['nc_s']:3d} "
              f"{r['gap_l']:+8.3f} {r['gap_s']:+8.3f} "
              f"{jg:+8.3f} {r['nl']:4d} {r['ns']:4d} "
              f"{r['plan_ratio_l']:5.1f} {r['plan_ratio_s']:5.1f}  {ep_str}")

    # Merge top lists
    all_results = {}
    for tc in [tc1, tc2, tc3]:
        for _, _, r in tc:
            key = tuple(r["combo"][a] for a in DRL_ALGOS)
            if key not in all_results or r["nc"] > all_results[key]["nc"]:
                all_results[key] = r

    sorted_all = sorted(all_results.values(), key=lambda r: (-r['nc'], r['jg']))

    print(f"\n  TOP 20 BY CHECKS (all phases):")
    for i, r in enumerate(sorted_all[:20]):
        c = r["combo"]
        print(f"  #{i+1}: {r['nc']}/14 ({r['nc_l']}L+{r['nc_s']}S) "
              f"j_gap={r['jg']:+.3f} nL={r['nl']} nS={r['ns']} "
              f"plan={r['plan_ratio_l']:.1f}x/{r['plan_ratio_s']:.1f}x")
        print(f"       " + " ".join(
            f"{a.split('-')[-1]}={c[a]}" for a in DRL_ALGOS))

    # Full eval of best
    if sorted_all:
        print_full_eval(sorted_all[0], "OVERALL BEST — Full Evaluation")
        # Also print #2 and #3 if they have same check count
        for i in range(1, min(3, len(sorted_all))):
            if sorted_all[i]["nc"] >= sorted_all[0]["nc"] - 1:
                print_full_eval(sorted_all[i],
                                f"RUNNER-UP #{i+1} ({sorted_all[i]['nc']}/14)")

    best_nc = sorted_all[0]["nc"] if sorted_all else 0
    print(f"\n{'='*80}")
    print(f"  OVERALL BEST: {best_nc}/14")
    if sorted_all:
        c = sorted_all[0]["combo"]
        print(f"  Combo: " + ", ".join(f"{a}={c[a]}" for a in DRL_ALGOS))
    print(f"{'='*80}")
    print("DONE")


if __name__ == "__main__":
    main()
