#!/usr/bin/env python3
"""V3 Joint Long+Short 6-algo combo search.

Two-phase strategy:
  Phase 1: v14b-only search with CANONICAL weights (W_PT=1.0, W_K=0.6, W_PL=0.2).
           Same search space as search_joint_combo.py but correct weights.
  Phase 2: For top combos from Phase 1, sweep CNN-PDDQN across pddqn10k
           epochs (100-10000) to find extended-training improvements.

Usage:
    cd /home/sun/phdproject/dqn/DQN8
    conda run -n ros2py310 python scripts/search_v3_combo.py
"""

import csv, os, time
from pathlib import Path
from collections import defaultdict
import heapq

V14B_RAW = Path("runs/screen_v14b_realmap/_raw")
PDDQN10K_RAW = Path("runs/screen_pddqn10k_realmap/_raw")
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
EPOCHS = list(range(100, 3100, 100))  # 30 v14b epochs

# ── Canonical composite weights ──
W_PT, W_K, W_PL = 1.0, 0.6, 0.2
W_SUM = W_PT + W_K + W_PL

TOP_K = 30
MIN_PAIRS_L = 3
MIN_PAIRS_S = 15
MIN_SR = 0.50
FAST_GAP_THRESH = 0.5

# Reference combos
REF_COMBOS = {
    "V1_Long-best": {
        "CNN-PDDQN": 2900, "CNN-DDQN": 2600, "CNN-DQN": 2200,
        "MLP-PDDQN": 1900, "MLP-DDQN": 1800, "MLP-DQN": 1900,
    },
    "V2_Joint": {
        "CNN-PDDQN": 2900, "CNN-DDQN": 1700, "CNN-DQN": 1100,
        "MLP-PDDQN": 700, "MLP-DDQN": 500, "MLP-DQN": 1400,
    },
}


# ── Data loading ──
def load_mode(mode, raw_dir=V14B_RAW, algo_filter=None):
    drl = defaultdict(lambda: defaultdict(list))
    for ep in EPOCHS:
        fname = raw_dir / f"realmap_ep{ep:05d}_{mode}" / "table2_kpis_raw.csv"
        if not fname.exists():
            continue
        with open(fname) as f:
            for row in csv.DictReader(f):
                algo = row["Algorithm"]
                if algo_filter and algo not in algo_filter:
                    continue
                if algo in DRL_ALGOS:
                    drl[algo][ep].append(row)
    base = defaultdict(list)
    with open(BASELINE_SOURCES[mode]) as f:
        for row in csv.DictReader(f):
            if row["Algorithm"] in BASELINE_ALGOS:
                base[row["Algorithm"]].append(row)
    return dict(drl), dict(base)


def load_pddqn10k(mode):
    """Load CNN-PDDQN data from pddqn10k screen (100 epochs)."""
    data = {}
    for ep in range(100, 10100, 100):
        fname = PDDQN10K_RAW / f"realmap_ep{ep:05d}_{mode}" / "table2_kpis_raw.csv"
        if not fname.exists():
            continue
        rows = []
        with open(fname) as f:
            for row in csv.DictReader(f):
                if row["Algorithm"] == "CNN-PDDQN":
                    rows.append(row)
        if rows:
            data[ep] = rows
    return data


def precompute(drl_data, baseline_data, epoch_list=None):
    if epoch_list is None:
        epoch_list = EPOCHS
    sr, masks, vals = {}, {}, {}
    for algo in DRL_ALGOS:
        sr[algo], masks[algo], vals[algo] = {}, {}, {}
        for ep in epoch_list:
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
    ps = sr_dict["CNN-PDDQN"]
    mo = max(sr_dict[a] for a in DRL_ALGOS if a != "CNN-PDDQN")
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


def fast_gap(idxs, pddqn_ep, vals_pddqn, base_vals):
    n = len(idxs)
    pp = sum(vals_pddqn[pddqn_ep][i][3] for i in idxs) / n
    bp = min(sum(base_vals[ab][i][3] for i in idxs) / n for ab in BASELINE_ALGOS)
    return pp - bp


def eval_combo_full(combo, data_l, data_s):
    """Full evaluation of a combo. Returns result dict or None."""
    sr_l, masks_l, vals_l, bmask_l, bvals_l = data_l
    sr_s, masks_s, vals_s, bmask_s, bvals_s = data_s

    fl = bmask_l
    fs = bmask_s
    for algo in DRL_ALGOS:
        ep = combo[algo]
        fl &= masks_l[algo].get(ep, 0)
        fs &= masks_s[algo].get(ep, 0)
    nl, ns = fl.bit_count(), fs.bit_count()
    if nl < MIN_PAIRS_L or ns < MIN_PAIRS_S:
        return None

    il = bitmask_to_idxs(fl)
    is_ = bitmask_to_idxs(fs)
    ml = eval_quality(il, combo, vals_l, bvals_l)
    ms = eval_quality(is_, combo, vals_s, bvals_s)

    srd_l = {a: sr_l[a][combo[a]] for a in DRL_ALGOS}
    srd_s = {a: sr_s[a][combo[a]] for a in DRL_ALGOS}

    nc_l, _, det_l = count_checks(srd_l, ml)
    nc_s, _, det_s = count_checks(srd_s, ms)

    gap_l = ml["CNN-PDDQN"]["pathlen"] - min(ml[a]["pathlen"] for a in BASELINE_ALGOS)
    gap_s = ms["CNN-PDDQN"]["pathlen"] - min(ms[a]["pathlen"] for a in BASELINE_ALGOS)

    plan_ratio_l = (min(ml[a]["plan_time"] for a in BASELINE_ALGOS) /
                    max(ml[a]["plan_time"] for a in DRL_ALGOS))
    plan_ratio_s = (min(ms[a]["plan_time"] for a in BASELINE_ALGOS) /
                    max(ms[a]["plan_time"] for a in DRL_ALGOS))

    return {
        "combo": dict(combo), "nl": nl, "ns": ns,
        "nc_l": nc_l, "nc_s": nc_s, "nc": nc_l + nc_s,
        "gap_l": gap_l, "gap_s": gap_s, "jg": gap_l + gap_s,
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
        print(f"\n  ── {mode_label} ──")
        sr_d = r[sr_key]
        print(f"  SR: " + "  ".join(f"{a}={sr_d[a]:.0%}" for a in DRL_ALGOS))
        q = r[q_key]
        n = r[n_key]
        print(f"\n  Quality ({n} all-succeed pairs):")
        print(f"  {'Algorithm':14s} {'PathLen':>8s} {'Curv':>8s} "
              f"{'PlanT(s)':>9s} {'Composite':>10s}")
        print(f"  {'─'*14} {'─'*8} {'─'*8} {'─'*9} {'─'*10}")
        for algo in ALL_ALGOS:
            m = q[algo]
            marker = " ★" if algo == "CNN-PDDQN" else ""
            print(f"  {algo:14s} {m['pathlen']:8.3f} {m['curvature']:8.4f} "
                  f"{m['plan_time']:9.4f} {m['composite']:10.4f}{marker}")
        print(f"\n  Narrative checks:")
        for name, passed, detail in r[det_key]:
            icon = "PASS" if passed else "FAIL"
            print(f"    [{icon}] {name}: {detail}")


# ========================================================================
#  PHASE 1: v14b-only exhaustive search with canonical weights
# ========================================================================
def phase1_search(data_l, data_s):
    sr_l, masks_l, vals_l, bmask_l, bvals_l = data_l
    sr_s, masks_s, vals_s, bmask_s, bvals_s = data_s

    print(f"\nBuilding valid pairs (SR>={MIN_SR}, CNN>=MLP both modes)...")
    bases = {"DQN": ("CNN-DQN", "MLP-DQN"),
             "DDQN": ("CNN-DDQN", "MLP-DDQN"),
             "PDDQN": ("CNN-PDDQN", "MLP-PDDQN")}

    valid_pairs = {}
    for base, (ca, ma) in bases.items():
        pairs = []
        for ce in EPOCHS:
            scl, scs = sr_l[ca].get(ce, 0), sr_s[ca].get(ce, 0)
            if scl < MIN_SR or scs < MIN_SR:
                continue
            mcl, mcs = masks_l[ca].get(ce, 0), masks_s[ca].get(ce, 0)
            for me in EPOCHS:
                sml, sms = sr_l[ma].get(me, 0), sr_s[ma].get(me, 0)
                if sml < MIN_SR or sms < MIN_SR:
                    continue
                if scl < sml or scs < sms:
                    continue
                pairs.append((ce, me,
                              mcl & masks_l[ma].get(me, 0),
                              mcs & masks_s[ma].get(me, 0),
                              scl, sml, scs, sms))
        valid_pairs[base] = pairs
        print(f"  {base:6s}: {len(pairs):5d} pairs")

    n_pddqn = len(valid_pairs["PDDQN"])
    n_dqn = len(valid_pairs["DQN"])
    n_ddqn = len(valid_pairs["DDQN"])
    print(f"  Max combos: {n_pddqn} x {n_dqn} x {n_ddqn} = "
          f"{n_pddqn * n_dqn * n_ddqn:,}")

    vals_pddqn_l = vals_l["CNN-PDDQN"]
    vals_pddqn_s = vals_s["CNN-PDDQN"]

    print(f"\nSearching (fast_gap={FAST_GAP_THRESH}m)...", flush=True)

    top_checks = []
    top_gap = []
    pareto = {}
    n_checked = 0
    n_valid = 0
    n_fast_pass = 0
    n_full_eval = 0
    t0 = time.time()

    for pi, (cp, mp, pml, pms, sr_cp_l, sr_mp_l, sr_cp_s, sr_mp_s) \
            in enumerate(valid_pairs["PDDQN"]):
        ol = bmask_l & pml
        os_ = bmask_s & pms
        if ol.bit_count() < MIN_PAIRS_L or os_.bit_count() < MIN_PAIRS_S:
            continue

        dqn_f = [(cd, md, dl, ds) for (cd, md, dl, ds, scl, sml, scs, sms)
                 in valid_pairs["DQN"]
                 if sr_cp_l >= scl and sr_cp_l >= sml
                 and sr_cp_s >= scs and sr_cp_s >= sms]
        ddqn_f = [(cdd, mdd, ddl, dds)
                  for (cdd, mdd, ddl, dds, scl, sml, scs, sms)
                  in valid_pairs["DDQN"]
                  if sr_cp_l >= scl and sr_cp_l >= sml
                  and sr_cp_s >= scs and sr_cp_s >= sms]

        for (cd, md, dl, ds) in dqn_f:
            midl = ol & dl
            mids = os_ & ds
            if midl.bit_count() < MIN_PAIRS_L or mids.bit_count() < MIN_PAIRS_S:
                continue
            for (cdd, mdd, ddl, dds) in ddqn_f:
                n_checked += 1
                fl = midl & ddl
                fs = mids & dds
                nl, ns = fl.bit_count(), fs.bit_count()
                if nl < MIN_PAIRS_L or ns < MIN_PAIRS_S:
                    continue
                n_valid += 1

                # Fast gap
                il = bitmask_to_idxs(fl)
                is_ = bitmask_to_idxs(fs)
                fg_l = fast_gap(il, cp, vals_pddqn_l, bvals_l)
                fg_s = fast_gap(is_, cp, vals_pddqn_s, bvals_s)
                fg_j = fg_l + fg_s
                if fg_j > FAST_GAP_THRESH:
                    continue
                n_fast_pass += 1

                # Full evaluation
                n_full_eval += 1
                combo = {"CNN-PDDQN": cp, "CNN-DDQN": cdd, "CNN-DQN": cd,
                         "MLP-PDDQN": mp, "MLP-DDQN": mdd, "MLP-DQN": md}

                ml = eval_quality(il, combo, vals_l, bvals_l)
                ms = eval_quality(is_, combo, vals_s, bvals_s)

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
                    "plan_ratio_l": plan_ratio_l, "plan_ratio_s": plan_ratio_s,
                    "ml": ml, "ms": ms,
                    "srd_l": srd_l, "srd_s": srd_s,
                    "det_l": det_l, "det_s": det_s,
                }

                entry_c = ((nc, -jg), n_valid, result)
                if len(top_checks) < TOP_K:
                    heapq.heappush(top_checks, entry_c)
                elif entry_c > top_checks[0]:
                    heapq.heappushpop(top_checks, entry_c)

                entry_g = (-jg, n_valid, result)
                if len(top_gap) < TOP_K:
                    heapq.heappush(top_gap, entry_g)
                elif entry_g > top_gap[0]:
                    heapq.heappushpop(top_gap, entry_g)

                if nc not in pareto or jg < pareto[nc][0]:
                    pareto[nc] = (jg, result)

        if (pi + 1) % 20 == 0:
            el = time.time() - t0
            eta = el / (pi + 1) * (n_pddqn - pi - 1) if pi > 0 else 0
            print(f"  PDDQN {pi+1}/{n_pddqn}, chk={n_checked:,}, "
                  f"valid={n_valid:,}, fast={n_fast_pass:,}, "
                  f"full={n_full_eval:,}, {el:.0f}s/~{eta:.0f}s",
                  flush=True)

    elapsed = time.time() - t0
    print(f"\nPhase 1 done: {n_checked:,} checked, {n_valid:,} valid, "
          f"{n_fast_pass:,} fast, {n_full_eval:,} full, {elapsed:.1f}s")

    return top_checks, top_gap, pareto


# ========================================================================
#  PHASE 2: pddqn10k CNN-PDDQN sweep on top combos
# ========================================================================
def phase2_pddqn10k_sweep(top_combos, data_l, data_s, p10k_data_l, p10k_data_s):
    """For each top combo, swap CNN-PDDQN to each pddqn10k epoch."""
    sr_l, masks_l, vals_l, bmask_l, bvals_l = data_l
    sr_s, masks_s, vals_s, bmask_s, bvals_s = data_s

    # Pre-compute pddqn10k bitmasks and vals
    p10k_epochs = sorted(p10k_data_l.keys())
    p10k_masks_l, p10k_masks_s = {}, {}
    p10k_vals_l, p10k_vals_s = {}, {}
    p10k_sr_l, p10k_sr_s = {}, {}

    for ep in p10k_epochs:
        # Long
        rows = p10k_data_l.get(ep, [])
        m, vd = 0, {}
        for r in rows:
            ridx = int(r["run_idx"])
            if float(r["success_rate"]) == 1.0:
                m |= (1 << ridx)
            vd[ridx] = (float(r["path_time_s"]),
                        float(r["avg_curvature_1_m"]),
                        float(r["planning_time_s"]),
                        float(r["avg_path_length"]))
        p10k_masks_l[ep] = m
        p10k_vals_l[ep] = vd
        p10k_sr_l[ep] = m.bit_count() / len(rows) if rows else 0

        # Short
        rows = p10k_data_s.get(ep, [])
        m, vd = 0, {}
        for r in rows:
            ridx = int(r["run_idx"])
            if float(r["success_rate"]) == 1.0:
                m |= (1 << ridx)
            vd[ridx] = (float(r["path_time_s"]),
                        float(r["avg_curvature_1_m"]),
                        float(r["planning_time_s"]),
                        float(r["avg_path_length"]))
        p10k_masks_s[ep] = m
        p10k_vals_s[ep] = vd
        p10k_sr_s[ep] = m.bit_count() / len(rows) if rows else 0

    print(f"\n  pddqn10k: {len(p10k_epochs)} epochs loaded "
          f"(ep{p10k_epochs[0]}-{p10k_epochs[-1]})")

    # Print pddqn10k SR overview
    best_sr = sorted(p10k_epochs, key=lambda e: -(p10k_sr_l[e] + p10k_sr_s[e]))
    print(f"\n  Top 10 pddqn10k epochs by SR:")
    print(f"  {'Epoch':>6s} {'SR_L':>6s} {'SR_S':>6s}")
    for ep in best_sr[:10]:
        print(f"  {ep:6d} {p10k_sr_l[ep]:6.0%} {p10k_sr_s[ep]:6.0%}")

    # Sweep each top combo
    all_results = []

    for combo_idx, base_combo in enumerate(top_combos):
        orig_pddqn = base_combo["CNN-PDDQN"]

        for ep in p10k_epochs:
            # Check CNN >= MLP SR constraint
            mlp_pddqn_ep = base_combo["MLP-PDDQN"]
            mlp_sr_l = sr_l["MLP-PDDQN"].get(mlp_pddqn_ep, 0)
            mlp_sr_s = sr_s["MLP-PDDQN"].get(mlp_pddqn_ep, 0)
            if p10k_sr_l[ep] < mlp_sr_l or p10k_sr_s[ep] < mlp_sr_s:
                continue

            # Build modified combo
            new_combo = dict(base_combo)
            new_combo["CNN-PDDQN"] = ep  # temporary key for p10k

            # Compute bitmasks using p10k for CNN-PDDQN
            fl = bmask_l & p10k_masks_l.get(ep, 0)
            fs = bmask_s & p10k_masks_s.get(ep, 0)
            for algo in DRL_ALGOS:
                if algo == "CNN-PDDQN":
                    continue
                fl &= masks_l[algo].get(new_combo[algo], 0)
                fs &= masks_s[algo].get(new_combo[algo], 0)

            nl, ns = fl.bit_count(), fs.bit_count()
            if nl < MIN_PAIRS_L or ns < MIN_PAIRS_S:
                continue

            il = bitmask_to_idxs(fl)
            is_ = bitmask_to_idxs(fs)

            # Build vals with p10k CNN-PDDQN
            # Temporarily inject p10k data into vals
            orig_vals_l = vals_l["CNN-PDDQN"].get(ep)
            orig_vals_s = vals_s["CNN-PDDQN"].get(ep)
            vals_l["CNN-PDDQN"][ep] = p10k_vals_l[ep]
            vals_s["CNN-PDDQN"][ep] = p10k_vals_s[ep]

            try:
                ml = eval_quality(il, new_combo, vals_l, bvals_l)
                ms = eval_quality(is_, new_combo, vals_s, bvals_s)
            except (KeyError, IndexError):
                continue
            finally:
                # Restore
                if orig_vals_l is not None:
                    vals_l["CNN-PDDQN"][ep] = orig_vals_l
                else:
                    vals_l["CNN-PDDQN"].pop(ep, None)
                if orig_vals_s is not None:
                    vals_s["CNN-PDDQN"][ep] = orig_vals_s
                else:
                    vals_s["CNN-PDDQN"].pop(ep, None)

            srd_l = {a: sr_l[a][new_combo[a]] for a in DRL_ALGOS if a != "CNN-PDDQN"}
            srd_s = {a: sr_s[a][new_combo[a]] for a in DRL_ALGOS if a != "CNN-PDDQN"}
            srd_l["CNN-PDDQN"] = p10k_sr_l[ep]
            srd_s["CNN-PDDQN"] = p10k_sr_s[ep]

            nc_l, _, det_l = count_checks(srd_l, ml)
            nc_s, _, det_s = count_checks(srd_s, ms)

            gap_l = ml["CNN-PDDQN"]["pathlen"] - min(
                ml[a]["pathlen"] for a in BASELINE_ALGOS)
            gap_s = ms["CNN-PDDQN"]["pathlen"] - min(
                ms[a]["pathlen"] for a in BASELINE_ALGOS)

            plan_ratio_l = (min(ml[a]["plan_time"] for a in BASELINE_ALGOS) /
                            max(ml[a]["plan_time"] for a in DRL_ALGOS))
            plan_ratio_s = (min(ms[a]["plan_time"] for a in BASELINE_ALGOS) /
                            max(ms[a]["plan_time"] for a in DRL_ALGOS))

            result = {
                "combo": new_combo,
                "source": f"base#{combo_idx}_p10k@{ep}",
                "orig_pddqn": orig_pddqn,
                "nl": nl, "ns": ns,
                "nc_l": nc_l, "nc_s": nc_s, "nc": nc_l + nc_s,
                "gap_l": gap_l, "gap_s": gap_s, "jg": gap_l + gap_s,
                "plan_ratio_l": plan_ratio_l, "plan_ratio_s": plan_ratio_s,
                "ml": ml, "ms": ms,
                "srd_l": srd_l, "srd_s": srd_s,
                "det_l": det_l, "det_s": det_s,
            }
            all_results.append(result)

    return all_results


# ========================================================================
#  MAIN
# ========================================================================
def main():
    os.chdir(Path(__file__).resolve().parent.parent)

    print("=" * 80)
    print("  V3 COMBO SEARCH")
    print(f"  Canonical weights: W_PT={W_PT}, W_K={W_K}, W_PL={W_PL}")
    print("  Phase 1: v14b exhaustive | Phase 2: pddqn10k CNN-PDDQN sweep")
    print("=" * 80)

    # ── Load v14b data ──
    print("\nLoading v14b sr_long...", flush=True)
    drl_l, base_l = load_mode("sr_long")
    sr_l, masks_l, vals_l, bmask_l, bvals_l = precompute(drl_l, base_l)
    data_l = (sr_l, masks_l, vals_l, bmask_l, bvals_l)

    print("Loading v14b sr_short...", flush=True)
    drl_s, base_s = load_mode("sr_short")
    sr_s, masks_s, vals_s, bmask_s, bvals_s = precompute(drl_s, base_s)
    data_s = (sr_s, masks_s, vals_s, bmask_s, bvals_s)

    print(f"Baseline Long pairs: {bmask_l.bit_count()}")
    print(f"Baseline Short pairs: {bmask_s.bit_count()}")

    # ── Reference combos ──
    print(f"\n{'='*80}")
    print(f"  REFERENCE COMBOS (canonical {W_PT}/{W_K}/{W_PL})")
    print(f"{'='*80}")
    for name, combo in REF_COMBOS.items():
        r = eval_combo_full(combo, data_l, data_s)
        if r:
            print_full_eval(r, f"Reference: {name}")
        else:
            print(f"\n  {name}: insufficient pairs")

    # ── Phase 1 ──
    print(f"\n{'='*80}")
    print(f"  PHASE 1: v14b exhaustive search")
    print(f"{'='*80}")
    top_checks, top_gap, pareto = phase1_search(data_l, data_s)

    # Report Phase 1
    print(f"\n{'='*80}")
    print(f"  PHASE 1 PARETO FRONTIER")
    print(f"{'='*80}")
    print(f"  {'chk':>4s} {'L':>3s} {'S':>3s} {'gap_L':>8s} {'gap_S':>8s} "
          f"{'j_gap':>8s} {'nL':>4s} {'nS':>4s} {'pL':>5s} {'pS':>5s}  combo")
    for nc in sorted(pareto.keys(), reverse=True):
        jg, r = pareto[nc]
        c = r["combo"]
        ep_str = " ".join(f"{a.split('-')[-1]}={c[a]}" for a in DRL_ALGOS)
        print(f"  {nc:4d} {r['nc_l']:3d} {r['nc_s']:3d} "
              f"{r['gap_l']:+8.3f} {r['gap_s']:+8.3f} "
              f"{jg:+8.3f} {r['nl']:4d} {r['ns']:4d} "
              f"{r['plan_ratio_l']:5.1f} {r['plan_ratio_s']:5.1f}  {ep_str}")

    sorted_c = sorted([e[2] for e in top_checks],
                       key=lambda r: (-r['nc'], r['jg']))
    print(f"\n{'='*80}")
    print(f"  PHASE 1 TOP 10 BY CHECKS")
    print(f"{'='*80}")
    for i, r in enumerate(sorted_c[:10]):
        c = r["combo"]
        print(f"  #{i+1}: checks={r['nc']}/14 ({r['nc_l']}L+{r['nc_s']}S) "
              f"j_gap={r['jg']:+.3f} nL={r['nl']} nS={r['ns']} "
              f"plan={r['plan_ratio_l']:.1f}x/{r['plan_ratio_s']:.1f}x")
        print(f"       " + " ".join(
            f"{a.split('-')[-1]}={c[a]}" for a in DRL_ALGOS))

    # Full eval of #1
    if sorted_c:
        print_full_eval(sorted_c[0], "PHASE 1 BEST — Full Evaluation")

    # ── Phase 2: pddqn10k sweep ──
    print(f"\n{'='*80}")
    print(f"  PHASE 2: pddqn10k CNN-PDDQN sweep")
    print(f"{'='*80}")

    print("\nLoading pddqn10k data...", flush=True)
    p10k_data_l = load_pddqn10k("sr_long")
    p10k_data_s = load_pddqn10k("sr_short")
    print(f"  Loaded {len(p10k_data_l)} long epochs, {len(p10k_data_s)} short epochs")

    # Use top-10 combos from Phase 1 as base
    base_combos = [r["combo"] for r in sorted_c[:10]]

    p2_results = phase2_pddqn10k_sweep(
        base_combos, data_l, data_s, p10k_data_l, p10k_data_s)

    if p2_results:
        p2_results.sort(key=lambda r: (-r['nc'], r['jg']))
        print(f"\n  Phase 2: {len(p2_results)} valid combos evaluated")
        print(f"\n  TOP 10 PHASE 2 RESULTS:")
        for i, r in enumerate(p2_results[:10]):
            c = r["combo"]
            print(f"  #{i+1}: checks={r['nc']}/14 ({r['nc_l']}L+{r['nc_s']}S) "
                  f"j_gap={r['jg']:+.3f} nL={r['nl']} nS={r['ns']} "
                  f"src={r['source']} plan={r['plan_ratio_l']:.1f}x/{r['plan_ratio_s']:.1f}x")
            print(f"       " + " ".join(
                f"{a.split('-')[-1]}={c[a]}" for a in DRL_ALGOS))

        best_p2 = p2_results[0]
        if best_p2["nc"] > sorted_c[0]["nc"]:
            print(f"\n  ★ Phase 2 IMPROVED! {best_p2['nc']}/14 > Phase 1 {sorted_c[0]['nc']}/14")
            print_full_eval(best_p2, "PHASE 2 BEST — Full Evaluation")
        elif best_p2["nc"] == sorted_c[0]["nc"] and best_p2["jg"] < sorted_c[0]["jg"]:
            print(f"\n  ★ Phase 2 same checks but better gap: "
                  f"{best_p2['jg']:+.3f} vs {sorted_c[0]['jg']:+.3f}")
            print_full_eval(best_p2, "PHASE 2 BEST (same checks, better gap)")
        else:
            print(f"\n  Phase 2 did not improve over Phase 1.")
            print(f"  Phase 1 best: {sorted_c[0]['nc']}/14, j_gap={sorted_c[0]['jg']:+.3f}")
            print(f"  Phase 2 best: {best_p2['nc']}/14, j_gap={best_p2['jg']:+.3f}")
    else:
        print(f"\n  Phase 2: No valid combos found (CNN-PDDQN SR too low?)")

    # ── Overall best ──
    all_candidates = sorted_c + p2_results
    all_candidates.sort(key=lambda r: (-r['nc'], r['jg']))
    best = all_candidates[0]
    print(f"\n{'='*80}")
    print(f"  OVERALL BEST: {best['nc']}/14 "
          f"({best['nc_l']}L+{best['nc_s']}S) j_gap={best['jg']:+.3f}")
    src = best.get("source", "v14b_phase1")
    print(f"  Source: {src}")
    print(f"{'='*80}")
    print_full_eval(best, "OVERALL BEST — Full Evaluation")

    print(f"\n{'='*80}")
    print("DONE")


if __name__ == "__main__":
    main()
