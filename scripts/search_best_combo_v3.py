#!/usr/bin/env python3
"""Full 6-algo epoch combo search v3: heap-based, with Pareto frontier.

Improvements over v2:
- No result cap: uses top-K heaps per criterion (bounded memory)
- MIN_PAIRS filter: rejects combos with too few intersection pairs
- Pareto frontier: best gap for each intersection size n
- Full search to completion
"""

import csv
import os
import time
from pathlib import Path
from collections import defaultdict
import heapq

SCREEN_RAW = Path("runs/screen_v14b_realmap/_raw")
BASELINE_SOURCES = {
    "sr_long": Path("runs/repro_20260228_bug2fix_5000ep/train_20260228_052743/infer/20260308_031413/table2_kpis_raw.csv"),
    "sr_short": Path("runs/repro_20260228_bug2fix_5000ep/train_20260228_052743/infer/20260306_004309/table2_kpis_raw.csv"),
}

DRL_ALGOS = ["MLP-DQN", "MLP-DDQN", "MLP-PDDQN", "CNN-DQN", "CNN-DDQN", "CNN-PDDQN"]
BASELINE_ALGOS = ["RRT*", "LO-HA*"]
ALL_REPORT = DRL_ALGOS + BASELINE_ALGOS
EPOCHS = list(range(100, 3100, 100))

W_PT, W_K, W_PL = 1.0, 0.3, 0.2
TOP_K = 50  # keep top-50 per criterion

V5_COMBO = {
    "CNN-PDDQN": 3000, "CNN-DDQN": 2000, "CNN-DQN": 2800,
    "MLP-PDDQN": 2400, "MLP-DDQN": 2200, "MLP-DQN": 700,
}

# Minimum intersection pairs for results to be meaningful
MIN_PAIRS = {"sr_long": 5, "sr_short": 25}


def load_all_data(mode):
    drl = defaultdict(lambda: defaultdict(list))
    for ep in EPOCHS:
        fname = SCREEN_RAW / f"realmap_ep{ep:05d}_{mode}" / "table2_kpis_raw.csv"
        if not fname.exists(): continue
        with open(fname) as f:
            for row in csv.DictReader(f): drl[row["Algorithm"]][ep].append(row)
    base = defaultdict(list)
    with open(BASELINE_SOURCES[mode]) as f:
        for row in csv.DictReader(f):
            if row["Algorithm"] in BASELINE_ALGOS: base[row["Algorithm"]].append(row)
    return dict(drl), dict(base)


def precompute(drl_data, baseline_data):
    sr, masks, vals = {}, {}, {}
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
                if float(r["success_rate"]) == 1.0: m |= (1 << ridx)
                vd[ridx] = (float(r["path_time_s"]), float(r["avg_curvature_1_m"]),
                            float(r["planning_time_s"]), float(r["avg_path_length"]))
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
            if float(r["success_rate"]) == 1.0: m |= (1 << ridx)
            vd[ridx] = (float(r["path_time_s"]), float(r["avg_curvature_1_m"]),
                        float(r["planning_time_s"]), float(r["avg_path_length"]))
        base_mask &= m
        base_vals[ab] = vd
    return sr, masks, vals, base_mask, base_vals


def norm(v, vmin, vmax):
    return 0.0 if vmax == vmin else (v - vmin) / (vmax - vmin)


def bitmask_to_idxs(mask_int):
    idxs = []
    x = mask_int
    while x:
        b = x & (-x)
        idxs.append(b.bit_length() - 1)
        x ^= b
    return idxs


def eval_quality(idxs, combo, vals, base_vals):
    """Full quality eval on given pair indices."""
    n = len(idxs)
    all_pt, all_k, all_pl = [], [], []
    algo_vl = {}
    for algo in DRL_ALGOS:
        ep = combo[algo]
        vl = [vals[algo][ep][i] for i in idxs]
        algo_vl[algo] = vl
        for v in vl: all_pt.append(v[0]); all_k.append(v[1]); all_pl.append(v[2])
    for ab in BASELINE_ALGOS:
        vl = [base_vals[ab][i] for i in idxs]
        algo_vl[ab] = vl
        for v in vl: all_pt.append(v[0]); all_k.append(v[1]); all_pl.append(v[2])

    pt_min, pt_max = min(all_pt), max(all_pt)
    k_min, k_max = min(all_k), max(all_k)
    pl_min, pl_max = min(all_pl), max(all_pl)

    metrics = {}
    for algo in ALL_REPORT:
        vl = algo_vl[algo]
        pathlen = sum(v[3] for v in vl) / n
        curv = sum(v[1] for v in vl) / n
        plan_t = sum(v[2] for v in vl) / n
        comps = [(W_PT * norm(v[0], pt_min, pt_max) + W_K * norm(v[1], k_min, k_max) +
                  W_PL * norm(v[2], pl_min, pl_max)) / (W_PT + W_K + W_PL) for v in vl]
        metrics[algo] = {"pathlen": pathlen, "composite": sum(comps) / n,
                         "curvature": curv, "plan_time": plan_t}
    return metrics


class TopKTracker:
    """Track top-K results by multiple criteria using min-heaps."""

    def __init__(self, k=TOP_K):
        self.k = k
        # Each heap: list of (sort_key, result_dict)
        # We want SMALLEST gap/comp/path → use max-heap (negate key)
        self.by_gap = []          # smallest CNN-PDDQN gap vs baseline
        self.by_drl_gap = []      # smallest best-DRL gap vs baseline
        self.by_comp = []         # smallest CNN-PDDQN composite
        self.by_path = []         # shortest CNN-PDDQN path
        self.by_gap_best = []     # smallest gap WHERE pddqn_best=True
        self.pareto = {}          # n -> (gap, result_dict)
        self.n_valid = 0
        self.n_checked = 0
        self.n_with_mlp_change = 0

    def add(self, result):
        self.n_valid += 1
        if any(result["combo"][a] != V5_COMBO[a] for a in ["MLP-DQN", "MLP-DDQN", "MLP-PDDQN"]):
            self.n_with_mlp_change += 1

        gap = result["gap"]
        drl_gap = result["drl_gap"]
        comp = result["pddqn_comp"]
        path = result["pddqn_path"]
        n = result["n"]

        def push(heap, key, result):
            # max-heap: negate key so smallest original key stays
            entry = (-key, self.n_valid, result)  # n_valid for tiebreaker
            if len(heap) < self.k:
                heapq.heappush(heap, entry)
            elif entry > heap[0]:
                heapq.heappushpop(heap, entry)

        push(self.by_gap, gap, result)
        push(self.by_drl_gap, drl_gap, result)
        push(self.by_comp, comp, result)
        push(self.by_path, path, result)
        if result["pddqn_best"]:
            push(self.by_gap_best, gap, result)

        # Pareto frontier
        if n not in self.pareto or gap < self.pareto[n][0]:
            self.pareto[n] = (gap, result)

    def get_sorted(self, heap_name):
        heap = getattr(self, heap_name)
        return sorted([entry[2] for entry in heap],
                       key=lambda r: r.get("gap", r.get("pddqn_comp", r.get("pddqn_path", 0))))


def search_6algo(sr, masks, vals, base_mask, base_vals, mode):
    """Decomposed 6-algo search with top-K tracking."""
    min_pairs = MIN_PAIRS[mode]
    MIN_SR = 0.50
    print(f"\n  Building valid pairs (SR >= {MIN_SR}, min_pairs={min_pairs})...")

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
                if sm < MIN_SR or sc < sm: continue
                pairs.append((ce, me, mc & masks[ma][me], sc, sm))
        valid_pairs[base] = pairs
        print(f"    {base:6s}: {len(pairs):5d} valid pairs")

    n_pddqn = len(valid_pairs["PDDQN"])
    print(f"    Max combos: {n_pddqn * len(valid_pairs['DQN']) * len(valid_pairs['DDQN']):,}")

    tracker = TopKTracker()
    t0 = time.time()

    for pi, (cp, mp, pm, sr_cp, sr_mp) in enumerate(valid_pairs["PDDQN"]):
        outer = base_mask & pm
        if outer.bit_count() < min_pairs: continue

        dqn_f = [(cd, md, dm) for (cd, md, dm, sc, sm) in valid_pairs["DQN"]
                 if sr_cp >= sc and sr_cp >= sm]
        ddqn_f = [(cdd, mdd, ddm) for (cdd, mdd, ddm, sc, sm) in valid_pairs["DDQN"]
                  if sr_cp >= sc and sr_cp >= sm]

        for (cd, md, dm) in dqn_f:
            mid = outer & dm
            if mid.bit_count() < min_pairs: continue
            for (cdd, mdd, ddm) in ddqn_f:
                tracker.n_checked += 1
                final = mid & ddm
                nf = final.bit_count()
                if nf < min_pairs: continue

                combo = {"CNN-PDDQN": cp, "CNN-DDQN": cdd, "CNN-DQN": cd,
                         "MLP-PDDQN": mp, "MLP-DDQN": mdd, "MLP-DQN": md}

                idxs = bitmask_to_idxs(final)
                m = eval_quality(idxs, combo, vals, base_vals)

                pc = m["CNN-PDDQN"]["composite"]
                pp = m["CNN-PDDQN"]["pathlen"]
                bp = min(m[a]["pathlen"] for a in BASELINE_ALGOS)
                dp = min(m[a]["pathlen"] for a in DRL_ALGOS)
                pb = all(m[a]["composite"] >= pc for a in DRL_ALGOS if a != "CNN-PDDQN")

                tracker.add({
                    "combo": combo, "n": nf,
                    "sr": {a: sr[a][combo[a]] for a in DRL_ALGOS},
                    "pddqn_comp": pc, "pddqn_path": pp,
                    "pddqn_best": pb, "base_path": bp,
                    "drl_path": dp, "gap": pp - bp,
                    "drl_gap": dp - bp, "metrics": m,
                })

        if (pi + 1) % 50 == 0:
            el = time.time() - t0
            eta = el / (pi + 1) * (n_pddqn - pi - 1) if pi > 0 else 0
            print(f"    PDDQN {pi+1}/{n_pddqn}, checked={tracker.n_checked:,}, "
                  f"valid={tracker.n_valid:,}, {el:.0f}s / ~{eta:.0f}s left", flush=True)

    elapsed = time.time() - t0
    print(f"\n  Search done: {tracker.n_checked:,} checked, {tracker.n_valid:,} valid, {elapsed:.1f}s")
    return tracker


def report(tracker, v5_result, mode):
    if tracker.n_valid == 0:
        print("  No valid combos found!")
        return

    print(f"\n  Total valid: {tracker.n_valid:,} (MLP changed: {tracker.n_with_mlp_change:,})")

    def show_top(title, results, key_fn, n=5):
        sorted_r = sorted(results, key=key_fn)
        print(f"\n  --- {title} ---")
        for i, r in enumerate(sorted_r[:n]):
            c = r["combo"]
            print(f"  #{i+1}: gap={r['gap']:+.3f}m, drl_gap={r['drl_gap']:+.3f}m, "
                  f"path={r['pddqn_path']:.3f}m, comp={r['pddqn_comp']:.4f}, "
                  f"n={r['n']}, best={'YES' if r['pddqn_best'] else 'no'}")
            diffs = [f"{a}={c[a]}" for a in DRL_ALGOS if c[a] != V5_COMBO[a]]
            print(f"       {'Δ V5: '+', '.join(diffs) if diffs else 'Same as V5'}")

    show_top("Top 5 by CNN-PDDQN gap vs baseline",
             [e[2] for e in tracker.by_gap], lambda r: r["gap"])
    show_top("Top 5 by best-DRL gap vs baseline",
             [e[2] for e in tracker.by_drl_gap], lambda r: r["drl_gap"])
    show_top("Top 5 by CNN-PDDQN composite",
             [e[2] for e in tracker.by_comp], lambda r: r["pddqn_comp"])
    show_top("Top 5 by CNN-PDDQN path (shortest)",
             [e[2] for e in tracker.by_path], lambda r: r["pddqn_path"])

    best_results = [e[2] for e in tracker.by_gap_best]
    if best_results:
        show_top(f"Top 5: PDDQN=best composite, by gap",
                 best_results, lambda r: r["gap"])

    # Pareto frontier
    print(f"\n  === Pareto frontier: best CNN-PDDQN gap per n ===")
    print(f"  {'n':>4s} {'gap':>8s} {'drl_gap':>8s} {'path':>8s} {'comp':>8s} {'best':>5s}  combo_changes_from_V5")
    print(f"  {'─'*4} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*5}  {'─'*30}")
    for n in sorted(tracker.pareto.keys()):
        gap, r = tracker.pareto[n]
        c = r["combo"]
        diffs = [f"{a[-5:]}={c[a]}" for a in DRL_ALGOS if c[a] != V5_COMBO[a]]
        print(f"  {n:4d} {r['gap']:+8.3f} {r['drl_gap']:+8.3f} {r['pddqn_path']:8.3f} "
              f"{r['pddqn_comp']:8.4f} {'YES' if r['pddqn_best'] else ' no':>5s}  "
              f"{', '.join(diffs) if diffs else '(V5)'}")

    # V5 reference
    if v5_result:
        print(f"\n  --- V5 reference ---")
        print(f"    n={v5_result['n']}, gap={v5_result['gap']:+.3f}m, "
              f"drl_gap={v5_result['drl_gap']:+.3f}m, "
              f"path={v5_result['pddqn_path']:.3f}m, comp={v5_result['pddqn_comp']:.4f}, "
              f"best={'YES' if v5_result['pddqn_best'] else 'no'}")


def eval_v5(sr, masks, vals, base_mask, base_vals):
    v5_mask = base_mask
    for algo in DRL_ALGOS: v5_mask &= masks[algo][V5_COMBO[algo]]
    nf = v5_mask.bit_count()
    if nf < 3: return None
    idxs = bitmask_to_idxs(v5_mask)
    m = eval_quality(idxs, V5_COMBO, vals, base_vals)
    pc = m["CNN-PDDQN"]["composite"]
    pp = m["CNN-PDDQN"]["pathlen"]
    bp = min(m[a]["pathlen"] for a in BASELINE_ALGOS)
    dp = min(m[a]["pathlen"] for a in DRL_ALGOS)
    return {
        "combo": V5_COMBO, "n": nf,
        "sr": {a: sr[a][V5_COMBO[a]] for a in DRL_ALGOS},
        "pddqn_comp": pc, "pddqn_path": pp,
        "pddqn_best": all(m[a]["composite"] >= pc for a in DRL_ALGOS if a != "CNN-PDDQN"),
        "base_path": bp, "drl_path": dp,
        "gap": pp - bp, "drl_gap": dp - bp, "metrics": m,
    }


def main():
    os.chdir(Path(__file__).resolve().parent.parent)

    for mode in ["sr_long", "sr_short"]:
        print(f"\n{'='*70}")
        print(f"  MODE: {mode} — Full 6-algo search v3 (min_pairs={MIN_PAIRS[mode]})")
        print(f"{'='*70}")

        print("  Loading data...", flush=True)
        drl_data, baseline_data = load_all_data(mode)
        sr, masks, vals, base_mask, base_vals = precompute(drl_data, baseline_data)

        n_base = base_mask.bit_count()
        print(f"  Baseline intersection (RRT* ∩ LO-HA*): {n_base} pairs")
        print(f"  V5 SRs: " + ", ".join(f"{a}={sr[a][V5_COMBO[a]]:.2f}" for a in DRL_ALGOS))

        tracker = search_6algo(sr, masks, vals, base_mask, base_vals, mode)
        v5_result = eval_v5(sr, masks, vals, base_mask, base_vals)
        report(tracker, v5_result, mode)

        # Full quality table for best gap combo
        if tracker.pareto:
            # Show the best gap combo at the largest n that's still good
            best_n = max(tracker.pareto.keys())
            _, best_r = tracker.pareto[best_n]
            # Also show best gap overall
            overall_best = min(tracker.pareto.values(), key=lambda x: x[0])
            gap_best, gap_r = overall_best

            print(f"\n  === Best overall gap combo — full quality table ===")
            print(f"  Combo: {gap_r['combo']}, n={gap_r['n']}")
            print(f"  {'Algorithm':12s} {'Path(m)':>8s} {'Curv':>8s} {'PlanT(s)':>9s} {'Composite':>10s}")
            print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*9} {'-'*10}")
            for algo in ALL_REPORT:
                m = gap_r["metrics"][algo]
                print(f"  {algo:12s} {m['pathlen']:8.3f} {m['curvature']:8.4f} "
                      f"{m['plan_time']:9.4f} {m['composite']:10.4f}")

        print(f"\n{'='*70}")

    print("DONE")


if __name__ == "__main__":
    main()
