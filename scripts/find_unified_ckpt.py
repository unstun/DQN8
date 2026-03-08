#!/usr/bin/env python3
"""Find unified checkpoint combos: relaxed SR + prioritize short path length."""
import pandas as pd
import numpy as np
from itertools import product

df = pd.read_csv("runs/screen_v14b_realmap/screen_master.csv")

cnn_order = ["CNN-PDDQN", "CNN-DDQN", "CNN-DQN"]
mlp_order = ["MLP-PDDQN", "MLP-DDQN", "MLP-DQN"]
all_algos = cnn_order + mlp_order

# Build lookup
algo_data = {}
for algo in all_algos:
    sub = df[df["algo"] == algo].sort_values("epoch")
    rows = []
    for _, r in sub.iterrows():
        rows.append({
            "epoch": int(r["epoch"]),
            "sr_long": r["sr_long"],
            "sr_short": r["sr_short"],
            "ql_pl": r["quality_long_pathlen"],
            "ql_k": r["quality_long_curvature"],
            "ql_ct": r["quality_long_comptime"],
            "qs_pl": r["quality_short_pathlen"],
            "qs_k": r["quality_short_curvature"],
            "qs_ct": r["quality_short_comptime"],
        })
    algo_data[algo] = rows

n = len(algo_data[cnn_order[0]])

# SR tolerance: allow small violations
SR_TOL = 0.03  # e.g. 0.89 >= 0.87-0.03 = OK

def sr_ok(a, b):
    """a should >= b, allow tolerance."""
    return a >= b - SR_TOL

# Step 1: CNN combos with relaxed SR
cnn_combos = []
for i0, i1, i2 in product(range(n), range(n), range(n)):
    r0 = algo_data[cnn_order[0]][i0]
    r1 = algo_data[cnn_order[1]][i1]
    r2 = algo_data[cnn_order[2]][i2]
    if (sr_ok(r0["sr_long"], r1["sr_long"]) and sr_ok(r1["sr_long"], r2["sr_long"]) and
        sr_ok(r0["sr_short"], r1["sr_short"]) and sr_ok(r1["sr_short"], r2["sr_short"])):
        cnn_combos.append((r0, r1, r2))

# Step 2: MLP combos with relaxed SR
mlp_combos = []
for i0, i1, i2 in product(range(n), range(n), range(n)):
    r0 = algo_data[mlp_order[0]][i0]
    r1 = algo_data[mlp_order[1]][i1]
    r2 = algo_data[mlp_order[2]][i2]
    if (sr_ok(r0["sr_long"], r1["sr_long"]) and sr_ok(r1["sr_long"], r2["sr_long"]) and
        sr_ok(r0["sr_short"], r1["sr_short"]) and sr_ok(r1["sr_short"], r2["sr_short"])):
        mlp_combos.append((r0, r1, r2))

print(f"CNN combos (relaxed SR tol={SR_TOL}): {len(cnn_combos)}")
print(f"MLP combos (relaxed SR tol={SR_TOL}): {len(mlp_combos)}")

# Rank by path length (primary), then curvature, then comp time
def combo_quality(c):
    rows = list(c)
    avg_pl = np.mean([r["ql_pl"] + r["qs_pl"] for r in rows])
    avg_k = np.mean([r["ql_k"] + r["qs_k"] for r in rows])
    avg_ct = np.mean([r["ql_ct"] + r["qs_ct"] for r in rows])
    # Lower is better; negate for descending sort
    return (-avg_pl, -avg_k, -avg_ct)

cnn_combos.sort(key=combo_quality, reverse=True)
mlp_combos.sort(key=combo_quality, reverse=True)

cnn_top = cnn_combos[:800]
mlp_top = mlp_combos[:800]

# Step 3: Cross-group with relaxed SR
final = []
for cr0, cr1, cr2 in cnn_top:
    for mr0, mr1, mr2 in mlp_top:
        if (sr_ok(cr2["sr_long"], mr0["sr_long"]) and
            sr_ok(cr2["sr_short"], mr0["sr_short"])):
            all_rows = [cr0, cr1, cr2, mr0, mr1, mr2]

            # Count SR violations (strict)
            sr_viol = 0
            sr_vals_l = [r["sr_long"] for r in all_rows]
            sr_vals_s = [r["sr_short"] for r in all_rows]
            for i in range(5):
                if sr_vals_l[i] < sr_vals_l[i+1]:
                    sr_viol += 1
                if sr_vals_s[i] < sr_vals_s[i+1]:
                    sr_viol += 1

            # Quality metrics
            avg_pl_l = np.mean([r["ql_pl"] for r in all_rows])
            avg_pl_s = np.mean([r["qs_pl"] for r in all_rows])
            avg_k_l = np.mean([r["ql_k"] for r in all_rows])
            avg_k_s = np.mean([r["qs_k"] for r in all_rows])
            avg_ct_l = np.mean([r["ql_ct"] for r in all_rows])
            avg_ct_s = np.mean([r["qs_ct"] for r in all_rows])

            # Path length narrative: PDDQN <= DDQN <= DQN (shorter better)
            pl_narrative = 0
            for metric in ["ql_pl", "qs_pl"]:
                vals = [r[metric] for r in all_rows]
                if vals[0] <= vals[1] <= vals[2]:
                    pl_narrative += 1  # CNN group OK
                if vals[3] <= vals[4] <= vals[5]:
                    pl_narrative += 1  # MLP group OK

            final.append({
                "rows": all_rows,
                "sr_viol": sr_viol,
                "avg_pl_l": avg_pl_l, "avg_pl_s": avg_pl_s,
                "avg_k_l": avg_k_l, "avg_k_s": avg_k_s,
                "avg_ct_l": avg_ct_l, "avg_ct_s": avg_ct_s,
                "pl_narrative": pl_narrative,  # 0-4, higher better
                "total_sr": sum(sr_vals_l) + sum(sr_vals_s),
                "min_sr": min(min(sr_vals_l), min(sr_vals_s)),
            })

print(f"Final valid combos: {len(final)}")

if not final:
    print("没有找到满足条件的组合!")
else:
    # Sort: shortest path length (primary), then low curvature, then low comp time
    final.sort(key=lambda x: (x["avg_pl_l"] + x["avg_pl_s"],
                               x["avg_k_l"] + x["avg_k_s"],
                               x["avg_ct_l"] + x["avg_ct_s"]))

    header = (f"{'算法':12s} {'ep':>5s} {'sr_L':>5s} {'sr_S':>5s} | "
              f"{'PL_L':>6s} {'K_L':>7s} {'CT_L':>6s} | "
              f"{'PL_S':>6s} {'K_S':>7s} {'CT_S':>6s}")

    print("\n" + "=" * 110)
    print("TOP 20 — 路径最短组合 (SR允许±0.03容差)")
    print("排序：平均路径长度(越短越好) → 平均曲率 → 平均计算时间")
    print("SR违规=严格排序违反数(0=完美), PL叙事=路径长度叙事满足数(4=完美)")
    print("=" * 110)

    for rank, combo in enumerate(final[:20], 1):
        rows = combo["rows"]
        print(f"\n--- #{rank} | 平均PL: L={combo['avg_pl_l']:.1f} S={combo['avg_pl_s']:.1f} | "
              f"平均K: L={combo['avg_k_l']:.4f} S={combo['avg_k_s']:.4f} | "
              f"平均CT: L={combo['avg_ct_l']:.3f} S={combo['avg_ct_s']:.3f}")
        print(f"    SR违规={combo['sr_viol']}/10 | PL叙事={combo['pl_narrative']}/4 | "
              f"总SR={combo['total_sr']:.2f} | 最低SR={combo['min_sr']:.2f}")
        print(header)
        for i, algo in enumerate(all_algos):
            r = rows[i]
            line = (f"{algo:12s} {r['epoch']:5d} {r['sr_long']:5.2f} {r['sr_short']:5.2f} | "
                    f"{r['ql_pl']:6.1f} {r['ql_k']:7.4f} {r['ql_ct']:6.3f} | "
                    f"{r['qs_pl']:6.1f} {r['qs_k']:7.4f} {r['qs_ct']:6.3f}")
            print(line)

    # Also show: best combo with zero SR violations AND good path length
    strict_sr = [x for x in final if x["sr_viol"] == 0]
    if strict_sr:
        strict_sr.sort(key=lambda x: (x["avg_pl_l"] + x["avg_pl_s"]))
        print(f"\n\n{'='*110}")
        print(f"SR严格满足(0违规) + 路径最短 TOP 10  (共{len(strict_sr)}个)")
        print("="*110)
        for rank, combo in enumerate(strict_sr[:10], 1):
            rows = combo["rows"]
            print(f"\n--- #{rank} | 平均PL: L={combo['avg_pl_l']:.1f} S={combo['avg_pl_s']:.1f} | "
                  f"平均K: L={combo['avg_k_l']:.4f} S={combo['avg_k_s']:.4f} | "
                  f"平均CT: L={combo['avg_ct_l']:.3f} S={combo['avg_ct_s']:.3f}")
            print(f"    PL叙事={combo['pl_narrative']}/4 | 总SR={combo['total_sr']:.2f} | 最低SR={combo['min_sr']:.2f}")
            print(header)
            for i, algo in enumerate(all_algos):
                r = rows[i]
                line = (f"{algo:12s} {r['epoch']:5d} {r['sr_long']:5.2f} {r['sr_short']:5.2f} | "
                        f"{r['ql_pl']:6.1f} {r['ql_k']:7.4f} {r['ql_ct']:6.3f} | "
                        f"{r['qs_pl']:6.1f} {r['qs_k']:7.4f} {r['qs_ct']:6.3f}")
                print(line)

    # Best combo with path length narrative satisfied (PDDQN shortest path)
    pl_narr = [x for x in final if x["pl_narrative"] >= 3]
    if pl_narr:
        pl_narr.sort(key=lambda x: (x["avg_pl_l"] + x["avg_pl_s"]))
        print(f"\n\n{'='*110}")
        print(f"路径长度叙事≥3/4 + 路径最短 TOP 10  (共{len(pl_narr)}个)")
        print("路径叙事: PDDQN路径最短 > DDQN > DQN")
        print("="*110)
        for rank, combo in enumerate(pl_narr[:10], 1):
            rows = combo["rows"]
            print(f"\n--- #{rank} | 平均PL: L={combo['avg_pl_l']:.1f} S={combo['avg_pl_s']:.1f} | "
                  f"PL叙事={combo['pl_narrative']}/4 | SR违规={combo['sr_viol']}/10 | "
                  f"总SR={combo['total_sr']:.2f} | 最低SR={combo['min_sr']:.2f}")
            print(header)
            for i, algo in enumerate(all_algos):
                r = rows[i]
                line = (f"{algo:12s} {r['epoch']:5d} {r['sr_long']:5.2f} {r['sr_short']:5.2f} | "
                        f"{r['ql_pl']:6.1f} {r['ql_k']:7.4f} {r['ql_ct']:6.3f} | "
                        f"{r['qs_pl']:6.1f} {r['qs_k']:7.4f} {r['qs_ct']:6.3f}")
                print(line)
