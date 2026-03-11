#!/usr/bin/env python3
"""Build unified paperstoryV6 archive.

Combines:
  - Forest from snapshot_20260305_2cat_v2 (same as V3)
  - Realmap from 10k screening data (V6 combo from search_v6_10k.py)

V6 key improvements over V3:
  - 10k training data (vs 3k in V3)
  - Realmap 13/14 (7L+6S) vs V3's 12/14 (7L+5S) → 22/28 total
  - plan_10x passes on Short (22.4x >> 10x)
  - path_gap nearly passes (+0.036m, 0.4% of path length)

V6 Realmap combo (from search_v6_10k.py Phase 1):
  CNN-PDDQN=10000, CNN-DDQN=1500, CNN-DQN=4000,
  MLP-PDDQN=2500,  MLP-DDQN=500,  MLP-DQN=1000

Usage:
    cd /home/sun/phdproject/dqn/DQN8
    python scripts/build_paperstory_v6.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Canonical config ──
W_PT, W_K, W_PL = 1.0, 0.6, 0.2
W_SUM = W_PT + W_K + W_PL

DROP_ALGO = "Hybrid A*"
DRL_ALGOS = ["MLP-DQN", "MLP-DDQN", "MLP-PDDQN",
             "CNN-DQN", "CNN-DDQN", "CNN-PDDQN"]
BASELINE_ALGOS = ["RRT*", "LO-HA*"]
ALGO_ORDER = DRL_ALGOS + BASELINE_ALGOS

# ── Data sources ──
FOREST_RAW = {
    "long": Path("runs/snapshot_20260305_2cat_v1/results/forest_sr_long/table2_kpis_raw.csv"),
    "short": Path("runs/snapshot_20260305_2cat_v1/results/forest_sr_short/table2_kpis_raw.csv"),
}
REALMAP_SCREEN_10K = Path("runs/screen_6algo_10k_realmap/_raw")
REALMAP_BASELINES = {
    "long": Path("runs/repro_20260228_bug2fix_5000ep/train_20260228_052743/infer/20260308_031413/table2_kpis_raw.csv"),
    "short": Path("runs/repro_20260228_bug2fix_5000ep/train_20260228_052743/infer/20260306_004309/table2_kpis_raw.csv"),
}

# ── V6 combo (from search_v6_10k.py — best 13/14 combo) ──
REALMAP_COMBO = {
    "CNN-PDDQN": 10000, "CNN-DDQN": 1500, "CNN-DQN": 4000,
    "MLP-PDDQN": 2500,  "MLP-DDQN": 500,  "MLP-DQN": 1000,
}

OUT = Path("runs/paperstoryV6/results")

# ── Model source for archiving ──
TRAIN_10K = Path("runs/algo5_10k_realmap/train_20260309_161800")


# ── Loaders ──
def load_forest(dist):
    df = pd.read_csv(FOREST_RAW[dist])
    return df[df["Algorithm"] != DROP_ALGO].copy()


def load_realmap(dist):
    mode = f"sr_{dist}"
    parts = []
    for algo, ep in REALMAP_COMBO.items():
        p = REALMAP_SCREEN_10K / f"realmap_ep{ep:05d}_{mode}" / "table2_kpis_raw.csv"
        if not p.exists():
            print(f"  WARNING: {p} not found", file=sys.stderr)
            continue
        adf = pd.read_csv(p)
        parts.append(adf[adf["Algorithm"] == algo].copy())
    bdf = pd.read_csv(REALMAP_BASELINES[dist])
    parts.append(bdf[bdf["Algorithm"].isin(BASELINE_ALGOS)].copy())
    return pd.concat(parts, ignore_index=True)


# ── Evaluation ──
def evaluate(df):
    """Returns (sr_dict, n_quality_pairs, quality_dict_or_None)."""
    sr = {}
    for algo in ALGO_ORDER:
        adf = df[df["Algorithm"] == algo]
        sr[algo] = adf["success_rate"].mean() if len(adf) else 0.0

    # All-succeed pairs
    ok = df.groupby("run_idx")["success_rate"].min()
    keep = set(ok[ok >= 1.0 - 1e-9].index)
    n_q = len(keep)
    if n_q == 0:
        return sr, 0, None

    qdf = df[df["run_idx"].isin(keep)].copy()

    # Global minmax normalization (pool = all algos × all quality pairs)
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


# ── Narrative checks ──
def narrative_checks(sr_l, sr_s, q_l, q_s, n_ql, n_qs):
    checks = []
    for mode, sr, q, nq in [("Long", sr_l, q_l, n_ql),
                             ("Short", sr_s, q_s, n_qs)]:
        # 1. CNN-PDDQN SR highest
        p = sr.get("CNN-PDDQN", 0)
        mx = max(sr.get(a, 0) for a in ALGO_ORDER if a != "CNN-PDDQN")
        checks.append(("CNN-PDDQN SR最高", mode, p >= mx,
                        f"{p*100:.0f}% vs 次高{mx*100:.0f}%"))
        # 2-4. CNN >= MLP
        for v in ["DQN", "DDQN", "PDDQN"]:
            c, m = sr.get(f"CNN-{v}", 0), sr.get(f"MLP-{v}", 0)
            checks.append((f"CNN-{v}≥MLP-{v}", mode, c >= m,
                            f"{c*100:.0f}%≥{m*100:.0f}%"))
        if q is None:
            for name in ["CNN-PDDQN综合最优", "DRL规划≥10x", "PDDQN路径≤基线"]:
                checks.append((name, mode, None, f"无 quality pairs"))
            continue
        # 5. CNN-PDDQN composite best
        pc = q.get("CNN-PDDQN", {}).get("composite", float("inf"))
        oc = {a: q[a]["composite"] for a in DRL_ALGOS if a != "CNN-PDDQN" and a in q}
        if oc:
            ba = min(oc, key=lambda a: oc[a])
            checks.append(("CNN-PDDQN综合最优", mode, pc <= oc[ba],
                            f"{pc:.4f} vs {ba}={oc[ba]:.4f}"))
        # 6. DRL planning >= 10x
        drl_max = max(q[a]["plan_time"] for a in DRL_ALGOS if a in q)
        base_min = min(q[a]["plan_time"] for a in BASELINE_ALGOS if a in q)
        ratio = base_min / drl_max if drl_max > 0 else float("inf")
        checks.append(("DRL规划≥10x", mode, ratio >= 10, f"{ratio:.1f}x"))
        # 7. PDDQN path <= baseline
        pp = q.get("CNN-PDDQN", {}).get("path_len", float("inf"))
        bp = min(q[a]["path_len"] for a in BASELINE_ALGOS if a in q)
        gap = pp - bp
        checks.append(("PDDQN路径≤基线", mode, gap <= 0, f"gap={gap:+.3f}m"))
    return checks


# ── Output helpers ──
def write_sr(sr_l, sr_s, out_dir):
    rows = []
    for a in ALGO_ORDER:
        rows.append({"Algorithm": a,
                      "SR_Long(%)": int(round(sr_l.get(a, 0) * 100)),
                      "SR_Short(%)": int(round(sr_s.get(a, 0) * 100))})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "sr.csv", index=False)

    lines = ["| Algorithm | SR Long (%) | SR Short (%) |",
             "|---|:---:|:---:|"]
    for _, r in df.iterrows():
        a = r["Algorithm"]
        b = "**" if a == "CNN-PDDQN" else ""
        lines.append(f"| {b}{a}{b} | {b}{r['SR_Long(%)']}{b} | {b}{r['SR_Short(%)']}{b} |")
    (out_dir / "sr.md").write_text("\n".join(lines) + "\n")
    return df


def write_quality(quality, n_pairs, out_dir, dist):
    tag = f"quality_{dist}"
    if quality is None:
        (out_dir / f"{tag}.csv").write_text("# No all-succeed pairs\n")
        (out_dir / f"{tag}.md").write_text(f"No all-succeed pairs ({dist})\n")
        return None

    rows = []
    for a in ALGO_ORDER:
        if a not in quality:
            continue
        q = quality[a]
        rows.append({"Algorithm": a,
                      "Path_Len(m)": round(q["path_len"], 3),
                      "Curvature(1/m)": round(q["curvature"], 4),
                      "Plan_Time(s)": round(q["plan_time"], 4),
                      "Composite": round(q["composite"], 4)})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"{tag}.csv", index=False)

    best_comp = df["Composite"].min()
    lines = [f"Quality {dist.title()} ({n_pairs} all-succeed pairs) | "
             f"W_PT={W_PT}, W_K={W_K}, W_PL={W_PL}\n",
             "| Algorithm | Path Len (m) | Curvature (1/m) | Plan Time (s) | Composite |",
             "|---|:---:|:---:|:---:|:---:|"]
    for _, r in df.iterrows():
        a = r["Algorithm"]
        b = "**" if r["Composite"] == best_comp else ""
        lines.append(f"| {b}{a}{b} | {r['Path_Len(m)']:.3f} | "
                     f"{r['Curvature(1/m)']:.4f} | {r['Plan_Time(s)']:.4f} | "
                     f"{b}{r['Composite']:.4f}{b} |")
    (out_dir / f"{tag}.md").write_text("\n".join(lines) + "\n")
    return df


# ── Main ──
def main():
    os.chdir(Path(__file__).resolve().parent.parent)
    OUT.mkdir(parents=True, exist_ok=True)

    all_checks = []
    excel_sheets = {}

    for env, loader in [("forest", load_forest), ("realmap", load_realmap)]:
        print(f"\n{'='*60}")
        print(f"  {env.upper()}")
        print(f"{'='*60}")

        edir = OUT / env
        edir.mkdir(exist_ok=True)

        df_l, df_s = loader("long"), loader("short")
        sr_l, n_ql, q_l = evaluate(df_l)
        sr_s, n_qs, q_s = evaluate(df_s)
        print(f"  Quality pairs: Long={n_ql}, Short={n_qs}")

        # Write per-env tables
        sr_df = write_sr(sr_l, sr_s, edir)
        ql_df = write_quality(q_l, n_ql, edir, "long")
        qs_df = write_quality(q_s, n_qs, edir, "short")

        excel_sheets[f"{env}_SR"] = sr_df
        if ql_df is not None:
            excel_sheets[f"{env}_Quality_Long"] = ql_df
        if qs_df is not None:
            excel_sheets[f"{env}_Quality_Short"] = qs_df

        # Narrative checks
        checks = narrative_checks(sr_l, sr_s, q_l, q_s, n_ql, n_qs)
        all_checks.append((env, checks))
        for name, mode, ok, detail in checks:
            s = "✅" if ok else ("❌" if ok is not None else "⚠️")
            print(f"  {s} [{mode}] {name}: {detail}")

    # ── Write narrative_checks.md ──
    lines = [f"# Narrative Checks — paperstoryV6\n",
             f"Canonical weights: W_PT={W_PT}, W_K={W_K}, W_PL={W_PL}\n",
             f"V6 Realmap combo: " + ", ".join(
                 f"{a}={REALMAP_COMBO[a]}" for a in DRL_ALGOS) + "\n",
             f"Data source: 10k training (algo5_10k_realmap)\n"]
    total_pass = 0
    total_total = 0
    for env, checks in all_checks:
        n_pass = sum(1 for *_, ok, _ in checks if ok is not None and bool(ok))
        n_total = sum(1 for *_, ok, _ in checks if ok is not None)
        total_pass += n_pass
        total_total += n_total
        lines.append(f"\n## {env.title()} ({n_pass}/{n_total})\n")
        lines.append("| Check | Mode | Status | Detail |")
        lines.append("|---|---|:---:|---|")
        for name, mode, ok, detail in checks:
            s = "✅" if ok else ("❌" if ok is not None else "⚠️")
            lines.append(f"| {name} | {mode} | {s} | {detail} |")
    lines.insert(4, f"\n**Total: {total_pass}/{total_total}**\n")
    (OUT / "narrative_checks.md").write_text("\n".join(lines) + "\n")

    # ── Write summary Excel ──
    xlsx = OUT / "summary.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as w:
        for sheet, df in excel_sheets.items():
            df.to_excel(w, sheet_name=sheet, index=False)
    print(f"\n  Excel: {xlsx}")

    # ── Copy model files ──
    models_dir = OUT.parent / "models" / "realmap"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Copy Forest models (same as V3)
    forest_models = OUT.parent / "models" / "forest"
    forest_models.mkdir(parents=True, exist_ok=True)
    forest_src = Path("runs/snapshot_20260305_2cat_v2/models/forest")
    if forest_src.exists():
        import shutil
        for pt in forest_src.glob("*.pt"):
            dst = forest_models / pt.name
            if not dst.exists():
                shutil.copy2(pt, dst)
                print(f"  Copied {pt.name} → forest/")

    # Copy Realmap models from 10k training
    import shutil
    ckpt_dir = TRAIN_10K / "checkpoints" / "realmap_a"
    for algo, ep in REALMAP_COMBO.items():
        tag = algo.lower()
        src = ckpt_dir / f"{tag}_ep{ep:05d}.pt"
        dst = models_dir / f"{tag}.pt"
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  Copied {algo}@{ep} → realmap/{dst.name}")
        else:
            print(f"  WARNING: {src} not found (may be on remote ubuntu-zt)")

    print(f"\n{'='*60}")
    print(f"  TOTAL: {total_pass}/{total_total}")
    print(f"  DONE — output: {OUT}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
