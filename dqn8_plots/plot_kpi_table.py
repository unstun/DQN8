"""KPI table rendered as a figure (best values bolded)."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common import ALG_COLORS, ALG_ORDER, OUTPUT_DPI, TABLE_FONT_SIZE, apply_rcparams, load_kpi

METRICS = [
    ("Success rate", "Success", "higher"),
    ("Average path length (m)", "Path len (m)", "lower"),
    ("Average curvature (1/m)", "Curvature (1/m)", "lower"),
    ("Planning time (s)", "Plan time (s)", "lower"),
    ("Composite score", "Score", "higher"),
]


def plot_kpi_table(
    kpi_csv: str | Path,
    out_dir: str | Path,
    env_filter: list[str] | None = None,
) -> list[Path]:
    apply_rcparams()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_kpi(kpi_csv)
    if "Algorithm name" in df.columns:
        df = df.rename(columns={"Algorithm name": "Algorithm"})
    if "env_case" not in df.columns:
        df["env_case"] = df["Environment"].apply(_env_to_case)

    avail = [(c, l, d) for c, l, d in METRICS if c in df.columns]
    if not avail:
        print("[table] no metrics – skip")
        return []

    saved: list[Path] = []
    for env_case in sorted(df["env_case"].unique()):
        if env_filter and env_case not in env_filter:
            continue
        env_df = df[df["env_case"] == env_case].copy()
        present = [a for a in ALG_ORDER if a in env_df["Algorithm"].values]
        env_df = env_df.set_index("Algorithm").reindex(present).reset_index()

        # Find best value per metric
        best_idx: dict[str, int] = {}
        for col, _, direction in avail:
            if col not in env_df.columns:
                continue
            vals = env_df[col].astype(float)
            best_idx[col] = int(vals.idxmax() if direction == "higher" else vals.idxmin())

        col_labels = ["Algorithm"] + [lbl for _, lbl, _ in avail]
        cell_text: list[list[str]] = []
        cell_colors: list[list[str]] = []
        for idx, row in env_df.iterrows():
            row_text = [str(row["Algorithm"])]
            row_colors = ["w"]
            for col, _, _ in avail:
                if col not in env_df.columns:
                    row_text.append("")
                    row_colors.append("w")
                    continue
                v = row[col]
                try:
                    txt = f"{float(v):.4f}"
                except Exception:
                    txt = str(v)
                if idx == best_idx.get(col):
                    txt = f"$\\bf{{{txt}}}$"
                row_text.append(txt)
                row_colors.append("w")
            cell_text.append(row_text)
            cell_colors.append(row_colors)

        fig_h = 2.0 + 0.4 * len(present)
        fig, ax = plt.subplots(1, 1, figsize=(12, fig_h), dpi=OUTPUT_DPI)
        ax.axis("off")
        tbl = ax.table(
            cellText=cell_text, colLabels=col_labels,
            loc="center", cellLoc="center", colLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(TABLE_FONT_SIZE)
        # Style header
        for j in range(len(col_labels)):
            tbl[0, j].set_facecolor("#d9e2f3")
            tbl[0, j].set_text_props(weight="bold")
        # Stripe rows & colour algo column
        for i in range(len(present)):
            algo = present[i]
            c = ALG_COLORS.get(algo, "#ffffff")
            tbl[i + 1, 0].set_facecolor(c + "22")  # light tint
            if i % 2 == 1:
                for j in range(1, len(col_labels)):
                    tbl[i + 1, j].set_facecolor("#f2f2f2")
        for cell in tbl.get_celld().values():
            cell.set_linewidth(0.5)
        tbl.scale(1.0, 1.3)
        fig.suptitle(env_case, fontsize=14, y=0.98)
        fig.tight_layout()
        slug = env_case.replace("::", "_")
        out_path = out_dir / f"table_kpis_{slug}.png"
        fig.savefig(str(out_path), bbox_inches="tight", dpi=OUTPUT_DPI)
        plt.close(fig)
        saved.append(out_path)
        print(f"  saved {out_path.name}")
    return saved


def _env_to_case(env_label: str) -> str:
    import re
    m = re.match(r"Env\.\s*\(([^)]+)\)(?:/(.+))?", str(env_label))
    if m:
        base = m.group(1).strip()
        suite = m.group(2).strip() if m.group(2) else None
        return f"{base}::{suite}" if suite else base
    return str(env_label)


def main() -> None:
    ap = argparse.ArgumentParser(description="Render KPI table as image.")
    ap.add_argument("--kpi-csv", type=str, required=True)
    ap.add_argument("--out-dir", type=str, default="figures")
    ap.add_argument("--env", type=str, nargs="*", default=None)
    args = ap.parse_args()
    plot_kpi_table(args.kpi_csv, args.out_dir, env_filter=args.env)


if __name__ == "__main__":
    main()
