"""KPI grouped bar charts."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common import ALG_COLORS, ALG_ORDER, OUTPUT_DPI, apply_rcparams, load_kpi

METRICS = [
    ("Average path length (m)", "Path length (m)", "lower"),
    ("Average curvature (1/m)", "Curvature (1/m)", "lower"),
    ("Planning time (s)", "Planning time (s)", "lower"),
    ("Composite score", "Composite score", "higher"),
]


def plot_kpi_bars(
    kpi_csv: str | Path,
    out_dir: str | Path,
    env_filter: list[str] | None = None,
) -> list[Path]:
    apply_rcparams()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_kpi(kpi_csv)

    # Normalise column names
    if "Algorithm name" in df.columns:
        df = df.rename(columns={"Algorithm name": "Algorithm"})

    # Build env_case from Environment column
    if "env_case" not in df.columns:
        df["env_case"] = df["Environment"].apply(_env_to_case)

    saved: list[Path] = []
    for env_case in sorted(df["env_case"].unique()):
        if env_filter and env_case not in env_filter:
            continue
        env_df = df[df["env_case"] == env_case].copy()
        env_df = env_df.set_index("Algorithm").reindex(
            [a for a in ALG_ORDER if a in env_df["Algorithm"].values]
        ).reset_index()

        fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=OUTPUT_DPI)
        axes_flat = np.atleast_1d(axes).flatten()
        for ax, (col, title, _) in zip(axes_flat, METRICS):
            if col not in env_df.columns:
                ax.set_visible(False)
                continue
            vals = env_df[col].to_numpy(dtype=float)
            colors = [ALG_COLORS.get(a, "#333333") for a in env_df["Algorithm"]]
            ax.bar(env_df["Algorithm"], vals, color=colors)
            ax.set_ylabel(title)
            ax.tick_params(axis="x", rotation=25)
        fig.suptitle(env_case, fontsize=14)
        fig.tight_layout()
        slug = env_case.replace("::", "_")
        out_path = out_dir / f"fig_kpi_bars_{slug}.png"
        fig.savefig(str(out_path), bbox_inches="tight", dpi=OUTPUT_DPI)
        plt.close(fig)
        saved.append(out_path)
        print(f"  saved {out_path.name}")
    return saved


def _env_to_case(env_label: str) -> str:
    """Convert 'Env. (forest_a)/short' → 'forest_a::short'."""
    import re
    m = re.match(r"Env\.\s*\(([^)]+)\)(?:/(.+))?", str(env_label))
    if m:
        base = m.group(1).strip()
        suite = m.group(2).strip() if m.group(2) else None
        return f"{base}::{suite}" if suite else base
    return str(env_label)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot KPI bar charts.")
    ap.add_argument("--kpi-csv", type=str, required=True, help="Path to table2_kpis_mean.csv or table2_kpis.csv.")
    ap.add_argument("--out-dir", type=str, default="figures")
    ap.add_argument("--env", type=str, nargs="*", default=None)
    args = ap.parse_args()
    plot_kpi_bars(args.kpi_csv, args.out_dir, env_filter=args.env)


if __name__ == "__main__":
    main()
