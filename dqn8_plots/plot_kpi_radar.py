"""KPI radar (spider) chart with normalised 0–1 axes."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import ALG_COLORS, ALG_ORDER, LEGEND_FONT_SIZE, OUTPUT_DPI, apply_rcparams, load_kpi

METRICS = [
    ("Average path length (m)", "Path len", "lower"),
    ("Average curvature (1/m)", "Curvature", "lower"),
    ("Planning time (s)", "Plan time", "lower"),
    ("Composite score", "Score", "higher"),
    ("Success rate", "Success", "higher"),
]


def plot_kpi_radar(
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

    # Use only metrics present in CSV
    avail_metrics = [(c, l, d) for c, l, d in METRICS if c in df.columns]
    if not avail_metrics:
        print("[radar] no metrics found – skip")
        return []

    saved: list[Path] = []
    for env_case in sorted(df["env_case"].unique()):
        if env_filter and env_case not in env_filter:
            continue
        env_df = df[df["env_case"] == env_case].copy()
        present = [a for a in ALG_ORDER if a in env_df["Algorithm"].values]
        env_df = env_df.set_index("Algorithm").reindex(present)

        # Normalise each metric to [0, 1]
        normed: dict[str, pd.Series] = {}
        for col, _, direction in avail_metrics:
            s = env_df[col].astype(float)
            lo, hi = s.min(), s.max()
            if math.isclose(hi, lo):
                normed[col] = pd.Series(1.0, index=s.index)
            elif direction == "higher":
                normed[col] = (s - lo) / (hi - lo)
            else:
                normed[col] = (hi - s) / (hi - lo)

        labels = [lbl for _, lbl, _ in avail_metrics]
        angles = np.linspace(0, 2 * math.pi, len(avail_metrics), endpoint=False)
        angles = np.concatenate([angles, [angles[0]]])

        fig = plt.figure(figsize=(8, 8), dpi=OUTPUT_DPI)
        ax = fig.add_subplot(111, polar=True)
        ax.set_theta_offset(math.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_thetagrids(np.degrees(angles[:-1]), labels)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8])
        ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8"])

        for algo in present:
            vals = np.array([normed[c].loc[algo] for c, _, _ in avail_metrics], dtype=float)
            vals = np.concatenate([vals, [vals[0]]])
            ax.plot(angles, vals, color=ALG_COLORS.get(algo, "#333"), linewidth=2, label=algo)
            ax.fill(angles, vals, color=ALG_COLORS.get(algo, "#333"), alpha=0.05)

        ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1.0),
                  frameon=True, fontsize=LEGEND_FONT_SIZE)
        fig.suptitle(env_case, fontsize=14)
        fig.tight_layout()
        slug = env_case.replace("::", "_")
        out_path = out_dir / f"fig_kpi_radar_{slug}.png"
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
    ap = argparse.ArgumentParser(description="Plot KPI radar charts.")
    ap.add_argument("--kpi-csv", type=str, required=True)
    ap.add_argument("--out-dir", type=str, default="figures")
    ap.add_argument("--env", type=str, nargs="*", default=None)
    args = ap.parse_args()
    plot_kpi_radar(args.kpi_csv, args.out_dir, env_filter=args.env)


if __name__ == "__main__":
    main()
