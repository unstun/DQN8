"""One-click runner for all DQN8 plots.

Usage:
    python run_all.py --base-dir /path/to/infer_output --out-dir figures/
    python run_all.py --base-dir /path/to/infer_output --out-dir figures/ --env forest_a::short forest_a::long
    python run_all.py --base-dir /path/to/infer_output --out-dir figures/ --train-dir /path/to/train_run
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure this directory is on sys.path for sibling imports
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))


def main() -> None:
    ap = argparse.ArgumentParser(description="Run all DQN8 plotting scripts.")
    ap.add_argument("--base-dir", type=str, required=True,
                    help="Inference output dir (contains traces/, maps/, table2_kpis*.csv).")
    ap.add_argument("--out-dir", type=str, default="figures",
                    help="Output directory for all generated figures.")
    ap.add_argument("--env", type=str, nargs="*", default=None,
                    help="Filter env_cases (e.g., forest_a::short forest_a::long).")
    ap.add_argument("--run-idxs", type=int, nargs="*", default=None,
                    help="Specific run indices to plot paths for.")
    ap.add_argument("--train-dir", type=str, default=None,
                    help="Training run dir for training curves (contains training_*.csv).")
    ap.add_argument("--smooth", type=int, default=20,
                    help="Smoothing window for training curves.")
    ap.add_argument("--skip", type=str, nargs="*", default=[],
                    help="Scripts to skip (paths, bars, radar, table, training, map_overview).")
    args = ap.parse_args()

    base_dir = Path(args.base_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    skip = set(s.lower() for s in (args.skip or []))

    # --- 1. Map overview ---
    if "map_overview" not in skip:
        maps_dir = base_dir / "maps"
        if maps_dir.exists():
            print("\n=== Map Overview ===")
            from plot_map_overview import plot_map_overview
            plot_map_overview(maps_dir, out_dir)

    # --- 2. Path comparison ---
    if "paths" not in skip:
        traces_dir = base_dir / "traces"
        if traces_dir.exists():
            print("\n=== Path Comparison ===")
            from plot_paths import plot_paths
            plot_paths(base_dir, out_dir, env_cases=args.env, run_idxs=args.run_idxs)

    # --- 3. KPI bar charts ---
    kpi_csv = _find_kpi_csv(base_dir)
    if "bars" not in skip and kpi_csv:
        print(f"\n=== KPI Bar Charts (from {kpi_csv.name}) ===")
        from plot_kpi_bars import plot_kpi_bars
        plot_kpi_bars(kpi_csv, out_dir, env_filter=args.env)

    # --- 4. KPI radar ---
    if "radar" not in skip and kpi_csv:
        print(f"\n=== KPI Radar Charts ===")
        from plot_kpi_radar import plot_kpi_radar
        plot_kpi_radar(kpi_csv, out_dir, env_filter=args.env)

    # --- 5. KPI table ---
    if "table" not in skip and kpi_csv:
        print(f"\n=== KPI Table ===")
        from plot_kpi_table import plot_kpi_table
        plot_kpi_table(kpi_csv, out_dir, env_filter=args.env)

    # --- 6. Training curves ---
    if "training" not in skip and args.train_dir:
        print(f"\n=== Training Curves ===")
        from plot_training import plot_training
        plot_training(args.train_dir, out_dir, smooth_window=args.smooth)

    print(f"\n=== Done. All figures in: {out_dir.resolve()} ===")


def _find_kpi_csv(base_dir: Path) -> Path | None:
    """Find the best KPI CSV in the inference output."""
    candidates = [
        base_dir / "table2_kpis_mean.csv",
        base_dir / "table2_kpis.csv",
        base_dir / "table2_kpis_mean_raw.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


if __name__ == "__main__":
    main()
