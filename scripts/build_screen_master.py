#!/usr/bin/env python3
"""Build screen_master.csv and per-algo folder structure from screening results.

Reads:  runs/screen_20260306/<env>/ep<XXXXX>/{sr_long,sr_short,quality_long,quality_short}.csv
Writes: runs/screen_20260306/screen_master.csv
        runs/screen_20260306/<env>/<algo>/<mode>/results.csv

Usage:
    python scripts/build_screen_master.py runs/screen_20260306
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path


MODES = ("sr_long", "sr_short", "quality_long", "quality_short")

# Columns to extract from raw KPI CSVs (column names in table2_kpis_mean_raw.csv)
SR_COLS = ["success_rate"]
QUALITY_COLS = [
    "success_rate",
    "avg_path_length",
    "avg_curvature_1_m",
    "inference_time_s",
    "composite_score",
]

# Master CSV column order
MASTER_FIELDS = [
    "env",
    "algo",
    "epoch",
    "sr_long",
    "sr_short",
    "quality_long_sr",
    "quality_long_pathlen",
    "quality_long_curvature",
    "quality_long_comptime",
    "quality_long_composite",
    "quality_short_sr",
    "quality_short_pathlen",
    "quality_short_curvature",
    "quality_short_comptime",
    "quality_short_composite",
]


def read_kpi_csv(csv_path: Path) -> dict[str, dict[str, str]]:
    """Read a table2_kpis_mean_raw.csv and return {Algorithm: {col: val}}."""
    result: dict[str, dict[str, str]] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            algo = row.get("Algorithm", "").strip()
            if algo:
                result[algo] = dict(row)
    return result


def build_master(screen_dir: Path) -> list[dict[str, object]]:
    """Scan all epoch directories and build the master row list."""
    rows: list[dict[str, object]] = []

    for env_name in ("forest", "realmap"):
        env_dir = screen_dir / env_name
        if not env_dir.is_dir():
            continue

        # Collect all epoch dirs
        ep_dirs = sorted(
            [d for d in env_dir.iterdir() if d.is_dir() and d.name.startswith("ep")],
            key=lambda d: int(d.name[2:]),
        )

        for ep_dir in ep_dirs:
            epoch = int(ep_dir.name[2:])

            # Read all 4 mode CSVs for this epoch
            mode_data: dict[str, dict[str, dict[str, str]]] = {}  # mode -> algo -> cols
            for mode in MODES:
                csv_path = ep_dir / f"{mode}.csv"
                if csv_path.exists():
                    mode_data[mode] = read_kpi_csv(csv_path)

            # Collect all algo names seen across modes
            all_algos: set[str] = set()
            for md in mode_data.values():
                all_algos.update(md.keys())

            # Build one row per algo
            for algo in sorted(all_algos):
                row: dict[str, object] = {
                    "env": env_name,
                    "algo": algo,
                    "epoch": epoch,
                }

                # SR modes
                for mode in ("sr_long", "sr_short"):
                    if mode in mode_data and algo in mode_data[mode]:
                        row[mode] = mode_data[mode][algo].get("success_rate", "")
                    else:
                        row[mode] = ""

                # Quality modes
                for mode in ("quality_long", "quality_short"):
                    if mode in mode_data and algo in mode_data[mode]:
                        d = mode_data[mode][algo]
                        row[f"{mode}_sr"] = d.get("success_rate", "")
                        row[f"{mode}_pathlen"] = d.get("avg_path_length", "")
                        row[f"{mode}_curvature"] = d.get("avg_curvature_1_m", "")
                        row[f"{mode}_comptime"] = d.get("inference_time_s", "")
                        row[f"{mode}_composite"] = d.get("composite_score", "")
                    else:
                        for suffix in ("_sr", "_pathlen", "_curvature", "_comptime", "_composite"):
                            row[f"{mode}{suffix}"] = ""

                rows.append(row)

    return rows


def write_master_csv(rows: list[dict[str, object]], out_path: Path) -> None:
    """Write the master CSV."""
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MASTER_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  screen_master.csv: {len(rows)} rows -> {out_path}")


def build_per_algo_folders(rows: list[dict[str, object]], screen_dir: Path) -> None:
    """Build per-algo folder structure with per-mode results.csv files."""
    # Group by (env, algo)
    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["env"]), str(row["algo"]))
        groups.setdefault(key, []).append(row)

    for (env_name, algo), algo_rows in sorted(groups.items()):
        algo_slug = algo.lower().replace(" ", "-")
        algo_rows_sorted = sorted(algo_rows, key=lambda r: int(r["epoch"]))  # type: ignore[arg-type]

        for mode in MODES:
            mode_dir = screen_dir / env_name / algo_slug / mode
            mode_dir.mkdir(parents=True, exist_ok=True)
            out_csv = mode_dir / "results.csv"

            if mode in ("sr_long", "sr_short"):
                fieldnames = ["epoch", "success_rate"]
                mode_rows = [
                    {"epoch": r["epoch"], "success_rate": r.get(mode, "")}
                    for r in algo_rows_sorted
                ]
            else:
                fieldnames = ["epoch", "success_rate", "avg_path_length",
                              "avg_curvature", "compute_time", "composite_score"]
                mode_rows = [
                    {
                        "epoch": r["epoch"],
                        "success_rate": r.get(f"{mode}_sr", ""),
                        "avg_path_length": r.get(f"{mode}_pathlen", ""),
                        "avg_curvature": r.get(f"{mode}_curvature", ""),
                        "compute_time": r.get(f"{mode}_comptime", ""),
                        "composite_score": r.get(f"{mode}_composite", ""),
                    }
                    for r in algo_rows_sorted
                ]

            with open(out_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(mode_rows)

    algo_count = len(groups)
    print(f"  Per-algo folders: {algo_count} algo(s) x {len(MODES)} modes")


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <screen_dir>")
        sys.exit(1)

    screen_dir = Path(sys.argv[1])
    if not screen_dir.is_dir():
        print(f"ERROR: {screen_dir} is not a directory")
        sys.exit(1)

    print(f"Scanning: {screen_dir}")
    rows = build_master(screen_dir)

    if not rows:
        print("No data found!")
        sys.exit(1)

    # Write master CSV
    write_master_csv(rows, screen_dir / "screen_master.csv")

    # Build per-algo folders
    build_per_algo_folders(rows, screen_dir)

    print("Done!")


if __name__ == "__main__":
    main()
