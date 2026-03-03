"""dqn8_plots — Post-hoc plotting suite for DQN8 experiments.

Standalone plotting package (not imported by amr_dqn training/inference).
Reads inference outputs (traces, CSVs, maps) and generates paper-quality figures.

Entry point:  python dqn8_plots/run_all.py --base-dir <infer_output> --out-dir figures/

Modules
-------
run_all.py           Master orchestrator: runs all plot scripts in sequence.
common.py            Shared config: algorithm colours, display names, style presets.
plot_training.py     Training curves (reward, success rate) from training_*.csv.
plot_paths.py        Path comparison: obstacle grid + trajectory overlay per run.
plot_kpi_bars.py     Grouped bar charts for KPI comparison across algorithms.
plot_kpi_radar.py    Radar/spider charts with normalised 0-1 axes.
plot_kpi_table.py    Table rendered as a figure with best values highlighted.
plot_map_overview.py Map visualisation with start/goal annotations.
map/                 Sub-package for map-specific utilities and mask visualisation.
"""
