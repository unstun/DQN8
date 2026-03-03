"""Command-line entrypoints.

Modules
-------
train.py                 Main training loop (1500+ LOC).
                         Supports multi-env, multi-algo, DQfD demo prefill,
                         Hybrid A* expert mixing, reward normalization, periodic eval.
                         Entry: main() -> build_parser() + train_one().

infer.py                 Inference & evaluation (2700+ LOC).
                         Loads trained models, runs rollouts, computes KPIs,
                         compares against Hybrid A* / RRT* baselines, generates
                         CSV tables + path/control figures.
                         Entry: main() -> build_parser().

benchmark.py             Orchestrates train -> infer pipeline with result validation.
config.py                Generates a combined train+infer JSON config template.
precompute_forest_paths.py  Offline Hybrid A* expert-path caching for forest maps.
"""

