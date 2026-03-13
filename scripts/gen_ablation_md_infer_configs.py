#!/usr/bin/env python3
"""Generate inference configs for MHA×Dueling ablation (18 configs: 9 variants × 2 suites)."""
import json
from pathlib import Path

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"

# (profile_name, models_dir, rl_algos)
VARIANTS = [
    ("cnn_dqn",       "abl_md_cnn_dqn",       ["cnn-dqn"]),
    ("cnn_dqn_mha",   "abl_md_cnn_dqn_mha",   ["cnn-dqn"]),
    ("cnn_dqn_duel",  "abl_md_cnn_dqn_duel",  ["cnn-dqn"]),
    ("cnn_dqn_md",    "abl_md_cnn_dqn_md",    ["cnn-dqn"]),
    ("cnn_ddqn",      "abl_md_cnn_ddqn",      ["cnn-ddqn"]),
    ("cnn_ddqn_mha",  "abl_md_cnn_ddqn_mha",  ["cnn-ddqn"]),
    ("cnn_ddqn_duel", "abl_md_cnn_ddqn_duel", ["cnn-ddqn"]),
    ("cnn_ddqn_md",   "abl_md_cnn_ddqn_md",   ["cnn-ddqn"]),
    ("mlp",           "abl_md_mlp",           ["mlp-dqn", "mlp-ddqn"]),
]

SUITES = {
    "sr_long":  {"rand_min_cost_m": 18.0, "rand_max_cost_m": 0.0},
    "sr_short": {"rand_min_cost_m": 6.0,  "rand_max_cost_m": 14.0},
}

BASE_INFER = {
    "envs": ["realmap_a"],
    "baselines": [],
    "rl_mpc_track": False,
    "random_start_goal": True,
    "rand_two_suites": False,
    "runs": 50,
    "plot_pair_runs": False,
    "rand_tries": 600,
    "rand_reject_unreachable": True,
    "filter_all_succeed": False,
    "max_steps": 600,
    "goal_tolerance": 1.0,
    "goal_speed_tol": 999.0,
    "edt_collision_margin": "half",
    "kpi_time_mode": "policy",
    "composite_w_path_time": 1.0,
    "composite_w_avg_curvature": 0.6,
    "composite_w_planning_time": 0.2,
    "device": "cuda",
    "cuda_device": 0,
    "seed": 42,
}

created = []
for short_name, models_dir, algos in VARIANTS:
    for suite_name, suite_params in SUITES.items():
        profile = f"ablation_20260312_infer_{short_name}_{suite_name}"
        needs_drop_edt = "cnn" in short_name
        cfg = {
            "_comment": f"Ablation infer: {short_name} {suite_name}, 50 runs realmap",
            "infer": {
                **BASE_INFER,
                "models": models_dir,
                "out": f"abl_md_infer_{short_name}",
                "rl_algos": algos,
                **suite_params,
            },
        }
        if needs_drop_edt:
            cfg["infer"]["cnn_drop_edt"] = True

        out_path = CONFIGS_DIR / f"{profile}.json"
        out_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n")
        created.append(profile)
        print(f"  {out_path.name}")

print(f"\nCreated {len(created)} inference configs.")
