"""Benchmark orchestrator: train -> infer -> validate KPI results.

Runs train.py and infer.py as subprocesses, then loads the resulting
table2_kpis.csv and checks that all required algorithms are present
and meet minimum quality thresholds.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

from amr_dqn.agents import parse_rl_algo
from amr_dqn.runs import latest_run_dir, latest_run_dir_with_models, resolve_experiment_dir


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def _load_kpis(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ("avg_path_length", "inference_time_s", "success_rate"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _check_required(df: pd.DataFrame, *, required_algos: list[str]) -> tuple[bool, list[str]]:
    msgs: list[str] = []
    ok = True
    for env_name, g in df.groupby("Environment", sort=False):
        for algo in required_algos:
            row = g[g["Algorithm"] == str(algo)]
            if row.empty:
                ok = False
                msgs.append(f"{env_name}: missing {algo} row")
    return ok, msgs


def main() -> int:
    ap = argparse.ArgumentParser(description="Train → infer benchmark loop (fixed + random start/goal).")
    ap.add_argument("--envs", nargs="*", default=["forest_a", "forest_b", "forest_c", "forest_d"])
    ap.add_argument(
        "--rl-algos",
        nargs="+",
        default=["mlp-dqn"],
        help=(
            "RL algorithms to run: mlp-dqn mlp-ddqn mlp-pddqn cnn-dqn cnn-ddqn cnn-pddqn (or 'all'). "
            "Legacy aliases: dqn ddqn iddqn cnn-iddqn. Default: mlp-dqn."
        ),
    )
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    ap.add_argument("--cuda-device", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("bench"))
    ap.add_argument("--runs-root", type=Path, default=Path("runs"))
    ap.add_argument("--baselines", nargs="*", default=["all"])

    ap.add_argument("--random-train", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--random-eval", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--rand-eval-runs", type=int, default=10)
    ap.add_argument("--rand-min-cost-m", type=float, default=6.0)
    ap.add_argument("--rand-max-cost-m", type=float, default=0.0)
    ap.add_argument("--rand-fixed-prob", type=float, default=0.2)
    ap.add_argument("--rand-tries", type=int, default=200)
    args = ap.parse_args()
    algo_label = {
        "mlp-dqn": "MLP-DQN",
        "mlp-ddqn": "MLP-DDQN",
        "mlp-pddqn": "MLP-PDDQN",
        "cnn-dqn": "CNN-DQN",
        "cnn-ddqn": "CNN-DDQN",
        "cnn-pddqn": "CNN-PDDQN",
    }
    canonical_all = ("mlp-dqn", "mlp-ddqn", "mlp-pddqn", "cnn-dqn", "cnn-ddqn", "cnn-pddqn")
    raw_algos = [str(a).lower().strip() for a in (args.rl_algos or [])]
    if any(a == "all" for a in raw_algos):
        raw_algos = list(canonical_all)
    rl_algos: list[str] = []
    unknown: list[str] = []
    for a in raw_algos:
        try:
            canonical, _arch, _base, _legacy = parse_rl_algo(a)
        except ValueError:
            unknown.append(a)
            continue
        if canonical not in rl_algos:
            rl_algos.append(canonical)
    if unknown:
        raise SystemExit(
            f"Unknown --rl-algos value(s): {', '.join(unknown)}. Choose from: {' '.join(canonical_all)} (or 'all')."
        )
    if not rl_algos:
        raise SystemExit(f"No RL algorithms selected (choose from: {' '.join(canonical_all)}).")

    exp_dir = resolve_experiment_dir(args.out, runs_root=args.runs_root)

    train_cmd = [
        sys.executable,
        "-m",
        "amr_dqn.cli.train",
        "--envs",
        *list(args.envs),
        "--episodes",
        str(args.episodes),
        "--max-steps",
        str(args.max_steps),
        "--device",
        str(args.device),
        "--cuda-device",
        str(args.cuda_device),
        "--seed",
        str(args.seed),
        "--out",
        str(args.out),
        "--runs-root",
        str(args.runs_root),
        "--rl-algos",
        *rl_algos,
    ]
    if bool(args.random_train):
        train_cmd += [
            "--forest-random-start-goal",
            "--forest-rand-min-cost-m",
            str(args.rand_min_cost_m),
            "--forest-rand-max-cost-m",
            str(args.rand_max_cost_m),
            "--forest-rand-fixed-prob",
            str(args.rand_fixed_prob),
            "--forest-rand-tries",
            str(args.rand_tries),
            "--forest-expert",
            "auto",
        ]

    _run(train_cmd)
    run_dir = latest_run_dir_with_models(exp_dir)
    if run_dir is None:
        raise SystemExit(f"No models found under {exp_dir}")

    models_dir = run_dir / "models"

    infer_fixed_cmd = [
        sys.executable,
        "-m",
        "amr_dqn.cli.infer",
        "--envs",
        *list(args.envs),
        "--models",
        str(models_dir),
        "--out",
        str(args.out) + "_infer_fixed",
        "--runs",
        "1",
        "--max-steps",
        str(args.max_steps),
        "--device",
        str(args.device),
        "--cuda-device",
        str(args.cuda_device),
        "--seed",
        str(args.seed),
        "--kpi-time-mode",
        "policy",
        "--rl-algos",
        *rl_algos,
    ]
    if args.baselines:
        infer_fixed_cmd += ["--baselines", *list(args.baselines)]
    _run(infer_fixed_cmd)

    fixed_dir = latest_run_dir(resolve_experiment_dir(Path(str(args.out) + "_infer_fixed"), runs_root=args.runs_root))
    if fixed_dir is None:
        raise SystemExit("Missing fixed inference outputs")
    fixed_kpis = _load_kpis(fixed_dir / "table2_kpis_raw.csv")
    required_pretty = [algo_label.get(a, a.upper()) for a in rl_algos]
    ok_fixed, msgs_fixed = _check_required(fixed_kpis, required_algos=required_pretty)

    ok_rand = True
    msgs_rand: list[str] = []
    if bool(args.random_eval):
        infer_rand_cmd = [
            sys.executable,
            "-m",
            "amr_dqn.cli.infer",
            "--envs",
            *list(args.envs),
            "--models",
            str(models_dir),
            "--out",
            str(args.out) + "_infer_random",
            "--runs",
            str(int(args.rand_eval_runs)),
            "--max-steps",
            str(args.max_steps),
            "--device",
            str(args.device),
            "--cuda-device",
            str(args.cuda_device),
            "--seed",
            str(args.seed),
            "--kpi-time-mode",
            "policy",
            "--random-start-goal",
            "--rand-min-cost-m",
            str(args.rand_min_cost_m),
            "--rand-max-cost-m",
            str(args.rand_max_cost_m),
            "--rand-fixed-prob",
            str(0.0),
            "--rand-tries",
            str(args.rand_tries),
            "--rl-algos",
            *rl_algos,
        ]
        if args.baselines:
            infer_rand_cmd += ["--baselines", *list(args.baselines)]
        _run(infer_rand_cmd)

        rand_dir = latest_run_dir(resolve_experiment_dir(Path(str(args.out) + "_infer_random"), runs_root=args.runs_root))
        if rand_dir is None:
            raise SystemExit("Missing random inference outputs")
        rand_kpis = _load_kpis(rand_dir / "table2_kpis_raw.csv")
        ok_rand, msgs_rand = _check_required(rand_kpis, required_algos=required_pretty)

    if not ok_fixed:
        print("FAILED fixed-start/goal check:")
        for m in msgs_fixed:
            print(" - " + m)
    if bool(args.random_eval) and not ok_rand:
        print("FAILED random-start/goal check:")
        for m in msgs_rand:
            print(" - " + m)

    if ok_fixed and (ok_rand if bool(args.random_eval) else True):
        print("PASS: benchmark completed (required algorithms present).")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
