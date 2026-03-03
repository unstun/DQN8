"""Training curves: reward + success rate from training CSVs."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import ALG_COLORS, ALG_ORDER, LEGEND_FONT_SIZE, OUTPUT_DPI, apply_rcparams

# Mapping from config algo key to display name
_ALGO_KEY_TO_LABEL = {
    "mlp-dqn": "MLP-DQN",
    "mlp-ddqn": "MLP-DDQN",
    "mlp-pddqn": "MLP-PDDQN",
    "cnn-dqn": "CNN-DQN",
    "cnn-ddqn": "CNN-DDQN",
    "cnn-pddqn": "CNN-PDDQN",
}


def _smooth(y: np.ndarray, window: int = 20) -> np.ndarray:
    """Simple moving average for curve smoothing."""
    if len(y) < window:
        return y
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="same")


def plot_training(
    train_dir: str | Path,
    out_dir: str | Path,
    smooth_window: int = 20,
) -> list[Path]:
    """Plot training reward and eval success-rate curves.

    *train_dir* should be a training run directory containing
    ``training_returns.csv`` and/or ``training_eval.csv``.
    """
    apply_rcparams()
    train_dir = Path(train_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    # --- Returns ---
    returns_csv = train_dir / "training_returns.csv"
    if returns_csv.exists():
        df_ret = pd.read_csv(returns_csv)
        envs = sorted(df_ret["env"].unique()) if "env" in df_ret.columns else [""]
        for env_name in envs:
            edf = df_ret[df_ret["env"] == env_name] if "env" in df_ret.columns else df_ret
            fig, ax = plt.subplots(figsize=(10, 5), dpi=OUTPUT_DPI)
            for col in edf.columns:
                if col in ("env", "episode"):
                    continue
                label_key = col.replace("_return", "")
                label = _ALGO_KEY_TO_LABEL.get(label_key, label_key.upper())
                eps = edf["episode"].to_numpy() if "episode" in edf.columns else np.arange(len(edf))
                vals = edf[col].to_numpy(dtype=float)
                ax.plot(eps, _smooth(vals, smooth_window), label=label,
                        color=ALG_COLORS.get(label, None), linewidth=1.5, alpha=0.9)
            ax.set_xlabel("Episode")
            ax.set_ylabel("Return")
            ax.set_title(f"Training Returns — {env_name}" if env_name else "Training Returns")
            ax.legend(fontsize=LEGEND_FONT_SIZE, loc="best")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            slug = env_name.replace("::", "_") if env_name else "all"
            p = out_dir / f"fig_training_returns_{slug}.png"
            fig.savefig(str(p), bbox_inches="tight", dpi=OUTPUT_DPI)
            plt.close(fig)
            saved.append(p)
            print(f"  saved {p.name}")

    # --- Eval success rate ---
    eval_csv = train_dir / "training_eval.csv"
    if eval_csv.exists():
        df_eval = pd.read_csv(eval_csv)
        envs = sorted(df_eval["env"].unique()) if "env" in df_eval.columns else [""]
        for env_name in envs:
            edf = df_eval[df_eval["env"] == env_name] if "env" in df_eval.columns else df_eval
            fig, ax = plt.subplots(figsize=(10, 5), dpi=OUTPUT_DPI)
            for algo_key in sorted(edf["algo"].unique()) if "algo" in edf.columns else []:
                adf = edf[edf["algo"] == algo_key]
                label = _ALGO_KEY_TO_LABEL.get(algo_key, algo_key.upper())
                eps = adf["episode"].to_numpy() if "episode" in adf.columns else np.arange(len(adf))
                sr = adf["success_rate"].to_numpy(dtype=float)
                ax.plot(eps, _smooth(sr, smooth_window), label=label,
                        color=ALG_COLORS.get(label, None), linewidth=1.5, alpha=0.9)
            ax.set_xlabel("Episode")
            ax.set_ylabel("Success Rate")
            ax.set_ylim(-0.05, 1.05)
            ax.set_title(f"Eval Success Rate — {env_name}" if env_name else "Eval Success Rate")
            ax.legend(fontsize=LEGEND_FONT_SIZE, loc="best")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            slug = env_name.replace("::", "_") if env_name else "all"
            p = out_dir / f"fig_training_success_{slug}.png"
            fig.savefig(str(p), bbox_inches="tight", dpi=OUTPUT_DPI)
            plt.close(fig)
            saved.append(p)
            print(f"  saved {p.name}")

    if not saved:
        print("[training] no training CSVs found in", train_dir)
    return saved


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot training curves.")
    ap.add_argument("--train-dir", type=str, required=True, help="Training run directory with training_*.csv files.")
    ap.add_argument("--out-dir", type=str, default="figures")
    ap.add_argument("--smooth", type=int, default=20, help="Moving average window size.")
    args = ap.parse_args()
    plot_training(args.train_dir, args.out_dir, smooth_window=args.smooth)


if __name__ == "__main__":
    main()
