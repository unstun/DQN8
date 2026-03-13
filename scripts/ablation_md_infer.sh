#!/bin/bash
# ============================================================
# Inference for MHA × Dueling ablation
# 18 jobs (9 variants × 2 suites), 4-parallel on single GPU
#
# Usage (on remote server, after training completes):
#   chmod +x scripts/ablation_md_infer.sh
#   nohup bash scripts/ablation_md_infer.sh > runs/ablation_md_infer_master.log 2>&1 &
# ============================================================
set -euo pipefail

PROJ="$HOME/DQN8"
ENV="ros2py310"
CONDA="$HOME/miniconda3/bin/conda"
MAX_PARALLEL=4
LOG_DIR="$PROJ/runs/ablation_md_infer_logs"
mkdir -p "$LOG_DIR"

# ---------- Job queue: 18 inference profiles ----------
PROFILES=(
  ablation_20260312_infer_cnn_dqn_sr_long
  ablation_20260312_infer_cnn_dqn_sr_short
  ablation_20260312_infer_cnn_ddqn_sr_long
  ablation_20260312_infer_cnn_ddqn_sr_short
  ablation_20260312_infer_mlp_sr_long
  ablation_20260312_infer_mlp_sr_short
  ablation_20260312_infer_cnn_dqn_mha_sr_long
  ablation_20260312_infer_cnn_dqn_mha_sr_short
  ablation_20260312_infer_cnn_ddqn_mha_sr_long
  ablation_20260312_infer_cnn_ddqn_mha_sr_short
  ablation_20260312_infer_cnn_dqn_duel_sr_long
  ablation_20260312_infer_cnn_dqn_duel_sr_short
  ablation_20260312_infer_cnn_ddqn_duel_sr_long
  ablation_20260312_infer_cnn_ddqn_duel_sr_short
  ablation_20260312_infer_cnn_dqn_md_sr_long
  ablation_20260312_infer_cnn_dqn_md_sr_short
  ablation_20260312_infer_cnn_ddqn_md_sr_long
  ablation_20260312_infer_cnn_ddqn_md_sr_short
)

echo "============================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Ablation MHA×Dueling INFERENCE starting"
echo "  Jobs: ${#PROFILES[@]}"
echo "  Max parallel: $MAX_PARALLEL"
echo "============================================"

running=0

for profile in "${PROFILES[@]}"; do
  while [ $running -ge $MAX_PARALLEL ]; do
    wait -n || true
    running=$((running - 1))
  done

  ts=$(date '+%Y%m%d_%H%M%S')
  logfile="$LOG_DIR/${profile}_${ts}.log"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] START  $profile → $logfile"

  $CONDA run --cwd "$PROJ" -n "$ENV" \
    python infer.py --profile "$profile" \
    > "$logfile" 2>&1 &

  running=$((running + 1))
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] All inference jobs launched. Waiting for remaining $running..."
wait
echo "============================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ALL INFERENCE DONE"
echo "============================================"
