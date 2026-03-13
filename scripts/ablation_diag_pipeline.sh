#!/bin/bash
# ============================================================
# Ablation DIAG: 9 variants × (train → infer_long + infer_short)
# All 9 training jobs start in parallel.
# When a training job finishes, its 2 inference jobs auto-launch.
#
# Usage (on remote server):
#   chmod +x scripts/ablation_diag_pipeline.sh
#   nohup bash scripts/ablation_diag_pipeline.sh > runs/ablation_diag_master.log 2>&1 &
# ============================================================
set -euo pipefail

PROJ="$HOME/DQN8"
ENV="ros2py310"
CONDA="$HOME/miniconda3/bin/conda"
LOG_DIR="$PROJ/runs/ablation_diag_logs"
mkdir -p "$LOG_DIR"

# 9 variants: (train_profile, infer_long_profile, infer_short_profile)
VARIANTS=(
  "ablation_20260313_diag_cnn_dqn|ablation_20260313_diag_infer_cnn_dqn_sr_long|ablation_20260313_diag_infer_cnn_dqn_sr_short"
  "ablation_20260313_diag_cnn_ddqn|ablation_20260313_diag_infer_cnn_ddqn_sr_long|ablation_20260313_diag_infer_cnn_ddqn_sr_short"
  "ablation_20260313_diag_mlp|ablation_20260313_diag_infer_mlp_sr_long|ablation_20260313_diag_infer_mlp_sr_short"
  "ablation_20260313_diag_cnn_dqn_mha|ablation_20260313_diag_infer_cnn_dqn_mha_sr_long|ablation_20260313_diag_infer_cnn_dqn_mha_sr_short"
  "ablation_20260313_diag_cnn_ddqn_mha|ablation_20260313_diag_infer_cnn_ddqn_mha_sr_long|ablation_20260313_diag_infer_cnn_ddqn_mha_sr_short"
  "ablation_20260313_diag_cnn_dqn_duel|ablation_20260313_diag_infer_cnn_dqn_duel_sr_long|ablation_20260313_diag_infer_cnn_dqn_duel_sr_short"
  "ablation_20260313_diag_cnn_ddqn_duel|ablation_20260313_diag_infer_cnn_ddqn_duel_sr_long|ablation_20260313_diag_infer_cnn_ddqn_duel_sr_short"
  "ablation_20260313_diag_cnn_dqn_md|ablation_20260313_diag_infer_cnn_dqn_md_sr_long|ablation_20260313_diag_infer_cnn_dqn_md_sr_short"
  "ablation_20260313_diag_cnn_ddqn_md|ablation_20260313_diag_infer_cnn_ddqn_md_sr_long|ablation_20260313_diag_infer_cnn_ddqn_md_sr_short"
)

# Worker function: train → infer_long → infer_short (sequential per variant)
run_variant() {
  local spec="$1"
  IFS='|' read -r train_prof infer_long infer_short <<< "$spec"
  local ts=$(date '+%Y%m%d_%H%M%S')

  echo "[$(date '+%H:%M:%S')] TRAIN START: $train_prof"
  $CONDA run --cwd "$PROJ" -n "$ENV" \
    python train.py --profile "$train_prof" \
    > "$LOG_DIR/${train_prof}_${ts}.log" 2>&1

  echo "[$(date '+%H:%M:%S')] TRAIN DONE: $train_prof → launching inference"

  # Launch both inferences in parallel
  $CONDA run --cwd "$PROJ" -n "$ENV" \
    python infer.py --profile "$infer_long" \
    > "$LOG_DIR/${infer_long}_${ts}.log" 2>&1 &
  local pid_long=$!

  $CONDA run --cwd "$PROJ" -n "$ENV" \
    python infer.py --profile "$infer_short" \
    > "$LOG_DIR/${infer_short}_${ts}.log" 2>&1 &
  local pid_short=$!

  wait $pid_long $pid_short
  echo "[$(date '+%H:%M:%S')] ALL DONE: $train_prof + infer"
}

echo "============================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Ablation DIAG pipeline starting"
echo "  Variants: ${#VARIANTS[@]}"
echo "  Mode: ALL 9 training in parallel"
echo "============================================"

# Launch all 9 variant pipelines in parallel
pids=()
for spec in "${VARIANTS[@]}"; do
  run_variant "$spec" &
  pids+=($!)
done

echo "[$(date '+%H:%M:%S')] All 9 variant pipelines launched (PIDs: ${pids[*]})"
echo "Waiting for all to complete..."
wait

echo "============================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ALL 9 VARIANTS COMPLETE"
echo "============================================"

# Summary
echo ""
echo "=== Training outputs ==="
ls -d "$PROJ"/runs/abl_diag_*/train_*/ 2>/dev/null || echo "(none found)"
echo ""
echo "=== Inference outputs ==="
ls "$PROJ"/runs/abl_diag_infer_*/*/table2_kpis_raw.csv 2>/dev/null || echo "(none found)"
