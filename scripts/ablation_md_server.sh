#!/bin/bash
# ============================================================
# Ablation: MHA × Dueling × Algo × Arch
# 9 jobs, 5-parallel on single RTX 4090
#
# Usage (on remote server):
#   chmod +x scripts/ablation_md_server.sh
#   nohup bash scripts/ablation_md_server.sh > runs/ablation_md_master.log 2>&1 &
# ============================================================
set -euo pipefail

PROJ="$HOME/DQN8"
ENV="ros2py310"
CONDA="$HOME/miniconda3/bin/conda"
MAX_PARALLEL=5
LOG_DIR="$PROJ/runs/ablation_md_logs"
mkdir -p "$LOG_DIR"

# ---------- Job queue (ordered: light jobs first) ----------
# MLP group runs 2 algos sequentially (~3h), CNN single-algo (~2.5h each)
# Start MLP + 2 CNN bases first so MLP finishes early and frees a slot.
PROFILES=(
  ablation_20260312_cnn_dqn
  ablation_20260312_cnn_ddqn
  ablation_20260312_mlp
  ablation_20260312_cnn_dqn_mha
  ablation_20260312_cnn_ddqn_mha
  ablation_20260312_cnn_dqn_duel
  ablation_20260312_cnn_ddqn_duel
  ablation_20260312_cnn_dqn_md
  ablation_20260312_cnn_ddqn_md
)

echo "============================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Ablation MHA×Dueling starting"
echo "  Jobs: ${#PROFILES[@]}"
echo "  Max parallel: $MAX_PARALLEL"
echo "============================================"

running=0
pids=()

for profile in "${PROFILES[@]}"; do
  # Wait if we've hit the parallel limit
  while [ $running -ge $MAX_PARALLEL ]; do
    # Wait for any child to finish
    wait -n || true
    running=$((running - 1))
  done

  ts=$(date '+%Y%m%d_%H%M%S')
  logfile="$LOG_DIR/${profile}_${ts}.log"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] START  $profile → $logfile"

  $CONDA run --cwd "$PROJ" -n "$ENV" \
    python train.py --profile "$profile" \
    > "$logfile" 2>&1 &

  pids+=($!)
  running=$((running + 1))
done

# Wait for all remaining jobs
echo "[$(date '+%Y-%m-%d %H:%M:%S')] All jobs launched. Waiting for remaining $running..."
wait
echo "============================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ALL DONE"
echo "============================================"

# Quick summary: list output directories
echo ""
echo "Output directories:"
for profile in "${PROFILES[@]}"; do
  out_name=$(python3 -c "import json; d=json.load(open('$PROJ/configs/${profile}.json')); print(d['train']['out'])")
  echo "  runs/$out_name/"
done
