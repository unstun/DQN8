#!/bin/bash
# Ablation experiment: CNN-DQN with/without EDT channel
# Run on remote server, two experiments in parallel

set -e
PROJ=/root/DQN8
ENV=ros2py310

echo "=== Starting EDT ablation experiment ==="
echo "$(date): Launching two parallel training jobs"

# Job 1: CNN-DQN WITHOUT EDT (2 channels)
echo "Starting: CNN-DQN drop EDT (2ch)"
nohup conda run --cwd $PROJ -n $ENV python -m amr_dqn.cli.train --profile ablation_20260311_cnn_drop_edt \
  > $PROJ/runs/ablation_drop_edt_$(date +%Y%m%d_%H%M%S).log 2>&1 &
PID1=$!
echo "  PID=$PID1"

# Job 2: CNN-DQN WITH EDT (3 channels, baseline)
echo "Starting: CNN-DQN keep EDT (3ch baseline)"
nohup conda run --cwd $PROJ -n $ENV python -m amr_dqn.cli.train --profile ablation_20260311_cnn_keep_edt \
  > $PROJ/runs/ablation_keep_edt_$(date +%Y%m%d_%H%M%S).log 2>&1 &
PID2=$!
echo "  PID=$PID2"

echo ""
echo "Both jobs launched! PIDs: $PID1, $PID2"
echo "Monitor with: tail -f $PROJ/runs/ablation_*.log"
echo "Check status: ps aux | grep train"
