#!/bin/bash
# Launch 12 split training processes (4 tasks x 3 algo-pairs)
# Each process trains 2 algos (CNN + MLP of same variant)
set -e
PROJ=/home/sun/phdproject/dqn/DQN8
ENV=ros2py310
LOGDIR=$PROJ/runs

PROFILES=(
  repro_20260312_forest_half_pddqn
  repro_20260312_forest_half_ddqn
  repro_20260312_forest_half_dqn
  repro_20260312_forest_diag_pddqn
  repro_20260312_forest_diag_ddqn
  repro_20260312_forest_diag_dqn
  repro_20260312_realmap_half_pddqn
  repro_20260312_realmap_half_ddqn
  repro_20260312_realmap_half_dqn
  repro_20260312_realmap_diag_pddqn
  repro_20260312_realmap_diag_ddqn
  repro_20260312_realmap_diag_dqn
)

echo "$(date): Launching 12 training processes..."
for p in "${PROFILES[@]}"; do
  LOG="$LOGDIR/${p}.log"
  nohup conda run --cwd "$PROJ" -n "$ENV" python train.py --profile "$p" \
    > "$LOG" 2>&1 &
  PID=$!
  echo "  Started $p (PID=$PID) -> $LOG"
  sleep 1  # stagger slightly
done

echo "$(date): All 12 launched. Monitor with: ps aux | grep train.py | grep -v grep | wc -l"
