#!/bin/bash
# Monitor 12 split training processes, merge when complete, then launch screen inference.
PROJ=/home/sun/phdproject/dqn/DQN8
ENV=ros2py310
CHECK_INTERVAL=300  # 5 minutes

# Task groups: out_dir -> list of profiles
declare -A TASK_PROFILES
TASK_PROFILES[algo6_10k_forest]="repro_20260312_forest_half_pddqn repro_20260312_forest_half_ddqn repro_20260312_forest_half_dqn"
TASK_PROFILES[algo6_10k_forest_diag]="repro_20260312_forest_diag_pddqn repro_20260312_forest_diag_ddqn repro_20260312_forest_diag_dqn"
TASK_PROFILES[algo6_10k_realmap_half]="repro_20260312_realmap_half_pddqn repro_20260312_realmap_half_ddqn repro_20260312_realmap_half_dqn"
TASK_PROFILES[algo6_10k_realmap_diag]="repro_20260312_realmap_diag_pddqn repro_20260312_realmap_diag_ddqn repro_20260312_realmap_diag_dqn"

# Track completed tasks
declare -A TASK_DONE

check_task_done() {
  local task="$1"
  local profiles="${TASK_PROFILES[$task]}"
  for p in $profiles; do
    if pgrep -f "train.py --profile $p" > /dev/null 2>&1; then
      return 1  # still running
    fi
  done
  return 0  # all done
}

get_latest_ep() {
  local task="$1"
  local algo="$2"
  ls -t "$PROJ/runs/$task"/train_20260312*/checkpoints/*/${algo}_ep*.pt 2>/dev/null | head -1 | grep -oP 'ep\d+' || echo "-"
}

while true; do
  echo "$(date): === Monitor Check ==="

  # Memory check
  MEM_USED=$(free -g | awk '/Mem:/{print $3}')
  SWAP_USED=$(free -g | awk '/Swap:/{print $3}')
  echo "  Memory: ${MEM_USED}GB used, Swap: ${SWAP_USED}GB"

  PROCS=$(pgrep -fc "python train.py" 2>/dev/null || echo 0)
  echo "  Training processes: $PROCS"

  # Per-task status
  ALL_DONE=true
  for task in algo6_10k_forest algo6_10k_forest_diag algo6_10k_realmap_half algo6_10k_realmap_diag; do
    if [ "${TASK_DONE[$task]}" = "1" ]; then
      continue
    fi

    if check_task_done "$task"; then
      echo "  $task: COMPLETED!"
      TASK_DONE[$task]=1

      # Merge split dirs
      echo "  Merging $task..."
      bash "$PROJ/scripts/merge_split_trains.sh" "$PROJ/runs/$task"

      # Launch screen inference if script exists
      SCREEN_SCRIPT="$PROJ/scripts/screen_6algo_10k_${task#algo6_10k_}.sh"
      if [ -f "$SCREEN_SCRIPT" ]; then
        echo "  Launching screen inference: $SCREEN_SCRIPT"
        nohup bash "$SCREEN_SCRIPT" > "$PROJ/runs/screen_${task}.log" 2>&1 &
      else
        echo "  No screen script found: $SCREEN_SCRIPT"
      fi
    else
      ALL_DONE=false
      # Show progress for each algo
      for algo in cnn-pddqn mlp-pddqn cnn-ddqn mlp-ddqn cnn-dqn mlp-dqn; do
        ep=$(get_latest_ep "$task" "$algo")
        if [ "$ep" != "-" ]; then
          echo "  $task/$algo: $ep"
        fi
      done
    fi
  done

  if [ "$ALL_DONE" = true ]; then
    echo "$(date): All tasks completed. Monitor exiting."
    break
  fi

  sleep $CHECK_INTERVAL
done
