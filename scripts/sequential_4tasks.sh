#!/bin/bash
# Sequential 4-task training: 1 task at a time, 3 parallel sub-processes per task.
# After each task completes, merge checkpoints, then start next task.
# Uses direct conda env activation (not conda run) so PID tracking works.
set -euo pipefail

PROJ="/home/sun/phdproject/dqn/DQN8"
ENV="ros2py310"
LOG_DIR="$PROJ/runs"
CONDA_PREFIX="$HOME/miniconda3"

# Source conda
eval "$($CONDA_PREFIX/bin/conda shell.bash hook)"

TASKS=("forest" "forest_diag" "realmap_half" "realmap_diag")
VARIANTS=("pddqn" "ddqn" "dqn")

echo "$(date): Starting sequential 4-task training"
echo "Memory before start:"
free -h

conda activate "$ENV"

for TASK in "${TASKS[@]}"; do
    echo ""
    echo "$(date): ======== Starting task: $TASK ========"
    free -h | head -2

    PIDS=()
    for VAR in "${VARIANTS[@]}"; do
        PROFILE="repro_20260312_${TASK}_${VAR}"
        LOGFILE="$LOG_DIR/${PROFILE}.log"
        echo "  Launching $PROFILE ..."
        cd "$PROJ"
        python train.py --profile "$PROFILE" > "$LOGFILE" 2>&1 &
        PIDS+=($!)
        sleep 2
    done

    echo "  PIDs: ${PIDS[*]}"
    echo "  Waiting for all 3 sub-processes to finish..."

    # Monitor loop: check PIDs every 10 minutes
    while true; do
        ALIVE=0
        for PID in "${PIDS[@]}"; do
            if kill -0 "$PID" 2>/dev/null; then
                ((ALIVE+=1))
            fi
        done
        if [ "$ALIVE" -eq 0 ]; then
            break
        fi

        MEM_USED=$(free -g | awk '/Mem:/{print $3}')
        SWAP_USED=$(free -g | awk '/Swap:/{print $3}')
        OUT_DIR="algo6_10k_${TASK}"
        LATEST=$(ls -t "$PROJ/runs/$OUT_DIR"/train_20260312*/checkpoints/*/*.pt 2>/dev/null | head -1 | grep -oP 'ep\d+' || echo "-")
        echo "  $(date): $ALIVE/3 alive, mem=${MEM_USED}G, swap=${SWAP_USED}G, latest=$LATEST"
        sleep 600
    done

    # Collect exit codes
    echo "$(date): Task $TASK complete!"
    for i in "${!PIDS[@]}"; do
        wait "${PIDS[$i]}" 2>/dev/null
        EC=$?
        echo "  ${VARIANTS[$i]}: exit code $EC"
    done

    # Merge
    OUT_DIR="algo6_10k_${TASK}"
    echo "  Merging checkpoints for $OUT_DIR ..."
    bash "$PROJ/scripts/merge_split_trains.sh" "$PROJ/runs/$OUT_DIR"

    # Launch screen inference in background if script exists
    SCREEN_SCRIPT="$PROJ/scripts/screen_6algo_10k_${TASK}.sh"
    if [ -f "$SCREEN_SCRIPT" ]; then
        echo "  Launching screen inference: $SCREEN_SCRIPT"
        nohup bash "$SCREEN_SCRIPT" > "$LOG_DIR/screen_${TASK}.log" 2>&1 &
        echo "  Screen PID: $!"
    fi

    echo "$(date): Moving to next task..."
    sleep 10
done

echo ""
echo "$(date): All 4 tasks complete!"
free -h
