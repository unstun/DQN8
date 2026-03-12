#!/bin/bash
# Sequential training: 4 tasks, 6 processes each (1 algo per process).
# Merge script combines checkpoints from this + any previous crashed runs.
set -eo pipefail

PROJ="/home/sun/phdproject/dqn/DQN8"
ENV="ros2py310"
LOG_DIR="$PROJ/runs"
CONDA_PREFIX="$HOME/miniconda3"

eval "$($CONDA_PREFIX/bin/conda shell.bash hook)"
conda activate "$ENV"
cd "$PROJ"

# task_name -> config suffix mapping
declare -A TASK_SUFFIX
TASK_SUFFIX[forest_diag]="forest_diag"
TASK_SUFFIX[forest]="forest_half"
TASK_SUFFIX[realmap_half]="realmap_half"
TASK_SUFFIX[realmap_diag]="realmap_diag"

TASKS=("forest_diag" "forest" "realmap_half" "realmap_diag")
ALGOS=("cnn_pddqn" "cnn_ddqn" "cnn_dqn" "mlp_pddqn" "mlp_ddqn" "mlp_dqn")

echo "$(date): Starting sequential 4-task training (6 processes per task)"
free -h

for TASK in "${TASKS[@]}"; do
    echo ""
    echo "$(date): ======== Starting task: $TASK (6 processes) ========"
    free -h | head -2

    SUFFIX="${TASK_SUFFIX[$TASK]}"
    PIDS=()
    for ALGO in "${ALGOS[@]}"; do
        PROFILE="repro_20260312_${SUFFIX}_${ALGO}"
        LOGFILE="$LOG_DIR/${PROFILE}.log"
        echo "  Launching $PROFILE ..."
        python train.py --profile "$PROFILE" > "$LOGFILE" 2>&1 &
        PIDS+=($!)
        sleep 2
    done

    echo "  PIDs: ${PIDS[*]}"
    echo "  Waiting for all 6 processes to finish..."

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
        echo "  $(date): $ALIVE/6 alive, mem=${MEM_USED}G, swap=${SWAP_USED}G, latest=$LATEST"
        sleep 600
    done

    echo "$(date): Task $TASK complete!"
    for i in "${!PIDS[@]}"; do
        wait "${PIDS[$i]}" 2>/dev/null
        EC=$?
        echo "  ${ALGOS[$i]}: exit code $EC"
    done

    OUT_DIR="algo6_10k_${TASK}"
    echo "  Merging checkpoints for $OUT_DIR ..."
    bash "$PROJ/scripts/merge_split_trains.sh" "$PROJ/runs/$OUT_DIR"

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
