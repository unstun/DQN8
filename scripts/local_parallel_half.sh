#!/bin/bash
# ============================================================
# LOCAL parallel training: forest_half + realmap_half
# Machine: local (RTX 3070 Ti, Python 3.10, 31G RAM)
# Margin: half (default, 0.5 * cell_size)
# Strategy: 3 processes in parallel per task (2 algos each = 6 algos)
# Tasks: forest_half -> realmap_half (sequential)
# ============================================================
set -eo pipefail

PROJ="/home/sun/phdproject/dqn/DQN8"
ENV="ros2py310"
LOG_DIR="$PROJ/runs"
CONDA_PREFIX="$HOME/miniconda3"

eval "$($CONDA_PREFIX/bin/conda shell.bash hook)"
conda activate "$ENV"
cd "$PROJ"

echo "============================================"
echo "LOCAL parallel training (half margin)"
echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null)"
echo "============================================"

TASKS=("forest_half" "realmap_half")
SPLITS=("pddqn" "ddqn" "dqn")

get_mem() { free -g | awk 'NR==2{print $3}'; }
get_swap() { free -g | awk 'NR==3{print $3}'; }

echo "$(date): Starting 2-task LOCAL training (half margin, 3 parallel per task)"
free -h | head -2

for TASK in "${TASKS[@]}"; do
    echo ""
    echo "$(date): ======== LOCAL task: $TASK (3 parallel processes, 6 algos) ========"
    echo "  Memory: mem=$(get_mem)G, swap=$(get_swap)G"

    PIDS=()
    for SPLIT in "${SPLITS[@]}"; do
        PROFILE="repro_20260312_${TASK}_${SPLIT}"
        LOGFILE="$LOG_DIR/${PROFILE}.log"
        echo "  Launching $PROFILE ..."
        python train.py --profile "$PROFILE" > "$LOGFILE" 2>&1 &
        PIDS+=($!)
        sleep 2
    done

    echo "  PIDs: ${PIDS[*]}"
    echo "  Waiting for all 3 processes to finish..."

    PEAK_MEM=0
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

        MEM=$(get_mem)
        SWAP=$(get_swap)
        if [ "$MEM" -gt "$PEAK_MEM" ]; then
            PEAK_MEM=$MEM
        fi

        # Find latest checkpoint
        TASK_OUT="algo6_10k_${TASK//_half/}"
        LATEST=$(find "$PROJ/runs/$TASK_OUT" -name "*.pt" 2>/dev/null | sort | tail -1 | grep -oP 'ep\d+' || echo "-")
        echo "  $(date): $ALIVE/3 alive, mem=${MEM}G, swap=${SWAP}G (peak=${PEAK_MEM}G), latest=$LATEST"
        sleep 300
    done

    echo ""
    echo "$(date): LOCAL task $TASK complete!"
    for i in "${!PIDS[@]}"; do
        wait "${PIDS[$i]}" 2>/dev/null
        EC=$?
        echo "  ${SPLITS[$i]}: exit code $EC"
    done

    echo "  Peak memory: ${PEAK_MEM}G"
    free -h | head -2

    echo "$(date): Moving to next task..."
    sleep 10
done

echo ""
echo "$(date): LOCAL training complete! (forest_half + realmap_half)"
free -h
