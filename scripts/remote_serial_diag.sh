#!/bin/bash
# ============================================================
# REMOTE serial training: forest_diag + realmap_diag
# Machine: ubuntu-zt (RTX 5070 Ti, Python 3.10, 30G RAM)
# Margin: diag (sqrt(2)/2 * cell_size)
# Strategy: 1 process at a time (serial) to avoid Blackwell segfault
# Tasks: forest_diag -> realmap_diag (sequential)
# Each task: 3 configs x 2 algos = 6 algos, run one config at a time
# ============================================================
set -eo pipefail

PROJ="/home/sun/phdproject/dqn/DQN8"
ENV="dqn_py310"
LOG_DIR="$PROJ/runs"
CONDA_PREFIX="$HOME/miniconda3"

eval "$($CONDA_PREFIX/bin/conda shell.bash hook)"
conda activate "$ENV"
cd "$PROJ"

echo "============================================"
echo "REMOTE serial training (diag margin)"
echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null)"
echo "============================================"

TASKS=("forest_diag" "realmap_diag")
SPLITS=("pddqn" "ddqn" "dqn")

get_mem() { free -g | awk 'NR==2{print $3}'; }
get_swap() { free -g | awk 'NR==3{print $3}'; }

echo "$(date): Starting 2-task REMOTE training (diag margin, serial 1 process at a time)"
free -h | head -2

TOTAL_START=$(date +%s)

for TASK in "${TASKS[@]}"; do
    echo ""
    echo "$(date): ======== REMOTE task: $TASK (serial, 3 configs x 2 algos) ========"

    TASK_START=$(date +%s)

    for SPLIT in "${SPLITS[@]}"; do
        PROFILE="repro_20260312_${TASK}_${SPLIT}"
        LOGFILE="$LOG_DIR/${PROFILE}.log"

        echo ""
        echo "  $(date): Starting $PROFILE (2 algos) ..."
        echo "  Memory: mem=$(get_mem)G, swap=$(get_swap)G"

        python train.py --profile "$PROFILE" > "$LOGFILE" 2>&1
        EC=$?

        ELAPSED=$(( $(date +%s) - TASK_START ))
        ELAPSED_MIN=$(( ELAPSED / 60 ))
        echo "  $(date): $PROFILE finished! exit code=$EC, task elapsed=${ELAPSED_MIN}min"
        echo "  Memory: mem=$(get_mem)G, swap=$(get_swap)G"

        if [ "$EC" -ne 0 ]; then
            echo "  WARNING: $PROFILE failed with exit code $EC"
            echo "  Last 5 lines of log:"
            tail -5 "$LOGFILE" 2>/dev/null | sed 's/^/    /'
        fi

        # Brief pause between processes
        sleep 5
    done

    TASK_ELAPSED=$(( $(date +%s) - TASK_START ))
    TASK_HOURS=$(echo "scale=1; $TASK_ELAPSED / 3600" | bc)
    echo ""
    echo "$(date): REMOTE task $TASK complete! (${TASK_HOURS}h)"
    free -h | head -2

    sleep 10
done

TOTAL_ELAPSED=$(( $(date +%s) - TOTAL_START ))
TOTAL_HOURS=$(echo "scale=1; $TOTAL_ELAPSED / 3600" | bc)

echo ""
echo "$(date): REMOTE training complete! (forest_diag + realmap_diag)"
echo "Total time: ${TOTAL_HOURS}h"
free -h
echo ""
echo "Checkpoints produced:"
for TASK in "${TASKS[@]}"; do
    TASK_SHORT="${TASK//_diag/}"
    OUT="algo6_10k_${TASK}"
    COUNT=$(find "$PROJ/runs/$OUT" -name "*.pt" 2>/dev/null | wc -l)
    echo "  $OUT: $COUNT checkpoints"
done
