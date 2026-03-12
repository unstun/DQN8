#!/bin/bash
# Test: launch 6 algo processes simultaneously (no pretrain) to measure parallel memory usage.
# 500 episodes, 1M buffer, forest_diag.
set -eo pipefail

PROJ="/home/sun/phdproject/dqn/DQN8"
ENV="ros2py310"
LOG_DIR="$PROJ/runs"
CONDA_PREFIX="$HOME/miniconda3"

eval "$($CONDA_PREFIX/bin/conda shell.bash hook)"
conda activate "$ENV"
cd "$PROJ"

ALGOS=("cnn_pddqn" "cnn_ddqn" "cnn_dqn" "mlp_pddqn" "mlp_ddqn" "mlp_dqn")

get_mem() { free -g | awk 'NR==2{print $3}'; }
get_swap() { free -g | awk 'NR==3{print $3}'; }

echo "$(date): Launching 6 processes simultaneously (500ep, 1M buffer, no pretrain)"
echo "  Memory before: mem=$(get_mem)G, swap=$(get_swap)G"
free -h

PIDS=()
for ALGO in "${ALGOS[@]}"; do
    PROFILE="test_staggered_forest_diag_${ALGO}"
    LOGFILE="$LOG_DIR/${PROFILE}.log"
    python train.py --profile "$PROFILE" > "$LOGFILE" 2>&1 &
    PIDS+=($!)
    echo "  Launched $ALGO PID: ${PIDS[-1]}"
    sleep 2
done

echo ""
echo "$(date): All 6 launched. PIDs: ${PIDS[*]}"
echo "  Memory after launch: mem=$(get_mem)G, swap=$(get_swap)G"

# Track peak memory
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
    echo "  $(date): $ALIVE/6 alive, mem=${MEM}G, swap=${SWAP}G (peak=${PEAK_MEM}G)"
    sleep 30
done

echo ""
echo "$(date): All processes finished!"
for i in "${!PIDS[@]}"; do
    wait "${PIDS[$i]}" 2>/dev/null
    EC=$?
    echo "  ${ALGOS[$i]}: exit code $EC"
done

echo ""
echo "Peak memory: ${PEAK_MEM}G"
echo "Final memory:"
free -h

echo ""
echo "Checkpoints:"
find "$PROJ/runs/test_staggered_forest_diag" -name "*.pt" 2>/dev/null | wc -l
