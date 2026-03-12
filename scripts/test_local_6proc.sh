#!/bin/bash
# Local test: 6 processes in parallel, 500ep, 200K buffer, no pretrain.
# Tests whether Python 3.10 avoids the segfault seen on remote (Python 3.13).
set -eo pipefail

PROJ="/home/sun/phdproject/dqn/DQN8"
ENV="ros2py310"
CONDA_PREFIX="$HOME/miniconda3"

eval "$($CONDA_PREFIX/bin/conda shell.bash hook)"
conda activate "$ENV"
cd "$PROJ"

echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"

ALGOS=("cnn_pddqn" "cnn_ddqn" "cnn_dqn" "mlp_pddqn" "mlp_ddqn" "mlp_dqn")

get_mem() { free -g | awk 'NR==2{print $3}'; }
get_swap() { free -g | awk 'NR==3{print $3}'; }

echo "$(date): Launching 6 processes (500ep, 200K buffer, no pretrain, Python 3.10)"
echo "  Memory before: mem=$(get_mem)G, swap=$(get_swap)G"

PIDS=()
for ALGO in "${ALGOS[@]}"; do
    PROFILE="test_local_6proc_${ALGO}"
    LOGFILE="$PROJ/runs/${PROFILE}.log"
    python train.py --profile "$PROFILE" > "$LOGFILE" 2>&1 &
    PIDS+=($!)
    echo "  Launched $ALGO PID: ${PIDS[-1]}"
    sleep 2
done

echo ""
echo "$(date): All 6 launched. PIDs: ${PIDS[*]}"

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
free -h
echo ""
echo "Checkpoints:"
find "$PROJ/runs/test_local_6proc" -name "*.pt" 2>/dev/null | wc -l
