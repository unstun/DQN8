#!/bin/bash
# Short test: staggered launch of 6 algo processes for forest_diag.
# Each process waits for the previous one to finish pretraining (first checkpoint)
# before launching the next. This avoids memory spikes from concurrent initialization.
# Uses 50 episodes + 1M buffer to test memory safety.
set -eo pipefail

PROJ="/home/sun/phdproject/dqn/DQN8"
ENV="ros2py310"
LOG_DIR="$PROJ/runs"
CONDA_PREFIX="$HOME/miniconda3"

eval "$($CONDA_PREFIX/bin/conda shell.bash hook)"
conda activate "$ENV"
cd "$PROJ"

ALGOS=("cnn_pddqn" "cnn_ddqn" "cnn_dqn" "mlp_pddqn" "mlp_ddqn" "mlp_dqn")
OUT_DIR="test_staggered_forest_diag"

get_mem() { free -g | awk 'NR==2{print $3}'; }
get_swap() { free -g | awk 'NR==3{print $3}'; }

echo "$(date): Starting staggered 6-process test (50ep, 1M buffer)"
free -h

PIDS=()

for i in "${!ALGOS[@]}"; do
    ALGO="${ALGOS[$i]}"
    ALGO_DASH="${ALGO//_/-}"
    PROFILE="test_staggered_forest_diag_${ALGO}"
    LOGFILE="$LOG_DIR/${PROFILE}.log"

    echo ""
    echo "$(date): Launching process $((i+1))/6: $ALGO"
    echo "  Memory before launch: mem=$(get_mem)G, swap=$(get_swap)G"

    python train.py --profile "$PROFILE" > "$LOGFILE" 2>&1 &
    PIDS+=($!)
    echo "  PID: ${PIDS[$i]}"

    # Wait for this process to produce its first checkpoint OR finish
    echo "  Waiting for first checkpoint from $ALGO_DASH ..."

    WAIT_START=$(date +%s)
    TIMEOUT=1800  # 30 min max wait per process
    while true; do
        sleep 10

        # Check for any checkpoint file from this algo (use find to avoid glob exit code issues)
        CKPT_COUNT=$(find "$PROJ/runs/$OUT_DIR" -name "${ALGO_DASH}_ep*.pt" 2>/dev/null | wc -l)
        if [ "$CKPT_COUNT" -gt 0 ]; then
            ELAPSED=$(( $(date +%s) - WAIT_START ))
            echo "  Checkpoint found after ${ELAPSED}s ($CKPT_COUNT files). mem=$(get_mem)G, swap=$(get_swap)G"
            break
        fi

        # Check if process died (after checkpoint check, so we catch fast completions)
        if ! kill -0 "${PIDS[$i]}" 2>/dev/null; then
            echo "  Process $ALGO finished/died. mem=$(get_mem)G, swap=$(get_swap)G"
            break
        fi

        # Timeout check
        ELAPSED=$(( $(date +%s) - WAIT_START ))
        if [ "$ELAPSED" -gt "$TIMEOUT" ]; then
            echo "  TIMEOUT: No checkpoint after ${TIMEOUT}s, launching next anyway"
            break
        fi
    done
done

echo ""
echo "$(date): All 6 processes launched. Waiting for completion..."
free -h

# Monitor until all done
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

    echo "  $(date): $ALIVE/6 alive, mem=$(get_mem)G, swap=$(get_swap)G"
    sleep 60
done

echo ""
echo "$(date): All processes finished!"
for i in "${!PIDS[@]}"; do
    wait "${PIDS[$i]}" 2>/dev/null
    EC=$?
    echo "  ${ALGOS[$i]}: exit code $EC"
done

echo ""
echo "Final memory:"
free -h

echo ""
echo "Checkpoints produced:"
find "$PROJ/runs/$OUT_DIR" -name "*.pt" 2>/dev/null | wc -l
