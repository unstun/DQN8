#!/bin/bash
# Screen 6 DRL algos × 10000ep (every 100ep) on Forest, sr_long + sr_short
set -euo pipefail

PROJ="/home/sun/phdproject/dqn/DQN8"
ENV_CONDA="ros2py310"
SCREEN_DIR="$PROJ/runs/screen_6algo_10k_forest"
SCREEN_RAW="$SCREEN_DIR/_raw"
mkdir -p "$SCREEN_RAW"

TRAIN_DIR="$PROJ/runs/algo6_10k_forest"
# Use train_merged if available, otherwise latest train dir
if [ -d "$TRAIN_DIR/train_merged" ]; then
    TRAIN_SUB="$TRAIN_DIR/train_merged"
else
    TRAIN_SUB=$(ls -d "$TRAIN_DIR"/train_* 2>/dev/null | sort | tail -1)
fi
if [ -z "$TRAIN_SUB" ]; then
    echo "ERROR: No train directory found under $TRAIN_DIR"
    exit 1
fi
MODEL_DIR="$TRAIN_SUB/models/forest_a"
CKPT_DIR="$TRAIN_SUB/checkpoints/forest_a"

ALGOS=("cnn-pddqn" "cnn-ddqn" "cnn-dqn" "mlp-pddqn" "mlp-ddqn" "mlp-dqn")
MODES=("sr_long" "sr_short")

# Backup original models
for ALGO in "${ALGOS[@]}"; do
    if [ -f "$MODEL_DIR/$ALGO.pt" ]; then
        cp "$MODEL_DIR/$ALGO.pt" "$MODEL_DIR/$ALGO.pt.bak_screen"
    fi
done

TOTAL=0
DONE=0

for EP in $(seq 100 100 10000); do
    EPSTR=$(printf "%05d" "$EP")
    EP_DIR="$SCREEN_DIR/forest/ep${EPSTR}"

    # Check if all modes already done for this epoch
    ALL_DONE=true
    for MODE in "${MODES[@]}"; do
        if [ ! -f "$EP_DIR/${MODE}.csv" ]; then
            ALL_DONE=false
            break
        fi
    done
    if [ "$ALL_DONE" = true ]; then
        ((DONE+=2))
        ((TOTAL+=2))
        continue
    fi

    # Check all 6 checkpoints exist for this epoch
    MISSING=false
    for ALGO in "${ALGOS[@]}"; do
        CKPT="$CKPT_DIR/${ALGO}_ep${EPSTR}.pt"
        if [ ! -f "$CKPT" ]; then
            echo "ep${EPSTR}: ${ALGO} checkpoint not found, skip epoch"
            MISSING=true
            break
        fi
    done
    if [ "$MISSING" = true ]; then
        continue
    fi

    # Copy all 6 checkpoints into models dir
    for ALGO in "${ALGOS[@]}"; do
        cp "$CKPT_DIR/${ALGO}_ep${EPSTR}.pt" "$MODEL_DIR/$ALGO.pt"
    done

    mkdir -p "$EP_DIR"

    for MODE in "${MODES[@]}"; do
        ((TOTAL+=1))
        if [ -f "$EP_DIR/${MODE}.csv" ]; then
            ((DONE+=1))
            continue
        fi

        PROFILE="screen_6algo10k_forest_${MODE}"
        echo -n "ep${EPSTR} ${MODE} [${DONE}/${TOTAL}]: "

        if ! conda run --cwd "$PROJ" -n "$ENV_CONDA" \
            python infer.py --profile "$PROFILE" > /dev/null 2>&1; then
            echo "FAILED"
            continue
        fi

        LATEST_INFER=$(cat "$PROJ/runs/screen_6algo_10k_forest/latest.txt" 2>/dev/null || echo "")
        LATEST_DIR="$PROJ/runs/screen_6algo_10k_forest/$LATEST_INFER"

        if [ -f "$LATEST_DIR/table2_kpis_mean_raw.csv" ]; then
            cp "$LATEST_DIR/table2_kpis_mean_raw.csv" "$EP_DIR/${MODE}.csv"
            mv "$LATEST_DIR" "$SCREEN_RAW/forest_ep${EPSTR}_${MODE}" 2>/dev/null || true
            ((DONE+=1))
            echo "OK"
        else
            echo "ERROR: no kpis csv"
        fi
    done
done

# Restore original models
for ALGO in "${ALGOS[@]}"; do
    if [ -f "$MODEL_DIR/$ALGO.pt.bak_screen" ]; then
        cp "$MODEL_DIR/$ALGO.pt.bak_screen" "$MODEL_DIR/$ALGO.pt"
        rm "$MODEL_DIR/$ALGO.pt.bak_screen"
    fi
done

echo ""
echo "Screen complete: $(date)"
echo "Total done: ${DONE}/${TOTAL}"
echo "Raw results: $(ls $SCREEN_RAW/ 2>/dev/null | wc -l)"
