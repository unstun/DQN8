#!/bin/bash
# Resume screening for remaining checkpoints (ep09400-ep10000)
set -euo pipefail

PROJ="/home/sun/phdproject/dqn/DQN8"
ENV_CONDA="ros2py310"
SCREEN_DIR="$PROJ/runs/screen_pddqn10k_realmap"
SCREEN_RAW="$SCREEN_DIR/_raw"
ALGO="cnn-pddqn"
ENV_BASE="realmap_a"

TRAIN_DIR="$PROJ/runs/pddqn10k_realmap/train_20260308_073353"
MODEL_DIR="$TRAIN_DIR/models/$ENV_BASE"
CKPT_DIR="$TRAIN_DIR/checkpoints/$ENV_BASE"
MODES=("sr_long" "sr_short")

echo "Resuming screen from remaining checkpoints..."
echo "Model dir: $MODEL_DIR"
echo "Ckpt dir: $CKPT_DIR"

cp "$MODEL_DIR/$ALGO.pt" "$MODEL_DIR/$ALGO.pt.bak_resume"

for EP in $(seq 100 100 10000); do
    EPSTR=$(printf "%05d" "$EP")
    EP_DIR="$SCREEN_DIR/realmap/ep${EPSTR}"

    ALL_DONE=true
    for MODE in "${MODES[@]}"; do
        if [ ! -f "$EP_DIR/${MODE}.csv" ]; then
            ALL_DONE=false
            break
        fi
    done
    if [ "$ALL_DONE" = true ]; then
        continue
    fi

    CKPT="$CKPT_DIR/${ALGO}_ep${EPSTR}.pt"
    if [ ! -f "$CKPT" ]; then
        echo "ep${EPSTR}: checkpoint not found, skip"
        continue
    fi

    cp "$CKPT" "$MODEL_DIR/$ALGO.pt"
    mkdir -p "$EP_DIR"

    for MODE in "${MODES[@]}"; do
        if [ -f "$EP_DIR/${MODE}.csv" ]; then
            continue
        fi

        PROFILE="screen_pddqn10k_${MODE}"
        echo -n "ep${EPSTR} ${MODE}: "

        if ! conda run --cwd "$PROJ" -n "$ENV_CONDA" \
            python infer.py --profile "$PROFILE" > /dev/null 2>&1; then
            echo "FAILED"
            continue
        fi

        LATEST_INFER=$(cat "$PROJ/runs/screen_pddqn10k_realmap/latest.txt" 2>/dev/null || echo "")
        LATEST_DIR="$PROJ/runs/screen_pddqn10k_realmap/$LATEST_INFER"

        if [ -f "$LATEST_DIR/table2_kpis_mean_raw.csv" ]; then
            cp "$LATEST_DIR/table2_kpis_mean_raw.csv" "$EP_DIR/${MODE}.csv"
            mv "$LATEST_DIR" "$SCREEN_RAW/realmap_ep${EPSTR}_${MODE}" 2>/dev/null || true
            echo "OK"
        else
            echo "ERROR"
        fi
    done
done

# Restore
if [ -f "$MODEL_DIR/$ALGO.pt.bak_resume" ]; then
    cp "$MODEL_DIR/$ALGO.pt.bak_resume" "$MODEL_DIR/$ALGO.pt"
    rm "$MODEL_DIR/$ALGO.pt.bak_resume"
fi

echo ""
echo "Screen resume complete: $(date)"
echo "Total raw results: $(ls $SCREEN_RAW/ | wc -l)"
