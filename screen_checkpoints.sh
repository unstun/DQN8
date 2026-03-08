#!/bin/bash
# 快速筛选 checkpoint：逐算法测试不同 checkpoint 的 Mode 1 (BK-Long) 成功率
# 用法: bash screen_checkpoints.sh <algo> <ep1> <ep2> ...
# 例如: bash screen_checkpoints.sh cnn-pddqn 500 1000 2000 3000 5000

set -e
PROJ="/home/sun/phdproject/dqn/DQN8"
TRAIN_DIR="$PROJ/runs/repro_20260228_bug2fix_5000ep/train_20260228_052743"
MODEL_DIR="$TRAIN_DIR/models/realmap_a"
CKPT_DIR="$TRAIN_DIR/checkpoints/realmap_a"
PROFILE="screen_realmap_bk_long_20runs"
ENV="ros2py310"

ALGO="$1"
shift
EPOCHS="$@"

if [ -z "$ALGO" ] || [ -z "$EPOCHS" ]; then
    echo "Usage: $0 <algo> <ep1> <ep2> ..."
    exit 1
fi

# Backup current model
cp "$MODEL_DIR/${ALGO}.pt" "$MODEL_DIR/${ALGO}.pt.bak_screen"
echo "=== Screening $ALGO ==="
echo "Backed up current model"

for EP in $EPOCHS; do
    EPSTR=$(printf "%05d" $EP)
    CKPT="$CKPT_DIR/${ALGO}_ep${EPSTR}.pt"

    if [ ! -f "$CKPT" ]; then
        echo "ep${EPSTR}: SKIP (checkpoint not found)"
        continue
    fi

    # Swap model
    cp "$CKPT" "$MODEL_DIR/${ALGO}.pt"

    # Run inference
    echo -n "ep${EPSTR}: running..."
    OUTPUT=$(conda run --cwd "$PROJ" -n "$ENV" python infer.py --profile "$PROFILE" 2>&1)

    # Find the latest infer directory
    LATEST_INFER=$(ls -td "$TRAIN_DIR"/infer/202* 2>/dev/null | head -1)

    if [ -z "$LATEST_INFER" ]; then
        echo " ERROR: no infer output found"
        continue
    fi

    # Extract success rates from table2_kpis.csv
    KPI_FILE="$LATEST_INFER/realmap_a/table2_kpis.csv"
    if [ -f "$KPI_FILE" ]; then
        echo " done -> $LATEST_INFER"
        # Print header once
        if [ "$EP" = "$(echo $EPOCHS | awk '{print $1}')" ]; then
            head -1 "$KPI_FILE" | awk -F',' '{printf "%-15s %-10s\n", $1, $2}'
            echo "----------------------------"
        fi
        # Print all algorithms' success rates
        awk -F',' 'NR>1{printf "%-15s %-10s\n", $1, $2}' "$KPI_FILE"
        echo "----------------------------"
    else
        echo " ERROR: KPI file not found at $KPI_FILE"
    fi
    echo ""
done

# Restore original model
cp "$MODEL_DIR/${ALGO}.pt.bak_screen" "$MODEL_DIR/${ALGO}.pt"
rm "$MODEL_DIR/${ALGO}.pt.bak_screen"
echo "Restored original model for $ALGO"
