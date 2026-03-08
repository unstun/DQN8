#!/bin/bash
# SR checkpoint screening: 扫描 cnn-pddqn 不同 epoch 的 BK-Long/Short 成功率
# 用法: bash screen_sr_ckpts.sh bk_long|bk_short <ep1> <ep2> ...
# 例如:
#   bash screen_sr_ckpts.sh bk_long 1000 1500 2000 2500 3000 3500 4000 4500 5000
#   bash screen_sr_ckpts.sh bk_short 1000 1500 2000 2500 3000 3500 4000 4500 5000

set -e

PROJ="/home/sun/phdproject/dqn/DQN8"
ENV="ros2py310"
ALGO="cnn-pddqn"
TRAIN_DIR="$PROJ/runs/repro_20260228_bug2fix_5000ep/train_20260228_052743"
ENV_BASE="realmap_a"
MODEL_DIR="$TRAIN_DIR/models/$ENV_BASE"
CKPT_DIR="$TRAIN_DIR/checkpoints/$ENV_BASE"
EP_FMT="%05d"

MODE="$1"
shift
EPOCHS="$@"

if [ -z "$MODE" ] || [ -z "$EPOCHS" ]; then
    echo "Usage: $0 bk_long|bk_short <ep1> <ep2> ..."
    exit 1
fi

if [ "$MODE" = "bk_long" ]; then
    PROFILE="screen_realmap_bk_long_20runs"
elif [ "$MODE" = "bk_short" ]; then
    PROFILE="screen_realmap_bk_short_20runs"
else
    echo "ERROR: MODE must be 'bk_long' or 'bk_short'"
    exit 1
fi

SUMMARY="$PROJ/runs/screen_checkpoints/realmap_${MODE}_sr_screening.csv"
mkdir -p "$PROJ/runs/screen_checkpoints"

echo "=== SR Checkpoint Screening: realmap $MODE ==="
echo "Profile: $PROFILE"
echo "Epochs: $EPOCHS"
echo ""

# Backup current model
cp "$MODEL_DIR/${ALGO}.pt" "$MODEL_DIR/${ALGO}.pt.bak_sr_screen"
echo "Backed up $MODEL_DIR/${ALGO}.pt"

echo "epoch,cnn_pddqn_sr,cnn_ddqn_sr,cnn_dqn_sr,mlp_pddqn_sr,mlp_ddqn_sr,mlp_dqn_sr" > "$SUMMARY"

for EP in $EPOCHS; do
    EPSTR=$(printf "$EP_FMT" $EP)
    CKPT="$CKPT_DIR/${ALGO}_ep${EPSTR}.pt"

    if [ ! -f "$CKPT" ]; then
        echo "ep${EPSTR}: SKIP (checkpoint not found: $CKPT)"
        continue
    fi

    # Swap model
    cp "$CKPT" "$MODEL_DIR/${ALGO}.pt"
    echo -n "ep${EPSTR}: running..."

    # Run inference
    conda run --cwd "$PROJ" -n "$ENV" python infer.py \
        --profile "$PROFILE" \
        2>&1 | grep -E "^\[|Saved|Error|Traceback|✓|✗|allsuc|n_runs" | tail -3

    # Find the latest infer directory
    LATEST_INFER=$(ls -td "$TRAIN_DIR"/infer/202* 2>/dev/null | head -1)

    if [ -z "$LATEST_INFER" ]; then
        echo " ERROR: no infer output found"
        echo "$EP,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR" >> "$SUMMARY"
        continue
    fi

    # Parse SR from table2_kpis_mean.csv
    KPI_FILE="$LATEST_INFER/table2_kpis_mean.csv"

    if [ ! -f "$KPI_FILE" ]; then
        echo " ERROR: KPI file not found at $KPI_FILE"
        echo "$EP,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR" >> "$SUMMARY"
        continue
    fi

    PARSED=$(conda run --cwd "$PROJ" -n "$ENV" python3 amr_dqn/utils/parse_sr_kpi.py "$KPI_FILE")

    echo "$EP,$PARSED" >> "$SUMMARY"
    CNN_PDDQN=$(echo "$PARSED" | cut -d',' -f1)
    CNN_DDQN=$(echo "$PARSED" | cut -d',' -f2)
    CNN_DQN=$(echo "$PARSED" | cut -d',' -f3)
    MLP_PDDQN=$(echo "$PARSED" | cut -d',' -f4)
    echo " done -> ep${EPSTR}: pddqn=$CNN_PDDQN | ddqn=$CNN_DDQN | dqn=$CNN_DQN | mlp_pddqn=$MLP_PDDQN"
done

# Restore original model
cp "$MODEL_DIR/${ALGO}.pt.bak_sr_screen" "$MODEL_DIR/${ALGO}.pt"
rm "$MODEL_DIR/${ALGO}.pt.bak_sr_screen"
echo ""
echo "=== Restored original model ==="
echo ""
echo "=== SR Screening Summary ==="
cat "$SUMMARY"
