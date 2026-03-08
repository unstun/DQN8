#!/bin/bash
# Quality checkpoint screening: 用固定 allsuc pairs 扫描 cnn-pddqn 不同 epoch 的 composite score
# 用法: bash screen_quality_ckpts.sh forest|realmap <ep1> <ep2> ...
# 例如:
#   bash screen_quality_ckpts.sh forest 100 200 300 400 500 600 700 800 900 1000
#   bash screen_quality_ckpts.sh realmap 500 1000 1500 2000 2500 3000 3500 4000 4500 5000

set -e

PROJ="/home/sun/phdproject/dqn/DQN8"
ENV="ros2py310"
ALGO="cnn-pddqn"

ENV_NAME="$1"
shift
EPOCHS="$@"

if [ -z "$ENV_NAME" ] || [ -z "$EPOCHS" ]; then
    echo "Usage: $0 forest|realmap <ep1> <ep2> ..."
    exit 1
fi

if [ "$ENV_NAME" = "forest" ]; then
    TRAIN_DIR="$PROJ/runs/repro_20260226_v14b_1000ep/train_20260227_010647"
    ENV_BASE="forest_a"
    PROFILE="screen_forest_quality_short"
    PAIRS="$PROJ/runs/snapshot_20260305_2cat_v1/pairs/forest_quality_short_allsuc_pairs.json"
    EP_FMT="%05d"
elif [ "$ENV_NAME" = "realmap" ]; then
    TRAIN_DIR="$PROJ/runs/repro_20260228_bug2fix_5000ep/train_20260228_052743"
    ENV_BASE="realmap_a"
    PROFILE="screen_realmap_quality_long"
    PAIRS="$PROJ/runs/snapshot_20260305_2cat_v1/pairs/realmap_quality_long_allsuc_pairs.json"
    EP_FMT="%05d"
elif [ "$ENV_NAME" = "realmap_short" ]; then
    TRAIN_DIR="$PROJ/runs/repro_20260228_bug2fix_5000ep/train_20260228_052743"
    ENV_BASE="realmap_a"
    PROFILE="screen_realmap_quality_short"
    PAIRS="$PROJ/runs/snapshot_20260305_2cat_v1/pairs/realmap_quality_short_allsuc_pairs.json"
    EP_FMT="%05d"
else
    echo "ERROR: ENV_NAME must be 'forest', 'realmap', or 'realmap_short'"
    exit 1
fi

MODEL_DIR="$TRAIN_DIR/models/$ENV_BASE"
CKPT_DIR="$TRAIN_DIR/checkpoints/$ENV_BASE"
SUMMARY="$PROJ/runs/screen_checkpoints/${ENV_NAME//\//_}_quality_screening.csv"

mkdir -p "$PROJ/runs/screen_checkpoints"

echo "=== Quality Checkpoint Screening: $ENV_NAME ==="
echo "Profile: $PROFILE"
echo "Pairs: $PAIRS"
echo "Epochs: $EPOCHS"
echo ""

# Backup current model
cp "$MODEL_DIR/${ALGO}.pt" "$MODEL_DIR/${ALGO}.pt.bak_screen"
echo "Backed up $MODEL_DIR/${ALGO}.pt"

echo "epoch,n_allsuc,cnn_pddqn_composite,cnn_ddqn_composite,cnn_dqn_composite,mlp_pddqn_composite" > "$SUMMARY"

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

    # Run inference with --load-pairs
    conda run --cwd "$PROJ" -n "$ENV" python infer.py \
        --profile "$PROFILE" \
        --load-pairs "$PAIRS" \
        2>&1 | grep -E "^\[|Saved|Error|Traceback|✓|✗|filter|allsuc|Kept" | tail -5

    # Find the latest infer directory
    LATEST_INFER=$(ls -td "$TRAIN_DIR"/infer/202* 2>/dev/null | head -1)

    if [ -z "$LATEST_INFER" ]; then
        echo " ERROR: no infer output found"
        echo "$EP,ERROR,ERROR,ERROR,ERROR,ERROR" >> "$SUMMARY"
        continue
    fi

    # Extract composite scores from table2_kpis_mean_filtered.csv (filter_all_succeed)
    KPI_FILE="$LATEST_INFER/table2_kpis_mean_filtered.csv"

    # Fallback to unfiltered mean if filtered not present
    if [ ! -f "$KPI_FILE" ]; then
        KPI_FILE="$LATEST_INFER/table2_kpis_mean.csv"
    fi

    if [ ! -f "$KPI_FILE" ]; then
        echo " ERROR: KPI file not found at $KPI_FILE"
        echo "$EP,ERROR,ERROR,ERROR,ERROR,ERROR" >> "$SUMMARY"
        continue
    fi

    # Parse composite scores and N_allsuc using Python helper
    PARSED=$(conda run --cwd "$PROJ" -n "$ENV" python3 amr_dqn/utils/parse_quality_kpi.py "$KPI_FILE")

    echo "$EP,$PARSED" >> "$SUMMARY"
    N_A=$(echo "$PARSED" | cut -d',' -f1)
    CNN_PDDQN=$(echo "$PARSED" | cut -d',' -f2)
    CNN_DDQN=$(echo "$PARSED" | cut -d',' -f3)
    CNN_DQN=$(echo "$PARSED" | cut -d',' -f4)
    echo " done -> ep${EPSTR} [n=${N_A}]: pddqn=$CNN_PDDQN | ddqn=$CNN_DDQN | dqn=$CNN_DQN"
done

# Restore original model
cp "$MODEL_DIR/${ALGO}.pt.bak_screen" "$MODEL_DIR/${ALGO}.pt"
rm "$MODEL_DIR/${ALGO}.pt.bak_screen"
echo ""
echo "=== Restored original model ==="
echo ""
echo "=== Screening Summary ==="
cat "$SUMMARY"
