#!/bin/bash
# =============================================================================
# train_and_screen_pddqn10k.sh
#
# Phase 1: Train CNN-PDDQN only for 10000 episodes on realmap_a
# Phase 2: Screen every 100 epochs (sr_long + sr_short, 100 runs each)
#
# Usage:
#   nohup bash scripts/train_and_screen_pddqn10k.sh \
#     > runs/pddqn10k_$(date +%Y%m%d_%H%M%S).log 2>&1 &
# =============================================================================
set -euo pipefail

PROJ="/home/sun/phdproject/dqn/DQN8"
ENV_CONDA="ros2py310"
SCREEN_DIR="$PROJ/runs/screen_pddqn10k_realmap"
SCREEN_RAW="$SCREEN_DIR/_raw"
ALGO="cnn-pddqn"
ENV_BASE="realmap_a"
MAX_EP=10000
STEP=100
MODES=("sr_long" "sr_short")

echo "============================================================"
echo " CNN-PDDQN 10k: Train + Screen"
echo " Started: $(date)"
echo "============================================================"

# ══════════════════════════════════════════════════════════════════
# Phase 1: Training
# ══════════════════════════════════════════════════════════════════
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Phase 1: Training CNN-PDDQN × 10000 episodes          ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

conda run --cwd "$PROJ" -n "$ENV_CONDA" \
    python train.py --profile repro_20260308_pddqn10k_realmap

echo ""
echo "[Phase 1] Training complete: $(date)"

# ── Find training output directory ──
EXPERIMENT_DIR="$PROJ/runs/pddqn10k_realmap"
LATEST_TRAIN=$(cat "$EXPERIMENT_DIR/latest.txt")
TRAIN_DIR="$EXPERIMENT_DIR/$LATEST_TRAIN"
MODEL_DIR="$TRAIN_DIR/models/$ENV_BASE"
CKPT_DIR="$TRAIN_DIR/checkpoints/$ENV_BASE"

echo "[Phase 1] Train dir: $TRAIN_DIR"
echo "[Phase 1] Checkpoints: $CKPT_DIR"

if [ ! -d "$CKPT_DIR" ]; then
    echo "ERROR: Checkpoint directory not found!"
    exit 1
fi

# ══════════════════════════════════════════════════════════════════
# Phase 2: Screening
# ══════════════════════════════════════════════════════════════════
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Phase 2: Screening 100 checkpoints × 2 modes          ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# Backup current model
if [ -f "$MODEL_DIR/$ALGO.pt" ]; then
    cp "$MODEL_DIR/$ALGO.pt" "$MODEL_DIR/$ALGO.pt.bak_screen"
fi

mkdir -p "$SCREEN_DIR" "$SCREEN_RAW"

TOTAL_EPOCHS=$(( MAX_EP / STEP ))
DONE=0

for EP in $(seq "$STEP" "$STEP" "$MAX_EP"); do
    EPSTR=$(printf "%05d" "$EP")
    EP_DIR="$SCREEN_DIR/realmap/ep${EPSTR}"
    DONE=$(( DONE + 1 ))

    # ── Skip if already done ──
    ALL_DONE=true
    for MODE in "${MODES[@]}"; do
        if [ ! -f "$EP_DIR/${MODE}.csv" ]; then
            ALL_DONE=false
            break
        fi
    done
    if [ "$ALL_DONE" = true ]; then
        echo "[Screen] ep${EPSTR} ($DONE/$TOTAL_EPOCHS) — already done, skip."
        continue
    fi

    # ── Check checkpoint exists ──
    CKPT="$CKPT_DIR/${ALGO}_ep${EPSTR}.pt"
    if [ ! -f "$CKPT" ]; then
        echo "[Screen] ep${EPSTR} ($DONE/$TOTAL_EPOCHS) — checkpoint not found, skip."
        continue
    fi

    # ── Swap model ──
    cp "$CKPT" "$MODEL_DIR/$ALGO.pt"

    mkdir -p "$EP_DIR"

    for MODE in "${MODES[@]}"; do
        if [ -f "$EP_DIR/${MODE}.csv" ]; then
            continue
        fi

        PROFILE="screen_pddqn10k_${MODE}"
        echo -n "[Screen] ep${EPSTR} ($DONE/$TOTAL_EPOCHS) ${MODE}: "

        if ! conda run --cwd "$PROJ" -n "$ENV_CONDA" \
            python infer.py --profile "$PROFILE" > /dev/null 2>&1; then
            echo "FAILED"
            continue
        fi

        # ── Collect results ──
        LATEST_INFER=$(cat "$PROJ/runs/screen_pddqn10k_realmap/latest.txt" 2>/dev/null || echo "")
        LATEST_DIR="$PROJ/runs/screen_pddqn10k_realmap/$LATEST_INFER"

        if [ -f "$LATEST_DIR/table2_kpis_mean_raw.csv" ]; then
            cp "$LATEST_DIR/table2_kpis_mean_raw.csv" "$EP_DIR/${MODE}.csv"
            # Move raw output (has per-pair table2_kpis_raw.csv)
            mv "$LATEST_DIR" "$SCREEN_RAW/realmap_ep${EPSTR}_${MODE}" 2>/dev/null || true
            echo "OK"
        else
            echo "ERROR (no output CSV)"
        fi
    done
done

# ── Restore original model ──
if [ -f "$MODEL_DIR/$ALGO.pt.bak_screen" ]; then
    cp "$MODEL_DIR/$ALGO.pt.bak_screen" "$MODEL_DIR/$ALGO.pt"
    rm "$MODEL_DIR/$ALGO.pt.bak_screen"
fi

echo ""
echo "============================================================"
echo " All done: $(date)"
echo " Screen results: $SCREEN_DIR/"
echo " Raw per-pair data: $SCREEN_RAW/"
echo "============================================================"
