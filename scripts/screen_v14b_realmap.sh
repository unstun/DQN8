#!/bin/bash
# =============================================================================
# screen_v14b_realmap.sh — v14b_realmap checkpoint 筛选
#
# 对所有 6 个 RL 算法，每 100 epoch 一个 checkpoint，跑 4 个评估模式。
# 结果归档到 runs/screen_v14b_realmap/realmap/ep<XXXXX>/{sr_long,sr_short,...}.csv
# 最后调用 Python 脚本生成 screen_master.csv + 按算法分文件夹。
#
# 用法:
#   bash scripts/screen_v14b_realmap.sh
#
# 支持断点续跑：如果某 epoch 的 4 个 CSV 都已存在，自动跳过。
# =============================================================================
set -euo pipefail

PROJ="/home/sun/phdproject/dqn/DQN8"
ENV_CONDA="ros2py310"
SCREEN_DIR="$PROJ/runs/screen_v14b_realmap"
SCREEN_RAW="$SCREEN_DIR/_raw"
ALGOS="mlp-dqn mlp-ddqn mlp-pddqn cnn-dqn cnn-ddqn cnn-pddqn"

TRAIN_DIR="$PROJ/runs/v14b_realmap/train_20260307_062153"
ENV_BASE="realmap_a"
MAX_EP=3000
STEP=100

MODEL_DIR="$TRAIN_DIR/models/$ENV_BASE"
CKPT_DIR="$TRAIN_DIR/checkpoints/$ENV_BASE"
OUT_BASE="$SCREEN_DIR/realmap"
MODES=("sr_long" "sr_short" "quality_long" "quality_short")

echo "============================================================"
echo " Screening: v14b_realmap  (ep $STEP .. $MAX_EP, step=$STEP)"
echo " Models:    $MODEL_DIR"
echo " Ckpts:     $CKPT_DIR"
echo " Output:    $OUT_BASE"
echo "============================================================"

# ── 1. 备份当前 models ──
echo "[v14b_realmap] Backing up current models..."
for a in $ALGOS; do
    if [ ! -f "$MODEL_DIR/$a.pt.bak_screen" ]; then
        cp "$MODEL_DIR/$a.pt" "$MODEL_DIR/$a.pt.bak_screen"
    fi
done

TOTAL_EPOCHS=$(( MAX_EP / STEP ))
DONE_EPOCHS=0

for EP in $(seq "$STEP" "$STEP" "$MAX_EP"); do
    EPSTR=$(printf "%05d" "$EP")
    EP_DIR="$OUT_BASE/ep${EPSTR}"
    DONE_EPOCHS=$(( DONE_EPOCHS + 1 ))

    # ── 断点续跑检查 ──
    ALL_DONE=true
    for MODE in "${MODES[@]}"; do
        if [ ! -f "$EP_DIR/${MODE}.csv" ]; then
            ALL_DONE=false
            break
        fi
    done
    if [ "$ALL_DONE" = true ]; then
        echo "[v14b_realmap] ep${EPSTR} ($DONE_EPOCHS/$TOTAL_EPOCHS) — already done, skipping."
        continue
    fi

    echo ""
    echo "[v14b_realmap] ══ Epoch ${EPSTR} ($DONE_EPOCHS/$TOTAL_EPOCHS) ══"

    # ── 2. 换模型 ──
    SKIP_EPOCH=false
    for a in $ALGOS; do
        CKPT="$CKPT_DIR/${a}_ep${EPSTR}.pt"
        if [ ! -f "$CKPT" ]; then
            echo "  WARNING: $CKPT not found — skipping epoch $EPSTR"
            SKIP_EPOCH=true
            break
        fi
        cp "$CKPT" "$MODEL_DIR/$a.pt"
    done
    if [ "$SKIP_EPOCH" = true ]; then
        continue
    fi

    mkdir -p "$EP_DIR"

    # ── 3. 跑 4 个模式 ──
    for MODE in "${MODES[@]}"; do
        # 如果该模式已跑完，跳过
        if [ -f "$EP_DIR/${MODE}.csv" ]; then
            echo "  $MODE: already done, skipping."
            continue
        fi

        PROFILE="screen_v14b_realmap_${MODE}"
        echo -n "  $MODE: running... "

        # 运行推理
        if ! conda run --cwd "$PROJ" -n "$ENV_CONDA" \
            python infer.py --profile "$PROFILE" > /dev/null 2>&1; then
            echo "FAILED (inference error)"
            continue
        fi

        # 获取最新输出目录
        LATEST_NAME=$(cat "$PROJ/runs/screen_v14b_realmap/latest.txt" 2>/dev/null || echo "")
        LATEST_DIR="$PROJ/runs/screen_v14b_realmap/$LATEST_NAME"

        if [ -f "$LATEST_DIR/table2_kpis_mean_raw.csv" ]; then
            cp "$LATEST_DIR/table2_kpis_mean_raw.csv" "$EP_DIR/${MODE}.csv"

            # 把原始输出移到 _raw 以保持顶层干净
            mkdir -p "$SCREEN_RAW"
            mv "$LATEST_DIR" "$SCREEN_RAW/realmap_ep${EPSTR}_${MODE}" 2>/dev/null || true

            echo "OK"
        else
            echo "ERROR: no table2_kpis_mean_raw.csv in $LATEST_DIR"
        fi
    done
done

# ── 4. 恢复原始模型 ──
echo ""
echo "[v14b_realmap] Restoring original models..."
for a in $ALGOS; do
    if [ -f "$MODEL_DIR/$a.pt.bak_screen" ]; then
        cp "$MODEL_DIR/$a.pt.bak_screen" "$MODEL_DIR/$a.pt"
        rm "$MODEL_DIR/$a.pt.bak_screen"
    fi
done

echo "[v14b_realmap] Done. Results in: $OUT_BASE/"

# ── 5. 构建 screen_master.csv + 按算法分文件夹 ──
echo ""
echo "Building screen_master.csv ..."
conda run --cwd "$PROJ" -n "$ENV_CONDA" \
    python scripts/build_screen_master.py "$SCREEN_DIR"

echo ""
echo "All done! Master CSV: $SCREEN_DIR/screen_master.csv"
