#!/bin/bash
# =============================================================================
# screen_all_ckpts.sh — 全量 checkpoint 筛选
#
# 对所有 6 个 RL 算法，每 100 epoch 一个 checkpoint，跑 4 个评估模式。
# 结果归档到 runs/screen_20260306/<env>/ep<XXXXX>/{sr_long,sr_short,...}.csv
# 最后调用 Python 脚本生成 screen_master.csv + 按算法分文件夹。
#
# 用法:
#   bash scripts/screen_all_ckpts.sh forest          # 只跑 Forest
#   bash scripts/screen_all_ckpts.sh realmap          # 只跑 Realmap
#   bash scripts/screen_all_ckpts.sh all              # 两个都跑
#
# 支持断点续跑：如果某 epoch 的 4 个 CSV 都已存在，自动跳过。
# =============================================================================
set -euo pipefail

PROJ="/home/sun/phdproject/dqn/DQN8"
ENV_CONDA="ros2py310"
SCREEN_DIR="$PROJ/runs/screen_20260306"
SCREEN_RAW="$SCREEN_DIR/_raw"       # 原始 infer 输出暂存，跑完可删
ALGOS="mlp-dqn mlp-ddqn mlp-pddqn cnn-dqn cnn-ddqn cnn-pddqn"

run_env() {
    local ENV_NAME="$1"
    local TRAIN_DIR ENV_BASE MAX_EP STEP

    if [ "$ENV_NAME" = "forest" ]; then
        TRAIN_DIR="$PROJ/runs/repro_20260226_v14b_1000ep/train_20260227_010647"
        ENV_BASE="forest_a"
        MAX_EP=1000
        STEP=100
    elif [ "$ENV_NAME" = "realmap" ]; then
        TRAIN_DIR="$PROJ/runs/repro_20260228_bug2fix_5000ep/train_20260228_052743"
        ENV_BASE="realmap_a"
        MAX_EP=5000
        STEP=100
    else
        echo "ERROR: Unknown env '$ENV_NAME'. Use: forest | realmap | all"
        exit 1
    fi

    local MODEL_DIR="$TRAIN_DIR/models/$ENV_BASE"
    local CKPT_DIR="$TRAIN_DIR/checkpoints/$ENV_BASE"
    local OUT_BASE="$SCREEN_DIR/$ENV_NAME"
    local MODES=("sr_long" "sr_short" "quality_long" "quality_short")

    echo "============================================================"
    echo " Screening: $ENV_NAME  (ep $STEP .. $MAX_EP, step=$STEP)"
    echo " Models:    $MODEL_DIR"
    echo " Ckpts:     $CKPT_DIR"
    echo " Output:    $OUT_BASE"
    echo "============================================================"

    # ── 1. 备份当前 models ──
    echo "[$ENV_NAME] Backing up current models..."
    for a in $ALGOS; do
        if [ ! -f "$MODEL_DIR/$a.pt.bak_screen" ]; then
            cp "$MODEL_DIR/$a.pt" "$MODEL_DIR/$a.pt.bak_screen"
        fi
    done

    local TOTAL_EPOCHS=$(( MAX_EP / STEP ))
    local DONE_EPOCHS=0

    for EP in $(seq "$STEP" "$STEP" "$MAX_EP"); do
        local EPSTR
        EPSTR=$(printf "%05d" "$EP")
        local EP_DIR="$OUT_BASE/ep${EPSTR}"
        DONE_EPOCHS=$(( DONE_EPOCHS + 1 ))

        # ── 断点续跑检查 ──
        local ALL_DONE=true
        for MODE in "${MODES[@]}"; do
            if [ ! -f "$EP_DIR/${MODE}.csv" ]; then
                ALL_DONE=false
                break
            fi
        done
        if [ "$ALL_DONE" = true ]; then
            echo "[$ENV_NAME] ep${EPSTR} ($DONE_EPOCHS/$TOTAL_EPOCHS) — already done, skipping."
            continue
        fi

        echo ""
        echo "[$ENV_NAME] ══ Epoch ${EPSTR} ($DONE_EPOCHS/$TOTAL_EPOCHS) ══"

        # ── 2. 换模型 ──
        local SKIP_EPOCH=false
        for a in $ALGOS; do
            local CKPT="$CKPT_DIR/${a}_ep${EPSTR}.pt"
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

            local PROFILE="screen_${ENV_NAME}_${MODE}"
            echo -n "  $MODE: running... "

            # 运行推理
            if ! conda run --cwd "$PROJ" -n "$ENV_CONDA" \
                python infer.py --profile "$PROFILE" > /dev/null 2>&1; then
                echo "FAILED (inference error)"
                continue
            fi

            # 获取最新输出目录
            local LATEST_NAME
            LATEST_NAME=$(cat "$PROJ/runs/screen_20260306/latest.txt" 2>/dev/null || echo "")
            local LATEST_DIR="$PROJ/runs/screen_20260306/$LATEST_NAME"

            if [ -f "$LATEST_DIR/table2_kpis_mean_raw.csv" ]; then
                cp "$LATEST_DIR/table2_kpis_mean_raw.csv" "$EP_DIR/${MODE}.csv"

                # 把原始输出移到 _raw 以保持顶层干净
                mkdir -p "$SCREEN_RAW"
                mv "$LATEST_DIR" "$SCREEN_RAW/${ENV_NAME}_ep${EPSTR}_${MODE}" 2>/dev/null || true

                echo "OK"
            else
                echo "ERROR: no table2_kpis_mean_raw.csv in $LATEST_DIR"
            fi
        done
    done

    # ── 4. 恢复原始模型 ──
    echo ""
    echo "[$ENV_NAME] Restoring original models..."
    for a in $ALGOS; do
        if [ -f "$MODEL_DIR/$a.pt.bak_screen" ]; then
            cp "$MODEL_DIR/$a.pt.bak_screen" "$MODEL_DIR/$a.pt"
            rm "$MODEL_DIR/$a.pt.bak_screen"
        fi
    done

    echo "[$ENV_NAME] Done. Results in: $OUT_BASE/"
}

# ── 主入口 ──
case "${1:-all}" in
    forest)
        run_env forest
        ;;
    realmap)
        run_env realmap
        ;;
    all)
        run_env forest
        run_env realmap
        ;;
    *)
        echo "Usage: $0 {forest|realmap|all}"
        exit 1
        ;;
esac

# ── 5. 构建 screen_master.csv + 按算法分文件夹 ──
echo ""
echo "Building screen_master.csv ..."
conda run --cwd "$PROJ" -n "$ENV_CONDA" \
    python scripts/build_screen_master.py "$SCREEN_DIR"

echo ""
echo "All done! Master CSV: $SCREEN_DIR/screen_master.csv"
