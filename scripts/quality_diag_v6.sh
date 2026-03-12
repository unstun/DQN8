#!/bin/bash
# Quality evaluation for Forest-diag & Realmap-diag using V6 combo checkpoints
# V6 combo: CPDDQN=10000, CDDQN=1500, CDQN=4000, MPDDQN=2500, MDDQN=500, MDQN=1000
set -euo pipefail

PROJ="/home/sun/phdproject/dqn/DQN8"
ENV_CONDA="ros2py310"

# V6 combo epochs
declare -A EPOCHS=(
  ["cnn-pddqn"]=10000
  ["cnn-ddqn"]=1500
  ["cnn-dqn"]=4000
  ["mlp-pddqn"]=2500
  ["mlp-ddqn"]=500
  ["mlp-dqn"]=1000
)

setup_models() {
    local ENV_NAME="$1"   # forest_a or realmap_a
    local TRAIN_DIR="$2"  # algo6_10k_forest_diag or algo6_10k_realmap_diag
    local ENV_SHORT="$3"  # forest or realmap

    local MERGED="$PROJ/runs/$TRAIN_DIR/train_merged"
    local MODEL_DIR="$MERGED/models/${ENV_NAME}"
    local CKPT_DIR="$MERGED/checkpoints/${ENV_NAME}"

    echo "Setting up models for $ENV_NAME from $TRAIN_DIR ..."
    mkdir -p "$MODEL_DIR"

    for ALGO in "${!EPOCHS[@]}"; do
        local EP=${EPOCHS[$ALGO]}
        local EPSTR=$(printf "%05d" "$EP")
        local CKPT="$CKPT_DIR/${ALGO}_ep${EPSTR}.pt"

        if [ ! -f "$CKPT" ]; then
            echo "ERROR: $CKPT not found!"
            exit 1
        fi
        cp "$CKPT" "$MODEL_DIR/$ALGO.pt"
        echo "  $ALGO <- ep${EPSTR}"
    done
    echo "Models ready."
}

run_profile() {
    local PROFILE="$1"
    local DESC="$2"
    echo ""
    echo "=========================================="
    echo "Running: $DESC ($PROFILE)"
    echo "Start: $(date)"
    echo "=========================================="

    conda run --cwd "$PROJ" -n "$ENV_CONDA" python infer.py --profile "$PROFILE"

    echo "Done: $(date)"
}

# === FOREST DIAG ===
setup_models "forest_a" "algo6_10k_forest_diag" "forest"
run_profile "quality_forest_diag_long"  "Forest-diag Quality Long (>=18m)"
run_profile "quality_forest_diag_short" "Forest-diag Quality Short (6-14m)"

# === REALMAP DIAG ===
setup_models "realmap_a" "algo6_10k_realmap_diag" "realmap"
run_profile "quality_realmap_diag_long"  "Realmap-diag Quality Long (>=18m)"
run_profile "quality_realmap_diag_short" "Realmap-diag Quality Short (6-14m)"

echo ""
echo "=========================================="
echo "All quality evaluations complete: $(date)"
echo "=========================================="
