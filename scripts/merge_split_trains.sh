#!/bin/bash
# Merge 3 split train directories into one unified directory per task.
# Usage: bash merge_split_trains.sh <task_out_dir>
# Example: bash merge_split_trains.sh /home/sun/phdproject/dqn/DQN8/runs/algo6_10k_forest
#
# Finds the 3 most recent train_* dirs, creates train_merged/ with symlinks.
set -e

TASK_DIR="$1"
if [ -z "$TASK_DIR" ]; then
  echo "Usage: $0 <task_out_dir>"
  exit 1
fi

MERGED="$TASK_DIR/train_merged"
mkdir -p "$MERGED/checkpoints" "$MERGED/configs"

echo "Merging split trains in $TASK_DIR..."

# Find all recent train_20260312* dirs (from today's split run)
TRAIN_DIRS=$(ls -d "$TASK_DIR"/train_20260312*/ 2>/dev/null | sort)
if [ -z "$TRAIN_DIRS" ]; then
  echo "No train_20260312* dirs found in $TASK_DIR"
  exit 1
fi

echo "Found train dirs:"
echo "$TRAIN_DIRS"

# Symlink all algo checkpoint dirs into merged
for td in $TRAIN_DIRS; do
  # Handle both structures: checkpoints/<env>/<algo>_ep*.pt or <algo>/checkpoints/<algo>_ep*.pt
  # Check structure 1: checkpoints/<env>/
  for env_dir in "$td"/checkpoints/*/; do
    if [ -d "$env_dir" ]; then
      env_name=$(basename "$env_dir")
      mkdir -p "$MERGED/checkpoints/$env_name"
      for f in "$env_dir"/*.pt; do
        [ -f "$f" ] && ln -sf "$f" "$MERGED/checkpoints/$env_name/$(basename $f)"
      done
      echo "  Linked checkpoints from $(basename $td)/$env_name"
    fi
  done

  # Copy configs
  for f in "$td"/configs/*; do
    [ -f "$f" ] && cp -n "$f" "$MERGED/configs/" 2>/dev/null || true
  done
done

echo "Merged into: $MERGED"
echo "Checkpoint count: $(find $MERGED/checkpoints -name '*.pt' | wc -l)"
