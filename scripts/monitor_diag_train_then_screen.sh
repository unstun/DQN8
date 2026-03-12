#!/bin/bash
# Monitor all 3 trainings (Forest 1/2 + Forest diag + Realmap diag),
# then auto-launch screen inference when each finishes.
# Run on remote: nohup bash scripts/monitor_diag_train_then_screen.sh > runs/monitor_diag.log 2>&1 &
set -euo pipefail

PROJ="/home/sun/phdproject/dqn/DQN8"
CHECK_INTERVAL=300  # 5 minutes

FOREST_HALF_DONE=false
FOREST_DIAG_DONE=false
REALMAP_DIAG_DONE=false
FOREST_HALF_SCREEN_LAUNCHED=false
FOREST_DIAG_SCREEN_LAUNCHED=false
REALMAP_DIAG_SCREEN_LAUNCHED=false

echo "$(date): Monitor started. Checking every ${CHECK_INTERVAL}s."

while true; do
    # ---- Forest 1/2 training ----
    if [ "$FOREST_HALF_DONE" = false ]; then
        if ! pgrep -f "train.py --profile repro_20260310_6algo_10k_forest" > /dev/null 2>&1; then
            FOREST_HALF_DONE=true
            echo "$(date): Forest 1/2 training COMPLETED."
        else
            FH_EP=$(ls "$PROJ/runs/algo6_10k_forest"/train_*/checkpoints/forest_a/cnn-pddqn_ep*.pt 2>/dev/null | sort | tail -1 | grep -oP 'ep\K\d+' || echo "?")
            echo "$(date): Forest 1/2 training running (latest cnn-pddqn ep: ${FH_EP})"
        fi
    fi

    # ---- Forest diag training ----
    if [ "$FOREST_DIAG_DONE" = false ]; then
        if ! pgrep -f "train.py --profile repro_20260311_6algo_10k_forest_diag" > /dev/null 2>&1; then
            FOREST_DIAG_DONE=true
            echo "$(date): Forest diag training COMPLETED."
        else
            FD_EP=$(ls "$PROJ/runs/algo6_10k_forest_diag"/train_*/checkpoints/forest_a/cnn-pddqn_ep*.pt 2>/dev/null | sort | tail -1 | grep -oP 'ep\K\d+' || echo "?")
            echo "$(date): Forest diag training running (latest cnn-pddqn ep: ${FD_EP})"
        fi
    fi

    # ---- Realmap diag training ----
    if [ "$REALMAP_DIAG_DONE" = false ]; then
        if ! pgrep -f "train.py --profile repro_20260311_6algo_10k_realmap_diag" > /dev/null 2>&1; then
            REALMAP_DIAG_DONE=true
            echo "$(date): Realmap diag training COMPLETED."
        else
            RD_EP=$(ls "$PROJ/runs/algo6_10k_realmap_diag"/train_*/checkpoints/realmap_a/cnn-pddqn_ep*.pt 2>/dev/null | sort | tail -1 | grep -oP 'ep\K\d+' || echo "?")
            echo "$(date): Realmap diag training running (latest cnn-pddqn ep: ${RD_EP})"
        fi
    fi

    # ---- Launch screens when training done ----

    # Forest 1/2 screen
    if [ "$FOREST_HALF_DONE" = true ] && [ "$FOREST_HALF_SCREEN_LAUNCHED" = false ]; then
        echo "$(date): Launching Forest 1/2 screen inference..."
        chmod +x "$PROJ/scripts/screen_6algo_10k_forest.sh"
        nohup bash "$PROJ/scripts/screen_6algo_10k_forest.sh" \
            > "$PROJ/runs/screen_forest_half.log" 2>&1 &
        FOREST_HALF_SCREEN_LAUNCHED=true
        echo "$(date): Forest 1/2 screen launched (PID $!)."
    fi

    # Forest diag screen
    if [ "$FOREST_DIAG_DONE" = true ] && [ "$FOREST_DIAG_SCREEN_LAUNCHED" = false ]; then
        echo "$(date): Launching Forest diag screen inference..."
        nohup bash "$PROJ/scripts/screen_6algo_10k_forest_diag.sh" \
            > "$PROJ/runs/screen_forest_diag.log" 2>&1 &
        FOREST_DIAG_SCREEN_LAUNCHED=true
        echo "$(date): Forest diag screen launched (PID $!)."
    fi

    # Realmap diag screen
    if [ "$REALMAP_DIAG_DONE" = true ] && [ "$REALMAP_DIAG_SCREEN_LAUNCHED" = false ]; then
        echo "$(date): Launching Realmap diag screen inference..."
        nohup bash "$PROJ/scripts/screen_6algo_10k_realmap_diag.sh" \
            > "$PROJ/runs/screen_realmap_diag.log" 2>&1 &
        REALMAP_DIAG_SCREEN_LAUNCHED=true
        echo "$(date): Realmap diag screen launched (PID $!)."
    fi

    # Exit when all 3 screens launched
    if [ "$FOREST_HALF_SCREEN_LAUNCHED" = true ] && \
       [ "$FOREST_DIAG_SCREEN_LAUNCHED" = true ] && \
       [ "$REALMAP_DIAG_SCREEN_LAUNCHED" = true ]; then
        echo "$(date): All 3 screen inferences launched. Monitor exiting."
        break
    fi

    sleep "$CHECK_INTERVAL"
done
