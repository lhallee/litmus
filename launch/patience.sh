#!/usr/bin/env bash
# On the host, beside a training run in its container: keep the checkpoint of every validation reading from reading
# START on, and stop training after PATIENCE readings in a row with no new best validation PR AUC among them (Logan,
# 2026-11-18: "After 1.0 passes minimum for all, start a patience timer. If does not improve within 10 patience, just
# take the best checkpoint in the region"). START is the first reading after a pass over every origin's training
# texts: with 1,024 texts a step, a third an origin, and a reading every 500 steps, reading ceil(rows / 341.3 / 500).
#   nohup bash patience.sh <run.log> <train folder> <out folder> <state log> [patience] [container] [start] &
# The run's `best.pt` is chosen by validation loss; the release's checkpoint is the kept reading of best PR AUC.
set -euo pipefail
LOG="$1"
TRAIN="$2"
OUT="$3"
STATE="$4"
PATIENCE="${5:-10}"
CONTAINER="${6:-litmus-train}"
START="${7:-0}"

running() {
    sudo docker ps --format '{{.Names}}' | grep -q "^$CONTAINER$"
}

sudo mkdir -p "$OUT"
for _ in $(seq 360); do  # up to 30 minutes for the training container to start
    running && break
    sleep 5
done
if ! running; then
    echo "$(date -u +%H:%M:%S) the container $CONTAINER never ran" >> "$STATE"
    exit 1
fi
seen=$(grep -c '"val/loss"' "$LOG" 2>/dev/null || true)
best=0
best_reading=0
since=0
echo "$(date -u +%H:%M:%S) START after reading $seen, patience $PATIENCE from reading $START" >> "$STATE"
while running; do
    count=$(grep -c '"val/loss"' "$LOG" 2>/dev/null || true)
    if (( count > seen )); then
        seen=$count
        if (( seen >= START )); then
            line=$(grep '"val/loss"' "$LOG" | tail -n 1)
            value=$(echo "$line" | grep -o '"val/pr_auc": [0-9.]*' | cut -d' ' -f2)
            loss=$(echo "$line" | grep -o '"val/loss": [0-9.]*' | cut -d' ' -f2)
            sudo cp "$TRAIN/last.pt" "$OUT/reading_$seen.pt"
            if awk -v a="$value" -v b="$best" 'BEGIN { exit !(a > b) }'; then
                best=$value
                best_reading=$seen
                since=0
            else
                since=$((since + 1))
            fi
            echo "$(date -u +%H:%M:%S) reading $seen pr_auc $value loss $loss best $best at reading $best_reading since $since" >> "$STATE"
            if (( since >= PATIENCE )); then
                sudo docker exec "$CONTAINER" pkill -INT -f litmus_student.train >> "$STATE" 2>&1 || true
                for _ in $(seq 18); do
                    running || break
                    sleep 5
                done
                if running; then
                    sudo docker stop "$CONTAINER" >> "$STATE" 2>&1
                fi
                echo "$(date -u +%H:%M:%S) STOPPED at reading $seen, best $best at reading $best_reading" >> "$STATE"
                exit 0
            fi
        else
            echo "$(date -u +%H:%M:%S) reading $seen before the region (starts at $START)" >> "$STATE"
        fi
    fi
    sleep 10
done
echo "$(date -u +%H:%M:%S) ENDED on its own after reading $seen, best $best at reading $best_reading" >> "$STATE"
