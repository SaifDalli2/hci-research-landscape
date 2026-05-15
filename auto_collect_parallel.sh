#!/bin/bash
# Auto-resume wrapper for parallel workers.
# Usage: bash auto_collect_parallel.sh <worker_id>
# Runs worker, restarts on failure, exits when all its queries are done.

cd /Users/saifalshammari/Documents/GitHub/Projects/hci-landscape
WORKER_ID=$1
PYTHON="./venv/bin/python3"
CHECKPOINT="data/checkpoint_w${WORKER_ID}.json"
LOG="data/auto_collect_w${WORKER_ID}.log"

if [ -z "$WORKER_ID" ]; then
    echo "Usage: bash auto_collect_parallel.sh <worker_id>"
    exit 1
fi

echo "$(date): Worker $WORKER_ID auto-collect started" >> "$LOG"

while true; do
    echo "$(date): Starting worker $WORKER_ID..." >> "$LOG"
    $PYTHON 01_collect_parallel.py worker $WORKER_ID >> "$LOG" 2>&1
    EXIT_CODE=$?

    # If checkpoint file is gone, all queries completed
    if [ ! -f "$CHECKPOINT" ]; then
        echo "$(date): Worker $WORKER_ID — all queries completed! Exiting." >> "$LOG"
        break
    fi

    echo "$(date): Worker $WORKER_ID exited (code $EXIT_CODE). Waiting 60s before resuming..." >> "$LOG"
    sleep 60
done

echo "$(date): Worker $WORKER_ID auto-collect finished" >> "$LOG"
