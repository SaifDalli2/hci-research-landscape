#!/bin/bash
# Auto-resume paper collection when it stops due to network errors.
# Keeps retrying until the checkpoint file is removed (all queries done).

cd /Users/saifalshammari/Documents/GitHub/Projects/hci-landscape
CHECKPOINT="data/checkpoint.json"
PYTHON="./venv/bin/python3"
LOG="data/auto_collect.log"

echo "$(date): Auto-collect started" >> "$LOG"

while true; do
    # Run the collection script
    echo "$(date): Starting collection..." >> "$LOG"
    $PYTHON 01_collect_data.py >> "$LOG" 2>&1
    EXIT_CODE=$?

    # If checkpoint file is gone, all queries completed
    if [ ! -f "$CHECKPOINT" ]; then
        echo "$(date): All queries completed! Exiting." >> "$LOG"
        break
    fi

    # Wait 60 seconds before retrying (let network recover)
    echo "$(date): Process exited (code $EXIT_CODE). Waiting 60s before resuming..." >> "$LOG"
    sleep 60
done

echo "$(date): Auto-collect finished" >> "$LOG"
