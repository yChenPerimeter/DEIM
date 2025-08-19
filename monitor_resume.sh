#!/bin/bash

# Script to monitor resumed training progress

LOG_DIR="logs"

# Find the most recent resume training log
LATEST_LOG=$(ls -t ${LOG_DIR}/resume_training_*.log 2>/dev/null | head -1)
LATEST_PID_FILE=$(ls -t ${LOG_DIR}/resume_training_*.pid 2>/dev/null | head -1)

if [ -z "$LATEST_LOG" ]; then
    echo "No resume training log found in ${LOG_DIR}/"
    exit 1
fi

if [ -z "$LATEST_PID_FILE" ]; then
    echo "No PID file found"
else
    PID=$(cat $LATEST_PID_FILE)
    echo "Training PID: $PID"
    
    if ps -p $PID > /dev/null; then
        echo "Training is RUNNING"
    else
        echo "Training has STOPPED"
    fi
    echo ""
fi

echo "Monitoring: $LATEST_LOG"
echo "Press Ctrl+C to stop monitoring"
echo "==============================================="

# Monitor the log file
tail -f $LATEST_LOG