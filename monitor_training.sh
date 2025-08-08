#!/bin/bash

# Script to monitor DEIM training progress

LOG_DIR="/home/ychen/Documents/project/DEIM/logs"

echo "========================================="
echo "DEIM Training Monitor"
echo "========================================="

# Find the latest PID file
LATEST_PID_FILE=$(ls -t $LOG_DIR/*.pid 2>/dev/null | head -n 1)

if [ -z "$LATEST_PID_FILE" ]; then
    echo "No training process found."
    echo "Start training with: ./train_nohup.sh"
    exit 1
fi

# Get PID and corresponding log file
PID=$(cat "$LATEST_PID_FILE")
LOG_FILE="${LATEST_PID_FILE%.pid}.log"

echo "PID: $PID"
echo "Log file: $LOG_FILE"
echo ""

# Check if process is running
if ps -p $PID > /dev/null; then
    echo "✅ Training is RUNNING"
    echo ""
    
    # Show GPU usage
    echo "GPU Usage:"
    nvidia-smi --query-gpu=gpu_name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | awk -F', ' '{printf "  %s: %s/%s MB (%s%%)\n", $1, $2, $3, $4}'
    echo ""
    
    # Show recent progress
    echo "Recent training progress:"
    echo "------------------------"
    tail -n 20 "$LOG_FILE" | grep -E "(epoch|loss|mAP|checkpoint|error)" || tail -n 10 "$LOG_FILE"
    echo ""
    
    echo "Commands:"
    echo "  Watch live: tail -f $LOG_FILE"
    echo "  Stop training: kill $PID"
    echo "  Check full log: less $LOG_FILE"
else
    echo "❌ Training is NOT running (process $PID not found)"
    echo ""
    echo "Last 20 lines from log:"
    echo "------------------------"
    tail -n 20 "$LOG_FILE"
    echo ""
    echo "To restart training: ./train_nohup.sh"
fi