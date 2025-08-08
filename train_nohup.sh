#!/bin/bash

# DEIM Training Script with nohup
# This script ensures training continues even after disconnection

# Set up environment
export CUDA_VISIBLE_DEVICES=0  # Use GPU 0, change if needed
export PYTHONUNBUFFERED=1      # Ensure real-time output to log

# Create logs directory if it doesn't exist
LOG_DIR="/home/ychen/Documents/project/DEIM/logs"
mkdir -p $LOG_DIR

# Generate timestamp for unique log files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/training_${TIMESTAMP}.log"
PID_FILE="$LOG_DIR/training_${TIMESTAMP}.pid"

echo "========================================="
echo "Starting DEIM Training"
echo "Timestamp: $TIMESTAMP"
echo "Log file: $LOG_FILE"
echo "PID file: $PID_FILE"
echo "========================================="

# Run training with nohup
nohup /home/ychen/Documents/project/torchENV_py312/bin/python train.py \
    -c configs/deim_dfine/deim_hgnetv2_l_mother_data_v2.yml \
    --use-amp \
    --seed=0 \
    -t /home/ychen/Documents/project/DEIM/downloads/pretrained/deim_dfine/deim_dfine_hgnetv2_l_coco_50e.pth \
    -u train_dataloader.total_batch_size=16 \
    > "$LOG_FILE" 2>&1 &

# Save the process ID
PID=$!
echo $PID > "$PID_FILE"

echo "Training started with PID: $PID"
echo "To monitor progress, use: tail -f $LOG_FILE"
echo "To check if still running: ps -p $PID"
echo "To stop training: kill $PID"

# Optional: Show first few lines of output to confirm start
sleep 3
echo ""
echo "Initial output:"
head -n 20 "$LOG_FILE"