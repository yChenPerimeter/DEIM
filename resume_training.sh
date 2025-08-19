#!/bin/bash

# Resume training script for DEIM with logging
# This script resumes training from the last checkpoint

# Configuration
CONFIG_FILE="configs/deim_dfine/deim_hgnetv2_l_mother_data_v2.yml"
RESUME_CHECKPOINT="outputs/deim_hgnetv2_l_mother_data_v2/last.pth"
LOG_DIR="logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/resume_training_${TIMESTAMP}.log"
PID_FILE="${LOG_DIR}/resume_training_${TIMESTAMP}.pid"

# Create log directory if it doesn't exist
mkdir -p ${LOG_DIR}

# Check if checkpoint exists
if [ ! -f "${RESUME_CHECKPOINT}" ]; then
    echo "Error: Checkpoint file ${RESUME_CHECKPOINT} not found!"
    exit 1
fi

echo "==============================================="
echo "Resuming DEIM training from checkpoint"
echo "Config: ${CONFIG_FILE}"
echo "Checkpoint: ${RESUME_CHECKPOINT}"
echo "Log file: ${LOG_FILE}"
echo "==============================================="

# Start training with nohup
nohup python -u train.py \
    --config ${CONFIG_FILE} \
    --resume ${RESUME_CHECKPOINT} \
    > ${LOG_FILE} 2>&1 &

# Save the process ID
echo $! > ${PID_FILE}
PID=$!

echo "Training resumed with PID: ${PID}"
echo "PID saved to: ${PID_FILE}"
echo ""
echo "To monitor the training progress:"
echo "  tail -f ${LOG_FILE}"
echo ""
echo "To check if training is still running:"
echo "  ps -p ${PID}"
echo ""
echo "To stop the training:"
echo "  kill ${PID}"
echo ""
echo "==============================================="

# Optional: Show first few lines of log to confirm it started
sleep 3
echo "First few lines of training log:"
head -20 ${LOG_FILE}