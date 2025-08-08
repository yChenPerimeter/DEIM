# DEIM Training Guide

This guide provides detailed instructions for training DEIM models, including background training, GPU optimization, and troubleshooting.

## 📋 Table of Contents
- [Quick Start](#quick-start)
- [Background Training](#background-training)
- [GPU Configuration](#gpu-configuration)
- [Mother Data Training](#mother-data-training)
- [Troubleshooting](#troubleshooting)
- [Advanced Configuration](#advanced-configuration)

## Quick Start

### Basic Training Command
```bash
python train.py \
  -c configs/deim_dfine/deim_hgnetv2_l_mother_data_v2.yml \
  --use-amp --seed=0 \
  -t downloads/pretrained/deim_dfine/deim_dfine_hgnetv2_l_coco_50e.pth
```

## Background Training

### Using Nohup for Long-Running Training

We provide automated scripts for disconnection-safe training that continues even if your SSH session ends.

#### 1. Start Training in Background
```bash
# Run the training script
./train_nohup.sh
```

This script will:
- Start training in the background using `nohup`
- Create timestamped log files in `logs/` directory
- Save the process ID for easy management
- Show initial output to confirm successful start

#### 2. Monitor Training Progress
```bash
# Check training status and recent progress
./monitor_training.sh
```

This will show:
- Whether training is running ✅ or stopped ❌
- GPU utilization and memory usage
- Recent training metrics (loss, mAP, epochs)
- Commands for further monitoring

#### 3. View Live Logs
```bash
# Watch training logs in real-time
tail -f logs/training_*.log

# View specific metrics
tail -f logs/training_*.log | grep -E "(epoch|loss|mAP)"
```

#### 4. Stop Training
```bash
# Get the process ID from the latest PID file
kill $(cat logs/*.pid | tail -1)
```

## GPU Configuration

### Batch Size Recommendations by GPU

| GPU Model | VRAM | HGNetv2-S | HGNetv2-M | HGNetv2-L | HGNetv2-X |
|-----------|------|-----------|-----------|-----------|-----------|
| **T4** | 16GB | 16 | 8 | 4-8 | 4 |
| **V100** | 32GB | 32 | 16 | 8-16 | 8 |
| **A100** | 40GB | 32-64 | 32 | **16-32** | 16 |
| **A100** | 80GB | 64-128 | 64 | 32-64 | 32 |
| **H100** | 80GB | 128 | 64-128 | 64 | 32-64 |

### Dynamic Batch Size Configuration

Override batch size without modifying config files:

```bash
# For A100 40GB with HGNetv2-L
python train.py \
  -c configs/deim_dfine/deim_hgnetv2_l_mother_data_v2.yml \
  --use-amp --seed=0 \
  -t downloads/pretrained/deim_dfine/deim_dfine_hgnetv2_l_coco_50e.pth \
  -u train_dataloader.total_batch_size=16

# For A100 40GB with aggressive settings
python train.py \
  -c configs/deim_dfine/deim_hgnetv2_l_mother_data_v2.yml \
  --use-amp --seed=0 \
  -t downloads/pretrained/deim_dfine/deim_dfine_hgnetv2_l_coco_50e.pth \
  -u train_dataloader.total_batch_size=32 val_dataloader.total_batch_size=32
```

### Multi-GPU Training

For multiple GPUs:
```bash
# 4 GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --master_port=7777 \
  --nproc_per_node=4 \
  train.py -c config.yml --use-amp --seed=0

# 8 GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
  --master_port=7777 \
  --nproc_per_node=8 \
  train.py -c config.yml --use-amp --seed=0
```

## Mother Data Training

### Configuration Overview

The `deim_hgnetv2_l_mother_data_v2.yml` configuration is optimized for medical image detection:

- **Model**: HGNetv2-L (Large variant, B4)
- **Classes**: 2 (DCIS and IDC)
- **Input Size**: 640×640
- **Default Batch Size**: 8 (conservative for compatibility)
- **Training Epochs**: 30
- **Learning Rate**: 0.00025 (AdamW)

### Dataset Structure

Ensure your Mother Data is in COCO format:
```
mother_data/
├── deim_coco_format/
│   ├── train/
│   │   └── *.jpg
│   ├── val/
│   │   └── *.jpg
│   └── annotations/
│       ├── train.json
│       └── val.json
```

### Training Commands

```bash
# Standard training
./train_nohup.sh

# With custom batch size for A100
python train.py \
  -c configs/deim_dfine/deim_hgnetv2_l_mother_data_v2.yml \
  --use-amp --seed=0 \
  -t downloads/pretrained/deim_dfine/deim_dfine_hgnetv2_l_coco_50e.pth \
  -u train_dataloader.total_batch_size=16
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Out of Memory (OOM) Error
**Solution**: Reduce batch size
```bash
-u train_dataloader.total_batch_size=4 val_dataloader.total_batch_size=4
```

#### 2. NotImplementedError in Transforms
**Issue**: Incompatible torchvision version
**Solution**: Downgrade torchvision
```bash
pip install torchvision==0.19.1
```

#### 3. Permission Denied Errors
**Solution**: Fix permissions using the provided script
```bash
sudo ./fix_permissions.sh
```

#### 4. Training Interrupted
**Solution**: Resume from checkpoint
```bash
python train.py \
  -c config.yml \
  --resume outputs/checkpoint.pth
```

### Monitoring GPU Usage

```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi

# Check specific GPU
nvidia-smi -i 0

# Monitor memory usage
nvidia-smi --query-gpu=memory.used,memory.total --format=csv -l 1
```

## Advanced Configuration

### Learning Rate Scaling

When changing batch size, scale learning rate accordingly:
```yaml
# For batch_size = 16 (2x default)
optimizer:
  lr: 0.0005  # 2x original 0.00025
  params:
    - params: '^(?=.*backbone)(?!.*norm|bn).*$'
      lr: 0.000025  # 2x original
```

### Mixed Precision Training

Always use `--use-amp` flag for:
- Faster training (up to 2x speedup)
- Lower memory usage
- Maintained accuracy

### Checkpointing Strategy

Checkpoints are saved to:
```
outputs/deim_hgnetv2_l_mother_data_v2/
├── checkpoint.pth        # Latest checkpoint
├── best_checkpoint.pth   # Best validation mAP
└── logs/                 # Training logs
```

### Custom Dataset Integration

To train on your own dataset:

1. Convert to COCO format
2. Update config file paths:
```yaml
train_dataloader:
  dataset:
    img_folder: /path/to/your/train/images
    ann_file: /path/to/your/train/annotations.json

val_dataloader:
  dataset:
    img_folder: /path/to/your/val/images
    ann_file: /path/to/your/val/annotations.json
```

3. Set number of classes:
```yaml
num_classes: YOUR_NUM_CLASSES
```

## Performance Tips

1. **Use SSD/NVMe storage** for datasets to avoid I/O bottlenecks
2. **Enable AMP** (Automatic Mixed Precision) for all training
3. **Increase num_workers** if CPU usage is low:
   ```bash
   -u train_dataloader.num_workers=8
   ```
4. **Monitor and adjust** batch size based on GPU memory usage
5. **Use gradient accumulation** for effective larger batch sizes:
   ```bash
   -u gradient_accumulation_steps=2
   ```

## Support

For issues or questions:
- Check the [main README](README.md)
- Review [GitHub Issues](https://github.com/ShihuaHuang95/DEIM/issues)
- Contact: shihuahuang95@gmail.com