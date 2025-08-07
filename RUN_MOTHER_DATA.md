# Running DEIM with Mother Data Dataset

This guide explains how to train DEIM models on your custom mother_data dataset.

## Prerequisites

1. Set up DEIM environment:
```bash
conda create -n deim python=3.11.9
conda activate deim
pip install -r requirements.txt
```

## Data Preparation

You have two options for preparing your data:

### Option A: Using JSON Manifest Files (Recommended)

If you have JSON manifest files from ODImgAssist's `build_manifest.py`, use the direct conversion script:

```bash
cd /home/ychen/Documents/project/ODImgAssist
python data/dataTools/prepare_deim_dataset.py \
    --manifest-dir /home/ychen/Documents/project/mother_data/source/manifest_data_sample/output/MayOD_520only \
    --output-dir /home/ychen/Documents/project/mother_data/deim_coco_format
```

This will:
- Read train and validation manifests
- Copy images and convert annotations to COCO format in one step
- Create the following structure:
  ```
  /home/ychen/Documents/project/mother_data/deim_coco_format/
  ├── train/          # Training images
  ├── val/            # Validation images
  └── annotations/
      ├── train.json  # Training annotations (COCO format)
      └── val.json    # Validation annotations (COCO format)
  ```

### Option B: Convert from YOLO Format

If you have raw YOLO format data (images/ and labels/ folders):

```bash
cd /home/ychen/Documents/project/DEIM
python tools/dataset/yolo_to_coco.py \
    --source_dir /home/ychen/Documents/project/mother_data/imgClear_experiment_data_40_positives \
    --output_dir /home/ychen/Documents/project/mother_data/coco_format \
    --train_split 0.8
```

This will:
- Convert YOLO annotations to COCO format
- Split data into 80% train, 20% validation
- Create the same directory structure as Option A

### Step 2: Update Dataset Configuration

After preparing the data, update the dataset configuration file:
`/home/ychen/Documents/project/DEIM/configs/dataset/mother_data_detection.yml`

Change the paths to match your output directory:
```yaml
train_dataloader:
  dataset:
    img_folder: /home/ychen/Documents/project/mother_data/deim_coco_format/train
    ann_file: /home/ychen/Documents/project/mother_data/deim_coco_format/annotations/train.json

val_dataloader:
  dataset:
    img_folder: /home/ychen/Documents/project/mother_data/deim_coco_format/val
    ann_file: /home/ychen/Documents/project/mother_data/deim_coco_format/annotations/val.json
```

### Step 3: Verify Dataset

Check that the conversion was successful:
```bash
# Check directory structure
ls -la /home/ychen/Documents/project/mother_data/deim_coco_format/

# Check annotation files
python -c "import json; print(len(json.load(open('/home/ychen/Documents/project/mother_data/deim_coco_format/annotations/train.json'))['images']), 'training images')"
python -c "import json; print(len(json.load(open('/home/ychen/Documents/project/mother_data/deim_coco_format/annotations/val.json'))['images']), 'validation images')"
```

## Training

### Option 1: Train DEIM-D-FINE Model from Scratch

For single GPU:
```bash
cd /home/ychen/Documents/project/DEIM
python train.py -c configs/deim_dfine/deim_hgnetv2_s_mother_data.yml --use-amp --seed=0
```

For multi-GPU (4 GPUs):
```bash
cd /home/ychen/Documents/project/DEIM
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=7777 --nproc_per_node=4 \
    train.py -c configs/deim_dfine/deim_hgnetv2_s_mother_data.yml --use-amp --seed=0
```

### Option 2: Fine-tune from Pre-trained Model (Recommended for Better Performance)

Using HGNetv2-L pretrained model:
```bash
cd /home/ychen/Documents/project/DEIM
python train.py -c configs/deim_dfine/deim_hgnetv2_l_mother_data.yml \
    --use-amp --seed=0 \
    -t /home/ychen/Documents/project/DEIM/downloads/pretrained/deim_dfine/deim_dfine_hgnetv2_l_coco_50e.pth
```

Or with custom Python environment:
```bash
cd /home/ychen/Documents/project/DEIM
/home/ychen/Documents/project/torchENV_py312/bin/python train.py \
    -c configs/deim_dfine/deim_hgnetv2_l_mother_data_v2.yml \
    --use-amp --seed=0 \
    -t /home/ychen/Documents/project/DEIM/downloads/pretrained/deim_dfine/deim_dfine_hgnetv2_l_coco_50e.pth
```

For multi-GPU training with pretrained model:
```bash
cd /home/ychen/Documents/project/DEIM
CUDA_VISIBLE_DEVICES=0,1 torchrun --master_port=7777 --nproc_per_node=2 \
    train.py -c configs/deim_dfine/deim_hgnetv2_l_mother_data.yml \
    --use-amp --seed=0 \
    -t /home/ychen/Documents/project/DEIM/downloads/pretrained/deim_dfine/deim_dfine_hgnetv2_l_coco_50e.pth
```

## Model Sizes

Choose different model sizes by modifying the config file name:
- `deim_hgnetv2_n_mother_data.yml` - Nano (4M params)
- `deim_hgnetv2_s_mother_data.yml` - Small (10M params) [Default]
- `deim_hgnetv2_m_mother_data.yml` - Medium (19M params)
- `deim_hgnetv2_l_mother_data.yml` - Large (31M params)
- `deim_hgnetv2_x_mother_data.yml` - Extra Large (62M params)

## Testing/Evaluation

Evaluate your trained model:
```bash
python train.py -c configs/deim_dfine/deim_hgnetv2_s_mother_data.yml \
    --test-only -r outputs/mother_data/checkpoint_best.pth
```

## Inference

### On Single Image:
```bash
python tools/inference/torch_inf.py \
    -c configs/deim_dfine/deim_hgnetv2_s_mother_data.yml \
    -r outputs/mother_data/checkpoint_best.pth \
    --input /path/to/test/image.png \
    --device cuda:0
```

### Batch Inference:
```bash
python tools/inference/torch_inf.py \
    -c configs/deim_dfine/deim_hgnetv2_s_mother_data.yml \
    -r outputs/mother_data/checkpoint_best.pth \
    --input /home/ychen/Documents/project/mother_data/test_images/ \
    --device cuda:0
```

## Visualization

Visualize predictions with FiftyOne:
```bash
python tools/visualization/fiftyone_vis.py \
    -c configs/deim_dfine/deim_hgnetv2_s_mother_data.yml \
    -r outputs/mother_data/checkpoint_best.pth
```

## Deployment

### Export to ONNX:
```bash
python tools/deployment/export_onnx.py --check \
    -c configs/deim_dfine/deim_hgnetv2_s_mother_data.yml \
    -r outputs/mother_data/checkpoint_best.pth
```

### Convert to TensorRT:
```bash
trtexec --onnx="outputs/mother_data/model.onnx" \
    --saveEngine="outputs/mother_data/model.engine" --fp16
```

## Troubleshooting

1. **Out of Memory**: Reduce batch size in the config file
2. **Dataset not found**: Ensure you ran the YOLO to COCO conversion
3. **Low accuracy**: Try training for more epochs or using a larger model
4. **Config parameter issues**: Note that DEIM uses `epoches` (with typo) instead of `epochs` in config files. This is consistent throughout the codebase
5. **Training doesn't start**: Make sure the dataset paths in the config files point to existing directories with valid COCO format annotations

## Dataset Classes

Your dataset contains 2 classes:
- Class 0: DCIS
- Class 1: IDC

## Notes

- The configuration uses mixed precision training (--use-amp) for faster training
- Checkpoints are saved every 10 epochs
- Evaluation runs every 5 epochs during training
- Training logs are saved to `outputs/mother_data/logs/`

## configed training (Chinese version)
单GPU训练：
  cd /home/ychen/Documents/project/DEIM
  python train.py -c configs/deim_dfine/deim_hgnetv2_s_mother_data.yml --use-amp --seed=0

  多GPU训练（如4个GPU）：
  cd /home/ychen/Documents/project/DEIM
  CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=7777 --nproc_per_node=4 \
      train.py -c configs/deim_dfine/deim_hgnetv2_s_mother_data.yml --use-amp --seed=0

  主要更改：
  - 训练图像路径：/home/ychen/Documents/project/mother_data/deim_coco_format/train
  - 训练标注路径：/home/ychen/Documents/project/mother_data/deim_coco_format/annotations/train.json
  - 验证图像路径：/home/ychen/Documents/project/mother_data/deim_coco_format/val
  - 验证标注路径：/home/ychen/Documents/project/mother_data/deim_coco_format/annotations/val.json