#!/usr/bin/env python3
"""
Convert YOLO format dataset to COCO format for DEIM training.
"""

import json
import os
from pathlib import Path
from PIL import Image
import numpy as np
from typing import Dict, List, Tuple
import argparse
from tqdm import tqdm
import shutil


def yolo_to_coco_bbox(yolo_bbox: List[float], img_width: int, img_height: int) -> List[float]:
    """
    Convert YOLO format bbox to COCO format.
    
    YOLO: [class_id, center_x, center_y, width, height] (normalized)
    COCO: [x_min, y_min, width, height] (absolute pixels)
    """
    _, cx, cy, w, h = yolo_bbox
    
    # Convert from normalized to absolute coordinates
    cx *= img_width
    cy *= img_height
    w *= img_width
    h *= img_height
    
    # Convert from center to top-left corner
    x_min = cx - w / 2
    y_min = cy - h / 2
    
    return [x_min, y_min, w, h]


def create_coco_format() -> Dict:
    """Create base COCO format structure."""
    return {
        "info": {
            "description": "Mother Data Medical Dataset",
            "version": "1.0",
            "year": 2025,
            "contributor": "Medical AI Team",
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 0, "name": "DCIS", "supercategory": "lesion"},
            {"id": 1, "name": "IDC", "supercategory": "lesion"}
        ]
    }


def process_dataset(
    img_dir: Path,
    label_dir: Path,
    output_dir: Path,
    split: str = "train"
) -> Dict:
    """Process a dataset split from YOLO to COCO format."""
    
    coco_data = create_coco_format()
    image_id = 1
    annotation_id = 1
    
    # Get all image files
    image_files = sorted(list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg")))
    
    # Create output directories
    output_img_dir = output_dir / split
    output_img_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing {split} split with {len(image_files)} images...")
    
    for img_path in tqdm(image_files):
        # Get corresponding label file
        label_path = label_dir / f"{img_path.stem}.txt"
        
        if not label_path.exists():
            print(f"Warning: No label file for {img_path.name}")
            continue
        
        # Copy image to output directory
        output_img_path = output_img_dir / img_path.name
        shutil.copy2(img_path, output_img_path)
        
        # Open image to get dimensions
        img = Image.open(img_path)
        img_width, img_height = img.size
        
        # Add image info to COCO
        image_info = {
            "id": image_id,
            "file_name": img_path.name,
            "width": img_width,
            "height": img_height
        }
        coco_data["images"].append(image_info)
        
        # Read YOLO annotations
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            
            class_id = int(parts[0])
            yolo_bbox = [float(x) for x in parts]
            
            # Convert to COCO bbox
            coco_bbox = yolo_to_coco_bbox(yolo_bbox, img_width, img_height)
            
            # Create annotation
            annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": class_id,
                "bbox": coco_bbox,
                "area": coco_bbox[2] * coco_bbox[3],
                "segmentation": [],
                "iscrowd": 0
            }
            coco_data["annotations"].append(annotation)
            annotation_id += 1
        
        image_id += 1
    
    return coco_data


def main():
    parser = argparse.ArgumentParser(description="Convert YOLO dataset to COCO format")
    parser.add_argument(
        "--source_dir",
        type=str,
        default="/home/ychen/Documents/project/mother_data/imgClear_experiment_data_40_positives",
        help="Path to source dataset directory with images/ and labels/ folders"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/ychen/Documents/project/mother_data/coco_format",
        help="Path to output COCO format dataset"
    )
    parser.add_argument(
        "--train_split",
        type=float,
        default=0.8,
        help="Proportion of data for training (rest goes to validation)"
    )
    args = parser.parse_args()
    
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    ann_dir = output_dir / "annotations"
    ann_dir.mkdir(exist_ok=True)
    
    # Get all images
    img_dir = source_dir / "images"
    label_dir = source_dir / "labels"
    
    if not img_dir.exists() or not label_dir.exists():
        raise FileNotFoundError(f"Expected 'images' and 'labels' directories in {source_dir}")
    
    # Get all image files and split
    all_images = sorted(list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg")))
    np.random.seed(42)
    np.random.shuffle(all_images)
    
    train_size = int(len(all_images) * args.train_split)
    train_images = all_images[:train_size]
    val_images = all_images[train_size:]
    
    print(f"Total images: {len(all_images)}")
    print(f"Train images: {len(train_images)}")
    print(f"Val images: {len(val_images)}")
    
    # Create temporary directories for split
    temp_train_img = output_dir / "temp_train_img"
    temp_train_label = output_dir / "temp_train_label"
    temp_val_img = output_dir / "temp_val_img"
    temp_val_label = output_dir / "temp_val_label"
    
    for d in [temp_train_img, temp_train_label, temp_val_img, temp_val_label]:
        d.mkdir(exist_ok=True)
    
    # Copy files to temporary directories
    for img_path in train_images:
        shutil.copy2(img_path, temp_train_img)
        label_path = label_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            shutil.copy2(label_path, temp_train_label)
    
    for img_path in val_images:
        shutil.copy2(img_path, temp_val_img)
        label_path = label_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            shutil.copy2(label_path, temp_val_label)
    
    # Process train split
    train_coco = process_dataset(temp_train_img, temp_train_label, output_dir, "train")
    with open(ann_dir / "train.json", 'w') as f:
        json.dump(train_coco, f, indent=2)
    
    # Process val split
    val_coco = process_dataset(temp_val_img, temp_val_label, output_dir, "val")
    with open(ann_dir / "val.json", 'w') as f:
        json.dump(val_coco, f, indent=2)
    
    # Clean up temporary directories
    shutil.rmtree(temp_train_img)
    shutil.rmtree(temp_train_label)
    shutil.rmtree(temp_val_img)
    shutil.rmtree(temp_val_label)
    
    print(f"\nConversion complete!")
    print(f"Output directory: {output_dir}")
    print(f"Train annotations: {ann_dir}/train.json")
    print(f"Val annotations: {ann_dir}/val.json")


if __name__ == "__main__":
    main()