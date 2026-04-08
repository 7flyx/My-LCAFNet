# LCAFNet Project Overview

## Goal
LCAFNet is a multispectral object detection project built on a YOLOv5-style framework. It focuses on RGB-IR feature fusion and custom attention/fusion blocks for improved detection in challenging conditions.

## Main Entry Scripts
- `train.py`: training pipeline, optimizer/scheduler setup, logging, checkpointing.
- `test.py`: evaluation pipeline and metric reporting.
- `detect_twostream.py`: dual-stream inference for visible + infrared inputs.
- `flops.py`: FLOPs and parameter profiling.
- `confluence.py`: utility script for model/feature analysis (project-specific usage).

## Core Directories
- `models/`: model graph definitions and building blocks.
- `utils/`: data loading, augmentations, losses, metrics, plotting, and helper tools.
- `data/`: dataset YAML and hyperparameter configs.
- `dataset/`: local dataset layout (visible/infrared/images/labels).
- `evaluation_script/`: COCO-style evaluation scripts and annotations.
- `transformer/`: YAML model configs for specific datasets and variants.
- `runs/`: training/test outputs and checkpoints.

## Model Config and Weights
- YAML model configs are under `models/` and `transformer/`.
- Pretrained/custom weights include `yolov5s.pt` and `LCAFNet_FLIR.pt`.

## Recommended Reading Order
1. `README.md`
2. `models/yolo_test.py` (dual-stream model execution path)
3. `models/common.py` (custom fusion and attention blocks)
4. `train.py` and `test.py`
5. `utils/datasets.py`, `utils/loss.py`, `utils/metrics.py`
