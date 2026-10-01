# Vessel Detection in 2D CT Scans

Automatic detection of vascular structures in 2D axial CT scan slices using [YOLO26s](https://docs.ultralytics.com/models/yolo26).

---

## Overview

This project implements an object detection pipeline to automatically localize 21 vascular structures in 2D axial CT scan images. Each CT series is a stack of 2D cross-sectional slices, and the goal is to predict bounding boxes around vessel structures on each individual slice.

The pipeline covers :
- Parsing JSON annotations into YOLO-compatible `.txt` label files
- Building a structured dataset compatible with the Ultralytics YOLO framework
- Training a YOLO26s model on a small dataset as a baseline
- Evaluating per-class detection performance

Given the one-day scope of the project, hyperparameter tuning was kept minimal in favour of establishing a solid baseline and thoroughly analysing the results. The notebook includes detailed interpretation of two key validation outputs: the Precision-Recall curve (mAP@0.5) and the normalized confusion matrix.

---

## Detected Classes

The model is trained to detect the following 21 vascular structures :

| # | Class |
|---|---|
| 0 | no_name |
| 1 | aorta |
| 2 | ascending_aorta |
| 3 | celiac_trunk |
| 4 | superior_mesenteric |
| 5 | left_renal |
| 6 | right_renal |
| 7 | left_common_iliac |
| 8 | left_external_iliac |
| 9 | left_internal_iliac |
| 10 | right_common_iliac |
| 11 | right_external_iliac |
| 12 | right_internal_iliac |
| 13 | left_subclavian |
| 14 | brachiocephalic_trunk |
| 15 | left_common_carotid |
| 16 | left_external_carotid |
| 17 | left_internal_carotid |
| 18 | right_common_carotid |
| 19 | right_external_carotid |
| 20 | right_internal_carotid |

---

## Model

**YOLO26s** was selected for the following reasons :
- Best mAP50-95 / inference latency tradeoff among the YOLO family on COCO
- 9.5M parameters — sufficient capacity to distinguish 21 classes without overfitting on a small dataset
- Native bounding box format `(center_x, center_y, width, height)` normalized — matches the annotation format exactly, eliminating any coordinate conversion
- Standard axis-aligned bounding boxes are appropriate here since vessel cross-sections in axial slices appear as circles or ellipses with no meaningful rotation angle

---

## Results

Trained on 4 patient series (2,389 images) and evaluated on 2 held-out patient series (1,572 images) after 10 epochs:

| Metric | Value |
|---|---|
| mAP50 | 0.533 |
| mAP50-95 | 0.311 |
| Precision | 0.53 |
| Recall | 0.571 |

Per-class results :

| Class | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|
| all | 1572 | 5256 | 0.529 | 0.570 | 0.533 | 0.311 |
| no_name | 702 | 1866 | 0.487 | 0.692 | 0.653 | 0.334 |
| aorta | 609 | 609 | 0.781 | 0.967 | 0.970 | 0.634 |
| ascending_aorta | 139 | 139 | 0.865 | 0.230 | 0.361 | 0.210 |
| celiac_trunk | 48 | 197 | 0.503 | 0.838 | 0.632 | 0.315 |
| superior_mesenteric | 83 | 98 | 0.721 | 0.541 | 0.702 | 0.408 |
| left_renal | 35 | 50 | 0.895 | 0.180 | 0.508 | 0.264 |
| right_renal | 26 | 45 | 0.261 | 0.711 | 0.466 | 0.229 |
| left_common_iliac | 44 | 44 | 0.063 | 0.091 | 0.092 | 0.053 |
| left_external_iliac | 166 | 166 | 0.380 | 0.502 | 0.377 | 0.278 |
| left_internal_iliac | 82 | 115 | 0.431 | 0.339 | 0.259 | 0.123 |
| right_common_iliac | 74 | 74 | 0.332 | 0.336 | 0.296 | 0.178 |
| right_external_iliac | 205 | 205 | 0.445 | 0.844 | 0.496 | 0.361 |
| right_internal_iliac | 82 | 109 | 0.226 | 0.505 | 0.199 | 0.089 |
| left_subclavian | 181 | 256 | 0.540 | 0.754 | 0.630 | 0.409 |
| brachiocephalic_trunk | 122 | 122 | 0.799 | 0.648 | 0.744 | 0.457 |
| left_common_carotid | 354 | 354 | 0.769 | 0.508 | 0.713 | 0.451 |
| left_external_carotid | 106 | 106 | 0.379 | 0.962 | 0.761 | 0.397 |
| left_internal_carotid | 180 | 181 | 0.482 | 0.547 | 0.491 | 0.247 |
| right_common_carotid | 243 | 248 | 0.538 | 0.802 | 0.723 | 0.519 |
| right_external_carotid | 84 | 85 | 0.576 | 0.412 | 0.537 | 0.313 |
| right_internal_carotid | 185 | 187 | 0.641 | 0.567 | 0.576 | 0.270 |


The model was trained for only 10 epochs on a very small dataset, and the loss curves showed the model had not yet converged — longer training and more data would be expected to yield substantially better results, particularly for underrepresented classes.

---

## Project Structure

```
project/
├── dataset_yolo/
│   ├── images/
│   │   ├── train/          # Training slices (.jpg)
│   │   └── val/            # Validation slices (.jpg)
│   └── labels/
│       ├── train/          # YOLO label files (.txt)
│       └── val/
├── dataset.yaml            # Dataset config for Ultralytics
├── one_day_project.ipynb   # Main notebook
└── README.md
```

---

## Installation

```bash
pip install ultralytics matplotlib numpy Pillow PyYAML
```

---

## Usage

The full pipeline is documented in `one_day_project.ipynb` and covers :

1. **Parsing annotations** — converting JSON annotations to YOLO `.txt` format (one file per slice)
2. **Building the dataset** — flattening the folder structure and generating `dataset.yaml`
3. **Training** — fine-tuning YOLO26s with `model.train()`
4. **Inference** — running predictions on new slices with `model.predict()`

---

## Training Setup

| Parameter | Value |
|---|---|
| Model | YOLO26s |
| Epochs | 10 |
| Batch size | 16 |
| Image size | 512×512 |
| Optimizer | AdamW (auto) |
| Learning rate | 0.0004 (auto) |
| Device | Apple MPS (M5) |
| Training time | ~40 min |

---

## Limitations and Perspectives

- **Dataset size** : trained on a very small number of patient series — the primary limiting factor for performance
- **Class imbalance** : some classes have very few instances in validation (e.g. `left_common_iliac`: 44), making their metrics unreliable
- **Convergence** : the model had not converged at epoch 10 — More epochs would be a natural next step
- **Augmentations** : Brightness variation to simulate different CT protocols could improve generalization
- **Evaluation** : a cross-validation strategy across patients would provide more reliable performance estimates

---

## Hardware

- **Device** : Apple MacBook Pro M5
- **Backend** : PyTorch MPS (Metal Performance Shaders)
- **RAM** : 16 GB unified memory
- **Environment** : Local
