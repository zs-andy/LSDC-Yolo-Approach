<div align="center">

# RSNA 2024 — Lumbar Spine Degenerative Classification (YOLO Ensemble)

**Silver Medal inference pipeline for Kaggle's RSNA 2024 Lumbar Spine Degenerative Classification competition, combining multi-scale YOLOv8 ensembles and 3D DICOM processing.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Kaggle](https://img.shields.io/badge/Kaggle-Silver%20Medal-94A3B8.svg)](#overview)
[![PyTorch](https://img.shields.io/badge/PyTorch-YOLOv8%20%7C%20MaxViT-EE4C2C.svg)](#pipeline-overview)
[![Medical Imaging](https://img.shields.io/badge/Medical%20AI-3D%20DICOM-0EA5E9.svg)](#pipeline-overview)

**English** | [简体中文](README.zh-CN.md)

</div>

---

## Overview

This repository contains the Kaggle **Silver Medal** inference notebook (`lsdc-yolo-approach.ipynb`) for the [RSNA 2024 Lumbar Spine Degenerative Classification](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification) challenge — predicting severity probabilities (`normal_mild`, `moderate`, `severe`) across lumbar spine conditions from multi-series MRI DICOM scans.

## Pipeline Overview

```text
3D MRI DICOM Studies (.dcm)
        │
        ▼  convert_img.py (pydicom · min-max normalization · multiprocessing)
Preprocessed 2D Slices (/images)
        │
        ├──► Multi-Scale YOLOv8s Ensemble (Fold 0 & 1 @ 384 / 512 / 768 px → pred1–6.csv)
        ├──► Multi-Fold YOLOv8m Ensemble  (Fold 0 & 1 @ 384 px            → pred7–8.csv)
        └──► 3D Volumetric MaxViT Stage   (maxvit3d.py @ 256×256)
        │
        ▼  Mean Probability Blending (grouped by row_id)
submission.csv (normal_mild · moderate · severe)
```

### Key Techniques

- **Parallel DICOM Conversion (`convert_img.py`)**: Reads raw `.dcm` pixel arrays via `pydicom`, normalizes intensity ranges, and converts slices in parallel with `multiprocessing.Pool`.
- **Multi-Scale, Multi-Backbone YOLOv8 Ensemble (`infer.py`)**: Runs **8 inference passes** combining `yolov8s` across three input resolutions (`384`, `512`, `768`) and two folds (`0`, `1`) plus `yolov8m` at `384` across folds `0` and `1` to capture both fine local stenosis cues and broader anatomical context.
- **Probability Averaging**: Concatenates all fold/scale prediction CSVs and averages class probabilities (`normal_mild`, `moderate`, `severe`) per `row_id` to produce a well-calibrated final `submission.csv`.

## Running the Notebook (Kaggle / Local GPU)

This notebook is designed for Kaggle's offline submission environment (installing `ultralytics` from a local wheel dataset with internet disabled).

### Required Inputs

- Competition dataset: `/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/`
- Offline `ultralytics` wheels: `/kaggle/input/ultralytics`
- Trained fold checkpoints:
  - `/kaggle/input/yolo-folds-model3/yolov8s_severe_last`
  - `/kaggle/input/yolo-folds-model4/yolov8m_last`

### Local Environment Setup

```sh
pip install ultralytics torch pydicom opencv-python pillow pandas numpy scikit-learn tqdm
jupyter notebook lsdc-yolo-approach.ipynb
```

## Contributing

Pull requests that improve reproducibility and clarity are welcome! Helpful additions include:

- Standalone **training scripts** and dataset-preparation configs for reproducing the YOLOv8 and MaxViT3D checkpoints.
- **Weighted ensemble optimization** (e.g., Nelder-Mead / Optuna blending weights on out-of-fold validation predictions).
- **Modular Python package refactor** extracting `convert_img.py`, `infer.py`, and `maxvit3d.py` out of notebook magic cells.

See [CONTRIBUTING.md](CONTRIBUTING.md) before opening a PR.

## License

Licensed under the [MIT License](LICENSE).
