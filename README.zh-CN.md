<div align="center">

# RSNA 2024 — 腰椎退行性病变分类（YOLO 多尺度集成方案）

**Kaggle RSNA 2024 Lumbar Spine Degenerative Classification 银牌推理方案 — 融合多尺度 YOLOv8 集成与 3D DICOM 影像处理。**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Kaggle](https://img.shields.io/badge/Kaggle-银牌方案-94A3B8.svg)](#项目简介)
[![PyTorch](https://img.shields.io/badge/PyTorch-YOLOv8%20%7C%20MaxViT-EE4C2C.svg)](#推理流程)
[![Medical Imaging](https://img.shields.io/badge/Medical%20AI-3D%20DICOM-0EA5E9.svg)](#推理流程)

[English](README.md) | **简体中文**

</div>

---

## 项目简介

本仓库包含 Kaggle [RSNA 2024 Lumbar Spine Degenerative Classification](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification) 竞赛的**银牌推理 Notebook**（`lsdc-yolo-approach.ipynb`），面向多序列腰椎 MRI DICOM 影像预测各部位病变严重程度概率（`normal_mild`、`moderate`、`severe`）。

## 推理流程

```text
3D MRI DICOM 影像 (.dcm)
        │
        ▼  convert_img.py (pydicom 像素归一化 · 多进程切片转换)
预处理 2D 影像切片 (/images)
        │
        ├──► 多尺度 YOLOv8s 集成 (Fold 0 & 1 @ 384 / 512 / 768 px → pred1–6.csv)
        ├──► 多折 YOLOv8m 集成   (Fold 0 & 1 @ 384 px            → pred7–8.csv)
        └──► 3D 体素 MaxViT 模块 (maxvit3d.py @ 256×256)
        │
        ▼  按 row_id 均值融合各类别概率
submission.csv (normal_mild · moderate · severe)
```

### 核心要点

- **并行 DICOM 预处理（`convert_img.py`）**：使用 `pydicom` 读取 `.dcm` 原始像素阵列并进行归一化，结合 `multiprocessing.Pool` 加速批量转换。
- **多尺度、多主干 YOLOv8 集成（`infer.py`）**：执行 **8 路推理**，组合 `yolov8s`（`384`/`512`/`768` 三尺度 × 双折）与 `yolov8m`（`384` × 双折），兼顾局部细微狭窄特征与全局解剖上下文。
- **概率均值融合**：汇总全部 `pred*.csv` 并按 `row_id` 对 `normal_mild`、`moderate`、`severe` 三个类别求均值，生成更稳健的 `submission.csv`。

## 运行方式（Kaggle / 本地 GPU）

Notebook 适配 Kaggle 离线提交环境（无外网条件下从本地数据集安装 `ultralytics`）。

### 所需数据与权重

- 官方竞赛数据：`/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/`
- 离线 `ultralytics` 安装包：`/kaggle/input/ultralytics`
- 训练好的多折权重：
  - `/kaggle/input/yolo-folds-model3/yolov8s_severe_last`
  - `/kaggle/input/yolo-folds-model4/yolov8m_last`

### 本地复现依赖

```sh
pip install ultralytics torch pydicom opencv-python pillow pandas numpy scikit-learn tqdm
jupyter notebook lsdc-yolo-approach.ipynb
```

## 参与贡献

欢迎提交提升可复现性与代码结构的 PR：

- 补充可复现的 **训练脚本** 与数据划分配置。
- 引入基于 OOF 验证集的 **加权融合策略**（如 Optuna / Nelder-Mead 寻优）。
- 将 Notebook 中的 `convert_img.py`、`infer.py`、`maxvit3d.py` 拆分为清晰的模块化 Python 包。

详见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 开源协议

基于 [MIT License](LICENSE) 开源。
