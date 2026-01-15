# Road Defect Detection using YOLOv9

> **Team InceptionJS** | Crackathon Hackathon Submission

## 🎯 Project Overview

This project implements an advanced **road defect detection system** using YOLOv9 (You Only Look Once version 9) deep learning architecture. The model is trained to detect and classify various types of road damage from images, enabling automated infrastructure assessment and maintenance prioritization.

## 📋 Problem Statement

Road infrastructure maintenance is critical for public safety and transportation efficiency. Manual inspection of road surfaces is:
- Time-consuming and labor-intensive
- Prone to human error and inconsistency
- Difficult to scale across large road networks

This solution automates the detection process using computer vision and deep learning.

## 🏷️ Detection Classes

The model identifies **5 types of road defects**:

| Class ID | Defect Type | Description |
|----------|-------------|-------------|
| 0 | **Longitudinal Crack** | Cracks running parallel to the road direction |
| 1 | **Transverse Crack** | Cracks running perpendicular to the road direction |
| 2 | **Alligator Crack** | Interconnected cracks resembling alligator skin |
| 3 | **Other Corruption** | Miscellaneous road surface damage |
| 4 | **Pothole** | Bowl-shaped depressions in the road surface |

## 🛠️ Technical Architecture

### Model Configuration

- **Base Model**: YOLOv9c (YOLOv9 Compact)
- **Pre-trained Weights**: `yolov9c.pt`
- **Image Size**: 768×768 pixels
- **Framework**: Ultralytics YOLO

### Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| Epochs | 80 |
| Batch Size | 16 |
| Image Size | 768 |
| Optimizer | SGD |
| Initial Learning Rate | 0.01 |
| Final Learning Rate | 0.0001 |
| Momentum | 0.937 |
| Weight Decay | 0.0005 |
| Warmup Epochs | 3.0 |

### Inference Settings

| Parameter | Value |
|-----------|-------|
| Confidence Threshold | 0.15 |
| IoU Threshold | 0.55 |
| Test Time Augmentation (TTA) | Enabled |

### TTA (Test Time Augmentation) Configuration

- **Scales**: [0.67, 0.83, 1.0]
- **Flip Augmentation**: Enabled
- **Sharpen**: Enabled
- **Noise Augmentation**: Enabled

## 📁 Project Structure

```
InceptionJS_Submissions/
├── README.md                 # Project documentation
├── best.pt                   # Trained model weights
├── trainingModel.ipynb       # Training pipeline notebook
├── Inference.ipynb           # Inference/prediction notebook
├── InceptionJSReport.pdf     # Detailed project report
└── predictions/              # Model predictions (YOLO format)
    ├── 000004.txt
    ├── 000019.txt
    └── ... (prediction files for test images)
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- Jupyter Notebook/Lab

### Installation

```bash
# Install required packages
pip install ultralytics>=8.1.0 tensorboard opencv-python-headless tqdm PyYAML kagglehub
```

### Dataset

The project uses the Crackathon dataset from Kaggle:

```python
import kagglehub

# Download dataset
path = kagglehub.dataset_download("anulayakhare/crackathon-data")
print("Path to dataset files:", path)
```

### Running Inference

1. Open `Inference.ipynb` in Jupyter
2. Run all cells sequentially
3. Predictions will be saved to the `predictions/` folder

```python
from ultralytics import YOLO

# Load trained model
model = YOLO("best.pt")

# Run inference
results = model.predict(
    source="path/to/test/images",
    save_txt=True,
    save_conf=True,
    conf=0.1,
    iou=0.55
)
```

### Training (Optional)

To retrain the model:

1. Open `trainingModel.ipynb`
2. Modify configuration settings as needed
3. Run the training pipeline

## 📊 Prediction Format

Predictions are saved in YOLO format (`.txt` files):

```
<class_id> <x_center> <y_center> <width> <height> <confidence>
```

- All coordinates are normalized (0-1)
- One file per image with the same base name
- Empty files indicate no detections

## 💡 Key Features

- ✅ **Multi-class Detection**: Identifies 5 different road defect types
- ✅ **High Accuracy**: Trained with optimized hyperparameters
- ✅ **TTA Support**: Test-time augmentation for improved predictions
- ✅ **Checkpoint Resume**: Training can be resumed from checkpoints
- ✅ **GPU Acceleration**: Optimized for CUDA-enabled GPUs

## 📈 Evaluation Metrics

The model is evaluated using:
- **mAP50**: Mean Average Precision at IoU threshold 0.50
- **mAP50-95**: Mean Average Precision averaged over IoU thresholds from 0.50 to 0.95

## 🔧 Reproducibility

For deterministic training:

```python
CONFIG = {
    "SEED": 42,
    "DETERMINISTIC": True,
    ...
}
```

## 📚 Files Description

| File | Description |
|------|-------------|
| `best.pt` | Trained model weights (best performance) |
| `trainingModel.ipynb` | Complete training pipeline with data preprocessing |
| `Inference.ipynb` | Inference notebook for running predictions |
| `predictions/` | Output directory containing YOLO format predictions |

## 🤝 Team InceptionJS

This project was developed as part of the **Crackathon** hackathon competition.

## 📄 License

This project is submitted as part of a hackathon competition. Please refer to the competition guidelines for usage terms.

---

**Note**: Ensure GPU availability for optimal performance. Training and inference can be performed on CPU but will be significantly slower.
