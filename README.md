# HELMET-DETECTION-YOLOV8


This is an enhanced helmet and vest detection system built on **YOLOv8**, upgraded with **SPD-Conv layers** and **EMA Attention** to improve feature extraction, spatial awareness, and detection robustness.  
This model is designed for construction-site safety monitoring to ensure workers comply with PPE (Personal Protective Equipment) norms.

---

## 🚀 Detection Results

### 🖼️ Image-Based Detection Demo

The model accurately detects **helmets** and **vests** across diverse construction-site environments using the improved YOLOv8 + SPD-EMA architecture.

**SAMPLE IMAGES**


<p align="center">
  <img src="sample_images/hard_hat_workers1346_png.rf.cbfddfe0bb6cfe30eefc3f11efc83ec2.jpg" alt="Sample 1" width="45%">
  <img src="sample_images/hard_hat_workers957_png.rf.963e0a153b3c8fb648135d3890dbdde6.jpg" alt="Sample 2" width="45%">
</p>



🔗 **[Google Drive – Full Results Folder](https://drive.google.com/drive/folders/1mN1S-gdZScozvR29-8DpZ5WK6loHT276?usp=drive_link)**  
Contains **500+ output images** generated from the model, showcasing full inference results.

---

## ⚙️ Model Architecture Enhancements

HelmetGuard improves YOLOv8 by integrating:

### 🔷 SPD-Conv (Space–Depth Convolution)
- Enhances multi-scale feature extraction  
- Improves spatial–channel fusion  
- Strengthens small-object detection (helmets)

### 🔶 EMA Attention (Exponential Moving Average Attention)
- Stabilizes feature activation  
- Reduces noise sensitivity  
- Helps model focus on crucial regions (helmet + vest zones)

Together, SPD-Conv + EMA significantly boost accuracy and feature consistency compared to baseline YOLOv8.

The idea/concept has been explained in detail in the report associated in this github repository.

---

## 📈 Training Metrics

<p align="center">
  <img src="yolo_results/confusion_matrix.png" width="45%"/>
  <img src="yolo_results/P_curve.png" width="45%"/>
  <br>
  <img src="yolo_results/R_curve.png" width="45%"/>
  <img src="yolo_results/F1_curve.png" width="45%"/>
</p>


**Final Reported Metrics:**

| Metric | Value | Meaning |
|-------|-------|---------|
| 🎯 **Precision** | 93% | Correct positive detections |
| 🔎 **Recall** | 92% | Ability to find all objects |
| 🏆 **mAP@0.5** | 92.4% | Overall detection performance |

This confirms the model’s strong capability for real-world construction-site safety monitoring.

---

## 🛠️ Installation Guide

```bash
# Clone this repository
git clone https://github.com/yourusername/HELMET_DETECTION-YOLOV8.git
cd HELMET_DETECTION-YOLOV8

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 📦 Requirements

The requirements.txt file contains all necessary Python packages for this project:

- `ultralytics` (for YOLOv8)
- `opencv-python` (OpenCV)
- And their dependencies

If installation causes issues, install core dependencies manually:

```bash
pip install ultralytics opencv-python
```
Train the model:

```bash
cd scripts
python train.py
```
