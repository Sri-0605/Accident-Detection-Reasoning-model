# 🚗 VisionAI: Explainable Traffic Accident Detection and Reasoning

## 📌 Overview

VisionAI is a hybrid deep learning framework designed to detect traffic accidents from video data and generate **human-understandable explanations**.

Unlike traditional black-box models, VisionAI explains:

- ✅ What happened  
- ✅ Which vehicles were involved  
- ✅ Where the accident occurred  
- ✅ Why the model predicted it  

---

## 🎯 Problem Statement

Most accident detection systems:

- ❌ Only classify accident vs non-accident  
- ❌ Do not explain the cause  
- ❌ Fail under domain shift  

### ✔ Our Goal

Build a system that can:

- Detect accident type  
- Identify involved objects  
- Highlight important regions  
- Generate explanations like:

> *“A car moving from the left collided with a motorcycle, causing a side collision.”*

---

## 🧠 Methodology

### 🔹 1. Frame Extraction
- Extract 16 frames per video  
- Resize to `224×224`  
- Normalize using ImageNet stats  

---

### 🔹 2. Spatial Feature Extraction
- Model: **ResNet18 (Pretrained)**
- Extract frame-level features  

---

### 🔹 3. Temporal Modeling
- Model: **BiLSTM + Attention**
- Captures motion patterns  
- Selects **critical frame**

---

### 🔹 4. Explainability
- Method: **Grad-CAM**
- Highlights accident-relevant regions  

---

### 🔹 5. Object Detection
- Model: **YOLOv8 Nano**
- Detects:
  - Cars 🚗
  - Bikes 🏍️
  - Trucks 🚚
  - Pedestrians 🚶  

---

### 🔹 6. Reasoning Module
- Model: **FLAN-T5**
- Generates explanation from:
  - Predicted class  
  - Detected objects  

---

## ⚙️ Architecture Flow
Video Input
↓
Frame Extraction
↓
ResNet18
↓
BiLSTM + Attention
↓
Classification
↓
Grad-CAM (Heatmap)
↓
YOLOv8 (Objects)
↓
FLAN-T5 (Explanation)
↓
Final Output


---

## 📊 Dataset

### 📁 HWID12 Dataset
- 12 accident classes  
- Real-world traffic videos  

### 📁 Roboflow Dataset
- 3 classes  
- Used for domain shift analysis  

---

## 📈 Results

### 🔹 Multi-Class Classification

| Metric | Value |
|------|------|
| Accuracy | **50.65%** |
| Macro F1 | **0.43** |

#### Observations:
- ✅ Best: **Other Crash (F1 = 0.71)**  
- ❌ Weak: Pedestrian Hit, Rear Collision  
- ⚠️ Confusion between similar classes  

---

### 🔹 Binary Classification

| Task | Accuracy |
|------|--------|
| Accident vs No Accident | **90.04%** ✅ |

👉 Insight:
> Binary detection is much easier than multi-class classification.

---

### 🔹 ROC Curve

- AUC Score: **1.00**
- Near-perfect separation

---

## 🔍 Explainability Output

VisionAI produces:

- 🔥 Grad-CAM heatmap  
- 📦 Bounding boxes  
- 🧾 Text explanation  

---

## 🧪 Domain Shift Analysis

### Problem:
Model trained on one dataset performs poorly on another.

### Reason:
- Different lighting  
- Camera angles  
- Dataset distribution  

👉 Conclusion:
> Domain shift is a major real-world challenge.

---

## 🚀 Key Contributions

- Hybrid CNN + LSTM + Attention architecture  
- Grad-CAM based explainability  
- Object-aware reasoning (YOLO + LLM)  
- Domain shift evaluation  

---

## ⚠️ Limitations

- Low accuracy for rare classes  
- Not real-time  
- Depends on object detection quality  
- Fixed frame length (16 frames)  

---

## 🔮 Future Work

- Transformer-based video models  
- Better domain generalization  
- Stronger vision-language models  
- Real-time deployment  

---

## 🛠️ Tech Stack

- Python  
- PyTorch  
- OpenCV  
- Ultralytics YOLOv8  
- HuggingFace Transformers  

---

## 📂 Project Setup

### Install Dependencies and run

```bash
pip install -r requirements.txt
python app.py
