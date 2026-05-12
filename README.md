# WasteNet ♻️  
### AI-Powered Waste Classification & Recycling Assistance System

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red?style=for-the-badge&logo=pytorch)
![Flask](https://img.shields.io/badge/Flask-Backend-black?style=for-the-badge&logo=flask)
![Accuracy](https://img.shields.io/badge/Accuracy-95.15%25-brightgreen?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

> Deep Learning waste classification system capable of detecting and classifying waste images into **6 categories** in real time using multiple neural network architectures, transfer learning, and transformer-based vision models.

---

# 📌 Project Overview

WasteNet is a university Deep Learning & Computer Vision project focused on solving a real-world environmental problem through AI.

The system classifies waste images into recyclable categories to support:

- Smart waste management
- Recycling assistance
- Environmental sustainability
- AI-based waste sorting systems

The project progressively explores **5 deep learning architectures** — from a custom CNN baseline to advanced transformer-based models.

### 🏆 Best Model

| Model | Accuracy | F1 Score |
|---|---|---|
| ViT-Small | **95.15%** | **0.9516** |

---

# 🗂 Waste Classes

The model classifies images into the following categories:

- ♻️ Plastic
- 📄 Paper
- 🔩 Metal
- 🍾 Glass
- 🍌 Organic Waste
- 💻 E-Waste

---

# 🧠 Models Used

| Model | Accuracy | Strategy |
|---|---|---|
| Baseline CNN | 50.24% | Built from scratch |
| Improved CNN | 61.65% | BatchNorm + Residual Blocks |
| EfficientNet-B0 | 94.42% | Transfer Learning |
| ViT-Small ⭐ BEST | **95.15%** | Vision Transformer |
| Hybrid CNN-Attention | 40.53% | ResNet50 + CBAM |

---

# 📂 Project Structure

```bash
wastenet_project/

├── notebooks
│   ├── 01_data_collection.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_baseline_cnn.ipynb
│   ├── 04_improved_cnn.ipynb
│   ├── 05_transfer_learning.ipynb
│   ├── 06_vision_transformer.ipynb
│   ├── 07_hybrid_attention.ipynb
│   └── 08_analysis_report.ipynb
│
└── dataset/
```

---

# 🚀 Quick Start

## 1️⃣ Clone Repository

```bash
git clone https://github.com/yourusername/WasteNet.git
cd WasteNet
```

---

## 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 3️⃣ Run AI Backend

```bash
python app.py
```

---

## 4️⃣ Start Dashboard Server

Open another terminal:

```bash
python -m http.server 8080
```

---

## 5️⃣ Open Dashboard

Open in browser:

```bash
http://127.0.0.1:8080/waste_nn_dashboard.html
```

---

# 📸 Dashboard Features

- Upload waste images
- Real-time camera detection
- Confidence score visualization
- Recycling recommendations
- Model comparison dashboard

---

# ⚙️ Tech Stack

| Technology | Usage |
|---|---|
| PyTorch | Deep Learning |
| Flask | Backend API |
| timm | Vision Transformer Models |
| HTML/CSS/JS | Dashboard |
| OpenCV | Image Processing |

---

# 📈 Results

| Metric | Best Score |
|---|---|
| Accuracy | 95.15% |
| F1 Score | 0.9516 |
| Best Model | ViT-Small |

---

# 🌍 Real-World Applications

- Smart recycling bins
- Waste sorting systems
- Environmental monitoring
- AI recycling assistants

---

# 👨‍💻 Authors

Developed as a Neural Networks & Deep Learning university project.

---

# 📜 License

MIT License
