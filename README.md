# 😷 Face Mask Detector

[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![MobileNetV2](https://img.shields.io/badge/Model-MobileNetV2-6A0DAD?logo=google&logoColor=white)](https://arxiv.org/abs/1801.04381)
[![Gradio UI](https://img.shields.io/badge/UI-Gradio-FFD21E?logo=gradio&logoColor=black)](https://gradio.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](https://opensource.org/licenses/MIT)

A two-stage deep learning pipeline that detects faces in any photo and classifies each one as **with mask** or **without mask** — with colored bounding boxes and confidence scores drawn directly on the output image.

---

## 📊 Performance Benchmark

| Metric | Score | Strategy |
| :--- | :--- | :--- |
| **Test Accuracy** | **~98–99%** | Transfer Learning + Fine-Tuning |
| **Final Test Loss** | **< 0.05** | Binary Cross-Entropy |
| **Training Duration** | ~5–8 Minutes | GPU Accelerated (T4 Runtime) |
| **Face Detection** | Multi-face | OpenCV Haar Cascade |

---

## 🔁 Pipeline Architecture

```text
         [ Input Photo ]
                │
    ┌───────────────────────┐
    │   Face Detection      │
    │   OpenCV Haar Cascade │
    │   → Bounding boxes    │
    └───────────┬───────────┘
                │  (one crop per face)
    ┌───────────────────────┐
    │   Preprocessing       │
    │   Resize to 224×224   │
    │   MobileNetV2 scaling │
    └───────────┬───────────┘
                │
    ┌───────────────────────┐       ┌──────────────────────────┐
    │   MobileNetV2 Base    │  +    │   Custom Head            │
    │   Pretrained frozen   │       │   GlobalAvgPool          │
    │   (Phase 1)           │       │   Dense(128) + Dropout   │
    │   Top 30 unfrozen     │       │   Dense(1) → Sigmoid     │
    │   (Phase 2)           │       │                          │
    └───────────┬───────────┘       └──────────────────────────┘
                │
       ┌────────┴────────┐
       ▼                 ▼
  ✅ With mask      ❌ No mask
  + confidence      + confidence
```

---

## ✨ Key Features

- **Transfer learning:** MobileNetV2 pretrained on 1.2M ImageNet images provides rich visual features out of the box — no need to train from scratch.
- **Two-phase training:** Phase 1 trains only the custom head (base frozen). Phase 2 fine-tunes the top 30 MobileNetV2 layers with a low learning rate (1e-5), squeezing out extra accuracy without destroying pretrained weights.
- **Multi-face detection:** Processes every face in a photo independently — works on group photos, not just single portraits.
- **Padded face crops:** Each detected face is padded by 10% before classification, giving the model more context around the face boundary.
- **Annotated output:** Bounding boxes and labels (with confidence %) are drawn directly on the output image. Green = mask on, red = no mask.
- **ReduceLROnPlateau:** Halves the learning rate during Phase 2 if validation loss stalls, ensuring smooth convergence.
- **Early stopping:** Restores best weights automatically if validation performance starts to degrade.

---

## 🛠️ Tech Stack

- **Neural network:** TensorFlow 2.x & Keras
- **Pretrained model:** MobileNetV2 (ImageNet weights)
- **Face detection:** OpenCV Haar Cascade
- **UI:** Gradio
- **Image processing:** Pillow (PIL), NumPy
- **Evaluation:** scikit-learn, Matplotlib, Seaborn

---

## 🗂️ Dataset

**Face Mask Detection** by [Chandrika Deb](https://github.com/chandrikadeb7/Face-Mask-Detection)

| Split | With mask | Without mask |
|-------|-----------|--------------|
| Train (80%) | 552 | 549 |
| Val (20%) | 138 | 137 |
| **Total** | **690** | **686** |

The dataset is balanced — no class weighting needed.

---

## 🚀 How to Run

### Google Colab (Recommended)

1. Upload `face_mask_detector.ipynb` to [Google Colab](https://colab.research.google.com)
2. Enable GPU: **Runtime → Change runtime type → T4 GPU**
3. **Runtime → Run all** (`Ctrl+F9`)
4. Scroll to the last cell and upload any photo to the Gradio app

### Local Machine

```bash
pip install tensorflow gradio opencv-python seaborn pillow scikit-learn

jupyter notebook face_mask_detector.ipynb
```

---

## 💡 What I Learned

- **Transfer learning** — reusing pretrained weights instead of training from scratch
- **Two-phase fine-tuning** — when and why to unfreeze layers
- **ImageDataGenerator** — loading, augmenting, and batching image datasets from folders
- **OpenCV** — face detection with Haar Cascade classifiers
- **Multi-model pipelines** — chaining a detection model with a classification model

---

## 👤 Author

**[Mohamed Ouledali]** — Engineering Student
