# 😷 Face Mask Detector

[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![MediaPipe](https://img.shields.io/badge/Detection-MediaPipe-00E676?logo=google&logoColor=white)](https://google.github.io/mediapipe/)
[![MobileNetV2](https://img.shields.io/badge/Model-MobileNetV2-6A0DAD?logo=google&logoColor=white)](https://arxiv.org/abs/1801.04381)
[![Gradio UI](https://img.shields.io/badge/UI-Gradio-FFD21E?logo=gradio&logoColor=black)](https://gradio.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](https://opensource.org/licenses/MIT)

A two-stage deep learning pipeline that detects faces in any photo and classifies each one as **with mask** or **without mask** — with colored bounding boxes and confidence scores drawn on the output image.

---

## 📊 Performance

| Metric | Score |
| :--- | :--- |
| **Validation Accuracy** | **~98–99%** |
| **Validation Loss** | **< 0.05** |
| **Training Time** | ~5–8 min (T4 GPU) |
| **Face Detection** | Multi-face, any angle |

---

## 🔁 Pipeline Architecture

```
        [ Input Image ]
               │
  ┌────────────▼────────────┐
  │   MediaPipe BlazeFace   │  ← handles partial faces,
  │   Face Detection        │    angles, group photos
  └────────────┬────────────┘
               │  (one padded crop per face)
  ┌────────────▼────────────┐
  │   Preprocessing         │  ← preprocess_input is embedded
  │   (inside the model)    │    inside the model graph,
  │   Resize → 224×224      │    not applied manually
  └────────────┬────────────┘
               │
  ┌────────────▼────────────┐     ┌─────────────────────────┐
  │   MobileNetV2 Backbone  │  +  │   Classification Head   │
  │   Phase 1: fully frozen │     │   GlobalAveragePooling  │
  │   Phase 2: top 30 layers│     │   Dense(128) + Dropout  │
  │           unfrozen      │     │   Dense(1) → Sigmoid    │
  └────────────┬────────────┘     └─────────────────────────┘
               │
      ┌────────┴────────┐
      ▼                 ▼
  ✅ With mask      ❌ No mask
  + confidence      + confidence
```

---

## ✨ Key Features

- **Transfer learning** — MobileNetV2 pretrained on ImageNet provides strong visual features without training from scratch.
- **Two-phase fine-tuning** — Phase 1 trains only the custom head with the backbone frozen. Phase 2 unfreezes the top 30 MobileNetV2 layers and continues at a low learning rate (`1e-5`) to avoid destroying pretrained weights.
- **MediaPipe face detection** — BlazeFace (MediaPipe's engine) replaces Haar Cascades. It handles angled faces, partial occlusions, and multiple faces in a single photo far more reliably.
- **Integrated preprocessing** — `preprocess_input` is embedded inside the Keras model graph rather than applied in the inference function. This eliminates a common deployment bug where the training and inference scaling steps get out of sync.
- **Fast data pipeline** — `tf.data.image_dataset_from_directory` with parallel mapping and `prefetch(AUTOTUNE)` replaces `ImageDataGenerator`, removing the CPU bottleneck during training.
- **Padded face crops** — Each detected face is padded by 15% before classification, giving the model context around the face boundary.
- **Early stopping + ReduceLROnPlateau** — Best weights are restored automatically if validation loss degrades. The learning rate halves during Phase 2 if validation loss stalls for 2 epochs.

---

## 🛠️ Tech Stack

| Component | Library |
|-----------|---------|
| Neural network | TensorFlow 2.x / Keras |
| Pretrained backbone | MobileNetV2 (ImageNet weights) |
| Face detection | MediaPipe (BlazeFace) |
| Data pipeline | `tf.data` |
| UI | Gradio Blocks |
| Image processing | Pillow, NumPy, OpenCV |
| Evaluation | scikit-learn, Matplotlib, Seaborn |

---

## 🗂️ Dataset

**Face Mask Detection** by [Chandrika Deb](https://github.com/chandrikadeb7/Face-Mask-Detection)

| Split | With Mask | Without Mask |
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
4. In the last cell, a Gradio public URL will appear — open it and upload any photo

### Local Machine

```bash
pip install tensorflow gradio mediapipe opencv-python-headless seaborn pillow scikit-learn

jupyter notebook face_mask_detector.ipynb
```

> **Note:** Remove `share=True` from `app.launch()` when running locally — it's only needed in Colab to generate the public tunnel URL.

---

## 💡 What I Learned

- **Transfer learning** — reusing pretrained ImageNet weights and adapting them to a new task
- **Two-phase fine-tuning** — when to freeze layers, when to unfreeze them, and why the learning rate must drop before you do
- **`tf.data` pipelines** — building fast, parallelized data loading that doesn't bottleneck GPU training
- **MediaPipe** — using a neural face detector instead of Haar Cascades, and understanding the difference in robustness
- **Model-integrated preprocessing** — why embedding `preprocess_input` inside the model graph is safer than doing it externally
- **Multi-model pipelines** — chaining a detection model (MediaPipe) with a classification model (MobileNetV2)
- **Evaluation** — reading a confusion matrix and classification report, not just a single accuracy number

---

## 👤 Author

**Mohamed Ouledali** — Engineering Student
