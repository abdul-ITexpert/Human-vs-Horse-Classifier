# 🐎 Horse vs Human Image Classifier using Transfer Learning



A deep learning image classification project that distinguishes **horses** from **humans** using **Transfer Learning with MobileNetV2**. This project demonstrates the complete machine learning workflow—from building an initial custom CNN that suffered from overfitting to developing a production-ready transfer learning model with **97.27% validation accuracy**.

The project also includes an interactive **Streamlit web application** for real-time image classification.

---

# 📖 Project Overview

The objective of this project was to build an accurate binary image classifier while exploring how different deep learning architectures affect model performance and generalization.

Instead of stopping after training a basic CNN, the project focused on understanding model limitations, reducing overfitting, and applying Transfer Learning to achieve production-level performance.

---

# 🚀 Project Journey

## Phase 1 – Custom CNN

The first version of the model was built from scratch using multiple convolutional layers followed by several fully connected layers.

### Architecture

* Multiple Convolution + MaxPooling blocks
* Flatten layer
* Dense Layers:

  * 72 neurons
  * 64 neurons
  * 32 neurons
  * 16 neurons
  * 8 neurons
  * 4 neurons
* Sigmoid output layer

### Result

Although the model achieved extremely high training accuracy, it failed to generalize to unseen images.

| Metric              |      Value |
| ------------------- | ---------: |
| Training Accuracy   | **99.42%** |
| Validation Accuracy | **69.92%** |
| Validation Loss     | **5.2291** |

### Issues Identified

* Severe overfitting
* Large train-validation performance gap
* Model memorized training data
* Poor generalization
* Consistently predicted one class on unseen images

---

## Phase 2 – Model Regularization

To improve generalization, several optimization techniques were introduced.

### Improvements

* Data Augmentation

  * Rotation
  * Width Shift
  * Height Shift
  * Shear
  * Zoom
* Dropout layers
* Early Stopping
* Better network design
* Reduced model complexity

### Performance

| Metric              |      Value |
| ------------------- | ---------: |
| Training Accuracy   | **81.30%** |
| Validation Accuracy | **83.20%** |
| Validation Loss     | **0.4254** |

The model became significantly more stable and generalized much better to unseen data.

---

## Phase 3 – Transfer Learning (Final Model)

To achieve production-quality performance, the project was upgraded using **MobileNetV2**, a lightweight convolutional neural network pre-trained on the ImageNet dataset.

### Key Improvements

* Pre-trained MobileNetV2 feature extractor
* Frozen convolutional base
* GlobalAveragePooling2D
* Dense layer (64 neurons)
* Dropout (0.4)
* Sigmoid output layer

This dramatically reduced the number of trainable parameters while improving feature extraction and overall accuracy.

### Final Results

| Metric              |      Value |
| ------------------- | ---------: |
| Training Accuracy   | **98.15%** |
| Validation Accuracy | **97.27%** |
| Validation Loss     | **0.0743** |

The model converged within only a few epochs and demonstrated excellent generalization performance.

---

# 🏗 Model Architecture

```text
Input Image
      │
      ▼
Resize (64 × 64)
      │
      ▼
MobileNetV2 Preprocessing
      │
      ▼
MobileNetV2 (Frozen Base)
      │
      ▼
GlobalAveragePooling2D
      │
      ▼
Dense (64, ReLU)
      │
      ▼
Dropout (0.4)
      │
      ▼
Dense (1, Sigmoid)
      │
      ▼
Prediction
Horse / Human
```

---

# ✨ Features

* Binary image classification
* Transfer Learning with MobileNetV2
* TensorFlow/Keras implementation
* ImageNet preprocessing pipeline
* Early stopping during training
* Streamlit web application
* Real-time image prediction
* Confidence score display
* Lightweight and fast inference

---

# 📊 Model Performance

| Model           | Training Accuracy | Validation Accuracy | Validation Loss | Status           |
| --------------- | ----------------: | ------------------: | --------------: | ---------------- |
| Custom CNN      |            99.42% |              69.92% |          5.2291 | Overfitted       |
| Regularized CNN |            81.30% |              83.20% |          0.4254 | Improved         |
| MobileNetV2     |        **98.15%** |          **97.27%** |      **0.0743** | Production Ready |

---

# 📁 Project Structure

```text
horse-human-classifier/
│
├── data/
│   ├── train/
│   │   ├── horses/
│   │   └── humans/
│   │
│   └── test_data/
│       ├── horses/
│       └── humans/
│
├── models/
│   └── horse_human_mobilenet.keras
│
├── notebooks/
│   └── training.ipynb
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

# ⚙ Installation

Clone the repository:

```bash
git clone https://github.com/abdul-ITexpert/Human-vs-Horse-Classifier.git
```

Move into the project directory:

```bash
cd Human-vs-Horse-Classifier
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If you don't have a requirements file:

```bash
pip install tensorflow streamlit pillow numpy
```

---

# ▶ Running the Application

Ensure the trained model (`horse_human_mobilenet.keras`) is located in the project directory.

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

---

# 🧠 Technologies Used

* Python
* TensorFlow
* Keras
* MobileNetV2
* NumPy
* Pillow
* Streamlit

---

# 🔍 Learning Outcomes

Through this project, I gained practical experience in:

* Building CNNs from scratch
* Identifying and reducing overfitting
* Data augmentation techniques
* Transfer Learning
* Model evaluation and comparison
* Hyperparameter tuning
* Deploying deep learning models with Streamlit
* Designing user-friendly ML applications

---

# 🚀 Future Improvements

* Fine-tune the upper layers of MobileNetV2 using a very low learning rate.
* Increase image resolution from **64×64** to **128×128** or **224×224**.
* Support batch image prediction.
* Deploy the application on Streamlit Community Cloud or Railway.
* Add Grad-CAM visualization for model explainability.

---

# 👨‍💻 Author

**Abdul Hanan**

**AI/ML Engineer | WordPress Developer**

If you found this project useful, consider giving it a ⭐ to support the repository.
