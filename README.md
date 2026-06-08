# Gold Component Detection in E-Waste using Deep Learning

An automated Computer Vision solution designed to detect and classify gold components within electronic waste (e-waste). By utilizing Transfer Learning with **MobileNetV2** and **TensorFlow/Keras**, this project aims to assist in the sustainable recycling process by identifying valuable metallic components from image data.

## 🚀 Project Overview
Electronic waste recycling relies heavily on sorting. This project builds a binary classification pipeline that separates images into "Gold" and "No Gold" categories. It leverages a pre-trained convolutional neural network (CNN) to detect distinct visual signatures such as metallic reflectivity, color gradients, and geometric patterns common in electronic components.

## 🛠️ Tech Stack
*   **Language:** Python
*   **Deep Learning Framework:** TensorFlow / Keras
*   **Image Processing:** PIL (Pillow), NumPy
*   **Visualization:** Matplotlib
*   **Base Architecture:** MobileNetV2 (Pre-trained on ImageNet)

## 📦 Pipeline & Features
1.  **Data Preprocessing:** Images are resized to `224x224` to match the model's input requirements. 
2.  **Pixel Normalization:** Implements mathematical scaling via `preprocess_input` to squash raw 0–255 pixel values down to a stable `-1 to 1` range, preventing exploding gradients and speeding up training.
3.  **Data Augmentation:** The dataset is pre-augmented to ensure model invariance against variations in orientation, rotation, and lighting conditions.
4.  **Transfer Learning:** Freezes the base MobileNetV2 layers to retain low-level feature extraction capabilities, adding a custom head (Global Average Pooling, Dense 128 with ReLU, and Dropout) for the final binary Sigmoid decision.

## 📂 Project Structure
```text
├── dataset/
│   ├── train/          # Pre-augmented training images (Gold vs No-Gold)
│   └── test/           # Validation/Test image directory
├── models/
│   └── best_gold_model_v2.keras  # Saved checkpoint of the highest-performing model
├── train.py            # Main training and evaluation script
└── README.md
⚡ How to Run
Clone the repository:

Bash
   git clone [https://github.com/YOUR_USERNAME/gold-e-waste-detection.git](https://github.com/YOUR_USERNAME/gold-e-waste-detection.git)
   cd gold-e-waste-detection
Install dependencies:

Bash
   pip install tensorflow matplotlib pillow numpy
Configure Paths:
Open train.py and update the TRAIN_DIR and TEST_DIR paths to point to your local dataset directories.

Train the Model:

Bash
   python train.py
The script includes ModelCheckpoint to automatically save the best performing model based on validation accuracy, and EarlyStopping to prevent overfitting.

📈 Optimization & Callbacks
Optimizer: Adam (Learning Rate: 0.0001)

Loss Function: binary_crossentropy

Early Stopping: Set to a patience of 5 epochs monitoring val_loss to capture the model at its peak generalization state.


---

### Tips for Customizing This:
*   Replace `YOUR_USERNAME` in the clone link with your actual GitHub username.
*   If you have a chart image from your Matplotlib output, you can save it as `metrics.png` and
