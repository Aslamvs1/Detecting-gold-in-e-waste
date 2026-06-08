import os
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# --- SETTINGS (use your existing paths) ---
TEST_DIR = r"C:/main pro/dataset/test"
MODEL_PATH = r"C:/main pro/models/best_gold_model_v2.keras"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SAVE_PLOT_PATH = r"C:/main pro/models/confusion_matrix_gold.png"  # change if needed

# --- 1. Load model ---
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

# --- 2. Prepare test generator (shuffle=False is critical) ---
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# --- 3. Predict on test set ---
steps = math.ceil(test_generator.samples / float(BATCH_SIZE))
print(f"Predicting on {test_generator.samples} images (steps={steps}) ...")
preds = model.predict(test_generator, steps=steps, verbose=1)

# preds shape: (N,1) for binary. Flatten to 1D array of probabilities.
preds = np.ravel(preds)

# Convert probabilities to class labels using 0.5 threshold
y_pred = (preds >= 0.5).astype(int)

# True labels from generator (already in the correct order because shuffle=False)
y_true = test_generator.classes[:len(y_pred)]

# --- 4. Class label names (ordered by numeric index) ---
# Example mapping: {'gold': 0, 'no_gold': 1}
class_indices = test_generator.class_indices
inv_map = {v: k for k, v in class_indices.items()}
labels = [inv_map[i] for i in sorted(inv_map.keys())]

print("Class indices:", class_indices)
print("Label order (index -> name):", list(enumerate(labels)))

# --- 5. Confusion Matrix & Report ---
cm = confusion_matrix(y_true, y_pred)
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=labels))

print("Confusion Matrix (rows=true, cols=predicted):")
print(cm)

# --- 6. Plot Confusion Matrix (raw and normalized) using matplotlib ---
def plot_confusion_matrix(cm, class_names, normalize=False, title='Confusion matrix', save_path=None):
    if normalize:
        # row-wise normalization (TPR per class). avoid division by zero
        with np.errstate(all='ignore'):
            row_sums = cm.sum(axis=1, keepdims=True)
            cm_norm = np.divide(cm, row_sums, where=(row_sums!=0))
        matrix = cm_norm
        fmt = '.2f'
        title = title + " (normalized)"
    else:
        matrix = cm
        fmt = 'd'

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(matrix, interpolation='nearest')
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(len(class_names)),
           yticks=np.arange(len(class_names)),
           xticklabels=class_names,
           yticklabels=class_names,
           ylabel='True label',
           xlabel='Predicted label',
           title=title)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # annotate cells
    thresh = matrix.max() / 2.0 if matrix.max() != 0 else 0.5
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if normalize:
                text = format(val, fmt)
            else:
                text = format(int(val), fmt)
            ax.text(j, i, text,
                    ha="center", va="center",
                    color="white" if val > thresh else "black")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved confusion matrix to: {save_path}")
    plt.show()

# Plot raw confusion matrix
plot_confusion_matrix(cm, labels, normalize=False, title='Confusion Matrix', save_path=SAVE_PLOT_PATH)

# Plot normalized confusion matrix (per-class proportions)
plot_confusion_matrix(cm, labels, normalize=True, title='Confusion Matrix', save_path=SAVE_PLOT_PATH.replace('.png', '_normalized.png'))
