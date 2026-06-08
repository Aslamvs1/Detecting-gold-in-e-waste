import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import os

# --- 1. SETTINGS ---
# PASTE YOUR IMAGE PATH HERE
IMAGE_PATH = r"C:/main pro/dataset/test/gold/163_bright.jpg"

# Path to your trained model
MODEL_PATH = r"C:/main pro/models/best_gold_model_v2.keras"

# --- 2. LOAD THE MODEL ---
# We load the model only once to save time
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model not found at {MODEL_PATH}")
    exit()

print("Loading model... please wait.")
model = tf.keras.models.load_model(MODEL_PATH)

# --- 3. PREDICTION FUNCTION ---
def predict_single_image(path):
    try:
        # Load the image and resize it to 224x224 (MobileNet standard)
        img = load_img(path, target_size=(224, 224))
        
        # Convert to array
        img_array = img_to_array(img)
        
        # Preprocess (Scale values to -1 to 1) - CRITICAL STEP
        img_array = preprocess_input(img_array)
        
        # Add batch dimension (Make it 1, 224, 224, 3)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make Prediction
        prediction = model.predict(img_array, verbose=0)[0][0]
        
        # --- INTERPRET RESULT ---
        # Alphabetical Order: Gold=0, No_Gold=1
        if prediction < 0.5:
            label = "GOLD"
            confidence = (1 - prediction) * 100
        else:
            label = "NO GOLD"
            confidence = prediction * 100
            
        print("\n" + "="*30)
        print(f"RESULT: {label}")
        print(f"Confidence: {confidence:.2f}%")
        print(f"Raw Score: {prediction:.4f}")
        print("="*30 + "\n")
        
    except Exception as e:
        print(f"Error processing image: {e}")

# --- 4. RUN ---
predict_single_image(IMAGE_PATH)