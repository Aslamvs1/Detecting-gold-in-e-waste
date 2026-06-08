import cv2
import numpy as np
import tensorflow as tf

# --- CONFIGURATION ---
BINARY_THRESHOLD = 100
MODEL_PATH = 'models/best_gold_model_v2.keras'

# Make it stricter (ignore mouse / noise)
CONFIDENCE_THRESHOLD = 0.90

# GOLD = 0 (as per your training)
GOLD_IS_ZERO = True

# DroidCam index
CAMERA_INDEX = 1


# --- LOAD MODEL ---
print("Loading AI Model...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("SUCCESS: Model loaded.")
except Exception as e:
    print(f"ERROR: Could not load model.\n{e}")
    exit()


# --- CAMERA ---
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)

print("SYSTEM READY. Place object on white paper.")


while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- 1. FIND OBJECT ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(
        blur, BINARY_THRESHOLD, 255, cv2.THRESH_BINARY_INV
    )

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    largest_contour = None
    max_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 3000:  # Filter small noise
            if area > max_area:
                max_area = area
                largest_contour = cnt

    # --- 2. PROCESS OBJECT ---
    if largest_contour is not None:
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Crop Object
        roi = frame[y:y+h, x:x+w]

        if roi.size > 0:
            try:
                # Prepare for AI
                roi_resized = cv2.resize(roi, (224, 224))
                roi_norm = roi_resized / 255.0
                roi_input = np.expand_dims(roi_norm, axis=0)

                # Predict
                prediction = model.predict(roi_input, verbose=0)
                score = prediction[0][0]

                # Decide Label
                is_gold = False

                if GOLD_IS_ZERO:
                    if score < CONFIDENCE_THRESHOLD:
                        is_gold = True
                    conf_val = (1 - score) if is_gold else score
                else:
                    if score > CONFIDENCE_THRESHOLD:
                        is_gold = True
                    conf_val = score if is_gold else (1 - score)

                # Set Text & Color
                if is_gold:
                    label = f"GOLD: {conf_val:.0%}"
                    color = (0, 255, 0)  # Green
                else:
                    label = f"NO GOLD: {conf_val:.0%}"
                    color = (0, 0, 255)  # Red

                # Ensure text doesn't go off-screen
                text_y = y - 10 if y - 10 > 25 else y + h + 25

                # Draw Black Background Box for Text
                (text_w, text_h), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
                )

                cv2.rectangle(
                    frame,
                    (x, text_y - text_h - 5),
                    (x + text_w, text_y + 5),
                    (0, 0, 0),
                    -1
                )

                # Draw Text
                cv2.putText(
                    frame,
                    label,
                    (x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2
                )

                # Draw Bounding Box
                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    color,
                    3
                )

                # Debug print
                print(f"Object Found: {label}")

            except Exception:
                pass

    cv2.imshow('Final Presentation', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
