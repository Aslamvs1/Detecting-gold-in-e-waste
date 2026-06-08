import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import os

# --- 1. CONFIGURATION ---
TRAIN_DIR = r'C:/main pro/dataset/train'
TEST_DIR = r'C:/main pro/dataset/test'

# We use the size you mentioned
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20

# --- 2. DATA LOADING (Simplified) ---
# Since you already augmented/resized, we REMOVE rotation/zoom parameters.
# We only keep 'preprocess_input' to ensure values are -1 to 1 for MobileNetV2.

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input 
    # No rotation_range, No zoom_range, etc. (Since you did it already)
)

test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

print("Loading your pre-processed Training Data...")
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True
)

print("\nLoading your pre-processed Validation Data...")
validation_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# --- 3. BUILD MODEL ---
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.2)(x) 
outputs = Dense(1, activation='sigmoid')(x) # Binary Output

model = Model(inputs=base_model.input, outputs=outputs)

# --- 4. COMPILE ---
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# --- 5. TRAIN ---
checkpoint = ModelCheckpoint(
    r'C:\main pro\models\best_gold_model_v2.keras',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

print("\n--- Starting Training on Pre-Augmented Data ---")
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=[checkpoint, early_stop]
)

# --- 6. PLOT ---
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(acc, label='Train Acc')
plt.plot(val_acc, label='Val Acc')
plt.title('Accuracy (Should be high quickly)')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Train Loss')
plt.plot(val_loss, label='Val Loss')
plt.title('Loss')
plt.legend()
plt.show()