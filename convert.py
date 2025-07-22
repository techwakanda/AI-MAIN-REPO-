import tensorflow as tf
import os

# --- Configuration ---
SAVED_MODEL_PATH = 'recyclable_item_classifier.h5'
TFLITE_MODEL_PATH = 'recyclable_item_classifier.tflite'

# --- 1. Load the Keras Model ---
if not os.path.exists(SAVED_MODEL_PATH):
    print(f"Error: Saved model not found at '{SAVED_MODEL_PATH}'")
    exit()

print(f"Loading model from {SAVED_MODEL_PATH}...")
model = tf.keras.models.load_model(SAVED_MODEL_PATH)
print("Model loaded successfully.")

# --- 2. Convert to TensorFlow Lite ---
print("Converting model to TensorFlow Lite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# --- 3. (Optional) Apply Optimizations ---
# Default optimization strategy (quantization)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
print("Applying default optimizations (quantization)...")

# --- 4. Perform Conversion ---
tflite_model = converter.convert()
print("Model conversion successful.")

# --- 5. Save the TFLite Model ---
with open(TFLITE_MODEL_PATH, 'wb') as f:
    f.write(tflite_model)
print(f"TFLite model saved to {TFLITE_MODEL_PATH}")

# --- 6. Verification ---
# You can verify the model size
keras_model_size = os.path.getsize(SAVED_MODEL_PATH) / (1024 * 1024) # in MB
tflite_model_size = os.path.getsize(TFLITE_MODEL_PATH) / 1024 # in KB
print(f"\nOriginal Keras model size: {keras_model_size:.2f} MB")
print(f"Converted TFLite model size: {tflite_model_size:.2f} KB")
print(f"Reduction ratio: {keras_model_size * 1024 / tflite_model_size:.2f}x smaller") 