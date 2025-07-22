import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# --- Configuration ---
TFLITE_MODEL_PATH = 'recyclable_item_classifier.tflite'
DATASET_DIR = 'dataset'
IMAGE_SIZE = (150, 150)
BATCH_SIZE = 32

# --- 1. Load the TFLite Model and Allocate Tensors ---
if not os.path.exists(TFLITE_MODEL_PATH):
    print(f"Error: TFLite model not found at '{TFLITE_MODEL_PATH}'")
    exit()

print("Loading TFLite model and allocating tensors...")
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()

# Get input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("Model loaded and tensors allocated.")

# --- 2. Prepare the Validation Data ---
if not os.path.exists(DATASET_DIR):
    print(f"Error: Dataset directory not found at '{DATASET_DIR}'")
    exit()

# Use ImageDataGenerator for rescaling, just like in training
# We use the 'validation' subset of the data for evaluation
validation_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

validation_generator = validation_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False  # Important: Do not shuffle for evaluation
)

# --- 3. Evaluate the Model ---
print("\nEvaluating TFLite model...")
correct_predictions = 0
total_predictions = 0

# Get the class indices mapping
class_indices = validation_generator.class_indices
labels = {v: k for k, v in class_indices.items()}

for i in range(len(validation_generator)):
    # Get a batch of images and their true labels
    images, true_labels = validation_generator[i]
    total_predictions += len(true_labels)

    for j in range(len(images)):
        # Get a single image and its label
        image = images[j]
        true_label_index = np.argmax(true_labels[j])

        # Preprocess the image and set it as input to the interpreter
        input_data = np.expand_dims(image, axis=0).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)

        # Run inference
        interpreter.invoke()

        # Get the prediction
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predicted_label_index = np.argmax(output_data)

        # Check if the prediction was correct
        if predicted_label_index == true_label_index:
            correct_predictions += 1

    print(f"Processed batch {i+1}/{len(validation_generator)}...")

# --- 4. Display Results ---
accuracy = (correct_predictions / total_predictions) * 100
print("\n--- TFLite Model Evaluation ---")
print(f"Total images evaluated: {total_predictions}")
print(f"Correct predictions: {correct_predictions}")
print(f"Accuracy: {accuracy:.2f}%")

# Example of a single prediction
print("\n--- Example Prediction ---")
sample_images, sample_labels = validation_generator[0]
sample_image = sample_images[0]
true_label_name = labels[np.argmax(sample_labels[0])]

# Run inference on the single sample image
input_data = np.expand_dims(sample_image, axis=0).astype(np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])
predicted_label_index = np.argmax(output_data)
predicted_label_name = labels[predicted_label_index]

print(f"True label: {true_label_name}")
print(f"Predicted label: {predicted_label_name}") 