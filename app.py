from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import tensorflow as tf
import json
import os
import traceback
import logging

app = Flask(__name__)
CORS(app)

# Set up logging
logging.basicConfig(level=logging.INFO)

# --- Configuration ---
MODEL_PATH = 'snake_model.tflite'
LABELS_PATH = 'labels.txt'
SNAKE_DATA_PATH = 'snake_data.json'
# --- NEW: Set a confidence threshold (e.g., 50%) ---
# If the model's confidence is below this, we'll consider it a wrong image.
CONFIDENCE_THRESHOLD = 0.68

# --- Load Resources ---
interpreter = None
class_labels = []
snake_data_map = {}
input_details = None
output_details = None

try:
    # Load the TFLite model and allocate tensors.
    if os.path.exists(MODEL_PATH):
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(f"✅ TFLite model loaded successfully from {MODEL_PATH}.")
    else:
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

    # Load class labels from the text file
    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH, 'r', encoding='utf-8') as f:
            class_labels = [line.strip() for line in f.readlines()]
        print(f"✅ Class labels loaded successfully. Found {len(class_labels)} classes.")
    else:
        raise FileNotFoundError(f"Labels file not found at {LABELS_PATH}")

    # Load snake details and create a robust lookup map
    if os.path.exists(SNAKE_DATA_PATH):
        with open(SNAKE_DATA_PATH, 'r', encoding='utf-8') as f:
            snake_details_list = json.load(f)
            for item in snake_details_list:
                # Clean key for robust matching
                key = item['ClassName'].lower().strip().replace('-', ' ')
                snake_data_map[key] = item
        print("✅ Snake details JSON loaded and indexed successfully.")
    else:
        raise FileNotFoundError(f"Snake data file not found at {SNAKE_DATA_PATH}")

except Exception as e:
    print(f"❌ An error occurred during server startup: {e}")
    interpreter = None # Ensure the app knows the model isn't ready

def preprocess_image(image, target_size=(224, 224)):
    """Resizes and normalizes the image for the TFLite model."""
    img = image.resize(target_size)
    img_array = np.array(img, dtype=np.float32)
    if img_array.shape[-1] == 4: # Handle PNG transparency
        img_array = img_array[..., :3]
    img_array = np.expand_dims(img_array, axis=0)
    # MobileNetV2 preprocessing is often baked into the TFLite model,
    # but applying it here is safer if the conversion didn't include it.
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    """Handles the image upload, prediction, and data lookup."""
    if interpreter is None or not class_labels or not snake_data_map:
        return jsonify({'error': 'Server resources are not loaded. Please check server startup logs.'}), 500
    if 'image' not in request.files:
        return jsonify({'error': 'No image file uploaded.'}), 400

    print("\n--- [DEBUG] Received a new prediction request ---")
    try:
        image_file = request.files['image']
        print("[DEBUG] Step 1: Image file received.")
        
        image = Image.open(image_file.stream).convert('RGB')
        print("[DEBUG] Step 2: Image opened and converted to RGB.")
        
        processed_image = preprocess_image(image)
        print(f"[DEBUG] Step 3: Image preprocessed. Shape: {processed_image.shape}")
        
        print("[DEBUG] Step 4: Invoking TFLite interpreter...")
        # Set the tensor to the model's input
        interpreter.set_tensor(input_details[0]['index'], processed_image)
        # Run inference
        interpreter.invoke()
        # Get the prediction result
        prediction = interpreter.get_tensor(output_details[0]['index'])[0]
        print("[DEBUG] Step 5: TFLite prediction successful.")
        
        predicted_index = np.argmax(prediction)
        confidence = float(prediction[predicted_index])
        predicted_class_name = class_labels[predicted_index]
        print(f"[DEBUG] Prediction result - Class: '{predicted_class_name}', Confidence: {confidence:.2f}")

        # --- NEW: Check if confidence is below the threshold ---
        if confidence < CONFIDENCE_THRESHOLD:
            print(f"❌ [ERROR] Low confidence ({confidence:.2f}). Rejecting prediction.")
            error_message = f"Could not confidently identify a snake in this image (Confidence: {confidence:.2%}). This may be the wrong type of image. Please try a clearer picture."
            return jsonify({'error': error_message}), 400

        lookup_key = predicted_class_name.lower().strip().replace('-', ' ')
        snake_details = snake_data_map.get(lookup_key)

        if not snake_details:
            print(f"❌ [ERROR] Details not found for key: '{lookup_key}'")
            return jsonify({
                'error': f"Prediction successful ('{predicted_class_name}'), but details not found in database."
            }), 404

        response_data = {**snake_details, 'Confidence': f"{confidence:.2%}"}
        return jsonify(response_data)

    except Exception as e:
        print(f"❌ [ERROR] An unexpected error occurred during prediction.")
        traceback.print_exc()
        return jsonify({'error': 'An internal server error occurred. Check the server console for details.'}), 500

@app.route('/welcome', methods=['GET'])
def welcome():
    """Returns a welcome message and logs the request."""
    app.logger.info(f"Request received: {request.method} {request.path}")
    return jsonify({"message": "Welcome to the Wildlife Safety API!"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

