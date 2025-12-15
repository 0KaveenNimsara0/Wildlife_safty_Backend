from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import json
import os
import traceback
import logging

app = Flask(__name__)
CORS(app)

# Set up logging
logging.basicConfig(level=logging.INFO)

# --- Configuration ---
MODEL_PATH = 'snake_model.h5'
LABELS_PATH = 'labels1.txt'
SNAKE_DATA_PATH = 'snake_data.json'
CONFIDENCE_THRESHOLD = 0.68

# --- Load Resources ---
model = None
class_labels = []
snake_data_map = {}

try:
    # Load the Keras model with custom objects to handle compatibility issues
    if os.path.exists(MODEL_PATH):
        # Try different loading approaches for compatibility
        try:
            model = load_model(MODEL_PATH, compile=False)
            print(f"✅ Keras model loaded successfully from {MODEL_PATH}.")
        except Exception as e:
            print(f"⚠️  Standard load failed, trying with custom objects: {e}")
            # Try loading with custom objects for compatibility
            model = load_model(
                MODEL_PATH,
                compile=False,
                custom_objects={
                    'Functional': tf.keras.Model,
                    'Adam': tf.keras.optimizers.Adam
                }
            )
            print(f"✅ Keras model loaded with custom objects from {MODEL_PATH}.")
    else:
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

    # Load class labels from the text file
    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH, 'r', encoding='utf-8') as f:
            class_labels = [line.strip() for line in f.readlines()]
        print(f"✅ Class labels loaded successfully. Found {len(class_labels)} classes.")
        print(f"Classes: {class_labels}")
    else:
        raise FileNotFoundError(f"Labels file not found at {LABELS_PATH}")

    # Load snake details from JSON
    if os.path.exists(SNAKE_DATA_PATH):
        with open(SNAKE_DATA_PATH, 'r', encoding='utf-8') as f:
            snake_details_list = json.load(f)
            for item in snake_details_list:
                # Create multiple lookup keys for robustness
                class_name = item['ClassName']
                key1 = class_name.lower().strip().replace('-', ' ').replace('_', ' ')
                key2 = class_name.lower().strip()
                key3 = class_name.replace('-', ' ').replace('_', ' ').lower().strip()
                
                snake_data_map[key1] = item
                snake_data_map[key2] = item
                snake_data_map[key3] = item
                
                # Also add common name variations
                if 'CommonEnglishNames' in item:
                    common_names = item['CommonEnglishNames'].split(',')
                    for common_name in common_names:
                        common_key = common_name.strip().lower()
                        snake_data_map[common_key] = item
        
        print(f"✅ Snake details JSON loaded successfully. Found {len(snake_details_list)} species.")
        print(f"Available species: {[item['ClassName'] for item in snake_details_list]}")
    else:
        print(f"⚠️  Snake data file not found at {SNAKE_DATA_PATH}")
        snake_data_map = {}

except Exception as e:
    print(f"❌ An error occurred during server startup: {e}")
    traceback.print_exc()
    model = None

def preprocess_image(image, target_size=(224, 224)):
    """Preprocess image for MobileNetV2 model"""
    img = image.resize(target_size)
    img_array = np.array(img, dtype=np.float32)
    
    # Handle PNG transparency
    if img_array.ndim == 3 and img_array.shape[-1] == 4:
        img_array = img_array[..., :3]
    
    # MobileNetV2 preprocessing: scale pixels between -1 and 1
    img_array = (img_array / 127.5) - 1.0
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def create_fallback_response(class_name, confidence):
    """Create fallback response when snake data is not available"""
    return {
        'ClassName': class_name,
        'Confidence': f"{confidence:.2%}",
        'CommonEnglishNames': 'Information not available',
        'ScientificName': 'Information not available',
        'Family': 'Information not available',
        'Venom': 'Information not available',
        'LocalNames': 'Information not available',
        'EndemicStatus': 'Information not available',
        'ConservationStatus': 'Information not available',
        'Description': f'This appears to be a {class_name}, but detailed information is not currently available in our database.',
        'Treatment': 'If this is a venomous snake, seek immediate medical attention for any bite. Clean any wound thoroughly and monitor for signs of infection or allergic reaction.'
    }

def find_snake_details(predicted_class_name):
    """Find snake details using multiple lookup strategies"""
    # Try multiple variations of the class name
    lookup_keys = [
        predicted_class_name.lower().strip(),
        predicted_class_name.lower().strip().replace('-', ' ').replace('_', ' '),
        predicted_class_name.lower().strip().replace(' ', '_'),
        predicted_class_name.lower().strip().replace(' ', '-')
    ]
    
    # Also try matching parts of the name
    name_parts = predicted_class_name.lower().split()
    if len(name_parts) > 1:
        lookup_keys.extend(name_parts)  # Add individual words
        lookup_keys.append(' '.join(name_parts[-2:]))  # Add last two words
    
    for key in lookup_keys:
        if key in snake_data_map:
            return snake_data_map[key]
    
    return None

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload, prediction, and data lookup"""
    if model is None or not class_labels:
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
        
        print("[DEBUG] Step 4: Running model prediction...")
        predictions = model.predict(processed_image, verbose=0)
        prediction = predictions[0]  # Get first batch element
        print("[DEBUG] Step 5: Model prediction successful.")
        
        predicted_index = np.argmax(prediction)
        confidence = float(prediction[predicted_index])
        predicted_class_name = class_labels[predicted_index]
        print(f"[DEBUG] Prediction result - Class: '{predicted_class_name}', Confidence: {confidence:.2f}")

        # Check if confidence is below the threshold
        if confidence < CONFIDENCE_THRESHOLD:
            error_message = f"Could not confidently identify a snake in this image (Confidence: {confidence:.2%}). Please try a clearer picture of the snake."
            print(f"❌ [ERROR] Low confidence: {error_message}")
            return jsonify({'error': error_message}), 400

        # Find snake details
        snake_details = find_snake_details(predicted_class_name)
        
        if not snake_details:
            print(f"⚠️ [WARNING] Details not found for: '{predicted_class_name}'. Using fallback data.")
            snake_details = create_fallback_response(predicted_class_name, confidence)

        # Prepare response data
        response_data = {
            'ClassName': snake_details.get('ClassName', predicted_class_name),
            'Confidence': f"{confidence:.2%}",
            'CommonEnglishNames': snake_details.get('CommonEnglishNames', 'Information not available'),
            'ScientificName': snake_details.get('ScientificName', 'Information not available'),
            'Family': snake_details.get('Family', 'Information not available'),
            'Venom': snake_details.get('Venom', 'Information not available'),
            'LocalNames': snake_details.get('LocalNames', 'Information not available'),
            'EndemicStatus': snake_details.get('EndemicStatus', 'Information not available'),
            'ConservationStatus': snake_details.get('ConservationStatus', 'Information not available'),
            'Description': snake_details.get('Description', 'Information not available'),
            'Treatment': snake_details.get('Treatment', 'Information not available')
        }

        print(f"[DEBUG] Sending response for: {response_data['ClassName']}")
        return jsonify(response_data)

    except Exception as e:
        print(f"❌ [ERROR] An unexpected error occurred during prediction:")
        traceback.print_exc()
        return jsonify({'error': 'An internal server error occurred. Please try again later.'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    status = {
        'model_loaded': model is not None,
        'labels_loaded': len(class_labels) > 0,
        'snake_data_loaded': len(snake_data_map) > 0,
        'num_classes': len(class_labels),
        'num_species': len(snake_data_map),
        'status': 'healthy' if model is not None and class_labels else 'unhealthy'
    }
    return jsonify(status)

@app.route('/species', methods=['GET'])
def list_species():
    """List all available snake species in the database"""
    species_list = []
    for key, details in snake_data_map.items():
        if 'ClassName' in details and details not in species_list:
            species_list.append(details)
    
    return jsonify({
        'count': len(species_list),
        'species': species_list
    })

@app.route('/welcome', methods=['GET'])
def welcome():
    """Welcome message endpoint"""
    return jsonify({
        "message": "Welcome to the Snake Identification API!",
        "version": "1.0.0",
        "endpoints": {
            "POST /predict": "Identify snake from image",
            "GET /health": "Server health check",
            "GET /species": "List all available species",
            "GET /welcome": "This welcome message"
        }
    })

if __name__ == '__main__':
    print("Starting Flask server...")
    print("Available routes:")
    print("  - POST /predict : Identify snake from image")
    print("  - GET  /health  : Server health check")
    print("  - GET  /species : List all available species")
    print("  - GET  /welcome : Welcome message")
    
    app.run(host='0.0.0.0', port=5000, debug=True)