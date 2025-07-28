from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import tensorflow as tf
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

# --- Configuration ---
TFLITE_MODEL_PATH = 'snake_classifier.tflite'
CLASSES_PATH = 'snake_classes_notebook.npy'
CSV_DATA_PATH = 'snake_details_final_filled.csv'

# --- Load Resources ---
interpreter = None
class_labels = []
animal_df = None

try:
    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
    interpreter.allocate_tensors()
    print("✅ TFLite model loaded successfully.")

    class_labels = np.load(CLASSES_PATH, allow_pickle=True)
    print(f"✅ Class labels loaded successfully. Found {len(class_labels)} classes.")

    # --- FINAL FIX: Change the separator from ';' to ',' ---
    # The debug logs showed the file is comma-separated.
    animal_df = pd.read_csv(CSV_DATA_PATH, sep=',', on_bad_lines='skip', encoding='utf-8-sig')
    
    print(f"DEBUG: Raw CSV columns from file: {animal_df.columns.tolist()}")
    
    # Clean column names to remove leading/trailing whitespace
    animal_df.columns = animal_df.columns.str.strip()
    print(f"DEBUG: Cleaned CSV columns: {animal_df.columns.tolist()}")
    
    # This line will now use the cleaned column name from the correctly parsed CSV
    animal_df['lookup_key'] = animal_df['Common English Name(s)'].str.lower().str.strip()
    animal_df.set_index('lookup_key', inplace=True)
    print("✅ Snake details CSV loaded and indexed successfully.")

except Exception as e:
    print(f"❌ An error occurred during server startup: {e}")


def preprocess_image(image, target_size=(224, 224)):
    """Resizes and normalizes the image for the model."""
    img = image.resize(target_size)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    """Handles the image upload, prediction, and data lookup."""
    if interpreter is None or len(class_labels) == 0 or animal_df is None:
        return jsonify({'error': 'Server resources are not available. Please check server logs.'}), 500
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    try:
        image_file = request.files['image']
        image = Image.open(image_file.stream).convert('RGB')
        
        processed_image = preprocess_image(image)
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.set_tensor(input_details[0]['index'], processed_image)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])[0]
        
        predicted_index = np.argmax(prediction)
        predicted_class = class_labels[predicted_index]
        lookup_key = predicted_class.lower().strip()

        snake_details = animal_df.loc[lookup_key].to_dict()
        
        response_data = {
            'Animal': snake_details.get('Common English Name(s)', predicted_class),
            'ScientificName': snake_details.get('Scientific Name & Authority', 'N/A'),
            'LocalNames': snake_details.get('Local Name(s) (Sinhala/Tamil)', 'N/A'),
            'Venom': snake_details.get('Venom & Medical Significance', 'No information available.'),
            'Description': snake_details.get('Description', 'No description available.'),
            'ConservationStatus': snake_details.get('Global IUCN Red List Status', 'Unknown')
        }
        print(f"✅ SUCCESS: Found details for '{lookup_key}'")
        return jsonify(response_data)
        
    except KeyError:
        print(f"❌ FAILED: Could not find '{lookup_key}' in the CSV's index. This means the model's label does not match any 'Common English Name(s)' in the CSV.")
        return jsonify({ 
            'Animal': predicted_class.replace('_', ' ').capitalize(), 
            'ScientificName': 'Details not found in our database.',
            'Venom': 'Information not available.',
            'Description': 'Could not retrieve detailed information for this species.',
            'ConservationStatus': 'Unknown',
            'LocalNames': 'N/A'
        })
    except Exception as e:
        print(f"❌ An unexpected error occurred during prediction: {e}")
        return jsonify({'error': f'An internal server error occurred: {e}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
# app.py - Flask application for snake classification and details retrieval
# This application uses a TensorFlow Lite model to classify snake images and retrieves details from a CSV file.
# It includes error handling for missing resources and provides detailed responses based on the classification results. 