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
# --- FINAL FIX: Using the new semicolon-separated CSV file ---
CSV_DATA_PATH = 'snake_details_final.csv'

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

    # --- FINAL FIX: Reading the CSV with a semicolon separator ---
    animal_df = pd.read_csv(CSV_DATA_PATH, sep=';')
    
    animal_df['lookup_key'] = animal_df['Common English Name(s)'].str.lower()
    animal_df.set_index('lookup_key', inplace=True)
    print("✅ Snake details CSV loaded and indexed successfully.")

except Exception as e:
    print(f"❌ An error occurred during server startup: {e}")


def preprocess_image(image, target_size=(224, 224)):
    img = image.resize(target_size)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    if interpreter is None or len(class_labels) == 0:
        return jsonify({'error': 'Server resources are not available'}), 500
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

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
    lookup_key = predicted_class.lower()

    try:
        snake_details = animal_df.loc[lookup_key]
        response_data = {
            'Animal': snake_details.get('Common English Name(s)', predicted_class),
            'ScientificName': snake_details.get('Scientific Name & Authority', 'N/A'),
            'Description': snake_details.get('Venom & Medical Significance', 'No description available.'),
            'ConservationStatus': snake_details.get('Global IUCN Red List Status', 'Unknown'),
            'FunFact': f"This species is from the '{snake_details.get('Family', 'Unknown')}' family."
        }
        print(f"✅ SUCCESS: Found details for '{lookup_key}'")
        return jsonify(response_data)
        
    except KeyError:
        print(f"❌ FAILED: Could not find '{lookup_key}' in the CSV's index.")
        return jsonify({ 'Animal': predicted_class.capitalize(), 'ScientificName': 'N/A' })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)