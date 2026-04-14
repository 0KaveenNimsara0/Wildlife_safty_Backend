"""
PredictionService - Contains all business logic for snake identification.
Orchestrates repositories to run the prediction pipeline.
"""

import numpy as np
from PIL import Image
from config import CONFIDENCE_THRESHOLD
from .image_service import ImageService


class PredictionService:
    def __init__(self, model_repository, label_repository, snake_data_repository):
        self.model_repo = model_repository
        self.label_repo = label_repository
        self.snake_data_repo = snake_data_repository
        self.image_service = ImageService()

    def is_ready(self):
        """Check if all resources are loaded and prediction is possible."""
        return self.model_repo.is_loaded() and self.label_repo.is_loaded()

    def predict(self, image_file):
        """
        Run the full prediction pipeline on an image file.
        Returns: (result_dict, error_message, status_code)
        """
        if not self.is_ready():
            return None, 'Server resources are not loaded. Please check server startup logs.', 500

        print("\n--- [DEBUG] Received a new prediction request ---")

        try:
            # Step 1: Open and convert image
            image = Image.open(image_file.stream).convert('RGB')
            print("[DEBUG] Step 1: Image file received and converted to RGB.")

            # Step 2: Preprocess
            processed_image = self.image_service.preprocess_image(image)
            print(f"[DEBUG] Step 2: Image preprocessed. Shape: {processed_image.shape}")

            # Step 3: Run model prediction
            print("[DEBUG] Step 3: Running model prediction...")
            predictions = self.model_repo.predict(processed_image)
            prediction = predictions[0]
            print("[DEBUG] Step 4: Model prediction successful.")

            # Step 4: Extract results
            predicted_index = np.argmax(prediction)
            confidence = float(prediction[predicted_index])
            predicted_class_name = self.label_repo.get_label(predicted_index)
            print(f"[DEBUG] Prediction result - Class: '{predicted_class_name}', Confidence: {confidence:.2f}")

            # Step 5: Check confidence threshold
            if confidence < CONFIDENCE_THRESHOLD:
                error_message = (
                    f"Could not confidently identify a snake in this image "
                    f"(Confidence: {confidence:.2%}). Please try a clearer picture of the snake."
                )
                print(f"❌ [ERROR] Low confidence: {error_message}")
                return None, error_message, 400

            # Step 6: Look up snake details
            snake_details = self.snake_data_repo.find_by_name(predicted_class_name)

            if not snake_details:
                print(f"⚠️ [WARNING] Details not found for: '{predicted_class_name}'. Using fallback data.")
                snake_details = self._create_fallback(predicted_class_name, confidence)

            # Step 7: Build response
            response_data = self._build_response(snake_details, predicted_class_name, confidence)
            print(f"[DEBUG] Sending response for: {response_data['ClassName']}")

            return response_data, None, 200

        except Exception as e:
            print(f"❌ [ERROR] An unexpected error occurred during prediction:")
            import traceback
            traceback.print_exc()
            return None, 'An internal server error occurred. Please try again later.', 500

    def _create_fallback(self, class_name, confidence):
        """Create fallback response when snake data is not available."""
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

    def _build_response(self, snake_details, predicted_class_name, confidence):
        """Build the standard API response from snake details."""
        return {
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

    def get_health_status(self):
        """Get server health status."""
        return {
            'model_loaded': self.model_repo.is_loaded(),
            'labels_loaded': self.label_repo.is_loaded(),
            'snake_data_loaded': self.snake_data_repo.is_loaded(),
            'num_classes': self.label_repo.count(),
            'num_species': self.snake_data_repo.count(),
            'status': 'healthy' if self.is_ready() else 'unhealthy'
        }

    def get_all_species(self):
        """Get all available snake species."""
        species_list = self.snake_data_repo.get_all_species()
        return {
            'count': len(species_list),
            'species': species_list
        }
