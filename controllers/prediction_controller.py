"""
PredictionController - Handles HTTP request/response.
Delegates all business logic to the service layer.
"""

from flask import request, jsonify


class PredictionController:
    def __init__(self, prediction_service):
        self.prediction_service = prediction_service

    def predict(self):
        """Handle image upload, prediction, and data lookup."""
        if 'image' not in request.files:
            return jsonify({'error': 'No image file uploaded.'}), 400

        image_file = request.files['image']
        result, error, status_code = self.prediction_service.predict(image_file)

        if error:
            return jsonify({'error': error}), status_code

        return jsonify(result)

    def health_check(self):
        """Health check endpoint."""
        status = self.prediction_service.get_health_status()
        return jsonify(status)

    def list_species(self):
        """List all available snake species in the database."""
        result = self.prediction_service.get_all_species()
        return jsonify(result)

    def welcome(self):
        """Welcome message endpoint."""
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
