"""
Wildlife Safety - Snake Identification API
==========================================
Layered Architecture:
  app.py (entry point) → routes/ → controllers/ → services/ → repositories/
"""

from flask import Flask
from flask_cors import CORS
import logging

from config import HOST, PORT, DEBUG
from repositories import ModelRepository, LabelRepository, SnakeDataRepository
from services import ImageService, PredictionService
from controllers import PredictionController
from routes import create_routes

# Set up logging
logging.basicConfig(level=logging.INFO)

# ---------------------
# Initialize Flask App
# ---------------------
app = Flask(__name__)
CORS(app)

# ---------------------
# Initialize Layers
# ---------------------

# Repository Layer (data access)
model_repository = ModelRepository()
label_repository = LabelRepository()
snake_data_repository = SnakeDataRepository()

# Service Layer (business logic)
prediction_service = PredictionService(model_repository, label_repository, snake_data_repository)

# Controller Layer (HTTP handling)
prediction_controller = PredictionController(prediction_service)

# Route Layer (URL mapping)
api_routes = create_routes(prediction_controller)
app.register_blueprint(api_routes)

# ---------------------
# Start Server
# ---------------------
if __name__ == '__main__':
    print("Starting Flask server...")
    print("Available routes:")
    print("  - POST /predict : Identify snake from image")
    print("  - GET  /health  : Server health check")
    print("  - GET  /species : List all available species")
    print("  - GET  /welcome : Welcome message")

    app.run(host=HOST, port=PORT, debug=DEBUG)