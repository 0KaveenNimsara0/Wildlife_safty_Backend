"""
Route Layer - Defines URL paths and maps them to controller methods.
"""

from flask import Blueprint


def create_routes(prediction_controller):
    """Create and return a Flask Blueprint with all API routes."""
    api = Blueprint('api', __name__)

    # Prediction endpoint
    api.add_url_rule('/predict', 'predict', prediction_controller.predict, methods=['POST'])

    # Health check endpoint
    api.add_url_rule('/health', 'health_check', prediction_controller.health_check, methods=['GET'])

    # Species listing endpoint
    api.add_url_rule('/species', 'list_species', prediction_controller.list_species, methods=['GET'])

    # Welcome endpoint
    api.add_url_rule('/welcome', 'welcome', prediction_controller.welcome, methods=['GET'])

    return api
