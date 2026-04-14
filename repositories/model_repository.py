"""
ModelRepository - Handles loading and accessing the ML model.
"""

import os
import traceback
import tensorflow as tf
from tensorflow.keras.models import load_model
from config import MODEL_PATH


class ModelRepository:
    def __init__(self):
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the Keras model with fallback for compatibility issues."""
        try:
            if os.path.exists(MODEL_PATH):
                try:
                    self.model = load_model(MODEL_PATH, compile=False)
                    print(f"✅ Keras model loaded successfully from {MODEL_PATH}.")
                except Exception as e:
                    print(f"⚠️  Standard load failed, trying with custom objects: {e}")
                    self.model = load_model(
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
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            traceback.print_exc()
            self.model = None

    def predict(self, processed_image):
        """Run model prediction on a preprocessed image."""
        if self.model is None:
            return None
        return self.model.predict(processed_image, verbose=0)

    def is_loaded(self):
        return self.model is not None
