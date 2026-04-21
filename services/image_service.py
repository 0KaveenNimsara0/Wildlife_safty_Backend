"""
ImageService - Handles image preprocessing logic.
"""

import numpy as np


class ImageService:
    @staticmethod
    def preprocess_image(image, target_size=(224, 224)):
        """Preprocess image for MobileNetV2 model."""
        img = image.resize(target_size)
        img_array = np.array(img, dtype=np.float32)

        # Handle PNG transparency
        if img_array.ndim == 3 and img_array.shape[-1] == 4:
            img_array = img_array[..., :3]

        # MobileNetV2 preprocessing: scale pixels between -1 and 1
        img_array = (img_array / 127.5) - 1.0
        img_array = np.expand_dims(img_array, axis=0)

        return img_array
