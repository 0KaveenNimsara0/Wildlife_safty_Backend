"""
LabelRepository - Handles loading and accessing class labels.
"""

import os
import traceback
from config import LABELS_PATH


class LabelRepository:
    def __init__(self):
        self.class_labels = []
        self._load_labels()

    def _load_labels(self):
        """Load class labels from text file."""
        try:
            if os.path.exists(LABELS_PATH):
                with open(LABELS_PATH, 'r', encoding='utf-8') as f:
                    self.class_labels = [line.strip() for line in f.readlines()]
                print(f"✅ Class labels loaded successfully. Found {len(self.class_labels)} classes.")
                print(f"Classes: {self.class_labels}")
            else:
                raise FileNotFoundError(f"Labels file not found at {LABELS_PATH}")
        except Exception as e:
            print(f"❌ Error loading labels: {e}")
            traceback.print_exc()
            self.class_labels = []

    def get_label(self, index):
        """Get class label by index."""
        if 0 <= index < len(self.class_labels):
            return self.class_labels[index]
        return None

    def is_loaded(self):
        return len(self.class_labels) > 0

    def count(self):
        return len(self.class_labels)
