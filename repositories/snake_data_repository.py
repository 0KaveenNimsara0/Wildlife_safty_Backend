"""
SnakeDataRepository - Handles loading and accessing snake species data.
"""

import os
import json
import traceback
from config import SNAKE_DATA_PATH


class SnakeDataRepository:
    def __init__(self):
        self.snake_data_map = {}
        self.species_list = []
        self._load_snake_data()

    def _load_snake_data(self):
        """Load snake details from JSON file."""
        try:
            if os.path.exists(SNAKE_DATA_PATH):
                with open(SNAKE_DATA_PATH, 'r', encoding='utf-8') as f:
                    snake_details_list = json.load(f)
                    self.species_list = snake_details_list

                    for item in snake_details_list:
                        class_name = item['ClassName']
                        # Create multiple lookup keys for robustness
                        key1 = class_name.lower().strip().replace('-', ' ').replace('_', ' ')
                        key2 = class_name.lower().strip()
                        key3 = class_name.replace('-', ' ').replace('_', ' ').lower().strip()

                        self.snake_data_map[key1] = item
                        self.snake_data_map[key2] = item
                        self.snake_data_map[key3] = item

                        # Also add common name variations
                        if 'CommonEnglishNames' in item:
                            common_names = item['CommonEnglishNames'].split(',')
                            for common_name in common_names:
                                common_key = common_name.strip().lower()
                                self.snake_data_map[common_key] = item

                print(f"[OK] Snake details JSON loaded successfully. Found {len(snake_details_list)} species.")
                print(f"Available species: {[item['ClassName'] for item in snake_details_list]}")
            else:
                print(f"[WARN] Snake data file not found at {SNAKE_DATA_PATH}")
        except Exception as e:
            print(f"[ERROR] Error loading snake data: {e}")
            traceback.print_exc()

    def find_by_name(self, predicted_class_name):
        """Find snake details using multiple lookup strategies."""
        lookup_keys = [
            predicted_class_name.lower().strip(),
            predicted_class_name.lower().strip().replace('-', ' ').replace('_', ' '),
            predicted_class_name.lower().strip().replace(' ', '_'),
            predicted_class_name.lower().strip().replace(' ', '-')
        ]

        # Also try matching parts of the name
        name_parts = predicted_class_name.lower().split()
        if len(name_parts) > 1:
            lookup_keys.extend(name_parts)
            lookup_keys.append(' '.join(name_parts[-2:]))

        for key in lookup_keys:
            if key in self.snake_data_map:
                return self.snake_data_map[key]

        return None

    def get_all_species(self):
        """Get all unique species."""
        species_list = []
        for key, details in self.snake_data_map.items():
            if 'ClassName' in details and details not in species_list:
                species_list.append(details)
        return species_list

    def is_loaded(self):
        return len(self.snake_data_map) > 0

    def count(self):
        return len(self.snake_data_map)
