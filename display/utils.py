from pathlib import Path
import json
import os

@staticmethod
def prompt_choice(prompt, valid_options):
    while True:
        user_input = input(prompt).strip()
        if user_input in valid_options or user_input is None:
            return user_input
        print(f"Invalid input. Choose from: {', '.join(valid_options)}")

BASE_DIR = Path(__file__).resolve().parent.parent    
properties_path_json= str(BASE_DIR / "face_detection" / "face_data" / "face_properties.json")
descriptors_path_json= str(BASE_DIR / "face_detection" / "face_data" / "face_descriptors.json")

def make_json_if_unavailable(specific_type):
    if specific_type == "properties":
        json_path = properties_path_json
        initial_data = {"faces": {}}
    elif specific_type == "descriptors":
        json_path = descriptors_path_json
        initial_data = {}

    if not os.path.exists(json_path):
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, 'w') as f:
            json.dump(initial_data, f, indent=4)
        print(f"Created new JSON file at {json_path}")


def load_json_descriptors():
    with open(descriptors_path_json, "r") as descriptors:
        face_descriptors= json.load(descriptors)
    return face_descriptors

def write_json_descriptors(face_descriptors):
    with open(descriptors_path_json, "w") as f:
        json.dump(face_descriptors, f)

def load_json_properties():
    with open(properties_path_json, "r") as people_data:
            face_properties= json.load(people_data)
    return face_properties

def write_json_properties(face_properties):
    with open(properties_path_json, "w") as f:
        json.dump(face_properties, f, indent=4)
