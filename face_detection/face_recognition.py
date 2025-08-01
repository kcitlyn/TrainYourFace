import sys
import os
import glob
import json
from pathlib import Path
import numpy as np

import dlib

class FaceRecognition():
    def __init__(self):
        self.BASE_DIR = Path(__file__).resolve()

        self.predictor_path = self.BASE_DIR / "faces_data" /"models" / "dlib_face_recognition_resnet_model_v1.dat"
        self.face_rec_model_path = self.BASE_DIR / "faces_data" /"models" / "shape_predictor_5_face_landmarks.dat"
        self.faces_folder_path = self.BASE_DIR / "face_data" / "faces"

        self.detector = dlib.get_frontal_face_detector()
        self.sp = dlib.shape_predictor(self.predictor_path)
        self.facerec = dlib.face_recognition_model_v1(self.face_rec_model_path)
        self.display_window = dlib.image_window()

        with open("face_properties.json", "r") as people_data:
            self.face_properties= json.load(people_data)
        with open("face_descriptors.json", "r") as descriptors:
            self.face_descriptors= json.load(descriptors)
    
    def register_face(self, face_name):
        relationship = input("what is this persons relationship to you? ")
        
        save_dir = Path(self.BASE_DIR / "face_data" / "faces" / "face_name")
        save_dir.mkdir(parents=True, exist_ok=True)  

        self.face_properties["faces"][face_name]={
            "user_id": len(self.face_properties["faces"]),
            "photo_count": 0,
            "relationship": relationship
        }
        self.face_descriptors[face_name]={}

        with open("face_properties.json", "w") as f:
            json.dump(self.face_properties, f, indent=4)

        with open("face_descriptors.json", "w") as f:
            json.dump(self.face_descriptors, f, indent=4)

    def get_face_descriptor(self, img):
        dets = self.detector(img, 1)
        print("Number of faces detected: {}".format(len(dets)))

        for k, d in enumerate(dets):
            print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                k, d.left(), d.top(), d.right(), d.bottom()))
            # Get the landmarks/parts for the face in box d.
            shape = self.sp(img, d)
            self.display_window.clear_overlay()
            self.display_window.add_overlay(shape) #if u want the box to appear around face

            face_descriptor = self.facerec.compute_face_descriptor(img, shape)
            return(face_descriptor, d)

    def find_face_match(new_descriptor, database, threshold=0.6):
        best_match = None
        lowest_distance = float("inf")

        for name, descriptors in database.items():
            for desc in descriptors:
                dist = np.linalg.norm(np.array(desc) - new_descriptor)
                if dist < lowest_distance:
                    lowest_distance = dist
                    best_match = name

        if lowest_distance < threshold:
            return best_match
        else:
            return "unknown"

    def train_face_images(self):
        for face_name in os.listdir(self.faces_folder_path):
            face_folder = os.path.join(self.faces_folder_path, face_name)
            if not os.path.isdir(face_folder):
                continue


            for img_path in glob.glob(os.path.join(face_folder, "*.jpg")):
                print(f"Processing file: {img_path}")
                img = dlib.load_rgb_image(img_path)

                dets = self.detector(img, 1)
                print(f"Number of faces detected: {len(dets)}")

                for k, d in enumerate(dets):
                    shape = self.sp(img, d)
                    self.display_window.clear_overlay()
                    self.display_window.add_overlay(d)
                    self.display_window.add_overlay(shape)
                    face_chip = dlib.get_face_chip(img, shape)
                    descriptor = list(self.facerec.compute_face_descriptor(face_chip))
                    self.face_descriptors[face_name].append(descriptor)
                    print("descriptor added to " + face_name)
                    dlib.hit_enter_to_continue()

        with open("face_descriptors.json", "w") as f:
            json.dump(self.face_descriptors, f)
