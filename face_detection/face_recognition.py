import numpy as np
import os
from pathlib import Path
import shutil

import dlib

from display import utils

class FaceRecognition():
    def __init__(self):
        self.BASE_DIR = Path(__file__).resolve().parent

        self.predictor_path = str(self.BASE_DIR / "face_data" / "models" / "shape_predictor_5_face_landmarks.dat")
        self.face_rec_model_path = str(self.BASE_DIR / "face_data" / "models" / "dlib_face_recognition_resnet_model_v1.dat")


        self.faces_folder_path = self.BASE_DIR / "face_data" / "faces"

        self.detector = dlib.get_frontal_face_detector()
        self.sp = dlib.shape_predictor(self.predictor_path)
        self.facerec = dlib.face_recognition_model_v1(self.face_rec_model_path)
        self.last_dets_count=0 #sets initial number of faces on window

        self.properties_path_json= str(self.BASE_DIR / "face_data" / "face_properties.json")
        self.descriptors_path_json= str(self.BASE_DIR / "face_data" / "face_descriptors.json")

        self.face_descriptors = utils.load_json_descriptors()
        self.face_properties = utils.load_json_properties()
    
    def register_face(self, face_name, purpose):
        if purpose == "initial reg":
            self.relationship = input("what is this persons relationship to you? ")
            
            save_dir = self.BASE_DIR / "face_data" / "faces" / face_name / f"{face_name}_unprocessed"
            processed_dir = self.BASE_DIR / "face_data" / "faces" / face_name / f"{face_name}_processed"

            save_dir.mkdir(parents=True, exist_ok=True)  
            processed_dir.mkdir(parents=True, exist_ok=True)
            print(f"Making directory: {save_dir}")
            if "faces" not in self.face_properties:
                self.face_properties["faces"] = {}
            self.face_properties["faces"][face_name]={
                "user_id": len(self.face_properties["faces"]),
                "photo_count": 0,
                "relationship": self.relationship
            }
            self.face_descriptors[face_name]=[]
        elif purpose == "updating info":
            self.face_properties["faces"][face_name]["photo_count"]+=1
            print("photo count added ", self.face_properties["faces"][face_name]["photo_count"])
        utils.write_json_properties(self.face_properties)
        utils.write_json_descriptors(self.face_descriptors)

    def get_face_descriptor(self, img):
        dets = self.detector(img, 1)

        # makes it so it only displays print message if there is a change in the number of faces on display
        if len(dets) != self.last_dets_count: 
            print(f"Number of faces detected: {len(dets)}")
    
        self.last_dets_count=len(dets)
        descriptors = []
        if dets:
            for k, d in enumerate(dets): #d represents rectangular box coord for where the face is
                shape = self.sp(img, d)
                face_descriptor = self.facerec.compute_face_descriptor(img, shape)
                descriptors.append((face_descriptor, d))
        
        if descriptors:
            return (descriptors, True)
        else:
            return (descriptors, False)

    def find_face_match(self, new_descriptor, threshold=0.4): #the lower the threshhold, the higher the accuracy
        self.face_descriptors = utils.load_json_descriptors()
        best_match = None
        lowest_distance = float("inf")
        # json files the descriptors are stored in terms of a list, but the new descriptor is a dlib vector
        converted_new_descriptor= np.array(new_descriptor) 
        if self.face_descriptors.items():
            for name, descriptors in self.face_descriptors.items():
                for desc in descriptors:
                    dist = np.linalg.norm(np.array(desc) - converted_new_descriptor)
                    if dist < lowest_distance:
                        lowest_distance = dist
                        best_match = name
            if lowest_distance < threshold:
                return best_match
            else:
                return None
        else:
            return None

    def train_face_images(self):
        self.face_descriptors = utils.load_json_descriptors()
        for face_name in os.listdir(self.faces_folder_path):
            save_dir = self.BASE_DIR / "face_data" / "faces" / face_name / f"{face_name}_unprocessed"
            processed_dir = self.BASE_DIR / "face_data" / "faces" / face_name / f"{face_name}_processed"

            for img_path in save_dir.glob("*.png"):
                print(f"Processing file: {img_path}")
                img = dlib.load_rgb_image(str(img_path))  # dlib needs string path

                dets = self.detector(img, 1)
                for k, d in enumerate(dets):
                    shape = self.sp(img, d)
                    face_chip = dlib.get_face_chip(img, shape)
                    descriptor = list(self.facerec.compute_face_descriptor(face_chip))
                    self.face_descriptors[face_name].append(descriptor)
                    shutil.move(str(img_path), str(processed_dir)) #moves newly processed photos to processed folder
            utils.write_json_descriptors(self.face_descriptors)
