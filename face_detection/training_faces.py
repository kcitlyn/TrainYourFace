import sys
import os
import glob
import json
from pathlib import Path
import numpy as np

import dlib

class FaceRecognition():
    def __init__(self, predictor_path, face_rec_model_path, faces_folder_path):
        self.faces_folder_path=faces_folder_path

        self.detector = dlib.get_frontal_face_detector()
        self.sp = dlib.shape_predictor(predictor_path)
        self.facerec = dlib.face_recognition_model_v1(face_rec_model_path)
        self.display_window = dlib.image_window()

        self.BASE_DIR = Path(__file__).resolve()
        with open("face_properties.json", "r") as people_data:
            self.face_properties= json.load(people_data)
        with open("face_descriptors.json", "r") as descriptors:
            self.face_descriptors= json.load(descriptors)
    
    def register_face(self, face_name):
        relationship = input("what is this persons relationship to you? ")
        
        save_dir = Path(self.BASE_DIR / "faces_of_people" / "face_name")
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
            return(face_descriptor)


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

    def train_from_images(self):# Now process all the images
        for f in glob.glob(os.path.join(self.faces_folder_path, "*.jpg")):
            print("Processing file: {}".format(f))
            img = dlib.load_rgb_image(f)

            self.display_window.clear_overlay()
            self.display_window.set_image(img)

            # Ask the detector to find the bounding boxes of each face. The 1 in the
            # second argument indicates that we should upsample the image 1 time. This
            # will make everything bigger and allow us to detect more faces.
            dets = self.detector(img, 1)
            print("Number of faces detected: {}".format(len(dets)))

            all_descriptors = []
            # Now process each face we found.
            for k, d in enumerate(dets):
                print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                    k, d.left(), d.top(), d.right(), d.bottom()))
                # Get the landmarks/parts for the face in box d.
                shape = self.sp(img, d)
                # Draw the face landmarks on the screen so we can see what face is currently being processed.
                self.display_window.clear_overlay()
                self.display_window.add_overlay(d)
                self.display_window.add_overlay(shape)

                face_descriptor = self.facerec.compute_face_descriptor(img, shape)
                print(face_descriptor)

                print("Computing descriptor on aligned image ..")
                
                # Let's generate the aligned image using get_face_chip
                face_chip = dlib.get_face_chip(img, shape)        

                # Now we simply pass this chip (aligned image) to the api
                face_descriptor_from_prealigned_image = list(self.facerec.compute_face_descriptor(face_chip))
                all_descriptors.append(face_descriptor_from_prealigned_image) #currently its based on all images in folder, not specific people, perhaps change it later so that its based on folder name of person???? FIX LATERRRRR

                #later create something that doesnt append the outliers if someone accidentally like mislabels a face or something 

                with open("face_descriptors.json", "w"):
                    json.dump(all_descriptors, f)           
                print(face_descriptor_from_prealigned_image)
                
                dlib.hit_enter_to_continue()
