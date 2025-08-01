from face_detection.face_capture import Capture
import cv2
from face_detection.face_recognition import FaceRecognition
import json

class DisplayManager:
    def __init__(self, window_name):
        self.name=window_name
        self.capture= Capture()
        self.recognizer= FaceRecognition()

        with open("face_descriptors.json", "r") as descriptors:
            self.face_descriptors= json.load(descriptors)["faces"]

    def general_camera(self):
        for img in self.capture.stream_frames():
            cv2.imshow("general", img)

    def training_display(self):
        for img in self.capture.stream_frames():
            cv2.imshow("training display", img)
            face_descriptor, det = self.recognizer.get_face_descriptor(img)
            face_name = self.recognizer.find_face_match(face_descriptor, self.face_descriptors.values())
            self.capture.text_overlay(img, det, face_name)
            self.capture.take_training_image(face_name, img)
        self.recognizer.train_face_images()
