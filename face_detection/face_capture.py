import dlib
import cv2
import json

from training_faces import FaceRecognition
from pathlib import Path

class Capture():
    def __init__(self):
        self.cam = cv2.VideoCapture(0)
        self.detector = dlib.get_frontal_face_detector()
        self.BASE_DIR = Path(__file__).resolve()
        self.face_recognition= FaceRecognition(predictor_path, face_rec_model_path, faces_folder_path)
        with open("face_properties.json", "r") as people_data:
            self.face_properties= json.load(people_data)

    def stream_frames(self):
        while True:
            ret, img = self.cam.read()
            if not ret:
                break
            if cv2.waitKey(1) == 27: #esc key:
                break
            yield img

    @staticmethod
    def text_overlay(img, det, name):
        x1, y1 = det.right(), det.bottom()
        location = (x1,y1)
        thickness= 2
        color= (102,0,235)
        font= cv2.FONT_HERSHEY_COMPLEX_SMALL
        font_scale=1
        cv2.putText(img, name, location, font, font_scale, color, thickness)

    def take_training_image(self, face_name):
        if self.face_properties["faces"].get(face_name) is None:
            self.face_recognition.register_face()

        image = Capture.stream_frames() # press space to capture image
        print("press space to capture")
        if cv2.waitKey(1) == 32: # space key
            photo_count = self.face_properties["faces"][face_name]["photo_count"]
            filename = f"{face_name}{photo_count}.png"
            save_path = self.BASE_DIR / "faces_of_people" / face_name / filename
            cv2.imwrite(str(save_path), image)
            print("image saved into " + save_path)
            self.face_properties["faces"][face_name]["photo_count"]+=1

