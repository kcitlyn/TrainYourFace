import dlib
import cv2
import json

from face_detection.face_recognition import FaceRecognition
from pathlib import Path
from display import utils
class Capture():
    def __init__(self):
        self.cam = cv2.VideoCapture(0)
        self.detector = dlib.get_frontal_face_detector()
        self.BASE_DIR = Path(__file__).resolve()

        self.face_recognition= FaceRecognition()
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
        cv2.destroyAllWindows()

    @staticmethod
    def text_overlay(img, det, name):
        x1, y1 = det.right(), det.bottom()
        location = (x1,y1)
        thickness= 2
        color= (102,0,235)
        font= cv2.FONT_HERSHEY_COMPLEX_SMALL
        font_scale=1
        cv2.putText(img, name, location, font, font_scale, color, thickness)

    def take_training_image(self, face_name, img):
        if self.face_properties["faces"].get(face_name) is None:
            answer = utils.prompt_choice("Do you want to register this persons voice? ", ["y", "n"])
            if answer == "y":
                self.face_recognition.register_face(face_name)
            if answer == "n":
                print("Registration declined. Exiting training image capture.")
                return

        print("press space to capture")
        if cv2.waitKey(1) == 32: # space key
            photo_count = self.face_properties["faces"][face_name]["photo_count"]
            filename = f"{face_name}{photo_count}.png"
            save_path = self.BASE_DIR / "face_data" / "faces" / face_name / filename
            cv2.imwrite(str(save_path), img)
            print("image saved into " + save_path)
            self.face_properties["faces"][face_name]["photo_count"]+=1

