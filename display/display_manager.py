from face_detection.face_capture import Capture
import cv2
from face_detection.face_recognition import FaceRecognition
import json
from pathlib import Path

class DisplayManager:
    def __init__(self):
        self.capture= Capture()
        self.recognizer= FaceRecognition()
        self.BASE_DIR = Path(__file__).resolve().parent.parent

        self.faces_folder_path = str(self.BASE_DIR / "face_detection" / "face_data" / "faces")
        self.descriptors_path_json= str(self.BASE_DIR / "face_detection" / "face_data" / "face_descriptors.json")
        try:
            with open(self.descriptors_path_json, "r") as descriptors:
                self.face_descriptors = json.load(descriptors)
        except (FileNotFoundError, json.JSONDecodeError, EOFError):
            print("face_descriptors.json is missing or invalid. Initializing as empty.")
            self.face_descriptors = {}

    def general_camera(self):
        for img in self.capture.stream_frames():
            resized_frame = cv2.resize(img, (320, 240))
            cv2.imshow("general", resized_frame)
            if cv2.waitKey(1) == 27: #esc key:
                break
        cv2.destroyAllWindows()  

    def training_display(self):
        print("press space to capture face for recognition")
        print("press escape to indicate you are done taking the training photos.")
        print("note: if the detected face is already registered within the database you cannot assign this face to a seperate identity")
        for img in self.capture.stream_frames():
            face_descriptor, det = self.recognizer.get_face_descriptor(img)
            resized_frame = cv2.resize(img, (640, 480)) # change according to your needs; the lower the res, the higher the fps
            
            if det != 0:  # only if a face is detected
                face_name = self.recognizer.find_face_match(face_descriptor)
                scaled_det = Capture.scale_rectangle(det, img.shape, resized_frame.shape)
                if face_name is not None:
                    self.capture.text_overlay(resized_frame, scaled_det, face_name)
                    self.capture.take_training_image(img, face_name)
                else:
                    self.capture.text_overlay(resized_frame, scaled_det, "UNKNOWN")
                    self.capture.take_training_image(img, face_name)
            cv2.imshow("training display", resized_frame)

            if cv2.waitKey(1) == 27: #esc key:
                break

        cv2.destroyAllWindows()

