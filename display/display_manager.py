from face_detection.face_capture import Capture
import cv2
from face_detection.face_recognition import FaceRecognition
import json
from pathlib import Path
from display import utils

class DisplayManager:
    def __init__(self):
        self.capture= Capture()
        self.recognizer= FaceRecognition()
        self.BASE_DIR = Path(__file__).resolve().parent.parent

        self.faces_folder_path = str(self.BASE_DIR / "face_detection" / "face_data" / "faces")
        self.descriptors_path_json= str(self.BASE_DIR / "face_detection" / "face_data" / "face_descriptors.json")
        try:
            self.face_descriptors = utils.load_json_descriptors()
        except (FileNotFoundError, json.JSONDecodeError, EOFError):
            print("face_descriptors.json is missing or invalid. Initializing as empty.")
            self.face_descriptors = {}
            
    def face_identifying_stream(self, display_name):
        for img in self.capture.stream_frames():
            face_descriptor, det, state = self.recognizer.get_face_descriptor(img)
            resized_frame = cv2.resize(img, (640, 480)) # change according to your needs; the lower the res, the higher the fps

            if state:  # only if a face is detected
                face_name = self.recognizer.find_face_match(face_descriptor)
                scaled_det = Capture.scale_rectangle(det, img.shape, resized_frame.shape)
                box_color= (212,255,228)
                cv2.rectangle(resized_frame, (scaled_det.left(), scaled_det.top()), (scaled_det.right(), scaled_det.bottom()), box_color, 1)
                if face_name is not None:
                    self.capture.text_overlay(resized_frame, scaled_det, face_name)
                else:
                    self.capture.text_overlay(resized_frame, scaled_det, "UNKNOWN")
                if display_name == "training display":
                    self.capture.take_training_image(img, face_name)
            cv2.imshow(display_name, resized_frame)
            if cv2.waitKey(1) & 0xFF == 27 : #esc key:
                break
        cv2.destroyAllWindows()

    def identification_display(self):
        print("IDENTIFICATION DISPLAY")
        print("press escape to exit this mode")
        self.face_identifying_stream("identification display")

    def training_display(self):
        print("TRAINING DISPLAY")
        print("press space to capture face for recognition")
        print("press escape twice to indicate you are done taking the training photos.")
        print("note: if the detected face is already registered within the database you cannot assign this face to a seperate identity")
        self.face_identifying_stream("training display")
