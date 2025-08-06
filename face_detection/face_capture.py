import dlib
import cv2
import json

from face_detection.face_recognition import FaceRecognition
from pathlib import Path
from display import utils
import threading

class Capture():
    def __init__(self):
        self.cam = cv2.VideoCapture(0)
        # fps = self.cam.get(cv2.CAP_PROP_FPS)
        # print(f"{fps} frames per second")

        self.given_name = "UNKNOWN" #temporary placeholder name for check_if_registered name method

        self.detector = dlib.get_frontal_face_detector()
        self.BASE_DIR = Path(__file__).resolve().parent
        self.face_recognition= FaceRecognition()
        
        self.properties_path_json= str(self.BASE_DIR / "face_data" / "face_properties.json")
        self.face_properties = utils.load_json_properties()

    def stream_frames(self):
        while True:
            ret, img = self.cam.read()
            if not ret and last_img:
                yield last_img
            elif not ret:
                break
            last_img = img
            yield img
        cv2.destroyAllWindows()

    @staticmethod
    def text_overlay(img, det, name):
        x1, y1 = det.right(), det.bottom()
        location = (x1,y1)
        thickness= 1
        color= (102,0,235)
        font= cv2.FONT_HERSHEY_COMPLEX_SMALL
        font_scale=1
        cv2.putText(img, name, location, font, font_scale, color, thickness)

    @staticmethod #this is for if window screen is resized; if window is resized, coords for things like text overlay must be too
    def scale_rectangle(det, original_shape, resized_shape):
        scale_x = resized_shape[1] / original_shape[1]
        scale_y = resized_shape[0] / original_shape[0]
        return dlib.rectangle(
            left=int(det.left() * scale_x),
            top=int(det.top() * scale_y),
            right=int(det.right() * scale_x),
            bottom=int(det.bottom() * scale_y),
        )

    def get_registration_info(self):
        while True:
            print("Registered names so far:", list(self.face_properties["faces"].keys()))
            print("Note: if the face on camera is already registered but appears as unknown, identify that person exactly as you have registered previously")
            print("this means that the face recognition needs more photo samples of your face")
            self.given_name = ((input("What is the name of the person you want to register? ")).upper()).strip()
            confirm = utils.prompt_choice(f"You entered '{self.given_name}'. Is this correct? (y/n): ", ["y", "n"])
            if confirm == 'y':
                break
            else:
                print("Reanswer the question please.\n")

        if self.given_name not in self.face_properties["faces"]:
            answer = utils.prompt_choice("Do you want to register this persons face? (y/n) ", ["y", "n"])
            if answer == "y":
                self.face_recognition.register_face(self.given_name, "initial reg")
            if answer == "n":
                print("Registration declined")
        else:
            self.face_recognition.register_face(self.given_name, "updating info")
    
    def take_training_image(self, img, face_name):
        if cv2.waitKey(1) == 32: # space key
            if face_name is None:
                thread = threading.Thread(target=self.get_registration_info())
                thread.start()
                thread.join()
                self.face_properties = utils.load_json_properties()
            else:
                self.given_name=face_name
                self.face_recognition.register_face(self.given_name, "updating info")

            self.face_properties = utils.load_json_properties() #loads info from registering face
            utils.write_json_properties(self.face_properties)
            self.face_properties = utils.load_json_properties()

            photo_count = self.face_properties["faces"][self.given_name]["photo_count"]
            print(photo_count)
            filename = f"{self.given_name}_{photo_count}.png"
            save_path = str(self.BASE_DIR / "face_data" / "faces" / self.given_name / f"{self.given_name}_unprocessed" / filename)
            cv2.imwrite(save_path, img)
            print("image saved into " + save_path)
            self.face_recognition.train_face_images() #after each image is taken it automatically trains the photo
            print("Face image(s) are being trained")
            