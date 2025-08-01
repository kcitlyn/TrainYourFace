from face_detection.face_capture import Capture
import cv2

class DisplayManager:
    def __init__(self, window_name):
        self.name=window_name
        self.capture= Capture()

    def general_camera(self):
        self.capture.stream_frames()
        cv2.imshow()

    def training_display(self):
        face_name=
        self.capture.take_training_image(face_name)
        cv2.imshow()