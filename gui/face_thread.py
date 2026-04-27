import cv2
from PyQt5.QtCore import QThread, pyqtSignal

from CORE.face_engine import load_db_features, detect_and_recognize


class FaceThread(QThread):
    frame_signal = pyqtSignal(object, object)  # frame, results
    error_signal = pyqtSignal(str)

    def __init__(self, camera_index=0):
        super().__init__()
        self.camera_index = camera_index
        self.running = False
        self.recognizing = False
        self.db_features = load_db_features()

    def run(self):
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            self.error_signal.emit("摄像头打开失败")
            return

        self.running = True
        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    continue

                results = []
                if self.recognizing:
                    results = detect_and_recognize(frame, self.db_features)

                self.frame_signal.emit(frame, results)
        finally:
            cap.release()

    def set_recognizing(self, enable):
        self.recognizing = enable

    def refresh_db_features(self):
        self.db_features = load_db_features()

    def stop(self):
        self.running = False
        self.wait()
