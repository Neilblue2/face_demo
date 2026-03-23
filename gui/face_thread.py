import cv2
from PyQt5.QtCore import QThread, pyqtSignal

from CORE.face_engine import load_db_features, detect_and_recognize


class FaceThread(QThread):

    frame_signal = pyqtSignal(object)

    def __init__(self):
        super().__init__()

        self.running = True

        self.cap = cv2.VideoCapture(0)

        # 加载人脸库
        self.db_features = load_db_features()

    def run(self):

        while self.running:

            ret, frame = self.cap.read()

            if not ret:
                continue

            results = detect_and_recognize(frame, self.db_features)

            for r in results:

                x1, y1, x2, y2 = r["bbox"]
                name = r["name"]
                score = r["score"]

                color = (0,255,0) if name!="Unknown" else (0,0,255)

                cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)

                cv2.putText(frame,
                            f"{name}:{score:.2f}",
                            (x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            color,
                            2)

            self.frame_signal.emit(frame)

    def stop(self):

        self.running = False
        self.cap.release()