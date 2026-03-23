import cv2
from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer

from CORE.face_engine import load_db_features, detect_and_recognize


class FacePage(QWidget):

    def __init__(self):
        super().__init__()

        self.label = QLabel()

        layout = QVBoxLayout()
        layout.addWidget(self.label)

        self.setLayout(layout)

        self.cap = cv2.VideoCapture(0)

        self.db_features = load_db_features()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def update_frame(self):

        ret, frame = self.cap.read()

        if not ret:
            return

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

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        h, w, ch = rgb.shape

        img = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888)

        self.label.setPixmap(QPixmap.fromImage(img))