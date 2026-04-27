import cv2
import numpy as np

from PyQt5.QtWidgets import (
    QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QLineEdit, QMessageBox
)

from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt, pyqtSignal

from CORE.face_engine import extract_faces
from CORE.db import get_conn


class RegisterPage(QWidget):
    register_success = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.blur_threshold = 80.0
        self.min_face_ratio = 0.08
        self.max_eye_tilt_ratio = 0.30
        self.max_nose_offset_ratio = 0.35

        # =========================
        # 输入框
        # =========================
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("姓名")

        self.id_input = QLineEdit()
        self.id_input.setPlaceholderText("学号")

        self.class_input = QLineEdit()
        self.class_input.setPlaceholderText("班级")

        self.major_input = QLineEdit()
        self.major_input.setPlaceholderText("专业")

        # =========================
        # 摄像头显示
        # =========================
        self.label = QLabel("等待摄像头")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setFixedSize(640, 480)
        self.label.setStyleSheet("background-color: #000000; color: #ffffff;")
        self.label.setScaledContents(True)

        # =========================
        # 按钮
        # =========================
        self.btn_start = QPushButton("开始采集")
        self.btn_register = QPushButton("注册用户")

        # 采集数量
        self.info_label = QLabel("已采集: 0/5")

        # =========================
        # 布局
        # =========================
        form_layout = QHBoxLayout()
        form_layout.addWidget(self.name_input)
        form_layout.addWidget(self.id_input)
        form_layout.addWidget(self.class_input)
        form_layout.addWidget(self.major_input)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.btn_start)
        btn_layout.addWidget(self.btn_register)

        layout = QVBoxLayout()

        layout.addLayout(form_layout)
        layout.addWidget(self.label)
        layout.addWidget(self.info_label)
        layout.addLayout(btn_layout)

        self.setLayout(layout)

        # =========================
        # 摄像头
        # =========================
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.current_faces = []
        self.last_frame = None

        # =========================
        # 数据
        # =========================
        self.embeddings = []

        # =========================
        # 信号
        # =========================
        self.btn_start.clicked.connect(self.start_collect)
        self.btn_register.clicked.connect(self.register_user)

    def _estimate_quality(self, frame, face):
        x1, y1, x2, y2 = face["bbox"]
        h, w = frame.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        face_area = max(0, x2 - x1) * max(0, y2 - y1)
        frame_area = h * w
        ratio = face_area / frame_area if frame_area > 0 else 0.0
        if ratio < self.min_face_ratio:
            return False, "人脸太小，请靠近摄像头"

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return False, "人脸区域无效，请重试"
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if blur_var < self.blur_threshold:
            return False, "画面偏模糊，请保持稳定后再采集"

        kps = face.get("kps")
        if kps is not None and len(kps) >= 3:
            left_eye, right_eye, nose = np.array(kps[0]), np.array(kps[1]), np.array(kps[2])
            eye_dist = np.linalg.norm(right_eye - left_eye)
            if eye_dist > 1e-6:
                eye_tilt_ratio = abs(left_eye[1] - right_eye[1]) / eye_dist
                eye_center_x = (left_eye[0] + right_eye[0]) / 2.0
                nose_offset_ratio = abs(nose[0] - eye_center_x) / eye_dist
                if eye_tilt_ratio > self.max_eye_tilt_ratio or nose_offset_ratio > self.max_nose_offset_ratio:
                    return False, "请尽量正视摄像头后再采集"

        return True, "质量通过"

    # =========================
    # 开始采集
    # =========================
    def start_collect(self):

        if len(self.embeddings) >= 5:
            QMessageBox.information(self, "提示", "已经采集 5 张人脸，请点击“注册用户”完成注册")
            return

        self.start_camera()

        if not self.timer.isActive():
            self.timer.start(30)

        self.capture_one_face()

    # =========================
    # 摄像头更新
    # =========================
    def update_frame(self):
        if self.cap is None or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()

        if not ret:
            return

        self.last_frame = frame.copy()
        faces = extract_faces(frame)

        for face in faces:
            bbox = face["bbox"]
            x1, y1, x2, y2 = bbox

            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

        self.current_faces = faces
        # 显示图像
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        h, w, ch = rgb.shape

        img = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888)

        self.label.setPixmap(QPixmap.fromImage(img))

    def capture_one_face(self):

        if len(self.embeddings) >= 5:
            QMessageBox.information(self, "提示", "已经采集 5 张人脸，请点击“注册用户”完成注册")
            return

        faces = self.current_faces

        if not faces and self.last_frame is not None:
            faces = extract_faces(self.last_frame)
            self.current_faces = faces

        if not faces:
            QMessageBox.warning(self, "提示", "未检测到人脸，请调整姿态再试")
            return

        if len(faces) > 1:
            QMessageBox.warning(self, "提示", "检测到多人，请仅保留一人入镜")
            return

        face = faces[0]
        frame_for_check = self.last_frame if self.last_frame is not None else None
        if frame_for_check is None:
            QMessageBox.warning(self, "提示", "采集帧不可用，请重试")
            return

        ok, msg = self._estimate_quality(frame_for_check, face)
        if not ok:
            self.info_label.setText(f"已采集: {len(self.embeddings)}/5 | {msg}")
            QMessageBox.warning(self, "提示", msg)
            return

        emb = face["embedding"]
        self.embeddings.append(emb)

        print("采集一张")
        self.info_label.setText(f"已采集: {len(self.embeddings)}/5")

    # =========================
    # 注册用户
    # =========================
    def register_user(self):

        name = self.name_input.text()
        student_id = self.id_input.text()
        class_name = self.class_input.text()
        major = self.major_input.text()

        if name == "" or student_id == "":
            QMessageBox.warning(self, "提示", "请输入姓名和学号")
            return

        if len(self.embeddings) < 5:
            QMessageBox.warning(self, "提示", "请先采集5张人脸")
            return

        conn = get_conn()
        cur = conn.cursor()

        # 插入用户
        cur.execute(
            "INSERT INTO users (name, student_id, class_name, major) VALUES (%s,%s,%s,%s)",
            (name, student_id, class_name, major)
        )

        user_id = cur.lastrowid

        for emb in self.embeddings:

            cur.execute(
                "INSERT INTO face_feature (user_id, embedding) VALUES (%s,%s)",
                (user_id, emb.astype(np.float32).tobytes())
            )

        conn.commit()
        conn.close()

        print("注册成功")

        self.info_label.setText("注册完成")

        self.embeddings = []
        self.info_label.setText("已采集: 0/5")
        self.register_success.emit()
        QMessageBox.information(self, "提示", "注册成功")

        self.name_input.setText("")
        self.id_input.setText("")
        self.class_input.setText("")
        self.major_input.setText("")

    # =========================
    # 摄像头控制
    # =========================
    def start_camera(self):

        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)

        if self.cap.isOpened() and not self.timer.isActive():
            self.timer.start(30)
            self.label.setText("")

    def stop_camera(self):

        if self.timer.isActive():
            self.timer.stop()

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        self.label.setPixmap(QPixmap())
        self.label.setText("等待摄像头")
