import cv2
import numpy as np

from PyQt5.QtWidgets import (
    QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QLineEdit, QMessageBox
)

from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt, pyqtSignal

from CORE.face_engine import app
from CORE.db import get_conn


class RegisterPage(QWidget):
    register_success = pyqtSignal()

    def __init__(self):
        super().__init__()

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
        faces = app.get(frame)

        for face in faces:

            bbox = face.bbox.astype(int)

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
            faces = app.get(self.last_frame)
            self.current_faces = faces

        if not faces:
            QMessageBox.warning(self, "提示", "未检测到人脸，请调整姿态再试")
            return
        face = faces[0]
        emb = face.embedding
        emb = emb / np.linalg.norm(emb)
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
