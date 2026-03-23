import cv2
import numpy as np

from PyQt5.QtWidgets import (
    QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QLineEdit
)

from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt

from CORE.face_engine import app
from CORE.db import get_conn


class RegisterPage(QWidget):

    def __init__(self):
        super().__init__()

        # =========================
        # 输入框
        # =========================
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("姓名")

        self.id_input = QLineEdit()
        self.id_input.setPlaceholderText("学号")

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
        self.cap = cv2.VideoCapture(0)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        # =========================
        # 数据
        # =========================
        self.embeddings = []
        self.collecting = False

        # =========================
        # 信号
        # =========================
        self.btn_start.clicked.connect(self.start_collect)
        self.btn_register.clicked.connect(self.register_user)

    # =========================
    # 开始采集
    # =========================
    def start_collect(self):

        self.embeddings = []
        self.collecting = True

        if not self.timer.isActive():
            self.timer.start(30)

        self.info_label.setText("已采集: 0/5")

    # =========================
    # 摄像头更新
    # =========================
    def update_frame(self):

        ret, frame = self.cap.read()

        if not ret:
            return

        faces = app.get(frame)

        for face in faces:

            bbox = face.bbox.astype(int)
            emb = face.embedding

            x1, y1, x2, y2 = bbox

            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

            # 采集 embedding
            if self.collecting and len(self.embeddings) < 5:

                emb = emb / np.linalg.norm(emb)
                self.embeddings.append(emb)

                print("采集一张")

                self.info_label.setText(
                    f"已采集: {len(self.embeddings)}/5"
                )

                # 防止一帧采集多次
                self.collecting = False

        # 显示图像
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        h, w, ch = rgb.shape

        img = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888)

        self.label.setPixmap(QPixmap.fromImage(img))

    # =========================
    # 注册用户
    # =========================
    def register_user(self):

        name = self.name_input.text()
        student_id = self.id_input.text()

        if name == "" or student_id == "":
            print("请输入姓名和学号")
            return

        if len(self.embeddings) < 5:
            print("请先采集5张人脸")
            return

        conn = get_conn()
        cur = conn.cursor()

        # 插入用户
        cur.execute(
            "INSERT INTO users (name, student_id) VALUES (%s,%s)",
            (name, student_id)
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
