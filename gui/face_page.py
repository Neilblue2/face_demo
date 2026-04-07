import cv2
import time
from datetime import datetime
from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout, QMessageBox, QPushButton
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer

from CORE.face_engine import load_db_features, detect_and_recognize
from CORE.db import get_conn


class FacePage(QWidget):

    def __init__(self):
        super().__init__()

        self.label = QLabel()
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("background-color: #000000;")

        self.loading_label = QLabel("正在加载中...")
        self.loading_label.setAlignment(Qt.AlignCenter)
        self.loading_label.setStyleSheet("color: #ffffff; font-size: 16px;")

        self.status_label = QLabel()
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: #ff4444; font-size: 18px;")

        self.btn_start = QPushButton("开始识别")

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.loading_label)
        layout.addWidget(self.status_label)
        layout.addWidget(self.btn_start)

        self.setLayout(layout)

        self.cap = None
        self.db_features = load_db_features()
        self.name_to_id = self._build_name_index(self.db_features)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.last_popup_time = {}
        self.recognizing = False

        self.btn_start.clicked.connect(self.start_recognition)

    def update_frame(self):
        if self.cap is None or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()

        if not ret:
            return

        if not self.recognizing:
            self.loading_label.setText("正在加载中...")
            self.status_label.setText("")
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            img = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888)
            self.label.setPixmap(QPixmap.fromImage(img))
            return

        self.loading_label.setText("")
        results = detect_and_recognize(frame, self.db_features)
        status_lines = []

        for r in results:

            x1, y1, x2, y2 = r["bbox"]
            name = r["name"]
            score = r["score"]
            user_id = r.get("user_id")

            if user_id is None and name != "Unknown":
                user_id = self.name_to_id.get(name)

            status_text, status_color = self._handle_checkin(user_id, name)
            self._maybe_show_popup(user_id, name, status_text)
            status_lines.append(status_text)

            color = (0,255,0) if name!="Unknown" else (0,0,255)

            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)

            cv2.putText(frame,
                        f"{name}:{score:.2f}",
                        (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        color,
                        2)

            # OpenCV 不稳定渲染中文，这里移到 Qt 标签展示

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        h, w, ch = rgb.shape

        img = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888)

        self.label.setPixmap(QPixmap.fromImage(img))
        self.status_label.setText("\n".join(status_lines))

    def refresh_db_features(self):
        self.db_features = load_db_features()
        self.name_to_id = self._build_name_index(self.db_features)

    # =========================
    # 摄像头控制
    # =========================
    def start_camera(self):

        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)

        if self.cap.isOpened() and not self.timer.isActive():
            self.timer.start(30)

    def stop_camera(self):

        if self.timer.isActive():
            self.timer.stop()

        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.recognizing = False
        self.loading_label.setText("正在加载中...")
        self.status_label.setText("")

    def start_recognition(self):
        self.recognizing = True

    def _build_name_index(self, db_features):
        index = {}
        for user in db_features:
            index[user["name"]] = user["id"]
        return index

    def _handle_checkin(self, user_id, name):
        if name == "Unknown" or user_id is None:
            return "人脸信息未注册", (0, 0, 255)

        now = datetime.now()

        conn = get_conn()
        cur = conn.cursor()

        # 1) 查询当前时间内的课程
        cur.execute("""
            SELECT id, name, start_time, end_time
            FROM courses
            WHERE start_time <= %s AND end_time >= %s
            ORDER BY start_time DESC
            LIMIT 1
        """, (now, now))

        active_course = cur.fetchone()

        if active_course:
            course_id, course_name, _, _ = active_course

            # 是否在课程名单中
            cur.execute(
                "SELECT 1 FROM course_roster WHERE course_id=%s AND user_id=%s",
                (course_id, user_id)
            )
            in_roster = cur.fetchone() is not None

            if not in_roster:
                conn.close()
                return f"{name}同学当前没有要签到的课程", (0, 0, 255)

            # 是否已签到
            cur.execute(
                "SELECT 1 FROM attendance WHERE course_id=%s AND user_id=%s",
                (course_id, user_id)
            )
            already = cur.fetchone() is not None

            if already:
                conn.close()
                return f"{name}同学请勿重复签到", (0, 0, 255)

            # 记录签到
            cur.execute(
                "INSERT INTO attendance (user_id, course_id, course_name, checkin_time) VALUES (%s,%s,%s,%s)",
                (user_id, course_id, course_name, now)
            )
            conn.commit()
            conn.close()
            return f"{name}同学成功签到{course_name}", (0, 255, 0)

        conn.close()
        return f"{name}同学当前没有要签到的课程", (0, 0, 255)

    def _maybe_show_popup(self, user_id, name, status_text):
        key = user_id if user_id is not None else name
        now = time.time()
        last_time = self.last_popup_time.get(key, 0)

        if now - last_time < 5:
            return

        self.last_popup_time[key] = now
        self._show_checkin_popup(status_text)

    def _show_checkin_popup(self, status_text):
        msg = QMessageBox(self)
        msg.setWindowTitle("签到提示")
        msg.setText(status_text)
        btn_home = msg.addButton("返回首页", QMessageBox.AcceptRole)
        btn_stay = msg.addButton("继续停留", QMessageBox.RejectRole)
        msg.exec_()

        if msg.clickedButton() == btn_home:
            self._go_home()
        else:
            QTimer.singleShot(5000, self._go_home)

    def _go_home(self):
        win = self.window()
        if hasattr(win, "stack"):
            win.stack.setCurrentIndex(0)
