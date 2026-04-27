import cv2
import time
from datetime import datetime
from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout, QHBoxLayout, QMessageBox, QPushButton
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap

from CORE.face_engine import load_db_features, get_engine_mode
from CORE.db import get_conn
from gui.face_thread import FaceThread


class FacePage(QWidget):

    def __init__(self):
        super().__init__()

        self.label = QLabel()
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("background-color: #000000;")

        self.engine_label = QLabel()
        self.engine_label.setAlignment(Qt.AlignCenter)
        self.engine_label.setStyleSheet("color: #666666; font-size: 13px;")

        self.runtime_label = QLabel("摄像头: 未开启 | 识别: 未开始 | 模型: 未加载")
        self.runtime_label.setAlignment(Qt.AlignCenter)
        self.runtime_label.setStyleSheet("color: #333333; font-size: 14px;")

        self.loading_label = QLabel("等待点击“开始识别”")
        self.loading_label.setAlignment(Qt.AlignCenter)
        self.loading_label.setStyleSheet("color: #ffffff; font-size: 16px;")

        self.status_label = QLabel()
        self.status_label.setTextFormat(Qt.RichText)
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("font-size: 18px;")

        self.btn_start = QPushButton("开始识别")
        self.btn_stop = QPushButton("停止识别")
        self.btn_stop.setEnabled(False)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.btn_start)
        btn_layout.addWidget(self.btn_stop)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.engine_label)
        layout.addWidget(self.runtime_label)
        layout.addWidget(self.loading_label)
        layout.addWidget(self.status_label)
        layout.addLayout(btn_layout)

        self.setLayout(layout)

        self.db_features = load_db_features()
        self.name_to_id = self._build_name_index(self.db_features)
        self.face_thread = None
        self.last_popup_time = {}
        self.recognizing = False
        self.camera_ready = False
        self.model_ready = False

        self.btn_start.clicked.connect(self.start_recognition)
        self.btn_stop.clicked.connect(self.stop_recognition)
        self._update_engine_label()
        self._update_runtime_label()

    def _on_thread_error(self, msg):
        self.loading_label.setText(msg)
        self.camera_ready = False
        self._update_runtime_label()

    def _on_frame(self, frame, results):
        if not self.recognizing:
            self.loading_label.setText("摄像头已开启，等待开始识别")
            self.status_label.setText("")
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            img = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888)
            self.label.setPixmap(QPixmap.fromImage(img))
            return

        self.model_ready = True
        self._update_runtime_label()
        self.loading_label.setText("")
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
            status_lines.append(self._format_status_line(status_text, status_color))

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
        self.status_label.setText("<br>".join(status_lines))

    def _update_engine_label(self):
        mode = get_engine_mode()
        if mode == "split":
            mode_name = "两段式 RetinaFace + ArcFace"
        elif mode == "yolov5face_dlib":
            mode_name = "YOLOv5-Face + dlib"
        else:
            mode_name = "一体化 buffalo_l"
        self.engine_label.setText(f"当前引擎: {mode_name}")

    def _update_runtime_label(self):
        cam = "已开启" if self.camera_ready else "未开启"
        rec = "运行中" if self.recognizing else "未开始"
        model = "已加载" if self.model_ready else "未加载"
        self.runtime_label.setText(f"摄像头: {cam} | 识别: {rec} | 模型: {model}")

    def _format_status_line(self, text, color):
        r, g, b = color
        return f'<span style="color: rgb({r},{g},{b});">{text}</span>'

    def refresh_db_features(self):
        self.db_features = load_db_features()
        self.name_to_id = self._build_name_index(self.db_features)
        if self.face_thread is not None:
            self.face_thread.refresh_db_features()

    # =========================
    # 摄像头控制
    # =========================
    def start_camera(self):
        if self.face_thread is not None and self.face_thread.isRunning():
            return
        self.face_thread = FaceThread(camera_index=0)
        self.face_thread.frame_signal.connect(self._on_frame)
        self.face_thread.error_signal.connect(self._on_thread_error)
        self.face_thread.set_recognizing(self.recognizing)
        self.face_thread.start()
        self.camera_ready = True
        self.model_ready = False
        self._update_runtime_label()

    def stop_camera(self):
        if self.face_thread is not None:
            self.face_thread.stop()
            self.face_thread = None
        self.recognizing = False
        self.camera_ready = False
        self.model_ready = False
        self.loading_label.setText("等待点击“开始识别”")
        self.status_label.setText("")
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self._update_runtime_label()

    def start_recognition(self):
        self.start_camera()
        self.recognizing = True
        if self.face_thread is not None:
            self.face_thread.set_recognizing(True)
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self._update_runtime_label()

    def stop_recognition(self):
        self.recognizing = False
        if self.face_thread is not None:
            self.face_thread.set_recognizing(False)
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.loading_label.setText("识别已停止，可点击“开始识别”继续")
        self._update_runtime_label()

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
