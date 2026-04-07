from PyQt5.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QTableWidget, QTableWidgetItem, QHeaderView
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from datetime import datetime

from CORE.db import get_conn


class HomePage(QWidget):

    def __init__(self):
        super().__init__()

        title = QLabel("课堂签到系统")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 24px; font-weight: bold;")

        self.course_label = QLabel("当前课程：无")
        self.course_label.setAlignment(Qt.AlignCenter)
        self.course_label.setStyleSheet("font-size: 18px; color: #333333;")

        # 统计
        self.total_label = QLabel("总人数：0")
        self.present_label = QLabel("实到人数：0")
        self.absent_label = QLabel("未到人数：0")
        self.present_label.setStyleSheet("color: #1b5e20;")
        self.absent_label.setStyleSheet("color: #d32f2f;")

        stats_layout = QHBoxLayout()
        stats_layout.addWidget(self.total_label)
        stats_layout.addWidget(self.present_label)
        stats_layout.addWidget(self.absent_label)

        # 表格
        self.present_table = QTableWidget()
        self.present_table.setColumnCount(3)
        self.present_table.setHorizontalHeaderLabels(["姓名", "学号", "班级"])
        self.present_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        self.absent_table = QTableWidget()
        self.absent_table.setColumnCount(3)
        self.absent_table.setHorizontalHeaderLabels(["姓名", "学号", "班级"])
        self.absent_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        table_layout = QHBoxLayout()
        table_layout.addWidget(self._wrap_table("实到学生", self.present_table))
        table_layout.addWidget(self._wrap_table("未到学生", self.absent_table))

        layout = QVBoxLayout()
        layout.addWidget(title)
        layout.addWidget(self.course_label)
        layout.addLayout(stats_layout)
        layout.addLayout(table_layout)

        self.setLayout(layout)

        self.refresh_data()

    def _wrap_table(self, title, table):
        box = QWidget()
        box_layout = QVBoxLayout()
        label = QLabel(title)
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("font-size: 16px; font-weight: bold;")
        if title == "实到学生":
            label.setStyleSheet("font-size: 16px; font-weight: bold; color: #1b5e20;")
        elif title == "未到学生":
            label.setStyleSheet("font-size: 16px; font-weight: bold; color: #d32f2f;")
        box_layout.addWidget(label)
        box_layout.addWidget(table)
        box.setLayout(box_layout)
        return box

    def refresh_data(self):
        conn = get_conn()
        cur = conn.cursor()
        now = datetime.now()

        cur.execute("""
            SELECT id, name, start_time, end_time
            FROM courses
            WHERE start_time <= %s AND end_time >= %s
            ORDER BY start_time DESC
            LIMIT 1
        """, (now, now))
        course = cur.fetchone()

        if not course:
            self.course_label.setText("当前课程：无")
            self.total_label.setText("总人数：0")
            self.present_label.setText("实到人数：0")
            self.absent_label.setText("未到人数：0")
            self.present_table.setRowCount(0)
            self.absent_table.setRowCount(0)
            conn.close()
            return

        course_id, course_name, start_time, end_time = course
        self.course_label.setText(
            f"当前课程：{course_name}（{start_time} - {end_time}）"
        )

        # 总名单
        cur.execute("""
            SELECT u.id, u.name, u.student_id, u.class_name
            FROM course_roster r
            JOIN users u ON r.user_id = u.id
            WHERE r.course_id = %s
            ORDER BY u.student_id
        """, (course_id,))
        roster = cur.fetchall()

        # 已签到
        cur.execute("""
            SELECT u.id, u.name, u.student_id, u.class_name
            FROM attendance a
            JOIN users u ON a.user_id = u.id
            WHERE a.course_id = %s
            ORDER BY u.student_id
        """, (course_id,))
        present = cur.fetchall()

        conn.close()

        roster_ids = {r[0] for r in roster}
        present_ids = {p[0] for p in present}
        absent = [r for r in roster if r[0] not in present_ids]

        self.total_label.setText(f"总人数：{len(roster_ids)}")
        self.present_label.setText(f"实到人数：{len(present_ids)}")
        self.absent_label.setText(f"未到人数：{len(absent)}")

        self._fill_table(self.present_table, present, "#1b5e20")
        self._fill_table(self.absent_table, absent, "#d32f2f")

    def _fill_table(self, table, rows, color):
        table.setRowCount(len(rows))
        for row_idx, (_, name, student_id, class_name) in enumerate(rows):
            item_name = QTableWidgetItem(name or "")
            item_sid = QTableWidgetItem(student_id or "")
            item_class = QTableWidgetItem(class_name or "")
            if color == "#1b5e20":
                item_name.setForeground(QColor("#1b5e20"))
                item_sid.setForeground(QColor("#1b5e20"))
                item_class.setForeground(QColor("#1b5e20"))
            else:
                item_name.setForeground(QColor("#d32f2f"))
                item_sid.setForeground(QColor("#d32f2f"))
                item_class.setForeground(QColor("#d32f2f"))
            table.setItem(row_idx, 0, item_name)
            table.setItem(row_idx, 1, item_sid)
            table.setItem(row_idx, 2, item_class)
