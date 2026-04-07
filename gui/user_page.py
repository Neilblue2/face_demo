from PyQt5.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QLineEdit, QPushButton, QTableWidget, QTableWidgetItem,
    QMessageBox, QAbstractItemView, QHeaderView
)
from PyQt5.QtCore import pyqtSignal

from CORE.db import get_conn


class UserPage(QWidget):
    users_changed = pyqtSignal()

    def __init__(self):
        super().__init__()

        # =========================
        # 表单
        # =========================
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("姓名")

        self.id_input = QLineEdit()
        self.id_input.setPlaceholderText("学号")

        self.class_input = QLineEdit()
        self.class_input.setPlaceholderText("班级")

        self.major_input = QLineEdit()
        self.major_input.setPlaceholderText("专业")

        self.btn_add = QPushButton("新增用户")
        self.btn_update = QPushButton("保存修改")
        self.btn_delete = QPushButton("删除所选")
        self.btn_refresh = QPushButton("刷新列表")

        form_layout = QHBoxLayout()
        form_layout.addWidget(self.name_input)
        form_layout.addWidget(self.id_input)
        form_layout.addWidget(self.class_input)
        form_layout.addWidget(self.major_input)
        form_layout.addWidget(self.btn_add)
        form_layout.addWidget(self.btn_update)
        form_layout.addWidget(self.btn_delete)
        form_layout.addWidget(self.btn_refresh)

        # =========================
        # 表格
        # =========================
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(["ID", "姓名", "学号", "班级", "专业", "Embedding数"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        layout = QVBoxLayout()
        layout.addLayout(form_layout)
        layout.addWidget(self.table)

        # =========================
        # 信号
        # =========================
        self.btn_add.clicked.connect(self.add_user)
        self.btn_update.clicked.connect(self.update_user)
        self.btn_delete.clicked.connect(self.delete_selected_user)
        self.btn_refresh.clicked.connect(self.load_users)
        self.table.itemSelectionChanged.connect(self._fill_form_from_selection)

        self.btn_add.setStyleSheet("color: #2e7d32;")
        self.btn_update.setStyleSheet("color: #1976d2;")
        self.btn_delete.setStyleSheet("color: #d32f2f;")

        self.setLayout(layout)

        self.load_users()
        self._selected_user_id = None

    # =========================
    # 加载用户列表
    # =========================
    def load_users(self):
        conn = get_conn()
        cur = conn.cursor()

        cur.execute("""
            SELECT u.id, u.name, u.student_id, u.class_name, u.major, COUNT(f.id) as feature_count
            FROM users u
            LEFT JOIN face_feature f ON u.id = f.user_id
            GROUP BY u.id, u.name, u.student_id, u.class_name, u.major
            ORDER BY u.id DESC
        """)

        rows = cur.fetchall()
        conn.close()

        self.table.setRowCount(len(rows))

        for row_idx, (user_id, name, student_id, class_name, major, feature_count) in enumerate(rows):
            self.table.setItem(row_idx, 0, QTableWidgetItem(str(user_id)))
            self.table.setItem(row_idx, 1, QTableWidgetItem(name or ""))
            self.table.setItem(row_idx, 2, QTableWidgetItem(student_id or ""))
            self.table.setItem(row_idx, 3, QTableWidgetItem(class_name or ""))
            self.table.setItem(row_idx, 4, QTableWidgetItem(major or ""))
            self.table.setItem(row_idx, 5, QTableWidgetItem(str(feature_count)))
        self._selected_user_id = None

    # =========================
    # 新增用户
    # =========================
    def add_user(self):
        name = self.name_input.text().strip()
        student_id = self.id_input.text().strip()
        class_name = self.class_input.text().strip()
        major = self.major_input.text().strip()

        if name == "" or student_id == "":
            QMessageBox.warning(self, "提示", "请输入姓名和学号")
            return

        conn = get_conn()
        cur = conn.cursor()

        cur.execute(
            "INSERT INTO users (name, student_id, class_name, major) VALUES (%s,%s,%s,%s)",
            (name, student_id, class_name, major)
        )

        conn.commit()
        conn.close()

        self.name_input.setText("")
        self.id_input.setText("")
        self.class_input.setText("")
        self.major_input.setText("")

        QMessageBox.information(self, "提示", "用户已新增（未采集人脸）")
        self.load_users()
        self.users_changed.emit()

    def update_user(self):
        if self._selected_user_id is None:
            QMessageBox.warning(self, "提示", "请先选择要编辑的用户")
            return

        name = self.name_input.text().strip()
        student_id = self.id_input.text().strip()
        class_name = self.class_input.text().strip()
        major = self.major_input.text().strip()

        if name == "" or student_id == "":
            QMessageBox.warning(self, "提示", "请输入姓名和学号")
            return

        conn = get_conn()
        cur = conn.cursor()
        cur.execute(
            "UPDATE users SET name=%s, student_id=%s, class_name=%s, major=%s WHERE id=%s",
            (name, student_id, class_name, major, self._selected_user_id)
        )
        conn.commit()
        conn.close()

        QMessageBox.information(self, "提示", "用户信息已更新")
        self.load_users()
        self.users_changed.emit()

    def _fill_form_from_selection(self):
        selected = self.table.selectionModel().selectedRows()
        if not selected:
            self._selected_user_id = None
            return

        row = selected[0].row()
        self._selected_user_id = int(self.table.item(row, 0).text())
        self.name_input.setText(self.table.item(row, 1).text())
        self.id_input.setText(self.table.item(row, 2).text())
        self.class_input.setText(self.table.item(row, 3).text())
        self.major_input.setText(self.table.item(row, 4).text())

    # =========================
    # 删除用户
    # =========================
    def delete_selected_user(self):
        selected = self.table.selectionModel().selectedRows()

        if not selected:
            QMessageBox.warning(self, "提示", "请先选择要删除的用户")
            return

        row = selected[0].row()
        user_id_item = self.table.item(row, 0)

        if user_id_item is None:
            return

        user_id = user_id_item.text()

        confirm = QMessageBox.question(
            self,
            "确认删除",
            f"确认删除用户 ID {user_id} 吗？这会删除该用户的人脸特征。",
            QMessageBox.Yes | QMessageBox.No
        )

        if confirm != QMessageBox.Yes:
            return

        conn = get_conn()
        cur = conn.cursor()

        cur.execute("DELETE FROM face_feature WHERE user_id=%s", (user_id,))
        cur.execute("DELETE FROM users WHERE id=%s", (user_id,))

        conn.commit()
        conn.close()

        QMessageBox.information(self, "提示", "用户已删除")
        self.load_users()
        self.users_changed.emit()
