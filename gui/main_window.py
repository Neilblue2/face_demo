import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton,
    QHBoxLayout, QVBoxLayout, QStackedWidget, QInputDialog, QMessageBox
)

from gui.face_page import FacePage
from gui.home_page import HomePage
from gui.course_page import CoursePage
from gui.register_page import RegisterPage
from gui.user_page import UserPage


class MainWindow(QWidget):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("人脸识别系统（普通用户）")
        self.resize(1100, 700)

        # =========================
        # 左侧菜单
        # =========================
        self.btn_home = QPushButton("返回首页")
        self.btn_face = QPushButton("签到打卡")
        self.btn_admin = QPushButton("后台管理")
        self.btn_register = QPushButton("用户注册")
        self.btn_course = QPushButton("课程管理")
        self.btn_users = QPushButton("用户管理")
        self.btn_exit_admin = QPushButton("退出后台管理")

        menu_layout = QVBoxLayout()
        menu_layout.setSpacing(0)
        menu_layout.setContentsMargins(0, 0, 0, 0)
        menu_layout.addWidget(self.btn_home, 1)
        menu_layout.addWidget(self.btn_face, 1)
        menu_layout.addWidget(self.btn_admin, 1)
        menu_layout.addWidget(self.btn_register, 1)
        menu_layout.addWidget(self.btn_course, 1)
        menu_layout.addWidget(self.btn_users, 1)
        menu_layout.addWidget(self.btn_exit_admin, 1)

        menu_widget = QWidget()
        menu_widget.setLayout(menu_layout)
        menu_widget.setFixedWidth(180)

        # =========================
        # 页面
        # =========================
        self.stack = QStackedWidget()

        self.page_home = HomePage()
        self.page_face = FacePage()
        self.page_register = RegisterPage()
        self.page_course = CoursePage()
        self.page_users = UserPage()

        self.stack.addWidget(self.page_home)     # 0
        self.stack.addWidget(self.page_face)     # 1
        self.stack.addWidget(self.page_register) # 2
        self.stack.addWidget(self.page_course)   # 3
        self.stack.addWidget(self.page_users)    # 4

        self.stack.currentChanged.connect(self.on_page_changed)
        self.on_page_changed(self.stack.currentIndex())
        self.page_register.register_success.connect(self.page_face.refresh_db_features)
        self.page_users.users_changed.connect(self.page_face.refresh_db_features)

        # =========================
        # 主布局
        # =========================
        main_layout = QHBoxLayout()

        main_layout.addWidget(menu_widget)
        main_layout.addWidget(self.stack)

        self.setLayout(main_layout)

        # =========================
        # 菜单点击
        # =========================
        self.btn_home.clicked.connect(lambda: self.stack.setCurrentIndex(0))
        self.btn_face.clicked.connect(lambda: self.stack.setCurrentIndex(1))
        self.btn_admin.clicked.connect(self.enter_admin_mode)
        self.btn_register.clicked.connect(lambda: self.stack.setCurrentIndex(2))
        self.btn_course.clicked.connect(lambda: self.stack.setCurrentIndex(3))
        self.btn_users.clicked.connect(lambda: self.stack.setCurrentIndex(4))
        self.btn_exit_admin.clicked.connect(self.exit_admin_mode)

        buttons = [
            self.btn_home, self.btn_face, self.btn_admin,
            self.btn_register, self.btn_course, self.btn_users, self.btn_exit_admin
        ]

        for btn in buttons:
            btn.setStyleSheet("font-size:16px;")
            btn.setMinimumHeight(60)

        self.btn_course.setStyleSheet("color: #1976d2; font-size:16px;")
        self.btn_users.setStyleSheet("color: #1976d2; font-size:16px;")
        self.btn_exit_admin.setStyleSheet("color: #d32f2f; font-size:16px;")

        self.admin_password = "admin123"
        self.admin_mode = False
        self._update_admin_menu()
        self._update_window_title()

    def on_page_changed(self, index):

        self.page_face.stop_camera()
        self.page_register.stop_camera()

        if index == 1:
            self.page_face.start_camera()
        elif index == 2:
            self.page_register.start_camera()
        elif index == 3:
            self.page_course.load_courses()
        elif index == 4:
            self.page_users.load_users()
        elif index == 0:
            self.page_home.refresh_data()

    def _update_admin_menu(self):
        self.btn_admin.setVisible(not self.admin_mode)
        self.btn_register.setVisible(self.admin_mode)
        self.btn_course.setVisible(self.admin_mode)
        self.btn_users.setVisible(self.admin_mode)
        self.btn_exit_admin.setVisible(self.admin_mode)

    def _update_window_title(self):
        role = "管理员" if self.admin_mode else "普通用户"
        self.setWindowTitle(f"人脸识别系统（{role}）")

    def enter_admin_mode(self):
        if self.admin_mode:
            return

        pwd, ok = QInputDialog.getText(self, "后台管理", "请输入管理员密码：")
        if not ok:
            return

        if pwd != self.admin_password:
            QMessageBox.warning(self, "提示", "管理员密码错误")
            return

        self.admin_mode = True
        self._update_admin_menu()
        self._update_window_title()

    def exit_admin_mode(self):
        self.admin_mode = False
        self._update_admin_menu()
        self._update_window_title()
        self.stack.setCurrentIndex(0)


if __name__ == "__main__":

    app = QApplication(sys.argv)

    win = MainWindow()
    win.show()

    sys.exit(app.exec_())
