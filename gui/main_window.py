import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton,
    QHBoxLayout, QVBoxLayout, QStackedWidget
)

from gui.face_page import FacePage
from gui.register_page import RegisterPage
from gui.user_page import UserPage


class MainWindow(QWidget):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("人脸识别系统")
        self.resize(1100, 700)

        # =========================
        # 左侧菜单
        # =========================
        self.btn_face = QPushButton("实时识别")
        self.btn_register = QPushButton("用户注册")
        self.btn_users = QPushButton("用户管理")

        menu_layout = QVBoxLayout()
        menu_layout.addWidget(self.btn_face)
        menu_layout.addWidget(self.btn_register)
        menu_layout.addWidget(self.btn_users)
        menu_layout.addStretch()

        menu_widget = QWidget()
        menu_widget.setLayout(menu_layout)
        menu_widget.setFixedWidth(150)

        # =========================
        # 页面
        # =========================
        self.stack = QStackedWidget()

        self.page_face = FacePage()
        self.page_register = RegisterPage()
        self.page_users = UserPage()

        self.stack.addWidget(self.page_face)
        self.stack.addWidget(self.page_register)
        self.stack.addWidget(self.page_users)

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
        self.btn_face.clicked.connect(lambda: self.stack.setCurrentIndex(0))
        self.btn_register.clicked.connect(lambda: self.stack.setCurrentIndex(1))
        self.btn_users.clicked.connect(lambda: self.stack.setCurrentIndex(2))


if __name__ == "__main__":

    app = QApplication(sys.argv)

    win = MainWindow()
    win.show()

    sys.exit(app.exec_())