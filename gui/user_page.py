from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout


class UserPage(QWidget):

    def __init__(self):
        super().__init__()

        layout = QVBoxLayout()

        layout.addWidget(QLabel("用户管理页面"))

        self.setLayout(layout)