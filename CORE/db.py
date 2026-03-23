import pymysql

def get_conn():
    return pymysql.connect(
        host="127.0.0.1",
        user="face",
        password="face123",
        database="face_db",
        port=3306,
        charset="utf8mb4"
    )
