import cv2
import numpy as np
from insightface.app import FaceAnalysis
from CORE.db import get_conn
from collections import deque, Counter

# =========================
# 参数
# =========================
THRESHOLD = 0.5
FRAME_WINDOW = 5

# =========================
# 多帧稳定缓存
# =========================
name_buffer = deque(maxlen=FRAME_WINDOW)
score_buffer = deque(maxlen=FRAME_WINDOW)

# =========================
# 初始化模型（只加载一次）
# =========================
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))


# =========================
# 加载数据库特征
# =========================
def load_db_features():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        SELECT u.id, u.name, f.embedding
        FROM users u
        JOIN face_feature f ON u.id = f.user_id
    """)

    rows = cur.fetchall()
    conn.close()

    user_dict = {}

    for user_id, name, emb_blob in rows:
        emb = np.frombuffer(emb_blob, dtype=np.float32)

        if user_id not in user_dict:
            user_dict[user_id] = {
                "name": name,
                "embeddings": []
            }

        user_dict[user_id]["embeddings"].append(emb)

    db_features = list(user_dict.values())

    print(f"[INFO] 已加载 {len(db_features)} 个用户")

    return db_features


# =========================
# 相似度计算
# =========================
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# =========================
# 单帧识别
# =========================
def recognize(emb, db_features):

    best_score = -1
    best_user = "Unknown"

    for user in db_features:
        for emb_db in user["embeddings"]:

            score = cosine_similarity(emb, emb_db)

            if score > best_score:
                best_score = score
                best_user = user["name"]

    return best_user, best_score


# =========================
# 多帧稳定识别
# =========================
def get_stable_result(name, score):

    name_buffer.append(name)
    score_buffer.append(score)

    most_common_name = Counter(name_buffer).most_common(1)[0][0]
    avg_score = sum(score_buffer) / len(score_buffer)

    if avg_score < THRESHOLD:
        return "Unknown", avg_score

    return most_common_name, avg_score


# =========================
# 检测 + 识别封装（推荐调用）
# =========================
def detect_and_recognize(frame, db_features):

    faces = app.get(frame)

    results = []

    for face in faces:

        emb = face.embedding
        bbox = face.bbox.astype(int)

        name, score = recognize(emb, db_features)
        stable_name, stable_score = get_stable_result(name, score)

        results.append({
            "bbox": bbox,
            "name": stable_name,
            "score": stable_score
        })

    return results