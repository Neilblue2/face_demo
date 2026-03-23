import cv2
import numpy as np
from insightface.app import FaceAnalysis
from CORE.db import get_conn

from collections import deque, Counter

# 多帧缓存
FRAME_WINDOW = 5
name_buffer = deque(maxlen=FRAME_WINDOW)
score_buffer = deque(maxlen=FRAME_WINDOW)

THRESHOLD = 0.5

# =========================
# 初始化模型（只初始化一次）
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

    # 👉 聚合
    user_dict = {}

    for user_id, name, emb_blob in rows:
        emb = np.frombuffer(emb_blob, dtype=np.float32)

        if user_id not in user_dict:
            user_dict[user_id] = {
                "name": name,
                "embeddings": []
            }

        user_dict[user_id]["embeddings"].append(emb)

    # 👉 转 list
    db_features = list(user_dict.values())

    print(f"[INFO] 已加载 {len(db_features)} 个用户特征")

    return db_features


# =========================
# 相似度
# =========================
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# =========================
# 匹配
# =========================
def recognize(emb, db_features):
    best_score = -1
    best_user = "Unknown"

    # ✅ 多 embedding 匹配
    best_score = -1
    best_user = "Unknown"

    for user in db_features:
        for emb_db in user["embeddings"]:
            score = cosine_similarity(emb, emb_db)

        if score > best_score:
            best_score = score
            best_user = user["name"]

    return best_user, best_score

def get_stable_result(name, score):
    name_buffer.append(name)
    score_buffer.append(score)

    # 投票机制
    most_common_name = Counter(name_buffer).most_common(1)[0][0]

    # 平均分数
    avg_score = sum(score_buffer) / len(score_buffer)

    # 阈值判断
    if avg_score < THRESHOLD:
        return "Unknown", avg_score

    return most_common_name, avg_score

# =========================
# 主程序
# =========================
def main():
    db_features = load_db_features()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ 摄像头打开失败")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 🔥 关键：直接用 InsightFace 检测
        faces = app.get(frame)

        for face in faces:
            emb = face.embedding
            bbox = face.bbox.astype(int)

            name, score = recognize(emb, db_features)
            stable_name, stable_score = get_stable_result(name, score)

            x1, y1, x2, y2 = bbox

            # 画框
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 显示名字
            cv2.putText(frame,
                        f"{stable_name} ({stable_score:.2f})",
                        (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2)

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()