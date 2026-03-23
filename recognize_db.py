import numpy as np
from face_utils import load_image, extract_embedding, cosine_similarity
from CORE.db import get_conn

THRESHOLD = 0.8

def recognize(img_path):
    img = load_image(img_path)
    emb = extract_embedding(img)

    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT user_id, embedding FROM face_embedding")
    records = cur.fetchall()

    best_score = 0
    best_user = None

    for user_id, emb_blob in records:
        db_emb = np.frombuffer(emb_blob, dtype=np.float32)
        score = cosine_similarity(emb, db_emb)

        if score > best_score:
            best_score = score
            best_user = user_id

    conn.close()

    print("最高相似度:", round(float(best_score), 4))

    if best_score > THRESHOLD:
        print(f"[SUCCESS] 识别为 {best_user}")
        return best_user
    else:
        print("[FAILED] 未识别到用户")
        return None

if __name__ == "__main__":
    recognize("data/images/test.jpg")
