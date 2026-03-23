import numpy as np
from CORE.db import get_conn

def evaluate_embedding_quality(emb):
    """
    简单质量评估（可扩展）
    """
    if emb is None:
        return 0.0

    # 1. L2 norm（InsightFace 正常≈1）
    norm = np.linalg.norm(emb)

    # 2. 稳定性评分（接近1更好）
    score = 1 - abs(norm - 1)

    return float(score)


def delete_user(user_id):
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("DELETE FROM face_feature WHERE user_id=%s", (user_id,))
    cur.execute("DELETE FROM users WHERE id=%s", (user_id,))

    conn.commit()
    conn.close()

    print(f"[OK] 用户 {user_id} 已删除")


def delete_low_quality_embeddings(threshold=0.5):
    conn = get_conn()
    cur = conn.cursor()

    cur.execute(
        "DELETE FROM face_feature WHERE quality < %s",
        (threshold,)
    )

    deleted = cur.rowcount

    conn.commit()
    conn.close()

    print(f"[OK] 删除低质量 embedding: {deleted} 条")


def limit_user_embeddings(max_per_user=5):
    conn = get_conn()
    cur = conn.cursor()

    # 找所有用户
    cur.execute("SELECT DISTINCT user_id FROM face_feature")
    users = cur.fetchall()

    total_deleted = 0

    for (user_id,) in users:
        cur.execute("""
            SELECT id FROM face_feature
            WHERE user_id=%s
            ORDER BY quality DESC
        """, (user_id,))

        rows = cur.fetchall()

        # 超出部分删除
        if len(rows) > max_per_user:
            to_delete = rows[max_per_user:]

            ids = [str(r[0]) for r in to_delete]
            id_str = ",".join(ids)

            cur.execute(f"DELETE FROM face_feature WHERE id IN ({id_str})")
            total_deleted += len(ids)

    conn.commit()
    conn.close()

    print(f"[OK] 限制完成，删除 {total_deleted} 条 embedding")


def insert_embedding(user_id, emb):
    quality = evaluate_embedding_quality(emb)

    conn = get_conn()
    cur = conn.cursor()

    cur.execute(
        "INSERT INTO face_feature (user_id, embedding, quality) VALUES (%s, %s, %s)",
        (user_id, emb.astype(np.float32).tobytes(), quality)
    )

    conn.commit()
    conn.close()