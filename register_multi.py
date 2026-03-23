import os
import numpy as np
from face_utils import load_image, extract_embedding
from CORE.db import get_conn


def register_users(student_id, name, img_dir):
    import numpy as np
    import os
    from embedding_manager import evaluate_embedding_quality

    embeddings = []
    qualities = []

    print(f"[INFO] 开始注册用户: {name}")
    print(f"[INFO] 图片目录: {img_dir}")

    # 1️⃣ 提取多张图片特征 + 质量评估
    for img_name in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img_name)

        print(f"[INFO] 处理图片: {img_name}")

        try:
            img = load_image(img_path)
            emb = extract_embedding(img)

            if emb is not None:
                emb = emb / np.linalg.norm(emb)

                # ⭐ 质量评分
                quality = evaluate_embedding_quality(emb)

                # ⭐ 过滤低质量（关键）
                if quality < 0.5:
                    print(f"  → 跳过（质量低: {quality:.2f}）")
                    continue

                embeddings.append(emb)
                qualities.append(quality)

                print(f"  → 提取成功（质量: {quality:.2f}）")

        except Exception as e:
            print(f"  → 跳过: {e}")

    if len(embeddings) == 0:
        print("[ERROR] 没有有效人脸")
        return

    print(f"[INFO] 共保留 {len(embeddings)} 个高质量特征")

    embeddings = np.array(embeddings)

    # 2️⃣ ⭐ 计算“均值 embedding”（核心优化）
    mean_emb = np.mean(embeddings, axis=0)
    mean_emb = mean_emb / np.linalg.norm(mean_emb)

    # 3️⃣ 数据库操作
    conn = get_conn()
    cur = conn.cursor()

    # 查用户
    cur.execute(
        "SELECT id FROM users WHERE student_id=%s",
        (student_id,)
    )
    result = cur.fetchone()

    if result:
        user_db_id = result[0]
        print(f"[INFO] 用户已存在，ID={user_db_id}")

        # ⭐ 清旧特征（推荐保留）
        cur.execute(
            "DELETE FROM face_feature WHERE user_id=%s",
            (user_db_id,)
        )
        print("[INFO] 已清除旧特征")

    else:
        cur.execute(
            "INSERT INTO users (name, student_id) VALUES (%s, %s)",
            (name, student_id)
        )
        user_db_id = cur.lastrowid
        print(f"[INFO] 新用户创建 ID={user_db_id}")

    # 4️⃣ ⭐ 存储多个 embedding（带质量）
    for emb, q in zip(embeddings, qualities):
        cur.execute(
            "INSERT INTO face_feature (user_id, embedding, quality) VALUES (%s, %s, %s)",
            (user_db_id, emb.astype(np.float32).tobytes(), float(q))
        )

    # 5️⃣ ⭐ 额外存一个“均值特征”（非常关键）
    cur.execute(
        "INSERT INTO face_feature (user_id, embedding, quality) VALUES (%s, %s, %s)",
        (user_db_id, mean_emb.astype(np.float32).tobytes(), 1.0)  # 均值设最高质量
    )

    conn.commit()
    conn.close()

    print(f"[OK] 注册完成，共存 {len(embeddings)+1} 条特征（含均值）")

if __name__ == "__main__":
    register_users(
        student_id="2023001",
        name="Neil",
        img_dir="data/register/neil"
    )