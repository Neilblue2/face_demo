import cv2
import numpy as np
from insightface.app import FaceAnalysis

# =========================
# 初始化模型（只执行一次）
# =========================
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))


# =========================
# 读取图片
# =========================
def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {path}")
    return img


# =========================
# 提取 embedding（关键函数）
# =========================
def extract_embedding(img):
    faces = app.get(img)

    if len(faces) == 0:
        return None   # ⚠️ 不要抛异常，识别阶段会更稳定

    # ✅ 选最大的人脸（比“第一张”更可靠）
    face = max(faces, key=lambda x: x.bbox[2] - x.bbox[0])

    emb = face.embedding

    # ✅ 归一化（必须）
    emb = emb / np.linalg.norm(emb)

    # ✅ 统一类型（非常重要）
    emb = emb.astype(np.float32)

    return emb


# =========================
# 余弦相似度
# =========================
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))