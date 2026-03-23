import numpy as np
from face_utils import load_image, extract_embedding, cosine_similarity

# 已注册用户
user_name = "neil2"
threshold = 0.8

db_embedding = np.load(f"data/embeddings/{user_name}.npy")

test_img = load_image("data/images/neil.jpg")
test_embedding = extract_embedding(test_img)

score = cosine_similarity(test_embedding, db_embedding)

print("相似度:", round(float(score), 4))

if score > threshold:
    print(f"[SUCCESS] 识别为 {user_name}")
else:
    print("[FAILED] 未识别到已注册用户")
