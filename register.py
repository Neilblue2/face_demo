import os
import numpy as np
from face_utils import load_image, extract_embedding

# 用户名
user_name = "neil2"

img_path = f"data/images/{user_name}.jpg"
save_path = f"data/embeddings/{user_name}.npy"

os.makedirs("data/embeddings", exist_ok=True)

img = load_image(img_path)
embedding = extract_embedding(img)

np.save(save_path, embedding)
print(f"[INFO] 用户 {user_name} 注册成功，特征已保存")
print("Embedding shape:", embedding.shape)
