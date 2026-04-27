# 基于深度学习人脸识别技术的课堂签到系统设计

本项目基于 InsightFace 提取人脸特征，PyQt5 构建桌面签到界面，MySQL 做后台用户与特征管理，目标是为课堂签到提供一个集识别、注册与管理于一体的原型系统。

## 项目核心文件
- `main.py`：程序入口，启动 PyQt5 `MainWindow` 并在桌面应用中加载各个功能页面。
- `gui/main_window.py`：左侧菜单 + `QStackedWidget` 页面容器，协调实时识别、注册与用户管理视图。
- `gui/face_page.py`：实时画面流 + `CORE.face_engine.detect_and_recognize`，在摄像头画面上叠加识别框与置信度。
- `gui/register_page.py`：采集 5 张高质量 embedding 并写入 `users` / `face_feature` 表，提供学号/姓名输入与注册按钮。
- `gui/user_page.py`：未来用户管理占位页，可以扩展删除/编辑操作。
- `CORE/face_engine.py`：封装 InsightFace 初始化、多帧稳定投票、数据库特征加载与识别逻辑，供 GUI 复用。
- `CORE/db.py`：MySQL 连接工厂，确保项目不同脚本都能使用统一的 `face_db` 账户与参数。
- `face_utils.py`：暴露 `load_image`、`extract_embedding` 与 `cosine_similarity`，对 InsightFace 的输入/输出做规范化，其他脚本可直接复用。
- `embedding_manager.py`：面对数据库中的 embedding 提供质量评估、删除低质量、限制每人数量、插入操作，可用于离线清洗。
- `register.py` / `register_multi.py`：单人/多人注册示例，`register_multi.py` 还包含采集多张图片并计算平均 embedding 的流程，适合作为批处理脚本。
- `recognize.py` / `recognize_muti.py`：离线图像与摄像头识别示例，`recognize_muti.py` 复用投票逻辑，并展示如何从数据库加载多条 embedding 做匹配。
- `data/`：
  - `data/images/`：原始人脸照片用于测试或单人注册脚本。
  - `data/embeddings/`：`np.save` 的 embedding 文件，可被 `recognize.py` 加载。
  - `data/register/`：批量注册时存放多个用户的图片子目录（每个子目录一名学生）。
- `note.md`：记录常用 MySQL 命令（`mysql -u face -p face_db`、`SHOW TABLES;`、`SELECT * FROM users;` 等），可直接复用进入数据库检查数据。

## 环境依赖
1. 安装 Python 库：
   ```bash
   pip install numpy opencv-python insightface PyQt5 pymysql
   ```
2. 如果要启用 `YOLOv5-Face + dlib` 引擎，额外安装：
   ```bash
   pip install onnxruntime dlib
   ```
3. 配置 MySQL：确保 `face_db` 数据库存在，并至少包含 `users` 与 `face_feature` 表（`note.md` 提供常用查询命令）。
   - `CORE/db.py` 默认连接 `127.0.0.1:3306`，用户 `face`，密码 `face123`。

## 数据库 & 表结构提示
- `users(name, student_id)` 存学号/姓名。
- `face_feature(user_id, embedding, quality)` 存 embedding blob 与可选质量分数。
- `embedding_manager` 中的 `evaluate_embedding_quality` 以 L2 norm 稳定度给出 0～1 分数，其他脚本可引用此函数统一判断。

## 运行命令
- 运行界面：`python main.py`，打开完整 CDT (Camera-Detection-Tracking) 界面。
- 单人注册：`python register.py`（直接从 `data/images/{name}.jpg` 抽取 embedding 并保存到 `data/embeddings`）。
- 批量注册：`python register_multi.py`，示例参数注册学号 `2023001`，可自行替换 `student_id`/`name`/`img_dir`。
- 单张图像识别：`python recognize.py`，对比 `data/embeddings/{user}.npy` 判断是否通过。
- 摄像头识别：`python recognize_muti.py`，内部使用 `detect_and_recognize` + 多帧投票逻辑，实时弹框识别结果。

## 引擎切换（可选）
- 默认引擎：`split`（RetinaFace + ArcFace）。
- 可用引擎：`unified`、`split`、`yolov5face_dlib`。
- 注意：`yolov5face_dlib` 生成的是 dlib 128 维特征，与 InsightFace 512 维特征不兼容。切换后需重新采集注册人脸数据。
- 通过环境变量切换：
  ```bash
  FACE_ENGINE_MODE=yolov5face_dlib \
  YOLOV5FACE_ONNX_PATH=models/yolov5s-face.onnx \
  DLIB_SHAPE_PREDICTOR_PATH=models/shape_predictor_68_face_landmarks.dat \
  DLIB_FACE_REC_PATH=models/dlib_face_recognition_resnet_model_v1.dat \
  python main.py
  ```

## 可选维护命令
- `python -c "from embedding_manager import limit_user_embeddings; limit_user_embeddings(5)"`：限制每位用户 embedding 数量。
- `python -c "from embedding_manager import delete_low_quality_embeddings; delete_low_quality_embeddings(0.6)"`：根据质量阈值清理旧 embedding。
- `python -c "from embedding_manager import delete_user; delete_user(1)"`：删除指定用户（需替换 ID）。

## 其他说明
- `gui/register_page.py` 与 `register_multi.py` 中都使用 `get_conn` 以事务方式写入 `users` + `face_feature`，注册后 `FacePage` 会在下一次加载时自动读取新数据。
- `data/embeddings` 与 `data/register` 可自由扩展，但最佳做法是每人专用文件夹保存多张图片，提高平均 embedding 质量。
