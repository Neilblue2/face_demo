# 基于深度学习人脸识别技术的课堂签到系统

本项目是一个基于 `PyQt5 + InsightFace + MySQL` 的课堂签到系统，支持实时人脸识别签到、用户注册、课程管理、课程名单维护和签到导出。

## 当前系统功能
- 实时签到：摄像头识别人脸，自动判断当前课程并写入签到记录。
- 用户注册：采集 5 张人脸特征后入库（`users` + `face_feature`）。
- 用户管理：新增、编辑、删除用户，并展示每位用户的 embedding 数量。
- 课程管理：创建/修改/删除课程，导入课程名单，手动加减课程成员。
- 签到导出：可按课程或全部导出签到记录（CSV/XLSX）。
- 首页看板：显示当前课程、实到/未到统计与名单。

## 项目结构（核心）
- `main.py`：程序入口。
- `gui/main_window.py`：主窗口、菜单、管理员模式切换。
- `gui/home_page.py`：当前课程统计看板（实到/未到）。
- `gui/face_page.py`：摄像头识别与签到逻辑。
- `gui/face_thread.py`：识别线程，循环采集并调用识别引擎。
- `gui/register_page.py`：人脸采集注册页（采集 5 张后写库）。
- `gui/course_page.py`：课程管理、名单导入、签到导出。
- `gui/user_page.py`：用户管理（增删改）。
- `CORE/face_engine.py`：检测识别引擎封装（支持多引擎模式）。
- `CORE/db.py`：MySQL 连接配置。
- `embedding_manager.py`：embedding 质量维护脚本。
- `register_multi.py`：离线批量注册示例脚本。

## 环境依赖
推荐先使用项目内的依赖文件安装：

```bash
pip install -r requirements.txt
```

如果 `requirements.txt` 未包含可选功能依赖，可按需补装：

- 课程名单 Excel 导入/签到导出 `.xlsx`：

```bash
pip install openpyxl
```

- 启用 `YOLOv5-Face + dlib` 引擎：

```bash
pip install onnxruntime dlib
```

## 数据库配置
数据库连接在 `CORE/db.py` 中，当前默认：
- host: `127.0.0.1`
- port: `3306`
- user: `face`
- password: `face123`
- database: `face_db`

可参考 `note.md` 里的常用 MySQL 命令检查数据。

## 必要数据表
当前系统代码依赖以下表：
- `users`：用户基本信息（至少包含 `id`, `name`, `student_id`, `class_name`, `major`）。
- `face_feature`：人脸特征（至少包含 `id`, `user_id`, `embedding`, `quality`）。
- `courses`：课程信息（`id`, `name`, `start_time`, `end_time`）。
- `course_roster`：课程名单（`course_id`, `user_id`）。
- `attendance`：签到记录（`id`, `user_id`, `course_id`, `course_name`, `checkin_time`）。

## 运行
启动桌面系统：

```bash
python main.py
```

管理员入口：
- 在主界面点击“后台管理”
- 密码默认为 `admin123`（定义在 `gui/main_window.py`）

## 签到流程说明（当前实现）
1. `FacePage` 读取摄像头帧并做人脸识别。
2. 根据识别用户查询“当前时间是否存在进行中的课程”。
3. 若用户在该课程名单中且未签到，则写入 `attendance`。
4. 若不在名单、重复签到或无当前课程，会给出对应提示。

## 人脸引擎模式
`CORE/face_engine.py` 支持 3 种模式：
- `split`（默认）：RetinaFace 检测 + ArcFace 识别。
- `unified`：InsightFace `buffalo_l` 一体化。
- `yolov5face_dlib`：YOLOv5-Face 检测 + dlib 识别。

通过环境变量切换：

```bash
FACE_ENGINE_MODE=yolov5face_dlib \
YOLOV5FACE_ONNX_PATH=models/yolov5s-face.onnx \
DLIB_SHAPE_PREDICTOR_PATH=models/shape_predictor_68_face_landmarks.dat \
DLIB_FACE_REC_PATH=models/dlib_face_recognition_resnet_model_v1.dat \
python main.py
```

注意：
- `yolov5face_dlib` 产生 128 维 dlib 特征。
- `split/unified` 使用 InsightFace 特征（通常 512 维）。
- 两类特征不兼容，切换引擎后需要重新采集注册数据。

## 其他脚本
- `register.py`：单图注册到 `data/embeddings` 的示例。
- `recognize.py`：单图识别示例。
- `recognize_muti.py`：旧版摄像头识别示例（独立于 GUI）。
- `register_multi.py`：按目录批量注册到数据库，含质量过滤与均值 embedding。
- `embedding_manager.py`：低质量清理、每人 embedding 限制、插入 embedding 等工具函数。

## 常见问题
- 摄像头打不开：检查是否被其他程序占用。
- 没有识别结果：确认数据库中已有 `face_feature` 数据，且引擎模式与特征维度一致。
- 导入/导出 xlsx 失败：安装 `openpyxl`。
