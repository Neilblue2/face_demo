import os
from collections import Counter, deque
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
from insightface.utils import face_align

from CORE.db import get_conn

try:
    import dlib
except Exception:
    dlib = None

try:
    import onnxruntime as ort
except Exception:
    ort = None

# =========================
# 参数
# =========================
THRESHOLD = 0.5
FRAME_WINDOW = 5
DETECTION_SIZE = (640, 640)

# 引擎模式：
# - "unified": FaceAnalysis(buffalo_l) 一体化（原有方案）
# - "split":   RetinaFace(检测) + ArcFace(识别) 两段式
# - "yolov5face_dlib": YOLOv5-Face(检测) + dlib(识别)
ENGINE_MODE = os.getenv("FACE_ENGINE_MODE", "split")
if ENGINE_MODE not in {"unified", "split", "yolov5face_dlib"}:
    print(f"[WARN] 未知引擎模式: {ENGINE_MODE}，已回退为 split")
    ENGINE_MODE = "split"

# 两段式模型名（InsightFace 预训练）
RETINAFACE_MODEL_NAME = "retinaface_r50_v1"
ARCFACE_MODEL_NAME = "arcface_r100_v1"
RETINAFACE_FALLBACKS = ["det_10g.onnx"]
ARCFACE_FALLBACKS = ["w600k_r50.onnx"]

# YOLOv5-Face + dlib 模型路径（可用环境变量覆盖）
YOLOV5FACE_ONNX_PATH = os.getenv(
    "YOLOV5FACE_ONNX_PATH",
    str(Path("models") / "yolov5s-face.onnx")
)
DLIB_SHAPE_PREDICTOR_PATH = os.getenv(
    "DLIB_SHAPE_PREDICTOR_PATH",
    str(Path("models") / "shape_predictor_68_face_landmarks.dat")
)
DLIB_FACE_REC_PATH = os.getenv(
    "DLIB_FACE_REC_PATH",
    str(Path("models") / "dlib_face_recognition_resnet_model_v1.dat")
)
YOLO_CONF_THRES = 0.35
YOLO_NMS_THRES = 0.45


def set_engine_mode(mode):
    """
    动态切换引擎模式：
    - "unified"
    - "split"
    - "yolov5face_dlib"
    """
    global ENGINE_MODE
    if mode not in {"unified", "split", "yolov5face_dlib"}:
        raise ValueError("mode 必须是 'unified'、'split' 或 'yolov5face_dlib'")
    ENGINE_MODE = mode


def get_engine_mode():
    return ENGINE_MODE


# =========================
# 多帧稳定缓存
# =========================
name_buffer = deque(maxlen=FRAME_WINDOW)
score_buffer = deque(maxlen=FRAME_WINDOW)


# =========================
# 初始化模型（只加载一次，按需懒加载）
# =========================
_unified_app = None
_detector = None
_recognizer = None
_yolo_session = None
_dlib_shape = None
_dlib_rec = None


def _prepare_model(model, ctx_id=0, det_size=None):
    """兼容不同模型 prepare 参数命名差异。"""
    if det_size is None:
        model.prepare(ctx_id=ctx_id)
        return
    try:
        model.prepare(ctx_id=ctx_id, input_size=det_size)
    except TypeError:
        model.prepare(ctx_id=ctx_id, det_size=det_size)


def _load_model_with_fallback(primary_name, fallback_files):
    """
    某些 insightface 版本里 get_model('retinaface_r50_v1') 可能返回 None。
    这里先尝试模型名，再回退到 ~/.insightface/models/** 下的 onnx 文件。
    """
    candidates = [primary_name, *fallback_files]

    for candidate in candidates:
        try:
            model = get_model(candidate)
            if model is not None:
                return model
        except Exception:
            pass

        if candidate.endswith(".onnx"):
            model_root = Path.home() / ".insightface" / "models"
            if model_root.exists():
                for onnx_path in model_root.rglob(candidate):
                    try:
                        model = get_model(str(onnx_path))
                        if model is not None:
                            return model
                    except Exception:
                        continue

    raise RuntimeError(
        f"无法加载模型: {primary_name}. "
        f"请确认已安装 insightface 并下载模型，或检查 ~/.insightface/models 下是否存在 {fallback_files}"
    )


def _resolve_model_path(path_str):
    p = Path(path_str)
    if p.exists():
        return p
    repo_rel = Path(__file__).resolve().parent.parent / path_str
    if repo_rel.exists():
        return repo_rel
    raise RuntimeError(f"模型文件不存在: {path_str}")


def init_unified_engine(ctx_id=0):
    """版本A：一体化方案（FaceAnalysis + buffalo_l）。"""
    global _unified_app
    if _unified_app is None:
        _unified_app = FaceAnalysis(name="buffalo_l")
        _prepare_model(_unified_app, ctx_id=ctx_id, det_size=DETECTION_SIZE)


def init_split_engine(ctx_id=0):
    """版本B：两段式方案（RetinaFace 检测 + ArcFace 识别）。"""
    global _detector, _recognizer
    if _detector is None:
        _detector = _load_model_with_fallback(RETINAFACE_MODEL_NAME, RETINAFACE_FALLBACKS)
        _prepare_model(_detector, ctx_id=ctx_id, det_size=DETECTION_SIZE)
    if _recognizer is None:
        _recognizer = _load_model_with_fallback(ARCFACE_MODEL_NAME, ARCFACE_FALLBACKS)
        _prepare_model(_recognizer, ctx_id=ctx_id)


def init_yolov5face_dlib_engine():
    """版本C：YOLOv5-Face 检测 + dlib 特征。"""
    global _yolo_session, _dlib_shape, _dlib_rec
    if ort is None:
        raise RuntimeError("缺少 onnxruntime 依赖，请先安装: pip install onnxruntime")
    if dlib is None:
        raise RuntimeError("缺少 dlib 依赖，请先安装 dlib")

    if _yolo_session is None:
        yolo_path = _resolve_model_path(YOLOV5FACE_ONNX_PATH)
        providers = ["CPUExecutionProvider"]
        _yolo_session = ort.InferenceSession(str(yolo_path), providers=providers)

    if _dlib_shape is None:
        shape_path = _resolve_model_path(DLIB_SHAPE_PREDICTOR_PATH)
        _dlib_shape = dlib.shape_predictor(str(shape_path))

    if _dlib_rec is None:
        rec_path = _resolve_model_path(DLIB_FACE_REC_PATH)
        _dlib_rec = dlib.face_recognition_model_v1(str(rec_path))


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
                "id": user_id,
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


def _normalize_embedding(emb):
    emb = emb.astype(np.float32)
    norm = np.linalg.norm(emb)
    if norm == 0:
        return emb
    return emb / norm


# =========================
# 单帧识别
# =========================
def recognize(emb, db_features):
    best_score = -1
    best_user = "Unknown"

    for user in db_features:
        for emb_db in user["embeddings"]:
            if emb_db.shape != emb.shape:
                continue
            score = cosine_similarity(emb, emb_db)
            if score > best_score:
                best_score = score
                best_user = user

    if best_user == "Unknown":
        return None, "Unknown", best_score

    return best_user["id"], best_user["name"], best_score


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


def _letterbox(image, new_shape=(640, 640), color=(114, 114, 114)):
    shape = image.shape[:2]  # h, w
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))  # w, h

    dw = (new_shape[1] - new_unpad[0]) / 2
    dh = (new_shape[0] - new_unpad[1]) / 2

    if shape[::-1] != new_unpad:
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return image, r, dw, dh


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _nms_xyxy(boxes, scores, iou_thres):
    if len(boxes) == 0:
        return []
    boxes = boxes.astype(np.float32)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-7)
        inds = np.where(ovr <= iou_thres)[0]
        order = order[inds + 1]
    return keep


def _decode_yolov5face(frame):
    init_yolov5face_dlib_engine()

    input_tensor, ratio, dw, dh = _letterbox(frame, DETECTION_SIZE)
    inp = cv2.cvtColor(input_tensor, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    inp = np.transpose(inp, (2, 0, 1))[None, ...]

    input_name = _yolo_session.get_inputs()[0].name
    outputs = _yolo_session.run(None, {input_name: inp})
    pred = outputs[0]

    if pred.ndim == 3:
        pred = pred[0]
    if pred.ndim != 2 or pred.shape[1] < 5:
        raise RuntimeError("YOLOv5-Face ONNX 输出格式不匹配，需为 [1,N,C] 且 C>=5 的解码后模型")

    obj = pred[:, 4]
    if np.any(obj < 0.0) or np.any(obj > 1.0):
        obj = _sigmoid(obj)

    if pred.shape[1] >= 16:
        cls_conf = pred[:, 15]
        if np.any(cls_conf < 0.0) or np.any(cls_conf > 1.0):
            cls_conf = _sigmoid(cls_conf)
        conf = obj * cls_conf
    else:
        conf = obj

    keep = conf > YOLO_CONF_THRES
    if not np.any(keep):
        return []

    pred = pred[keep]
    conf = conf[keep]

    xywh = pred[:, :4].copy()
    if np.max(xywh) <= 2.5:
        xywh[:, [0, 2]] *= DETECTION_SIZE[1]
        xywh[:, [1, 3]] *= DETECTION_SIZE[0]

    boxes = np.zeros((xywh.shape[0], 4), dtype=np.float32)
    boxes[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
    boxes[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
    boxes[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
    boxes[:, 3] = xywh[:, 1] + xywh[:, 3] / 2

    boxes[:, [0, 2]] -= dw
    boxes[:, [1, 3]] -= dh
    boxes /= ratio

    h, w = frame.shape[:2]
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h - 1)

    kps = None
    if pred.shape[1] >= 15:
        kps = pred[:, 5:15].reshape(-1, 5, 2).copy()
        if np.max(kps) <= 2.5:
            kps[:, :, 0] *= DETECTION_SIZE[1]
            kps[:, :, 1] *= DETECTION_SIZE[0]
        kps[:, :, 0] = (kps[:, :, 0] - dw) / ratio
        kps[:, :, 1] = (kps[:, :, 1] - dh) / ratio
        kps[:, :, 0] = np.clip(kps[:, :, 0], 0, w - 1)
        kps[:, :, 1] = np.clip(kps[:, :, 1], 0, h - 1)

    keep_idx = _nms_xyxy(boxes, conf, YOLO_NMS_THRES)
    results = []
    for idx in keep_idx:
        x1, y1, x2, y2 = boxes[idx].astype(int)
        if x2 <= x1 or y2 <= y1:
            continue
        results.append({
            "bbox": np.array([x1, y1, x2, y2], dtype=np.int32),
            "kps": None if kps is None else kps[idx].astype(np.float32),
            "det_score": float(conf[idx]),
        })
    return results


def _shape_to_quality_kps(shape):
    pts = np.array([[shape.part(i).x, shape.part(i).y] for i in range(shape.num_parts)], dtype=np.float32)
    if shape.num_parts >= 68:
        left_eye = pts[36:42].mean(axis=0)
        right_eye = pts[42:48].mean(axis=0)
        nose = pts[30]
        return np.stack([left_eye, right_eye, nose], axis=0)
    if shape.num_parts >= 5:
        return pts[:3]
    return None


def _dlib_embedding_from_bbox(frame, bbox):
    init_yolov5face_dlib_engine()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    x1, y1, x2, y2 = bbox.tolist()
    rect = dlib.rectangle(int(x1), int(y1), int(x2), int(y2))
    shape = _dlib_shape(rgb, rect)
    descriptor = _dlib_rec.compute_face_descriptor(rgb, shape)
    emb = np.array(descriptor, dtype=np.float32)
    emb = _normalize_embedding(emb)
    return emb, _shape_to_quality_kps(shape)


def detect_and_recognize_unified(frame, db_features):
    """版本A：FaceAnalysis(buffalo_l) 一体化。"""
    init_unified_engine()

    faces = _unified_app.get(frame)
    results = []
    for face in faces:
        emb = _normalize_embedding(face.embedding)
        bbox = face.bbox.astype(int)

        user_id, name, score = recognize(emb, db_features)
        stable_name, stable_score = get_stable_result(name, score)
        results.append({
            "bbox": bbox,
            "user_id": user_id,
            "name": stable_name,
            "score": stable_score
        })
    return results


def detect_and_recognize_split(frame, db_features):
    """版本B：RetinaFace(检测) + ArcFace(识别) 两段式。"""
    init_split_engine()

    bboxes, kpss = _detector.detect(frame, max_num=0, metric="default")
    if bboxes is None or len(bboxes) == 0:
        return []

    results = []
    for i in range(len(bboxes)):
        x1, y1, x2, y2 = bboxes[i, :4].astype(int)
        kps = kpss[i]
        face_chip = face_align.norm_crop(frame, landmark=kps)
        emb = _recognizer.get_feat(face_chip).flatten()
        emb = _normalize_embedding(emb)

        user_id, name, score = recognize(emb, db_features)
        stable_name, stable_score = get_stable_result(name, score)
        results.append({
            "bbox": np.array([x1, y1, x2, y2]),
            "user_id": user_id,
            "name": stable_name,
            "score": stable_score
        })
    return results


def detect_and_recognize_yolov5face_dlib(frame, db_features):
    """版本C：YOLOv5-Face(检测) + dlib(识别)。"""
    dets = _decode_yolov5face(frame)
    results = []
    for det in dets:
        emb, _ = _dlib_embedding_from_bbox(frame, det["bbox"])
        user_id, name, score = recognize(emb, db_features)
        stable_name, stable_score = get_stable_result(name, score)
        results.append({
            "bbox": det["bbox"],
            "user_id": user_id,
            "name": stable_name,
            "score": stable_score
        })
    return results


def extract_faces(frame):
    """
    提取当前帧的人脸框 + 归一化 embedding。
    供注册页采集 embedding 使用，内部自动兼容 unified/split/yolov5face_dlib。
    返回:
    [
      {"bbox": np.array([x1,y1,x2,y2]), "embedding": np.ndarray(float32)}
    ]
    """
    if ENGINE_MODE == "split":
        init_split_engine()
        bboxes, kpss = _detector.detect(frame, max_num=0, metric="default")
        if bboxes is None or len(bboxes) == 0:
            return []

        faces = []
        for i in range(len(bboxes)):
            x1, y1, x2, y2 = bboxes[i, :4].astype(int)
            kps = kpss[i]
            face_chip = face_align.norm_crop(frame, landmark=kps)
            emb = _recognizer.get_feat(face_chip).flatten()
            emb = _normalize_embedding(emb)
            faces.append({
                "bbox": np.array([x1, y1, x2, y2]),
                "embedding": emb,
                "kps": kps,
                "det_score": float(bboxes[i, 4]) if bboxes.shape[1] > 4 else 0.0
            })
        return faces

    if ENGINE_MODE == "yolov5face_dlib":
        dets = _decode_yolov5face(frame)
        faces = []
        for det in dets:
            emb, shape_kps = _dlib_embedding_from_bbox(frame, det["bbox"])
            faces.append({
                "bbox": det["bbox"],
                "embedding": emb,
                "kps": det["kps"] if det["kps"] is not None else shape_kps,
                "det_score": det["det_score"],
            })
        return faces

    init_unified_engine()
    raw_faces = _unified_app.get(frame)
    faces = []
    for face in raw_faces:
        faces.append({
            "bbox": face.bbox.astype(int),
            "embedding": _normalize_embedding(face.embedding),
            "kps": getattr(face, "kps", None),
            "det_score": float(getattr(face, "det_score", 0.0))
        })
    return faces


class _LegacyAppProxy:
    """
    兼容旧代码: app.get(frame) -> 返回含 bbox/embedding 属性的对象列表。
    """

    def get(self, frame):
        faces = extract_faces(frame)
        return [
            SimpleNamespace(bbox=f["bbox"], embedding=f["embedding"])
            for f in faces
        ]


# 兼容旧导入方式：from CORE.face_engine import app
app = _LegacyAppProxy()


# =========================
# 检测 + 识别封装（推荐调用）
# =========================
def detect_and_recognize(frame, db_features):
    """
    统一入口：
    - ENGINE_MODE = "unified" -> 一体化版本
    - ENGINE_MODE = "split" -> 两段式版本
    - ENGINE_MODE = "yolov5face_dlib" -> YOLOv5-Face + dlib 版本
    """
    if ENGINE_MODE == "split":
        return detect_and_recognize_split(frame, db_features)
    if ENGINE_MODE == "yolov5face_dlib":
        return detect_and_recognize_yolov5face_dlib(frame, db_features)
    return detect_and_recognize_unified(frame, db_features)
