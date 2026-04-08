import cv2
import numpy as np
import pickle
import os
from insightface.app import FaceAnalysis
from schemas import DetectedFace

# تهيئة النموذج
app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))

DB_PATH = "face_db.pkl"

# قاعدة البيانات — dict اسم: list of embeddings
face_db: dict[str, list[np.ndarray]] = {}

def load_db():
    global face_db
    if os.path.exists(DB_PATH):
        with open(DB_PATH, "rb") as f:
            face_db = pickle.load(f)
        print(f"✅ تم تحميل {len(face_db)} شخص من قاعدة البيانات")

def save_db():
    with open(DB_PATH, "wb") as f:
        pickle.dump(face_db, f)

def register_face(name: str, image: np.ndarray) -> bool:
    """تسجيل وجه جديد في قاعدة البيانات"""
    faces = app.get(image)
    if not faces:
        return False
    embedding = faces[0].normed_embedding
    if name not in face_db:
        face_db[name] = []
    face_db[name].append(embedding)
    save_db()
    return True

def recognize_faces(frame: np.ndarray) -> list[DetectedFace]:
    """التعرف على الوجوه في الـ frame"""
    if not face_db:
        return []

    faces = app.get(frame)
    results = []
    h, w = frame.shape[:2]

    for face in faces:
        embedding = face.normed_embedding
        best_name = None
        best_score = 0.0

        for name, embeddings in face_db.items():
            # مقارنة مع كل صور الشخص وناخد الأعلى
            scores = [float(np.dot(embedding, e)) for e in embeddings]
            score = max(scores)
            if score > best_score:
                best_score = score
                best_name = name

        # threshold 0.45 — لو أقل مش متأكد ومش بنقول اسم
        if best_score < 0.45:
            continue

        # تقدير المسافة من حجم الوجه
        box = face.bbox
        face_height = box[3] - box[1]
        ratio = face_height / h

        if ratio > 0.35:
            dist = "متر تقريباً"
        elif ratio > 0.2:
            dist = "مترين تقريباً"
        elif ratio > 0.1:
            dist = "3 متر تقريباً"
        else:
            dist = "بعيد"

        results.append(DetectedFace(
            name=best_name,
            distance_meters=dist
        ))

    return results

# تحميل الـ DB عند import الملف
load_db()