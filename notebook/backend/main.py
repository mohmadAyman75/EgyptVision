import cv2
import time
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

from schemas import AnalysisResult
from detector import detect_objects
from face_recognition import recognize_faces, register_face

app = FastAPI(title="Egypt Vision AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def determine_priority(objects, faces) -> str:
    """تحديد أولوية الرسالة الصوتية"""
    if any(o.distance == "قريب جداً" and o.approaching for o in objects):
        return "high"
    if any(o.distance == "قريب جداً" for o in objects):
        return "high"
    if faces:
        return "high"
    if any(o.distance == "قريب" for o in objects):
        return "medium"
    return "low"

def build_scene_description(objects, faces) -> str:
    """بناء جملة عربية وصفية للمشهد"""
    parts = []

    # الوجوه أول — أهم حاجة
    for face in faces:
        parts.append(f"قدامك {face.name} على بعد {face.distance_meters}")

    # الأجسام المقتربة
    approaching = [o for o in objects if o.approaching]
    for obj in approaching[:2]:  # أهم اتنين بس
        parts.append(f"تحذير: {obj.label} {obj.direction} بيقرب")

    # باقي الأجسام القريبة
    close = [o for o in objects if not o.approaching and o.distance in ("قريب جداً", "قريب")]
    for obj in close[:3]:
        parts.append(f"في {obj.label} {obj.direction}")

    if not parts:
        return "الطريق واضح"

    return "، ".join(parts)

@app.post("/analyze", response_model=AnalysisResult)
async def analyze_frame(file: UploadFile = File(...)):
    """استقبال frame وإرجاع التحليل كامل"""
    start = time.time()

    # قراءة الصورة
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return AnalysisResult(
            objects=[], faces=[],
            scene_description="خطأ في قراءة الصورة",
            priority="low",
            processing_time_ms=0
        )

    # تشغيل الـ models
    objects = detect_objects(frame)
    faces = recognize_faces(frame)

    elapsed = (time.time() - start) * 1000

    return AnalysisResult(
        objects=objects,
        faces=faces,
        scene_description=build_scene_description(objects, faces),
        priority=determine_priority(objects, faces),
        processing_time_ms=round(elapsed, 2)
    )

@app.post("/register-face")
async def register_face_endpoint(
    name: str = Form(...),
    file: UploadFile = File(...)
):
    """تسجيل وجه جديد"""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    success = register_face(name, image)
    if success:
        return {"status": "ok", "message": f"تم تسجيل {name} بنجاح"}
    return {"status": "error", "message": "مش لاقي وجه في الصور"}

@app.get("/health")
def health():
    return {"status": "ok", "message": "server is running"}