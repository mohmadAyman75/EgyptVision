import cv2
import time
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

from schemas import AnalysisResult
from detector import detect_objects, analyze_scene
from face_recognition import recognize_faces, register_face

app = FastAPI(title="Egypt Vision AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def determine_priority(objects, faces, scene) -> str:
    # خطر فوري — على الشارع أو في سلم
    if scene and (scene.on_road or scene.stairs_ahead):
        return "high"
    if any(o.distance in ("ملاصق", "قريب جداً") and o.approaching for o in objects):
        return "high"
    if any(o.distance in ("ملاصق", "قريب جداً") for o in objects):
        return "high"
    if faces:
        return "high"
    if any(o.distance == "قريب" for o in objects):
        return "medium"
    if scene and scene.crossing_ahead:
        return "medium"
    return "low"


def build_scene_description(objects, faces, scene) -> str:
    parts = []

    # ── البيئة أول — أهم حاجة للمكفوف ──
    if scene:
        if scene.stairs_ahead:
            parts.append("تحذير: في سلم قدامك")
        if scene.on_road:
            parts.append("تحذير: أنت على الشارع، ابعد عن الطريق")
        if scene.crossing_ahead:
            parts.append("في ممر مشاة قدامك")
        if scene.surface != "غير معروف" and not scene.on_road and not scene.stairs_ahead:
            parts.append(f"أنت على {scene.surface}")

    # ── الوجوه ──
    for face in faces:
        parts.append(f"قدامك {face.name} على بعد {face.distance_meters}")

    # ── الأجسام المقتربة ──
    approaching = [o for o in objects if o.approaching]
    for obj in approaching[:2]:
        parts.append(f"تحذير: {obj.label} {obj.direction} بيقرب")

    # ── الأجسام القريبة ──
    close = [o for o in objects if not o.approaching and o.distance in ("ملاصق", "قريب جداً", "قريب")]
    for obj in close[:3]:
        parts.append(f"في {obj.label} {obj.direction}")

    return "، ".join(parts) if parts else "الطريق واضح"


@app.post("/analyze", response_model=AnalysisResult)
async def analyze_frame(file: UploadFile = File(...)):
    start = time.time()

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return AnalysisResult(
            objects=[], faces=[], scene=None,
            scene_description="خطأ في قراءة الصورة",
            priority="low", processing_time_ms=0
        )

    objects = detect_objects(frame)
    faces   = recognize_faces(frame)
    scene   = analyze_scene(frame)

    elapsed = (time.time() - start) * 1000

    return AnalysisResult(
        objects=objects,
        faces=faces,
        scene=scene,
        scene_description=build_scene_description(objects, faces, scene),
        priority=determine_priority(objects, faces, scene),
        processing_time_ms=round(elapsed, 2)
    )


@app.post("/register-face")
async def register_face_endpoint(
    name: str = Form(...),
    file: UploadFile = File(...)
):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    success = register_face(name, image)
    if success:
        return {"status": "ok", "message": f"تم تسجيل {name} بنجاح"}
    return {"status": "error", "message": "مش لاقي وجه في الصورة"}


@app.get("/health")
def health():
    return {"status": "ok", "message": "server is running"}