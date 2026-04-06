import cv2
import numpy as np
from ultralytics import YOLO
from schemas import DetectedObject

model = YOLO("yolov8l.pt")  # هيتحمل أوتوماتيك أول مرة (~87MB)

LABEL_MAP = {
    "person":        "شخص",
    "car":           "سيارة",
    "truck":         "شاحنة",
    "motorcycle":    "دراجة نارية",
    "bus":           "أتوبيس",
    "bicycle":       "عجلة",
    "traffic light": "إشارة مرور",
    "stop sign":     "علامة وقوف",
    "dog":           "كلب",
    "chair":         "كرسي",
    "bench":         "بنش",
}

# ذاكرة الـ tracking — بنحفظ آخر حجم لكل object
_prev_heights: dict[int, float] = {}

def get_direction(x_center: float, frame_width: int) -> str:
    third = frame_width / 3
    if x_center < third:
        return "على اليسار"
    elif x_center > 2 * third:
        return "على اليمين"
    return "أمامك"

def get_distance(box_height: float, frame_height: int) -> str:
    ratio = box_height / frame_height
    if ratio > 0.55:
        return "قريب جداً"
    elif ratio > 0.30:
        return "قريب"
    elif ratio > 0.12:
        return "متوسط"
    return "بعيد"

def is_approaching(track_id: int, current_height: float) -> bool:
    prev = _prev_heights.get(track_id)
    _prev_heights[track_id] = current_height
    if prev is None:
        return False
    return current_height > prev * 1.03  # بيكبر بأكتر من 3% = بيقرب

def detect_objects(frame: np.ndarray) -> list[DetectedObject]:
    # track=True هو اللي بيفعّل الـ tracking
    # tracker="bytetrack.yaml" أفضل tracker في YOLOv8
    results = model.track(
        frame,
        tracker="bytetrack.yaml",
        persist=True,       # مهم: يحتفظ بالـ IDs بين الـ frames
        verbose=False,
        conf=0.45,          # تجاهل أي detection تحت 45% confidence
        iou=0.5,
    )[0]

    detected = []
    h, w = frame.shape[:2]

    if results.boxes.id is None:
        return []  # مفيش tracking IDs لو الـ frame فاضية

    for box, track_id in zip(results.boxes, results.boxes.id.int().tolist()):
        cls_id   = int(box.cls[0])
        label_en = model.names[cls_id]
        label_ar = LABEL_MAP.get(label_en, label_en)
        conf     = float(box.conf[0])

        if conf < 0.45:
            continue

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x_center   = (x1 + x2) / 2
        box_height = y2 - y1

        detected.append(DetectedObject(
            track_id   = track_id,
            label      = label_ar,
            direction  = get_direction(x_center, w),
            distance   = get_distance(box_height, h),
            approaching= is_approaching(track_id, box_height),
        ))

    # ترتيب حسب الأولوية — الأقرب والمقترب أول
    detected.sort(key=lambda o: (
        not o.approaching,
        ["قريب جداً","قريب","متوسط","بعيد"].index(o.distance)
    ))

    return detected