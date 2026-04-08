import cv2
import numpy as np
from ultralytics import YOLO
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image
import torch
from schemas import DetectedObject, SceneContext

# ── YOLOv8 ──────────────────────────────────────────────
model = YOLO("yolov8l.pt")

# ── Segformer (ADE20K - 150 class) ───────────────────────
# هيتحمل أول مرة تلقائي (~230MB)
_seg_processor = SegformerImageProcessor.from_pretrained(
    "nvidia/segformer-b2-finetuned-ade-512-512"
)
_seg_model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b2-finetuned-ade-512-512"
)
_seg_model.eval()

# ADE20K class IDs اللي بيهمنا
ADE20K_ROAD      = 6    # road
ADE20K_SIDEWALK  = 11   # sidewalk / pavement
ADE20K_STAIRS    = 53   # stairs / staircase
ADE20K_FLOOR     = 3    # floor (داخل مبنى)
ADE20K_CROSSING  = 136  # crosswalk

LABEL_MAP = {
    "person": "شخص", "bicycle": "دراجة هوائية", "car": "سيارة",
    "motorcycle": "دراجة نارية", "airplane": "طيارة", "bus": "أتوبيس",
    "train": "قطر", "truck": "شاحنة", "boat": "قارب",
    "traffic light": "إشارة مرور", "fire hydrant": "صنبور حريق",
    "stop sign": "إشارة قف", "parking meter": "عداد موقف", "bench": "بنش",
    "bird": "طائر", "cat": "قطة", "dog": "كلب", "horse": "حصان",
    "sheep": "خروف", "cow": "بقرة", "elephant": "فيل", "bear": "دب",
    "zebra": "حمار وحشي", "giraffe": "زرافة", "backpack": "شنطة ظهر",
    "umbrella": "شمسية", "handbag": "شنطة يد", "tie": "كرافتة",
    "suitcase": "شنطة سفر", "frisbee": "فريسبي", "skis": "تزلج على الجليد",
    "snowboard": "لوح تزلج", "sports ball": "كرة رياضية", "kite": "طائرة ورقية",
    "baseball bat": "مضرب بيسبول", "baseball glove": "قفازة بيسبول",
    "skateboard": "لوح تزلج أرضي", "surfboard": "لوح تزلج مائي",
    "tennis racket": "مضرب تنس", "bottle": "زجاجة", "wine glass": "كأس نبيذ",
    "cup": "كوباية", "fork": "شوكة", "knife": "سكينة", "spoon": "معلقة",
    "bowl": "طبق", "banana": "موزة", "apple": "تفاحة", "sandwich": "سندوتش",
    "orange": "برتقالة", "broccoli": "بروكلي", "carrot": "جزرة",
    "hot dog": "هوت دوج", "pizza": "بيتزا", "donut": "دونات", "cake": "كيكة",
    "chair": "كرسي", "couch": "كنبة", "potted plant": "نبتة في أصيص",
    "bed": "سرير", "dining table": "ترابيزة أكل", "toilet": "حمام",
    "tv": "تلفزيون", "laptop": "لاب توب", "mouse": "ماوس", "remote": "ريموت",
    "keyboard": "كيبورد", "cell phone": "موبايل", "microwave": "ميكرويف",
    "oven": "فرن", "toaster": "توستر", "sink": "حوض", "refrigerator": "تلاجة",
    "book": "كتاب", "clock": "ساعة حيطة", "vase": "فازة", "scissors": "مقص",
    "teddy bear": "دبدوب", "hair drier": "مجفف شعر", "toothbrush": "فرشة أسنان",
}

DISTANCE_ORDER = ["ملاصق", "قريب جداً", "قريب", "متوسط", "بعيد", "بعيد جداً"]

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
    if ratio > 0.75:
        return "ملاصق"
    elif ratio > 0.55:
        return "قريب جداً"
    elif ratio > 0.38:
        return "قريب"
    elif ratio > 0.22:
        return "متوسط"
    elif ratio > 0.10:
        return "بعيد"
    return "بعيد جداً"


def is_approaching(track_id: int, current_height: float) -> bool:
    prev = _prev_heights.get(track_id)
    _prev_heights[track_id] = current_height
    if prev is None:
        return False
    return current_height > prev * 1.03


def detect_objects(frame: np.ndarray) -> list[DetectedObject]:
    results = model.track(
        frame,
        tracker="bytetrack.yaml",
        persist=True,
        verbose=False,
        conf=0.45,
        iou=0.5,
    )[0]

    detected = []
    h, w = frame.shape[:2]

    if results.boxes.id is None:
        return []

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

    detected.sort(key=lambda o: (
        not o.approaching,
        DISTANCE_ORDER.index(o.distance)
    ))

    return detected


def analyze_scene(frame: np.ndarray) -> SceneContext:
    """
    بيحلل البيئة الكاملة باستخدام Segformer
    بيشوف: شارع، رصيف، سلم، ممر مشاة، داخل مبنى
    """
    h, w = frame.shape[:2]

    # تحويل الـ frame لـ PIL عشان Segformer
    image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    inputs = _seg_processor(images=image_pil, return_tensors="pt")

    with torch.no_grad():
        outputs = _seg_model(**inputs)

    # upscale الـ segmentation map لنفس حجم الـ frame
    logits = outputs.logits  # (1, num_classes, H/4, W/4)
    seg_map = torch.argmax(logits, dim=1).squeeze().numpy()
    seg_map = cv2.resize(
        seg_map.astype(np.uint8),
        (w, h),
        interpolation=cv2.INTER_NEAREST
    )

    # ── تحليل المنطقة الأمامية (الثلث السفلي من الصورة) ──
    # الكاميرا في النظارة بتبص للأمام، فالثلث السفلي هو اللي قدام المستخدم
    front_zone = seg_map[h // 2:, :]          # نص الصورة للأسفل
    center_zone = seg_map[h // 2:, w//4: 3*w//4]  # المنتصف بالتحديد

    def zone_has(zone, class_id, threshold=0.05) -> bool:
        """بيشوف لو نسبة معينة من المنطقة فيها الـ class ده"""
        return np.mean(zone == class_id) > threshold

    on_road     = zone_has(front_zone, ADE20K_ROAD)
    on_sidewalk = zone_has(front_zone, ADE20K_SIDEWALK)
    stairs      = zone_has(center_zone, ADE20K_STAIRS)
    crossing    = zone_has(front_zone, ADE20K_CROSSING)
    indoors     = zone_has(front_zone, ADE20K_FLOOR, threshold=0.15)

    # تحديد السطح
    if indoors:
        surface = "داخل مبنى"
    elif on_sidewalk:
        surface = "رصيف"
    elif on_road:
        surface = "شارع"
    else:
        surface = "غير معروف"

    return SceneContext(
        on_road        = on_road,
        on_sidewalk    = on_sidewalk,
        crossing_ahead = crossing,
        stairs_ahead   = stairs,
        obstacle_low   = False,   # ممكن تضيفه لاحقاً بـ depth model
        surface        = surface,
    )