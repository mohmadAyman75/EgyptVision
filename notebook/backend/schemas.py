from pydantic import BaseModel
from typing import List

class DetectedObject(BaseModel):
    track_id: int        # رقم ثابت للـ object طول ما هو في الكاميرا
    label: str           # "سيارة", "شخص", "دراجة"
    direction: str       # "يمين", "يسار", "أمام"
    distance: str        # "قريب جداً", "قريب", "متوسط", "بعيد"
    approaching: bool    # بيقرب ولا بيبعد؟

class DetectedFace(BaseModel):
    name: str
    distance_meters: str

class AnalysisResult(BaseModel):
    objects: List[DetectedObject]
    faces: List[DetectedFace]
    scene_description: str
    priority: str              # "high", "medium", "low"
    processing_time_ms: float