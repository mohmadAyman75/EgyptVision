from pydantic import BaseModel
from typing import List, Optional

class DetectedObject(BaseModel):
    track_id: int
    label: str
    direction: str
    distance: str
    approaching: bool

class DetectedFace(BaseModel):
    name: str
    distance_meters: str

class SceneContext(BaseModel):
    on_road: bool           # الشخص على الشارع؟
    on_sidewalk: bool       # على الرصيف؟
    crossing_ahead: bool    # في ممر مشاة قدامه؟
    stairs_ahead: bool      # في سلم؟
    obstacle_low: bool      # في عقبة على الأرض؟
    surface: str            # "شارع", "رصيف", "داخل مبنى", "غير معروف"

class AnalysisResult(BaseModel):
    objects: List[DetectedObject]
    faces: List[DetectedFace]
    scene: Optional[SceneContext]
    scene_description: str
    priority: str
    processing_time_ms: float