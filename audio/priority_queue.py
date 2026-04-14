"""
priority_queue.py — Egypt Vision AI
====================================
مسؤول عن:
- تحديد إمتى ننطق (مش كل response)
- ترتيب الأولويات (خطر > تحذير > وصف)
- منع التكرار باستخدام track_id
- cooldown per object
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional


# ─────────────────────────────────────────
# Priority Levels
# ─────────────────────────────────────────

class Priority(IntEnum):
    HIGH   = 1   # خطر فوري — يقاطع أي كلام
    MEDIUM = 2   # تحذير — ينطق لو مفيش HIGH شغال
    LOW    = 3   # وصف عام — ينطق لو الـ queue فاضية


# ─────────────────────────────────────────
# Message dataclass
# ─────────────────────────────────────────

@dataclass(order=True)
class AudioMessage:
    priority: Priority
    text: str          = field(compare=False)
    track_id: Optional[int] = field(default=None, compare=False)
    timestamp: float   = field(default_factory=time.time, compare=False)

    # رسالة تنتهي صلاحيتها بعد كذا ثانية لو لسه في الـ queue
    ttl_seconds: float = field(default=5.0, compare=False)

    def is_expired(self) -> bool:
        return (time.time() - self.timestamp) > self.ttl_seconds


# ─────────────────────────────────────────
# State Tracker — قلب المنطق كله
# ─────────────────────────────────────────

class StateTracker:
    """
    يتذكر آخر حالة لكل object عن طريق track_id.
    يقرر: هل في حاجة جديدة تستاهل ننطقها؟
    """

    COOLDOWN = {
        Priority.HIGH:   2.0,
        Priority.MEDIUM: 4.0,
        Priority.LOW:    10.0,
    }

    def __init__(self):
        self._object_states: dict[int, dict] = {}
        self._cooldowns: dict[tuple, float] = {}
        self._last_scene: str = ""
        self._last_scene_time: float = 0.0
        self._seen_faces: set[str] = set()

    def should_speak_object(self, obj: dict) -> tuple[bool, Priority]:
        track_id = obj["track_id"]
        prev = self._object_states.get(track_id)

        if prev is None:
            self._object_states[track_id] = {
                "distance":  obj["distance"],
                "direction": obj["direction"],
                "approaching": obj["approaching"],
            }
            priority = Priority.HIGH if obj["approaching"] else Priority.MEDIUM
            return self._check_cooldown(track_id, priority), priority

        if obj["approaching"] and obj["distance"] == "قريب جداً":
            changed = (
                prev["distance"]   != obj["distance"] or
                prev["approaching"] != obj["approaching"]
            )
            self._object_states[track_id] = {
                "distance":  obj["distance"],
                "direction": obj["direction"],
                "approaching": obj["approaching"],
            }
            if changed:
                return self._check_cooldown(track_id, Priority.HIGH), Priority.HIGH

        distance_changed  = prev["distance"]  != obj["distance"]
        direction_changed = prev["direction"] != obj["direction"]

        if distance_changed or direction_changed:
            self._object_states[track_id] = {
                "distance":  obj["distance"],
                "direction": obj["direction"],
                "approaching": obj["approaching"],
            }
            priority = Priority.HIGH if obj["approaching"] else Priority.MEDIUM
            return self._check_cooldown(track_id, priority), priority

        return False, Priority.LOW

    def should_speak_face(self, name: str) -> bool:
        if name not in self._seen_faces:
            self._seen_faces.add(name)
            return True
        return False

    def should_speak_scene(self, description: str) -> bool:
        now = time.time()
        if (description != self._last_scene and
                now - self._last_scene_time > self.COOLDOWN[Priority.LOW]):
            self._last_scene = description
            self._last_scene_time = now
            return True
        return False

    def _check_cooldown(self, track_id: int, priority: Priority) -> bool:
        key = (track_id, priority)
        now = time.time()
        last = self._cooldowns.get(key, 0.0)
        if now - last >= self.COOLDOWN[priority]:
            self._cooldowns[key] = now
            return True
        return False

    def cleanup_stale(self, active_ids: set[int]):
        stale = set(self._object_states.keys()) - active_ids
        for track_id in stale:
            self._object_states.pop(track_id, None)
            for priority in Priority:
                self._cooldowns.pop((track_id, priority), None)


# ─────────────────────────────────────────
# Priority Queue
# ─────────────────────────────────────────

class AudioQueue:
    def __init__(self):
        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._current_priority: Optional[Priority] = None
        self._interrupt_event = asyncio.Event()

    async def put(self, message: AudioMessage):
        if message.priority == Priority.LOW and not self._queue.empty():
            return

        if message.priority == Priority.HIGH:
            self._interrupt_event.set()

        await self._queue.put(message)

    async def get(self) -> Optional[AudioMessage]:
        while True:
            message = await self._queue.get()
            if not message.is_expired():
                self._current_priority = message.priority
                self._interrupt_event.clear()
                return message

    @property
    def should_interrupt(self) -> bool:
        return self._interrupt_event.is_set()

    def qsize(self) -> int:
        return self._queue.qsize()

    def clear(self):
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        self._interrupt_event.clear()
        self._current_priority = None


# ─────────────────────────────────────────
# Text Generator
# ─────────────────────────────────────────

class ArabicTextGenerator:

    DISTANCE_URGENT = {"قريب جداً", "قريب"}

    def generate_object_message(self, obj: dict) -> str:
        label     = obj["label"]
        direction = obj["direction"]
        distance  = obj["distance"]
        approach  = obj["approaching"]

        if approach and distance == "قريب جداً":
            return f"تحذير! {label} قادم من {direction}"

        if approach and distance == "قريب":
            return f"انتبه، {label} يقترب من {direction}"

        if distance in self.DISTANCE_URGENT:
            return f"{label} على {direction}، {distance}"

        return f"في {label} على {direction}"

    def generate_face_message(self, name: str, distance_meters: str) -> str:
        return f"أمامك {name} على بعد {distance_meters}"

    def generate_scene_message(self, description: str) -> str:
        return description


# ─────────────────────────────────────────
# Audio Processor
# ─────────────────────────────────────────

class AudioProcessor:

    def __init__(self):
        self.state_tracker = StateTracker()
        self.queue         = AudioQueue()
        self.text_gen      = ArabicTextGenerator()

    async def process(self, result: dict):

        active_ids = {obj["track_id"] for obj in result.get("objects", [])}
        self.state_tracker.cleanup_stale(active_ids)

        for obj in result.get("objects", []):
            should, priority = self.state_tracker.should_speak_object(obj)
            if should:
                text = self.text_gen.generate_object_message(obj)
                await self.queue.put(AudioMessage(
                    priority=priority,
                    text=text,
                    track_id=obj["track_id"],
                ))

        for face in result.get("faces", []):
            if self.state_tracker.should_speak_face(face["name"]):
                text = self.text_gen.generate_face_message(
                    face["name"], face["distance_meters"]
                )
                await self.queue.put(AudioMessage(
                    priority=Priority.MEDIUM,
                    text=text,
                ))

        scene = result.get("scene_description", "")
        if scene and result.get("priority") != "high":
            if self.state_tracker.should_speak_scene(scene):
                text = self.text_gen.generate_scene_message(scene)
                await self.queue.put(AudioMessage(
                    priority=Priority.LOW,
                    text=text,
                    ttl_seconds=8.0,
                ))

    def get_status(self) -> dict:
        return {
            "queue_length": self.queue.qsize(),
            "interrupt_pending": self.queue.should_interrupt,
        }