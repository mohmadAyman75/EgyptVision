"""
audio_module.py — Egypt Vision AI
====================================
الملف الرئيسي للـ Audio Layer.

مسؤول عن:
- استقبال AnalysisResult من الـ backend
- تمريره لـ AudioProcessor (priority_queue.py)
- تشغيل الـ consumer loop اللي يجيب من الـ queue وينطق
- FastAPI endpoints للـ frontend
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from priority_queue import AudioProcessor, AudioMessage, Priority
from tts_engine import TTSEngine

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────
# Schemas (مطابقة لـ schemas.py في الـ backend)
# ─────────────────────────────────────────

class DetectedObject(BaseModel):
    track_id:   int
    label:      str
    direction:  str
    distance:   str
    approaching: bool

class DetectedFace(BaseModel):
    name:             str
    distance_meters:  str

class AnalysisResult(BaseModel):
    objects:            list[DetectedObject]
    faces:              list[DetectedFace]
    scene_description:  str
    priority:           str   # "high" | "medium" | "low"
    processing_time_ms: float


# ─────────────────────────────────────────
# AudioModule — القلب بتاع الـ Layer ده
# ─────────────────────────────────────────

class AudioModule:
    """
    يربط الـ AudioProcessor بالـ TTSEngine.

    Flow:
        result (AnalysisResult)
            ↓
        AudioProcessor.process()   ← يقرر إيه ينطق
            ↓
        AudioQueue.get()           ← يجيب الـ message
            ↓
        TTSEngine.speak()          ← ينطق
    """

    def __init__(self):
        self.processor  = AudioProcessor()
        self.tts        = TTSEngine()
        self._consumer_task: Optional[asyncio.Task] = None
        self._running   = False

        # Stats
        self._total_spoken   = 0
        self._total_received = 0
        self._start_time     = time.time()

    # ─── Lifecycle ────────────────────────

    async def start(self):
        """يبدأ الـ module: يحمل TTS ويشغل consumer loop"""
        logger.info("AudioModule starting...")
        await self.tts.initialize()

        self._running = True
        self._consumer_task = asyncio.create_task(
            self._consumer_loop(),
            name="audio-consumer"
        )
        logger.info("AudioModule started ✓")

    async def stop(self):
        """يوقف كل حاجة بنظافة"""
        logger.info("AudioModule stopping...")
        self._running = False

        await self.tts.stop()

        if self._consumer_task:
            self._consumer_task.cancel()
            try:
                await self._consumer_task
            except asyncio.CancelledError:
                pass

        logger.info("AudioModule stopped ✓")

    # ─── Main Entry Point ─────────────────

    async def handle_result(self, result: AnalysisResult):
        """
        استقبل نتيجة من الـ backend وحطها في الـ pipeline.
        ده اللي بينادي عليه الـ FastAPI endpoint.
        """
        self._total_received += 1

        result_dict = {
            "objects": [obj.model_dump() for obj in result.objects],
            "faces":   [f.model_dump() for f in result.faces],
            "scene_description": result.scene_description,
            "priority": result.priority,
            "processing_time_ms": result.processing_time_ms,
        }

        await self.processor.process(result_dict)

        # لو HIGH priority → interrupt فوري
        if result.priority == "high":
         await self.tts.stop()  # وقف فوراً بغض النظر عن is_speaking

    # ─── Consumer Loop ────────────────────

    async def _consumer_loop(self):
        """
        Loop دايم يجيب messages من الـ queue وينطقها.
        ده بيشتغل في background طول ما الـ module شغال.
        """
        logger.info("Consumer loop started")

        while self._running:
            try:
                # استنى message من الـ queue
                message: AudioMessage = await asyncio.wait_for(
                    self.processor.queue.get(),
                    timeout=1.0  # نفحص الـ _running flag كل ثانية
                )

                if message is None:
                    continue

                # لو جه interrupt في النص → وقف وخلي الـ HIGH ييجي
                if self.processor.queue.should_interrupt and message.priority != Priority.HIGH:
                    logger.debug(f"Skipping {message.priority.name} — interrupt pending")
                    continue

                logger.info(
                    f"Speaking [{message.priority.name}]: {message.text}"
                )

                success = await self.tts.speak(message.text)
                if success:
                    self._total_spoken += 1

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Consumer loop error: {e}")
                await asyncio.sleep(0.5)

        logger.info("Consumer loop stopped")

    # ─── Status ───────────────────────────

    def get_status(self) -> dict:
        uptime = time.time() - self._start_time
        return {
            "running":          self._running,
            "is_speaking":      self.tts.is_speaking,
            "tts_ready":        self.tts.is_ready,
            "queue_length":     self.processor.queue.qsize(),
            "interrupt_pending": self.processor.queue.should_interrupt,
            "total_received":   self._total_received,
            "total_spoken":     self._total_spoken,
            "uptime_seconds":   round(uptime, 1),
        }


audio_module = AudioModule()

@asynccontextmanager
async def lifespan(app: FastAPI):
    await audio_module.start()
    yield
    await audio_module.stop()

app = FastAPI(
    title="Egypt Vision — Audio Module",
    description="Audio layer لمساعدة ضعاف البصر في الشوارع المصرية",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/audio/process")
async def process_result(result: AnalysisResult):
    try:
        await audio_module.handle_result(result)
        status = audio_module.processor.get_status()
        return {
            "status": "queued",
            "queue_length": status["queue_length"],
            "interrupt_pending": status["interrupt_pending"],
        }
    except Exception as e:
        logger.error(f"/audio/process error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/audio/status")
async def get_status():
    return audio_module.get_status()

@app.post("/audio/stop")
async def stop_audio():
    await audio_module.tts.stop()
    audio_module.processor.queue.clear()
    return {"status": "stopped"}

@app.get("/health")
async def health():
    return {"status": "ok", "module": "audio"}

if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    uvicorn.run("audio_module:app", host="0.0.0.0", port=8001)