"""
tts_engine.py — Egypt Vision AI
=================================
مسؤول عن:
- تحويل النص العربي لصوت طبيعي (gTTS)
- تشغيل الصوت على السماعة (أو Bluetooth)
- دعم الـ interrupt الفوري لو جه HIGH priority
- async بالكامل عشان ميبلوكش الـ pipeline
"""

import asyncio
import io
import logging
import time
from pathlib import Path
from typing import Optional
from gtts import gTTS
import tempfile
import os
import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────
# Audio Player — تشغيل الصوت الفعلي
# ─────────────────────────────────────────

class AudioPlayer:
    """
    يشغل numpy array كـ audio على الـ output device.
    بيستخدم sounddevice عشان يدعم Bluetooth و ALSA.
    """

    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
        self._stop_event = asyncio.Event()
        self._playing   = False

        # lazy import — sounddevice مش دايماً متاح في كل بيئة
        try:
            import sounddevice as sd
            self._sd = sd
            logger.info("sounddevice loaded ✓")
        except ImportError:
            self._sd = None
            logger.warning("sounddevice مش متاح — هيشتغل في silent mode")

    @property
    def is_playing(self) -> bool:
        return self._playing

    async def play(self, audio: np.ndarray):
        if self._sd is None:
            duration = len(audio) / self.sample_rate
            await self._interruptible_sleep(duration)
            return

        self._stop_event.clear()
        self._playing = True
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(None, self._blocking_play, audio)
        finally:
            self._playing = False

    def _blocking_play(self, audio: np.ndarray):
        try:
            chunk_size = int(self.sample_rate * 0.1)
            for i in range(0, len(audio), chunk_size):
                if self._stop_event.is_set():
                    break
                chunk = audio[i: i + chunk_size]
                self._sd.play(chunk, samplerate=self.sample_rate, blocking=True)
        except Exception as e:
            logger.error(f"Audio playback error: {e}")

    async def _interruptible_sleep(self, duration: float):
        step = 0.05
        elapsed = 0.0
        while elapsed < duration:
            if self._stop_event.is_set():
                break
            await asyncio.sleep(step)
            elapsed += step

    def stop(self):
        self._stop_event.set()
        self._playing = False
        if self._sd is not None:
            try:
                self._sd.stop()
            except Exception:
                pass


# ─────────────────────────────────────────
# TTS Engine — القلب بتاع الموضوع
# ─────────────────────────────────────────

class TTSEngine:
    """
    gTTS wrapper بالعربي.

    الاستخدام:
        engine = TTSEngine()
        await engine.initialize()
        await engine.speak("تحذير! سيارة قادمة من اليمين")
    """

    def __init__(self, use_gpu: bool = False):
        self.use_gpu     = use_gpu
        self.sample_rate = 22050

        self._player     = AudioPlayer(self.sample_rate)
        self._lock       = asyncio.Lock()
        self._ready      = False
        self._speaking   = False

    async def initialize(self):
        logger.info("Using gTTS engine ✓")
        self._ready = True

    async def speak(self, text: str) -> bool:
        if not text or not text.strip():
            return False

        if not self._ready:
            logger.info(f"[TTS-SILENT] {text}")
            return True

        async with self._lock:
            self._speaking = True
            try:
                audio = await self._synthesize(text)
                if audio is None:
                    return False
                await self._player.play(audio)
                return True
            except asyncio.CancelledError:
                logger.debug("speak() cancelled")
                return False
            except Exception as e:
                logger.error(f"speak() error: {e}")
                return False
            finally:
                self._speaking = False

    async def stop(self):
        self._player.stop()
        self._speaking = False
        logger.debug("TTS stopped (interrupt)")

    async def _synthesize(self, text: str):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._gtts_generate, text)

    def _gtts_generate(self, text: str):
        try:
            tts = gTTS(text=text, lang="ar")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                temp_path = fp.name
            tts.save(temp_path)

            import sounddevice as sd
            import soundfile as sf
            data, samplerate = sf.read(temp_path)
            sd.play(data, samplerate)
            sd.wait()

            os.remove(temp_path)
            return data
        except Exception as e:
            logger.error(f"gTTS error: {e}")
            return None

    @property
    def is_speaking(self) -> bool:
        return self._speaking or self._player.is_playing

    @property
    def is_ready(self) -> bool:
        return self._ready


# ─────────────────────────────────────────
# Quick Test
# ─────────────────────────────────────────

async def _test():
    logging.basicConfig(level=logging.DEBUG)
    engine = TTSEngine()
    await engine.initialize()

    print("Testing: تحذير! سيارة قادمة من اليمين")
    await engine.speak("تحذير! سيارة قادمة من اليمين")

    print("Testing interrupt...")
    task = asyncio.create_task(
        engine.speak("هذا نص طويل جداً للتجربة فقط وليس له معنى حقيقي")
    )
    await asyncio.sleep(0.5)
    await engine.stop()
    await task

    print("Done ✓")


if __name__ == "__main__":
    asyncio.run(_test())