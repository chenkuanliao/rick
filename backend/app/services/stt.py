from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from ..config import Settings


class SpeechToTextService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._model: Any | None = None
        self._load_error: str | None = None

    @property
    def status(self) -> dict[str, Any]:
        return {
            "model": self.settings.stt_model,
            "fallback_model": self.settings.stt_fallback_model,
            "loaded": self._model is not None,
            "available": self._load_error is None,
            "error": self._load_error,
        }

    async def transcribe(self, wav_path: Path) -> str:
        model = await self._get_model()
        return await asyncio.to_thread(self._transcribe_sync, model, wav_path)

    async def warmup(self) -> None:
        await self._get_model()

    async def _get_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from faster_whisper import WhisperModel

            device = self.settings.stt_device
            compute_type = self.settings.stt_compute_type
            kwargs: dict[str, str] = {}
            if device != "auto":
                kwargs["device"] = device
            if compute_type != "auto":
                kwargs["compute_type"] = compute_type
            self._model = await asyncio.to_thread(
                WhisperModel,
                self.settings.stt_model,
                **kwargs,
            )
            self._load_error = None
            return self._model
        except Exception as exc:  # noqa: BLE001
            self._load_error = str(exc)
            raise RuntimeError(
                "Could not load faster-whisper. Install backend requirements and ensure "
                f"the model is available: {self.settings.stt_model}"
            ) from exc

    def _transcribe_sync(self, model: Any, wav_path: Path) -> str:
        segments, _info = model.transcribe(
            str(wav_path),
            beam_size=self.settings.stt_beam_size,
            vad_filter=self.settings.stt_vad_filter,
            language=self.settings.stt_language,
            condition_on_previous_text=False,
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
        return text
