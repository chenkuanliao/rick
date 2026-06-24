from __future__ import annotations

import asyncio
import re
import tempfile
from pathlib import Path
from typing import Any
from uuid import uuid4

from ..config import DATA_DIR, Settings


class TextToSpeechService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._model: Any | None = None
        self._load_error: str | None = None
        self._lock = asyncio.Lock()

    @property
    def status(self) -> dict[str, Any]:
        return {
            "engine": "Chatterbox Turbo",
            "loaded": self._model is not None,
            "available": self._load_error is None,
            "voice_prompt_saved": self.settings.voice_prompt_path.exists(),
            "error": self._load_error,
        }

    async def synthesize(self, text: str) -> Path:
        model = await self._get_model()
        async with self._lock:
            return await asyncio.to_thread(self._synthesize_sync, model, text)

    async def synthesize_chunk(self, text: str) -> Path:
        model = await self._get_model()
        async with self._lock:
            return await asyncio.to_thread(self._synthesize_sync, model, text, False)

    async def _get_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from chatterbox.tts_turbo import ChatterboxTurboTTS

            device = self._select_device()
            self._model = await asyncio.to_thread(
                ChatterboxTurboTTS.from_pretrained,
                device=device,
            )
            self._load_error = None
            return self._model
        except Exception as exc:  # noqa: BLE001
            self._load_error = str(exc)
            raise RuntimeError(
                "Could not load Chatterbox Turbo TTS. Install Chatterbox/Torch in the active Conda environment."
            ) from exc

    def _select_device(self) -> str:
        if self.settings.tts_device != "auto":
            return self.settings.tts_device
        import torch

        if not torch.cuda.is_available():
            return "cpu"
        try:
            test = torch.ones((8, 8), device="cuda")
            _ = test @ test
            torch.cuda.synchronize()
        except Exception as exc:  # noqa: BLE001
            self._load_error = f"CUDA is visible but failed a Torch smoke test; using CPU. {exc}"
            return "cpu"
        return "cuda"

    def _synthesize_sync(self, model: Any, text: str, split_text: bool = True) -> Path:
        import numpy as np
        import librosa
        import soundfile as sf

        prompt_path = None
        prompt_file: Path | None = None
        try:
            if self.settings.voice_prompt_path.exists():
                audio, sample_rate = librosa.load(
                    self.settings.voice_prompt_path,
                    sr=None,
                    mono=True,
                )
                prompt_file = Path(tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name)
                sf.write(prompt_file, audio.astype("float32"), sample_rate)
                prompt_path = str(prompt_file)

            chunks = split_tts_text(text, self.settings.tts_max_chunk_chars) if split_text else [text.strip()]
            pause = np.zeros(int(model.sr * 0.18), dtype="float32")
            audio_parts = []
            for index, chunk in enumerate(chunks):
                wav = model.generate(
                    text=chunk,
                    audio_prompt_path=prompt_path,
                    temperature=self.settings.tts_temperature,
                    top_p=self.settings.tts_top_p,
                    top_k=self.settings.tts_top_k,
                    repetition_penalty=self.settings.tts_repetition_penalty,
                    norm_loudness=self.settings.tts_norm_loudness,
                )
                audio = wav.squeeze().detach().cpu().numpy().astype("float32")
                audio_parts.append(audio)
                if index < len(chunks) - 1:
                    audio_parts.append(pause)

            output_audio = np.concatenate(audio_parts) if audio_parts else np.zeros(1, dtype="float32")
            output_path = DATA_DIR / "audio" / f"{uuid4().hex}.wav"
            sf.write(output_path, output_audio, model.sr)
            return output_path
        finally:
            if prompt_file is not None:
                prompt_file.unlink(missing_ok=True)


def split_tts_text(text: str, max_chars: int) -> list[str]:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return [""]

    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_split_long_sentence(sentence, max_chars))
            continue
        candidate = f"{current} {sentence}".strip()
        if current and len(candidate) > max_chars:
            chunks.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks or [cleaned[:max_chars]]


def _split_long_sentence(sentence: str, max_chars: int) -> list[str]:
    pieces = [piece.strip() for piece in re.split(r"([,;:])", sentence) if piece.strip()]
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        candidate = f"{current}{piece}" if piece in {",", ";", ":"} else f"{current} {piece}".strip()
        if current and len(candidate) > max_chars:
            chunks.append(current)
            current = piece
        else:
            current = candidate
    if current:
        chunks.append(current)

    final: list[str] = []
    for chunk in chunks:
        if len(chunk) <= max_chars:
            final.append(chunk)
        else:
            words = chunk.split()
            current_words = ""
            for word in words:
                candidate = f"{current_words} {word}".strip()
                if current_words and len(candidate) > max_chars:
                    final.append(current_words)
                    current_words = word
                else:
                    current_words = candidate
            if current_words:
                final.append(current_words)
    return final
