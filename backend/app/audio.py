from __future__ import annotations

import asyncio
import base64
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from uuid import uuid4

from .config import DATA_DIR, Settings


class AudioError(RuntimeError):
    pass


def decode_audio_data_url_or_base64(payload: str) -> bytes:
    if "," in payload and payload.strip().startswith("data:"):
        payload = payload.split(",", 1)[1]
    try:
        return base64.b64decode(payload, validate=False)
    except Exception as exc:  # noqa: BLE001
        raise AudioError("Audio payload was not valid base64.") from exc


async def convert_to_wav_16k_mono(
    audio_bytes: bytes,
    *,
    mime_type: str | None,
    settings: Settings,
) -> Path:
    suffix = _suffix_for_mime(mime_type)
    source = Path(tempfile.NamedTemporaryFile(suffix=suffix, delete=False).name)
    target = DATA_DIR / "tmp" / f"{uuid4().hex}.wav"
    source.write_bytes(audio_bytes)
    ffmpeg = resolve_ffmpeg(settings.ffmpeg_path)
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "wav",
        str(target),
    ]
    try:
        await asyncio.to_thread(
            subprocess.run,
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise AudioError("ffmpeg was not found. Install ffmpeg or set FFMPEG_PATH.") from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode("utf-8", errors="replace").strip()
        raise AudioError(f"ffmpeg could not decode the recording: {detail}") from exc
    finally:
        source.unlink(missing_ok=True)
    return target


def resolve_ffmpeg(configured_path: str) -> str:
    found = shutil.which(configured_path)
    if found:
        return found

    configured = Path(configured_path).expanduser()
    if configured.is_file():
        return str(configured)

    for base in (os.environ.get("CONDA_PREFIX"), sys.prefix):
        if not base:
            continue
        candidate = Path(base) / "bin" / "ffmpeg"
        if candidate.is_file():
            return str(candidate)

    return configured_path


async def normalize_voice_prompt(
    audio_bytes: bytes,
    *,
    mime_type: str | None,
    settings: Settings,
) -> Path:
    wav = await convert_to_wav_16k_mono(audio_bytes, mime_type=mime_type, settings=settings)
    settings.voice_prompt_path.parent.mkdir(parents=True, exist_ok=True)
    settings.voice_prompt_path.write_bytes(wav.read_bytes())
    wav.unlink(missing_ok=True)
    return settings.voice_prompt_path


def _suffix_for_mime(mime_type: str | None) -> str:
    if not mime_type:
        return ".webm"
    if "wav" in mime_type:
        return ".wav"
    if "mp4" in mime_type or "m4a" in mime_type:
        return ".m4a"
    if "mpeg" in mime_type or "mp3" in mime_type:
        return ".mp3"
    if "ogg" in mime_type or "opus" in mime_type:
        return ".ogg"
    return ".webm"
