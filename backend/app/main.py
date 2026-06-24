from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from .audio import AudioError, convert_to_wav_16k_mono, decode_audio_data_url_or_base64, normalize_voice_prompt
from .config import FRONTEND_DIST, ensure_data_dirs, get_settings
from .services.llm import ProviderRegistry, cuda_status
from .services.chats import ChatStore
from .services.stt import SpeechToTextService
from .services.tts import TextToSpeechService, split_tts_text


settings = get_settings()
ensure_data_dirs()

app = FastAPI(title=settings.app_title)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def redirect_insecure_public_hosts(request: Request, call_next):
    if settings.public_https_url and request.url.scheme == "http":
        host = request.headers.get("host", "").split(":", 1)[0]
        redirect_hosts = {item.strip() for item in settings.insecure_redirect_hosts.split(",") if item.strip()}
        if host in redirect_hosts:
            target = f"{settings.public_https_url.rstrip('/')}{request.url.path}"
            if request.url.query:
                target = f"{target}?{request.url.query}"
            return RedirectResponse(target, status_code=307)
    return await call_next(request)

stt = SpeechToTextService(settings)
tts = TextToSpeechService(settings)
providers = ProviderRegistry(settings)
chats = ChatStore()

audio_dir = Path(__file__).resolve().parents[2] / "data" / "audio"
app.mount("/audio", StaticFiles(directory=audio_dir), name="audio")


@app.on_event("startup")
async def preload_speech_models() -> None:
    if not settings.preload_speech_models:
        return
    asyncio.create_task(_preload_speech_models())


async def _preload_speech_models() -> None:
    await asyncio.gather(stt.warmup(), tts.warmup(), return_exceptions=True)


class PromptPayload(BaseModel):
    system_prompt: str


class ChatUpdatePayload(BaseModel):
    title: str | None = None
    pinned: bool | None = None


@app.get("/health")
async def health() -> dict[str, Any]:
    provider_status = await providers.status()
    default_status = provider_status["providers"].get(settings.default_provider, {})
    return {
        "ok": True,
        "cuda": await cuda_status(),
        "stt": stt.status,
        "tts": tts.status,
        "llm": provider_status,
        "config_ok": bool(default_status.get("configured")),
    }


@app.get("/api/config")
async def config() -> dict[str, Any]:
    return {
        "default_provider": settings.default_provider,
        "system_prompt": providers.get_system_prompt(),
        "tts_output_prompt": settings.tts_output_prompt,
        "providers": await providers.status(include_models=False),
        "stt": stt.status,
        "tts": tts.status,
    }


@app.get("/api/providers/{provider_key}/models")
async def provider_models(provider_key: str) -> dict[str, Any]:
    return {"models": await providers.provider_models(provider_key)}


@app.post("/api/system-prompt")
async def system_prompt(payload: PromptPayload) -> dict[str, Any]:
    prompt = providers.save_system_prompt(payload.system_prompt)
    return {"saved": True, "system_prompt": prompt}


@app.get("/api/chats")
async def list_chats() -> dict[str, Any]:
    items = chats.list_chats()
    if not items:
        chat = chats.create_chat()
        items = chats.list_chats()
        return {"chats": items, "active_chat": chat}
    return {"chats": items}


@app.post("/api/chats")
async def create_chat() -> dict[str, Any]:
    chat = chats.create_chat()
    return {"chat": chat, "chats": chats.list_chats()}


@app.get("/api/chats/{chat_id}")
async def get_chat(chat_id: str) -> dict[str, Any]:
    chat = chats.get_chat(chat_id)
    if chat["id"] != chat_id:
        raise HTTPException(status_code=404, detail="Chat not found")
    return {"chat": chat}


@app.patch("/api/chats/{chat_id}")
async def update_chat(chat_id: str, payload: ChatUpdatePayload) -> dict[str, Any]:
    chat = chats.update_chat(chat_id, title=payload.title, pinned=payload.pinned)
    return {"chat": chat, "chats": chats.list_chats()}


@app.delete("/api/chats/{chat_id}")
async def delete_chat(chat_id: str) -> dict[str, Any]:
    chats.delete_chat(chat_id)
    return {"deleted": True, "chats": chats.list_chats()}


@app.post("/api/voice-prompt")
async def voice_prompt(file: UploadFile = File(...)) -> dict[str, Any]:
    content = await file.read()
    path = await normalize_voice_prompt(content, mime_type=file.content_type, settings=settings)
    return {"saved": True, "path": str(path), "tts": tts.status}


@app.websocket("/ws/chat")
async def chat_socket(websocket: WebSocket) -> None:
    await websocket.accept()
    active_turn: asyncio.Task[None] | None = None
    partial_transcript: asyncio.Task[None] | None = None
    try:
        while True:
            payload = await websocket.receive_json()
            if payload.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
                continue
            if payload.get("type") == "cancel":
                if active_turn and not active_turn.done():
                    active_turn.cancel()
                    await websocket.send_json({"type": "cancelled"})
                    await websocket.send_json({"type": "status", "phase": "idle"})
                if partial_transcript and not partial_transcript.done():
                    partial_transcript.cancel()
                else:
                    await websocket.send_json({"type": "status", "phase": "idle"})
                continue
            if payload.get("type") == "live_transcript":
                if partial_transcript and not partial_transcript.done():
                    partial_transcript.cancel()
                partial_transcript = asyncio.create_task(_handle_live_transcript(websocket, payload))
                partial_transcript.add_done_callback(_consume_cancelled_turn)
                continue
            if payload.get("type") not in {"turn", "text_turn"}:
                await websocket.send_json({"type": "error", "message": "Unsupported websocket message type."})
                continue
            if active_turn and not active_turn.done():
                active_turn.cancel()
                await websocket.send_json({"type": "cancelled"})
            if partial_transcript and not partial_transcript.done():
                partial_transcript.cancel()
            active_turn = asyncio.create_task(_handle_turn(websocket, payload))
            active_turn.add_done_callback(_consume_cancelled_turn)
    except WebSocketDisconnect:
        if active_turn and not active_turn.done():
            active_turn.cancel()
        if partial_transcript and not partial_transcript.done():
            partial_transcript.cancel()
        return
    except RuntimeError as exc:
        if "Cannot call \"send\" once a close message has been sent" in str(exc):
            return
        raise


def _consume_cancelled_turn(task: asyncio.Task[None]) -> None:
    try:
        task.result()
    except asyncio.CancelledError:
        return
    except Exception:
        return


async def _handle_turn(websocket: WebSocket, payload: dict[str, Any]) -> None:
    wav_path: Path | None = None
    try:
        chat = chats.get_chat(payload.get("chatId"))
        chat_id = chat["id"]
        await websocket.send_json({"type": "chat", "chat": chat})
        if payload.get("type") == "text_turn":
            user_text = str(payload.get("text", "")).strip()
            if not user_text:
                await websocket.send_json({"type": "error", "message": "Type a message before sending."})
                return
        else:
            await websocket.send_json({"type": "status", "phase": "transcribing"})
            audio_bytes = decode_audio_data_url_or_base64(payload.get("audio", ""))
            wav_path = await convert_to_wav_16k_mono(
                audio_bytes,
                mime_type=payload.get("mimeType"),
                settings=settings,
            )
            user_text = await stt.transcribe(wav_path)
            wav_path.unlink(missing_ok=True)
            if not user_text:
                await websocket.send_json({"type": "error", "message": "No speech was detected in that recording."})
                return
        await _handle_user_text(websocket, payload, chat_id, user_text)
    except asyncio.CancelledError:
        if wav_path is not None:
            wav_path.unlink(missing_ok=True)
        raise
    except AudioError as exc:
        await websocket.send_json({"type": "error", "message": str(exc)})
    except Exception as exc:  # noqa: BLE001
        await websocket.send_json({"type": "error", "message": str(exc)})


async def _handle_live_transcript(websocket: WebSocket, payload: dict[str, Any]) -> None:
    wav_path: Path | None = None
    try:
        audio_bytes = decode_audio_data_url_or_base64(payload.get("audio", ""))
        wav_path = await convert_to_wav_16k_mono(
            audio_bytes,
            mime_type=payload.get("mimeType"),
            settings=settings,
        )
        user_text = await stt.transcribe(wav_path)
        if user_text:
            await websocket.send_json(
                {
                    "type": "partial_transcript",
                    "role": "user",
                    "text": user_text,
                    "sequence": payload.get("sequence"),
                }
            )
    except asyncio.CancelledError:
        raise
    except Exception:
        return
    finally:
        if wav_path is not None:
            wav_path.unlink(missing_ok=True)


async def _handle_user_text(websocket: WebSocket, payload: dict[str, Any], chat_id: str, user_text: str) -> None:
    chat = chats.append_message(chat_id, role="user", text=user_text)
    await websocket.send_json({"type": "transcript", "role": "user", "text": user_text})
    await websocket.send_json({"type": "chat", "chat": chat, "chats": chats.list_chats()})

    await websocket.send_json({"type": "status", "phase": "thinking"})
    stream, provider_key, model = await providers.stream_chat(
        payload.get("provider"),
        payload.get("model"),
        chats.llm_messages(chat_id),
        user_text,
    )
    assistant_id = uuid4().hex
    audio_queue: asyncio.Queue[str | None] = asyncio.Queue()
    audio_task = asyncio.create_task(_stream_tts_audio(websocket, audio_queue, provider_key, model))
    response_parts: list[str] = []
    tts_chunker = StreamingTtsChunker()
    started = False
    try:
        async for delta in stream:
            if not started:
                started = True
                await websocket.send_json(
                    {
                        "type": "assistant_start",
                        "id": assistant_id,
                        "provider": provider_key,
                        "model": model,
                    }
                )
            response_parts.append(delta)
            await websocket.send_json({"type": "assistant_delta", "id": assistant_id, "text": delta})
            for chunk in tts_chunker.push(delta):
                await _queue_tts_text(audio_queue, chunk)
        for chunk in tts_chunker.finish():
            await _queue_tts_text(audio_queue, chunk)
    finally:
        await audio_queue.put(None)
        await audio_task

    response_text = "".join(response_parts).strip()
    if not response_text:
        raise RuntimeError("The provider returned an empty response.")
    chat = chats.append_message(
        chat_id,
        role="assistant",
        text=response_text,
        provider=provider_key,
        model=model,
    )
    await websocket.send_json({"type": "assistant_done", "id": assistant_id})
    await websocket.send_json({"type": "chat", "chat": chat, "chats": chats.list_chats()})
    await websocket.send_json({"type": "status", "phase": "idle"})


class StreamingTtsChunker:
    def __init__(self) -> None:
        self.buffer = ""
        self.pending = ""
        self.pending_sentences = 0
        self.first_chunk_sent = False

    def push(self, delta: str) -> list[str]:
        chunks: list[str] = []
        self.buffer += delta

        sentences, self.buffer = _pop_complete_tts_sentences(self.buffer)
        for sentence in sentences:
            chunks.extend(self._append_sentence(sentence))

        if not self.first_chunk_sent:
            fragment, self.buffer = _pop_speakable_fragment(
                self.buffer,
                min_chars=settings.tts_first_chunk_min_chars,
                max_chars=settings.tts_max_chunk_chars,
            )
            if fragment:
                chunks.extend(self._append_sentence(fragment))
        return chunks

    def finish(self) -> list[str]:
        remaining = f"{self.pending} {self.buffer.strip()}".strip()
        self.pending = ""
        self.buffer = ""
        self.pending_sentences = 0
        if not remaining:
            return []
        self.first_chunk_sent = True
        return [remaining]

    def _append_sentence(self, sentence: str) -> list[str]:
        sentence = sentence.strip()
        if not sentence:
            return []

        chunks: list[str] = []
        candidate = f"{self.pending} {sentence}".strip()
        if self.pending and len(candidate) > settings.tts_max_chunk_chars:
            chunks.append(self.pending)
            self.first_chunk_sent = True
            self.pending = sentence
            self.pending_sentences = 1
        else:
            self.pending = candidate
            self.pending_sentences += 1

        max_sentences = (
            settings.tts_next_chunk_max_sentences
            if self.first_chunk_sent
            else settings.tts_first_chunk_max_sentences
        )
        sentence_limit_ready = self.pending_sentences >= max_sentences and len(self.pending) >= self._minimum_flush_chars()
        if len(self.pending) >= self._target_chars() or sentence_limit_ready:
            chunks.append(self.pending)
            self.first_chunk_sent = True
            self.pending = ""
            self.pending_sentences = 0
        return chunks

    def _target_chars(self) -> int:
        return settings.tts_next_chunk_target_chars if self.first_chunk_sent else settings.tts_first_chunk_min_chars

    def _minimum_flush_chars(self) -> int:
        return max(120, int(self._target_chars() * 0.75))


async def _stream_tts_audio(
    websocket: WebSocket,
    audio_queue: asyncio.Queue[str | None],
    provider_key: str,
    model: str | None,
) -> None:
    while True:
        text = await audio_queue.get()
        try:
            if text is None:
                return
            await websocket.send_json({"type": "status", "phase": "synthesizing"})
            audio_path = await tts.synthesize_chunk(text)
            await websocket.send_json(
                {
                    "type": "audio",
                    "url": f"/audio/{audio_path.name}",
                    "text": text,
                    "provider": provider_key,
                    "model": model,
                    "streaming": True,
                }
            )
        finally:
            audio_queue.task_done()


async def _queue_tts_text(audio_queue: asyncio.Queue[str | None], text: str) -> None:
    for chunk in split_tts_text(text, settings.tts_max_chunk_chars):
        cleaned = chunk.strip()
        if cleaned:
            await audio_queue.put(cleaned)


def _pop_complete_tts_sentences(text: str) -> tuple[list[str], str]:
    matches = list(re.finditer(r"(?<=[.!?])\s+", text))
    if not matches:
        return [], text
    cut = matches[-1].end()
    complete = text[:cut].strip()
    remaining = text[cut:]
    sentences = [item.strip() for item in re.split(r"(?<=[.!?])\s+", complete) if item.strip()]
    return sentences, remaining


def _pop_speakable_fragment(text: str, *, min_chars: int, max_chars: int) -> tuple[str, str]:
    cleaned = re.sub(r"\s+", " ", text)
    if len(cleaned) < min_chars:
        return "", text

    limit = min(len(cleaned), max_chars)
    preferred_window_end = min(limit, min_chars + 80)
    preferred = list(re.finditer(r"[,;:]\s+", cleaned[:preferred_window_end]))
    if preferred and preferred[-1].end() >= min_chars:
        cut = preferred[-1].end()
        return cleaned[:cut].strip(), cleaned[cut:]

    whitespace = [match for match in re.finditer(r"\s+", cleaned[:preferred_window_end]) if match.end() >= min_chars]
    if whitespace:
        cut = whitespace[-1].end()
        return cleaned[:cut].strip(), cleaned[cut:]

    fallback = cleaned.rfind(" ", 0, limit)
    if fallback >= min_chars:
        cut = fallback + 1
        return cleaned[:cut].strip(), cleaned[cut:]
    if len(cleaned) >= max_chars:
        return cleaned[:max_chars].strip(), cleaned[max_chars:]
    return "", text


if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")


@app.get("/{path:path}")
async def frontend(path: str) -> FileResponse:
    requested = FRONTEND_DIST / path
    if path and requested.is_file():
        return FileResponse(requested)
    index = FRONTEND_DIST / "index.html"
    if index.exists():
        return FileResponse(index)
    return FileResponse(Path(__file__).resolve().parents[2] / "backend" / "static" / "dev.html")
