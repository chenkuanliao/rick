from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

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
from .services.tts import TextToSpeechService


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
        "providers": await providers.status(),
        "stt": stt.status,
        "tts": tts.status,
    }


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
    items = chats.list_chats()
    active_chat = chats.create_chat() if not items else None
    return {"deleted": True, "chats": chats.list_chats(), "active_chat": active_chat}


@app.post("/api/voice-prompt")
async def voice_prompt(file: UploadFile = File(...)) -> dict[str, Any]:
    content = await file.read()
    path = await normalize_voice_prompt(content, mime_type=file.content_type, settings=settings)
    return {"saved": True, "path": str(path), "tts": tts.status}


@app.websocket("/ws/chat")
async def chat_socket(websocket: WebSocket) -> None:
    await websocket.accept()
    active_turn: asyncio.Task[None] | None = None
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
                else:
                    await websocket.send_json({"type": "status", "phase": "idle"})
                continue
            if payload.get("type") not in {"turn", "text_turn"}:
                await websocket.send_json({"type": "error", "message": "Unsupported websocket message type."})
                continue
            if active_turn and not active_turn.done():
                active_turn.cancel()
                await websocket.send_json({"type": "cancelled"})
            active_turn = asyncio.create_task(_handle_turn(websocket, payload))
            active_turn.add_done_callback(_consume_cancelled_turn)
    except WebSocketDisconnect:
        if active_turn and not active_turn.done():
            active_turn.cancel()
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


async def _handle_user_text(websocket: WebSocket, payload: dict[str, Any], chat_id: str, user_text: str) -> None:
    chat = chats.append_message(chat_id, role="user", text=user_text)
    await websocket.send_json({"type": "transcript", "role": "user", "text": user_text})
    await websocket.send_json({"type": "chat", "chat": chat, "chats": chats.list_chats()})

    await websocket.send_json({"type": "status", "phase": "thinking"})
    response_text, provider_key, model = await providers.chat(
        payload.get("provider"),
        payload.get("model"),
        chats.llm_messages(chat_id),
        user_text,
    )
    chat = chats.append_message(
        chat_id,
        role="assistant",
        text=response_text,
        provider=provider_key,
        model=model,
    )
    await websocket.send_json(
        {
            "type": "transcript",
            "role": "assistant",
            "text": response_text,
            "provider": provider_key,
            "model": model,
        }
    )
    await websocket.send_json({"type": "chat", "chat": chat, "chats": chats.list_chats()})

    await websocket.send_json({"type": "status", "phase": "synthesizing"})
    audio_path = await tts.synthesize(response_text)
    await websocket.send_json(
        {
            "type": "audio",
            "url": f"/audio/{audio_path.name}",
            "text": response_text,
            "provider": provider_key,
            "model": model,
        }
    )


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
