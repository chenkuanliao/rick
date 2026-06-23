from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from ..config import DATA_DIR


CHAT_DIR = DATA_DIR / "chats"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ChatStore:
    def __init__(self, root: Path = CHAT_DIR):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def list_chats(self) -> list[dict[str, Any]]:
        chats = [self._summary(chat) for chat in self._all()]
        return sorted(chats, key=lambda item: (item["pinned"], item["updated_at"]), reverse=True)

    def create_chat(self, title: str = "Untitled chat") -> dict[str, Any]:
        timestamp = now_iso()
        chat = {
            "id": uuid4().hex,
            "title": title,
            "pinned": False,
            "created_at": timestamp,
            "updated_at": timestamp,
            "messages": [],
        }
        self._write(chat)
        return chat

    def get_chat(self, chat_id: str | None) -> dict[str, Any]:
        if not chat_id:
            return self.create_chat()
        path = self._path(chat_id)
        if not path.exists():
            return self.create_chat()
        return json.loads(path.read_text(encoding="utf-8"))

    def update_chat(self, chat_id: str, *, title: str | None = None, pinned: bool | None = None) -> dict[str, Any]:
        chat = self.get_chat(chat_id)
        if title is not None:
            cleaned = title.strip()
            if cleaned:
                chat["title"] = cleaned
        if pinned is not None:
            chat["pinned"] = pinned
        chat["updated_at"] = now_iso()
        self._write(chat)
        return chat

    def delete_chat(self, chat_id: str) -> None:
        self._path(chat_id).unlink(missing_ok=True)

    def append_message(
        self,
        chat_id: str,
        *,
        role: str,
        text: str,
        provider: str | None = None,
        model: str | None = None,
        error: bool = False,
    ) -> dict[str, Any]:
        chat = self.get_chat(chat_id)
        message = {
            "id": uuid4().hex,
            "role": role,
            "text": text,
            "provider": provider,
            "model": model,
            "error": error,
            "time": now_iso(),
        }
        chat["messages"].append(message)
        if chat["title"] == "Untitled chat" and role == "user":
            chat["title"] = text[:48].strip() or chat["title"]
        chat["updated_at"] = now_iso()
        self._write(chat)
        return chat

    def llm_messages(self, chat_id: str) -> list[dict[str, str]]:
        chat = self.get_chat(chat_id)
        result = []
        for message in chat.get("messages", []):
            if message.get("role") in {"user", "assistant"} and not message.get("error"):
                result.append({"role": message["role"], "content": message["text"]})
        return result

    def _all(self) -> list[dict[str, Any]]:
        return [json.loads(path.read_text(encoding="utf-8")) for path in self.root.glob("*.json")]

    def _summary(self, chat: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": chat["id"],
            "title": chat["title"],
            "pinned": chat["pinned"],
            "created_at": chat["created_at"],
            "updated_at": chat["updated_at"],
            "message_count": len(chat.get("messages", [])),
        }

    def _path(self, chat_id: str) -> Path:
        safe_id = "".join(ch for ch in chat_id if ch.isalnum() or ch in {"_", "-"})
        return self.root / f"{safe_id}.json"

    def _write(self, chat: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self._path(chat["id"]).write_text(json.dumps(chat, indent=2), encoding="utf-8")
