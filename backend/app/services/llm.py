from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from ..config import Settings


Message = dict[str, str]
DEGENERATE_TOKEN_PATTERNS = (
    re.compile(r"(\[\[\[\[C\]\]\]\]C\)){2,}"),
    re.compile(r"(>\s*\d+(?:\.\d+)?s\)){2,}"),
    re.compile(r"(.{8,}?)\1{4,}", re.DOTALL),
)


class Provider(Protocol):
    name: str
    model: str | None

    def status(self) -> dict[str, Any]:
        ...

    async def chat(self, messages: list[Message]) -> str:
        ...

    async def models(self) -> list[str]:
        ...


@dataclass
class OpenAICompatibleProvider:
    name: str
    base_url: str | None
    api_key: str | None
    model: str | None
    required_api_key: bool = True

    def status(self) -> dict[str, Any]:
        missing: list[str] = []
        if not self.base_url:
            missing.append("base_url")
        if not self.model:
            missing.append("model")
        if self.required_api_key and not self.api_key:
            missing.append("api_key")
        return {
            "name": self.name,
            "model": self.model,
            "configured": not missing,
            "missing": missing,
            "base_url": self.base_url,
        }

    async def chat(self, messages: list[Message]) -> str:
        return await self.chat_with_model(messages, self.model)

    async def chat_with_model(self, messages: list[Message], model: str | None) -> str:
        missing = self.status()["missing"]
        if "model" in missing and model:
            missing = [item for item in missing if item != "model"]
        if missing:
            raise RuntimeError(f"{self.name} is not configured: missing {', '.join(missing)}")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        url = f"{self.base_url.rstrip('/')}/chat/completions"  # type: ignore[union-attr]
        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 700,
        }
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
        return data["choices"][0]["message"]["content"].strip()

    async def models(self) -> list[str]:
        if not self.base_url:
            return []
        headers: dict[str, str] = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            async with httpx.AsyncClient(timeout=8) as client:
                response = await client.get(f"{self.base_url.rstrip('/')}/models", headers=headers)
                response.raise_for_status()
                data = response.json()
        except Exception:  # noqa: BLE001
            return []
        values = data.get("data", [])
        return sorted(item.get("id", "") for item in values if item.get("id"))


@dataclass
class OllamaProvider:
    base_url: str
    model: str | None
    name: str = "Ollama"

    def status(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model": self.model,
            "configured": bool(self.model),
            "missing": [] if self.model else ["model"],
            "base_url": self.base_url,
        }

    async def available_models(self) -> list[str]:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(f"{self.base_url.rstrip('/')}/api/tags")
                response.raise_for_status()
                data = response.json()
        except Exception:  # noqa: BLE001
            return []
        return [item.get("name", "") for item in data.get("models", []) if item.get("name")]

    async def chat(self, messages: list[Message]) -> str:
        return await self.chat_with_model(messages, self.model)

    async def chat_with_model(self, messages: list[Message], model: str | None) -> str:
        if not model:
            models = await self.available_models()
            for candidate in ("gpt-oss:20b", "gemma4:26b"):
                if candidate in models:
                    model = candidate
                    break
        if not model:
            raise RuntimeError("Ollama is not configured: set OLLAMA_MODEL or install gpt-oss:20b/gemma4:26b.")
        payload = {"model": model, "messages": messages, "stream": False}
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(f"{self.base_url.rstrip('/')}/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()
        return data["message"]["content"].strip()

    async def models(self) -> list[str]:
        return await self.available_models()


@dataclass
class GoogleProvider:
    api_key: str | None
    model: str | None
    name: str = "Google Gemini"

    def status(self) -> dict[str, Any]:
        missing = []
        if not self.api_key:
            missing.append("api_key")
        if not self.model:
            missing.append("model")
        return {"name": self.name, "model": self.model, "configured": not missing, "missing": missing}

    async def chat(self, messages: list[Message]) -> str:
        return await self.chat_with_model(messages, self.model)

    async def chat_with_model(self, messages: list[Message], model: str | None) -> str:
        missing = self.status()["missing"]
        if "model" in missing and model:
            missing = [item for item in missing if item != "model"]
        if missing:
            raise RuntimeError(f"Google is not configured: missing {', '.join(missing)}")
        prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent?key={self.api_key}"
        )
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
        parts = data["candidates"][0]["content"]["parts"]
        return "".join(part.get("text", "") for part in parts).strip()

    async def models(self) -> list[str]:
        if not self.api_key:
            return []
        try:
            async with httpx.AsyncClient(timeout=8) as client:
                response = await client.get(f"https://generativelanguage.googleapis.com/v1beta/models?key={self.api_key}")
                response.raise_for_status()
                data = response.json()
        except Exception:  # noqa: BLE001
            return []
        return sorted(
            item["name"].split("/", 1)[-1]
            for item in data.get("models", [])
            if "generateContent" in item.get("supportedGenerationMethods", [])
        )


@dataclass
class DisabledProvider:
    name: str
    reason: str
    model: str | None = None

    def status(self) -> dict[str, Any]:
        return {"name": self.name, "model": self.model, "configured": False, "disabled": True, "reason": self.reason}

    async def chat(self, messages: list[Message]) -> str:
        raise RuntimeError(f"{self.name} is disabled: {self.reason}")

    async def models(self) -> list[str]:
        return []


class ProviderRegistry:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.providers: dict[str, Provider] = {
            "nvidia": OpenAICompatibleProvider(
                "NVIDIA NIM",
                settings.nvidia_base_url,
                settings.nvidia_api_key,
                settings.nvidia_model,
            ),
            "openai": OpenAICompatibleProvider(
                "OpenAI",
                settings.openai_base_url,
                settings.openai_api_key,
                settings.openai_model,
            ),
            "lmstudio": OpenAICompatibleProvider(
                "LM Studio",
                settings.lm_studio_base_url,
                "lm-studio",
                settings.lm_studio_model,
                required_api_key=False,
            ),
            "openai_compatible": OpenAICompatibleProvider(
                settings.openai_compatible_name,
                settings.openai_compatible_base_url,
                settings.openai_compatible_api_key,
                settings.openai_compatible_model,
                required_api_key=False,
            ),
            "ollama": OllamaProvider(settings.ollama_base_url, settings.ollama_model),
            "google": GoogleProvider(settings.google_api_key, settings.google_model),
            "opencode": DisabledProvider(
                "Opencode",
                "No stable callable CLI/API contract was configured for this app.",
            ),
        }

    def get_system_prompt(self) -> str:
        if self.settings.prompt_store_path.exists():
            saved = self.settings.prompt_store_path.read_text(encoding="utf-8").strip()
            if saved:
                return saved
        return self.settings.system_prompt

    def save_system_prompt(self, prompt: str) -> str:
        cleaned = prompt.strip()
        if not cleaned:
            raise RuntimeError("System prompt cannot be empty.")
        self.settings.prompt_store_path.parent.mkdir(parents=True, exist_ok=True)
        self.settings.prompt_store_path.write_text(cleaned, encoding="utf-8")
        return cleaned

    async def status(self) -> dict[str, Any]:
        result = {key: provider.status() for key, provider in self.providers.items()}
        ollama = self.providers["ollama"]
        if isinstance(ollama, OllamaProvider):
            result["ollama"]["available_models"] = await ollama.available_models()
        for key, provider in self.providers.items():
            if "available_models" not in result[key]:
                result[key]["available_models"] = await provider.models()
        return {
            "default": self.settings.default_provider,
            "providers": result,
        }

    async def chat(
        self,
        provider_key: str | None,
        model_override: str | None,
        transcript: list[Message],
        user_text: str,
    ) -> tuple[str, str, str | None]:
        key = provider_key or self.settings.default_provider
        provider = self.providers.get(key)
        if provider is None:
            raise RuntimeError(f"Unknown provider: {key}")
        messages = [
            {
                "role": "system",
                "content": f"{self.get_system_prompt()}\n\n{self.settings.tts_output_prompt}",
            }
        ]
        messages.extend(transcript[-12:])
        messages.append({"role": "user", "content": user_text})
        if hasattr(provider, "chat_with_model"):
            response = await provider.chat_with_model(messages, model_override or provider.model)  # type: ignore[attr-defined]
        else:
            response = await provider.chat(messages)
        if _looks_degenerate(response):
            raise RuntimeError(
                f"{provider.name} returned malformed text with model {model_override or provider.model}. "
                "Choose a different model."
            )
        return response, key, model_override or provider.model


async def cuda_status() -> dict[str, Any]:
    def inspect() -> dict[str, Any]:
        try:
            import torch

            available = torch.cuda.is_available()
            result = {
                "torch_imported": True,
                "available": available,
                "device": torch.cuda.get_device_name(0) if available else None,
                "capability": torch.cuda.get_device_capability(0) if available else None,
                "torch_version": torch.__version__,
            }
            if available:
                try:
                    x = torch.ones((8, 8), device="cuda")
                    _ = x @ x
                    torch.cuda.synchronize()
                    result["cuda_smoke_test"] = True
                except Exception as exc:  # noqa: BLE001
                    result["cuda_smoke_test"] = False
                    result["warning"] = f"CUDA is visible but failed a Torch smoke test: {exc}"
            return result
        except Exception as exc:  # noqa: BLE001
            return {"torch_imported": False, "available": False, "error": str(exc)}

    return await asyncio.to_thread(inspect)


def _looks_degenerate(text: str) -> bool:
    compact = text.strip()
    if not compact:
        return True
    return any(pattern.search(compact) for pattern in DEGENERATE_TOKEN_PATTERNS)
