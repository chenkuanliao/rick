from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
FRONTEND_DIST = BASE_DIR / "frontend" / "dist"
DEFAULT_SYSTEM_PROMPT = (
    "You are Rick, an AI live chat companion. Be direct, funny, skeptical, "
    "and inventive, but stay useful. "
    "Challenge weak reasoning, surface hidden assumptions, and keep the conversation moving."
)
TTS_OUTPUT_PROMPT = (
    "Format every assistant reply for text-to-speech. Use natural spoken language, "
    "short paragraphs, no markdown tables, no code fences unless explicitly requested, "
    "no emojis, and avoid visual-only formatting. Keep the answer concise unless the user asks "
    "for depth."
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_host: str = "127.0.0.1"
    app_port: int = 8000
    app_title: str = "Rick AI Live Chat"
    public_https_url: str | None = None
    insecure_redirect_hosts: str = ""
    preload_speech_models: bool = False

    default_provider: Literal[
        "nvidia",
        "openai",
        "google",
        "ollama",
        "lmstudio",
        "openai_compatible",
        "opencode",
    ] = "nvidia"
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    tts_output_prompt: str = TTS_OUTPUT_PROMPT
    prompt_store_path: Path = Field(default=DATA_DIR / "memory" / "system_prompt.txt")
    llm_max_tokens: int = 8192
    llm_retry_max_tokens: int = 16384

    nvidia_api_key: str | None = None
    nvidia_model: str | None = "meta/llama-3.3-70b-instruct"
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"

    openai_api_key: str | None = None
    openai_model: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"

    google_api_key: str | None = None
    google_model: str | None = "gemini-2.5-flash"

    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str | None = None

    lm_studio_base_url: str = "http://127.0.0.1:1234/v1"
    lm_studio_model: str | None = None

    openai_compatible_name: str = "OpenAI-compatible"
    openai_compatible_base_url: str | None = None
    openai_compatible_api_key: str | None = None
    openai_compatible_model: str | None = None

    opencode_api_key: str | None = None
    opencode_model: str | None = "deepseek-v4-flash"
    opencode_base_url: str = "https://opencode.ai/zen/v1"

    stt_model: str = "Systran/faster-whisper-medium"
    stt_fallback_model: str = "openai/whisper-base"
    stt_device: str = "auto"
    stt_compute_type: str = "auto"
    stt_language: str | None = None
    stt_beam_size: int = 1
    stt_vad_filter: bool = True

    tts_temperature: float = 0.8
    tts_top_p: float = 0.95
    tts_top_k: int = 1000
    tts_repetition_penalty: float = 1.2
    tts_norm_loudness: bool = True
    tts_device: str = "auto"
    tts_max_chunk_chars: int = 900
    tts_first_chunk_min_chars: int = 180
    tts_next_chunk_target_chars: int = 650
    tts_first_chunk_max_sentences: int = 3
    tts_next_chunk_max_sentences: int = 6
    voice_prompt_path: Path = Field(default=DATA_DIR / "uploads" / "voice_prompt.wav")

    ffmpeg_path: str = "ffmpeg"

    @field_validator(
        "nvidia_api_key",
        "nvidia_model",
        "openai_api_key",
        "openai_model",
        "google_api_key",
        "google_model",
        "ollama_model",
        "lm_studio_model",
        "openai_compatible_base_url",
        "openai_compatible_api_key",
        "openai_compatible_model",
        "opencode_api_key",
        "opencode_model",
        "stt_language",
        "public_https_url",
        mode="before",
    )
    @classmethod
    def blank_optional_string_to_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()


def ensure_data_dirs() -> None:
    for path in (DATA_DIR / "audio", DATA_DIR / "uploads", DATA_DIR / "tmp"):
        path.mkdir(parents=True, exist_ok=True)
