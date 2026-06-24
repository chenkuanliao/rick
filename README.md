# Rick AI Live Chat

Single-port browser app for live AI chat with an AI named Rick.

Browser mic or typed message -> local faster-whisper STT when needed -> selected LLM provider -> local Chatterbox TTS -> browser playback.

The chat path is chunked for low latency: LLM deltas stream to the browser immediately, the first speakable TTS phrase is queued before the full reply is complete, later TTS chunks synthesize while earlier chunks are playing, and stale live-transcript work is cancelled when the final recording is submitted.

## Setup

```bash
./scripts/install.sh
```

Set `NVIDIA_API_KEY` in `.env` for the default provider. `NVIDIA_MODEL` defaults to `meta/llama-3.3-70b-instruct`.
Use `./scripts/install.sh` as the canonical installer; it installs Chatterbox first, then force-upgrades Torch/torchaudio to a CUDA build that supports the RTX 5070 Ti. The installer also installs ffmpeg in the selected Conda environment and writes its absolute path to `FFMPEG_PATH` in `.env`.

Use your active Conda environment, or set `ENV_NAME` explicitly:

```bash
conda activate your-env
./scripts/install.sh
./scripts/run.sh
```

## Run

Development, with Vite frontend and FastAPI backend:

```bash
./scripts/dev.sh
```

Single-port run, with FastAPI serving the built frontend:

```bash
./scripts/run.sh
```

By default the app binds to `127.0.0.1`. Open `http://127.0.0.1:8000` on the same machine.

For Tailscale or LAN access, set `APP_HOST=0.0.0.0` deliberately and protect the service at the network layer. Microphone capture requires HTTPS or localhost in modern browsers. The app can redirect a configured raw Tailscale IP host to the HTTPS MagicDNS URL.

## Notes

- Use a Python 3.12 Conda environment.
- `/health` reports CUDA, STT, TTS, and provider configuration without loading Whisper or Chatterbox.
- `PRELOAD_SPEECH_MODELS=true` loads Whisper and Chatterbox in the background at startup so the first real turn does not pay the full model-load cost.
- Interactive STT defaults to `STT_BEAM_SIZE=1` for lower latency. Increase it if you prefer slower, more exhaustive decoding.
- TTS chunking is controlled by `TTS_FIRST_CHUNK_MIN_CHARS`, `TTS_NEXT_CHUNK_TARGET_CHARS`, and `TTS_MAX_CHUNK_CHARS`. The defaults favor continuous speech with fewer sentence gaps; smaller chunks reduce time-to-first-audio but can sound choppy.
- Reasoning/thinking model IDs are filtered from provider model lists and blocked at request time. OpenCode Zen requests include disabled thinking mode so DeepSeek V4 Flash returns final answer content instead of hidden thinking only.
- The UI can save a system prompt for personality/memory. The default is Rick, an AI live chat companion.
- The UI accepts an MP3 or other audio file as the Chatterbox voice template and normalizes it to WAV.
- The installer force-installs a recent CUDA PyTorch stack after Chatterbox so RTX 50-series GPUs can be used despite Chatterbox's older Torch pin. TTS uses CUDA automatically when a Torch CUDA smoke test passes.

## Security

- `.env`, generated audio, uploads, chat history, build output, and local caches are ignored by git.
- Keep `APP_HOST=127.0.0.1` unless you explicitly need remote access.
- Rotate provider API keys if they were ever stored in a shared project directory or committed.
