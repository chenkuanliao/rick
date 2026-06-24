#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${ENV_NAME:-}" ]]; then
  if [[ -n "${CONDA_DEFAULT_ENV:-}" && "$CONDA_DEFAULT_ENV" != "base" ]]; then
    ENV_NAME="$CONDA_DEFAULT_ENV"
  else
    echo "Activate a Conda environment or set ENV_NAME before running this script." >&2
    exit 1
  fi
fi
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}"
CONDA_BIN="${CONDA_BIN:-}"

if [[ -z "$CONDA_BIN" ]]; then
  if command -v mamba >/dev/null 2>&1; then
    CONDA_BIN="mamba"
  elif command -v conda >/dev/null 2>&1; then
    CONDA_BIN="conda"
  else
    echo "mamba or conda is required to create the Python environment." >&2
    exit 1
  fi
fi

if ! "$CONDA_BIN" env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  "$CONDA_BIN" create -y -n "$ENV_NAME" -c conda-forge python=3.12 pip ffmpeg
fi

"$CONDA_BIN" install -y -n "$ENV_NAME" -c conda-forge ffmpeg
"$CONDA_BIN" run -n "$ENV_NAME" python -m pip install --upgrade pip
"$CONDA_BIN" run -n "$ENV_NAME" python -m pip install -r backend/requirements.txt
"$CONDA_BIN" run -n "$ENV_NAME" python -m pip install --no-build-isolation -r backend/requirements-tts.txt
"$CONDA_BIN" run -n "$ENV_NAME" python -m pip install --upgrade --force-reinstall \
  torch torchaudio \
  --index-url "$TORCH_INDEX_URL"
npm --prefix frontend install

if [[ ! -f .env && -f .env.example ]]; then
  cp .env.example .env
fi

FFMPEG_BIN="$("$CONDA_BIN" run -n "$ENV_NAME" python -c 'import shutil; path = shutil.which("ffmpeg"); assert path, "ffmpeg was installed, but it is not visible inside the conda environment."; print(path)')"

if [[ -f .env ]]; then
  if grep -q '^FFMPEG_PATH=' .env; then
    sed -i.bak "s|^FFMPEG_PATH=.*|FFMPEG_PATH=$FFMPEG_BIN|" .env
    rm -f .env.bak
  else
    printf '\nFFMPEG_PATH=%s\n' "$FFMPEG_BIN" >> .env
  fi
fi

echo "Configured FFMPEG_PATH=$FFMPEG_BIN"
