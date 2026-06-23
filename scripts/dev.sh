#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-rick-live-chat}"
HOST="${APP_HOST:-127.0.0.1}"
PORT="${APP_PORT:-8000}"
CONDA_BIN="${CONDA_BIN:-}"

if [[ -z "$CONDA_BIN" ]]; then
  if command -v mamba >/dev/null 2>&1; then
    CONDA_BIN="mamba"
  elif command -v conda >/dev/null 2>&1; then
    CONDA_BIN="conda"
  else
    echo "mamba or conda is required to run the Python environment." >&2
    exit 1
  fi
fi

"$CONDA_BIN" run -n "$ENV_NAME" uvicorn backend.app.main:app --host "$HOST" --port "$PORT" --reload &
BACKEND_PID=$!
npm --prefix frontend run dev
kill "$BACKEND_PID" 2>/dev/null || true
