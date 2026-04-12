#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYTHON_EXE="${TADA_PYTHON:-}"

is_supported_python() {
  "$1" -c "import sys; raise SystemExit(0 if sys.version_info[:2] >= (3, 10) else 1)" >/dev/null 2>&1
}

if [[ -n "$PYTHON_EXE" ]] && ! is_supported_python "$PYTHON_EXE"; then
  PYTHON_EXE=""
fi

if [[ -z "$PYTHON_EXE" && -x "$ROOT/.venv/bin/python" ]]; then
  if is_supported_python "$ROOT/.venv/bin/python" && "$ROOT/.venv/bin/python" -m pip --version >/dev/null 2>&1; then
    PYTHON_EXE="$ROOT/.venv/bin/python"
  fi
fi

if [[ -z "$PYTHON_EXE" && -x "$ROOT/env/bin/python" ]]; then
  if is_supported_python "$ROOT/env/bin/python" && "$ROOT/env/bin/python" -m pip --version >/dev/null 2>&1; then
    PYTHON_EXE="$ROOT/env/bin/python"
  fi
fi

if [[ -z "$PYTHON_EXE" ]]; then
  PYTHON_EXE="$(command -v python3 || command -v python || true)"
fi

if [[ -z "$PYTHON_EXE" ]]; then
  echo "[ERROR] Python was not found. Run install.sh first." >&2
  exit 1
fi

if ! is_supported_python "$PYTHON_EXE"; then
  echo "[ERROR] Python 3.10 or newer is required. Run install.sh with a supported interpreter." >&2
  exit 1
fi

if [[ ! -f "$ROOT/frontend/dist/index.html" ]]; then
  echo "[ERROR] frontend/dist/index.html is missing. Run install.sh first." >&2
  exit 1
fi

if [[ ! -f "$ROOT/backend/vendor/tada/modules/tada.py" ]]; then
  echo "[ERROR] backend/vendor is missing. Run install.sh first." >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TADA_DEVICE="${TADA_DEVICE:-cuda:0}"
export TADA_ENCODER_DEVICE="${TADA_ENCODER_DEVICE:-cpu}"
export TADA_DISABLE_TORCH_COMPILE="${TADA_DISABLE_TORCH_COMPILE:-1}"
export TADA_ENABLE_CPU_OFFLOAD="${TADA_ENABLE_CPU_OFFLOAD:-0}"
export TADA_DEFAULT_STEPS="${TADA_DEFAULT_STEPS:-10}"
export TADA_MODEL_NAME="${TADA_MODEL_NAME:-HumeAI/tada-3b-ml}"
export HF_HOME="$ROOT/.hf_cache"
export HF_HUB_CACHE="$ROOT/.hf_cache/hub"
export HF_XET_CACHE="$ROOT/.hf_cache/xet"
export HF_ASSETS_CACHE="$ROOT/.hf_cache/assets"
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
RUNTIME_PACKAGE_MODE="$(cat "$ROOT/.runtime_package_mode" 2>/dev/null || true)"
PYTHONPATH_BASE="$ROOT"
if [[ "$RUNTIME_PACKAGE_MODE" == "target" && -d "$ROOT/.python_packages" ]]; then
  PYTHONPATH_BASE="$ROOT/.python_packages:$PYTHONPATH_BASE"
fi
if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="$PYTHONPATH_BASE:$PYTHONPATH"
else
  export PYTHONPATH="$PYTHONPATH_BASE"
fi
export PYTHONUTF8=1

TADA_SERVER_HOST="127.0.0.1"
TADA_SERVER_PORT="7878"
TADA_ALLOW_LAN_ACCESS="false"
while IFS='=' read -r key value; do
  case "$key" in
    TADA_SERVER_HOST) TADA_SERVER_HOST="$value" ;;
    TADA_SERVER_PORT) TADA_SERVER_PORT="$value" ;;
    TADA_ALLOW_LAN_ACCESS) TADA_ALLOW_LAN_ACCESS="$value" ;;
  esac
done < <("$PYTHON_EXE" "$ROOT/tools/print_launch_settings.py")

export TADA_STARTUP_ADMIN_KEY_TTL_SECONDS="${TADA_STARTUP_ADMIN_KEY_TTL_SECONDS:-300}"
export TADA_STARTUP_ADMIN_KEY_DISPLAY_SECONDS="${TADA_STARTUP_ADMIN_KEY_DISPLAY_SECONDS:-15}"
export TADA_STARTUP_ADMIN_KEY="$("$PYTHON_EXE" "$ROOT/tools/generate_startup_admin_key.py" 2>/dev/null || true)"

if [[ -n "$TADA_STARTUP_ADMIN_KEY" ]]; then
  echo
  echo "============================================================"
  echo "Temporary startup admin key (valid for $TADA_STARTUP_ADMIN_KEY_TTL_SECONDS seconds after server start):"
  echo "$TADA_STARTUP_ADMIN_KEY"
  echo "Copy it now if you need emergency admin access in the browser."
  echo "This screen clears automatically in $TADA_STARTUP_ADMIN_KEY_DISPLAY_SECONDS seconds..."
  echo "============================================================"
  sleep "$TADA_STARTUP_ADMIN_KEY_DISPLAY_SECONDS"
  if command -v clear >/dev/null 2>&1; then
    clear
  fi
else
  echo "[WARN] Temporary startup admin key could not be generated." >&2
fi

echo "Using Python: $PYTHON_EXE"
if [[ "$TADA_ALLOW_LAN_ACCESS" == "true" ]]; then
  echo "Starting TADA server on http://$TADA_SERVER_HOST:$TADA_SERVER_PORT (LAN enabled; use this machine's IP address from other devices)"
else
  echo "Starting TADA server on http://$TADA_SERVER_HOST:$TADA_SERVER_PORT"
fi
exec "$PYTHON_EXE" -m uvicorn backend.server_app:app --host "$TADA_SERVER_HOST" --port "$TADA_SERVER_PORT"
