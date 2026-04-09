#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

VENV_DIR="$ROOT/.venv"
PYTHON_EXE="${TADA_PYTHON:-}"

is_supported_python() {
  "$1" -c "import sys; raise SystemExit(0 if sys.version_info[:2] >= (3, 10) else 1)" >/dev/null 2>&1
}

if [[ -n "$PYTHON_EXE" ]] && ! is_supported_python "$PYTHON_EXE"; then
  PYTHON_EXE=""
fi

if [[ -z "$PYTHON_EXE" && -x "$VENV_DIR/bin/python" ]]; then
  if is_supported_python "$VENV_DIR/bin/python" && "$VENV_DIR/bin/python" -m pip --version >/dev/null 2>&1; then
    PYTHON_EXE="$VENV_DIR/bin/python"
  fi
fi

if [[ -z "$PYTHON_EXE" && -x "$ROOT/env/bin/python" ]]; then
  if is_supported_python "$ROOT/env/bin/python" && "$ROOT/env/bin/python" -m pip --version >/dev/null 2>&1; then
    PYTHON_EXE="$ROOT/env/bin/python"
  fi
fi

if [[ -z "$PYTHON_EXE" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    echo "[0/3] Creating local virtual environment in .venv ..."
    if python3 -m venv "$VENV_DIR"; then
      if "$VENV_DIR/bin/python" -m pip --version >/dev/null 2>&1; then
        PYTHON_EXE="$VENV_DIR/bin/python"
      else
        echo "[WARN] The new .venv does not contain a usable pip. Falling back to the system interpreter."
        PYTHON_EXE="$(command -v python3)"
      fi
    else
      echo "[WARN] Could not create .venv via python3 -m venv. Falling back to the system interpreter."
      PYTHON_EXE="$(command -v python3)"
    fi
  elif command -v python >/dev/null 2>&1; then
    echo "[0/3] Creating local virtual environment in .venv ..."
    if python -m venv "$VENV_DIR"; then
      if "$VENV_DIR/bin/python" -m pip --version >/dev/null 2>&1; then
        PYTHON_EXE="$VENV_DIR/bin/python"
      else
        echo "[WARN] The new .venv does not contain a usable pip. Falling back to the system interpreter."
        PYTHON_EXE="$(command -v python)"
      fi
    else
      echo "[WARN] Could not create .venv via python -m venv. Falling back to the system interpreter."
      PYTHON_EXE="$(command -v python)"
    fi
  else
    echo "[ERROR] No usable Python interpreter was found. Install Python 3 and re-run install.sh." >&2
    exit 1
  fi
fi

export PIP_DISABLE_PIP_VERSION_CHECK=1
export PYTHONUTF8=1

echo "[1/3] Using Python: $PYTHON_EXE"
"$PYTHON_EXE" -c "import sys; print(sys.version)"
echo "[2/3] Running shared runtime installer..."
"$PYTHON_EXE" "$ROOT/tools/install_runtime.py"
echo "[3/3] Install finished successfully."
echo "Run ./start.sh to launch the server."
