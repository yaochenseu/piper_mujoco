#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PIPERX_PYTHON:-/home/yzc/miniconda3/envs/piper_mujoco/bin/python}"

if [ ! -x "$PYTHON" ]; then
    echo "Missing Python: $PYTHON" >&2
    echo "Set PIPERX_PYTHON to the piper_mujoco environment Python." >&2
    exit 2
fi

exec "$PYTHON" "$ROOT/main_piperx_vision.py" --real "$@"
