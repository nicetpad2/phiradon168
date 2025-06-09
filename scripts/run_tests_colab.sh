#!/usr/bin/env bash
# [Patch] Add Colab-compatible test runner script
set -e
# Mount Google Drive if not already mounted
if [ ! -d "/content/drive/MyDrive" ]; then
  python3 - <<'PY'
from google.colab import drive; drive.mount('/content/drive')
PY
fi
# Change to project directory
cd /content/drive/MyDrive/Phiradon168
# Run pytest in project root
pytest -q --disable-warnings --maxfail=1

