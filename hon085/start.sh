#!/bin/sh
cd "$(dirname "$0")" || exit 1
python3 -c "import fastapi, uvicorn, sklearn, mediapipe" 2>/dev/null || python3 -m pip install -r requirements.txt || exit 1
python3 scripts/fetch_assets.py || exit 1
exec python3 run.py "$@"
