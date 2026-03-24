#!/usr/bin/env bash
set -euo pipefail

PORT="${1:-8787}"

if [[ -z "${ARK_API_KEY:-}" ]]; then
  echo "[warn] ARK_API_KEY is empty. Real engine may fail."
fi

if [[ -z "${ARK_BASE_URL:-}" ]]; then
  export ARK_BASE_URL="https://ark.cn-beijing.volces.com/api/v3"
fi

if [[ -z "${OPENAI_TRUST_ENV_PROXY:-}" ]]; then
  export OPENAI_TRUST_ENV_PROXY="0"
fi

unset OPENAI_MODEL_NAME || true

echo "[info] starting demo_api on 127.0.0.1:${PORT}"
python -m uvicorn demo_api:app --host 127.0.0.1 --port "${PORT}" --reload
