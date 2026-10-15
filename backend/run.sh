#!/usr/bin/env bash
# Start the WBGT FastAPI backend on the port Apache reverse-proxies to.
#
# Apache (rhialarm.sdsu.edu-ssl.conf) forwards:
#     ProxyPass "/api/" "http://127.0.0.1:8080/api/"
# so the backend MUST listen on 127.0.0.1:8080 and serve routes under /api.
#
# Bind to 127.0.0.1 (not 0.0.0.0): only Apache on the same host should reach
# it; the public TLS termination is Apache's job.
set -euo pipefail

cd "$(dirname "$0")"

exec ./venv/bin/uvicorn app.main:app \
    --host 127.0.0.1 \
    --port 8080 \
    --workers 2 \
    --timeout-keep-alive 120
