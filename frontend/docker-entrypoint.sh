#!/bin/sh
set -e

# ── Parse BACKEND_URL ────────────────────────────────────────────────────────
# Default: http://localhost:8000
BACKEND_URL="${BACKEND_URL:-http://localhost:8000}"

# Strip scheme and extract host
BACKEND_HOST=$(echo "$BACKEND_URL" | sed -E 's~^https?://~~' | sed -E 's~(:[0-9]+)?(/.*)?$~~')
# Extract port (if any); fall back to 80 for http / 443 for https
_BACKEND_PORT=$(echo "$BACKEND_URL" | grep -oE ':[0-9]+' | tr -d ':')
if [ -n "$_BACKEND_PORT" ]; then
    BACKEND_PORT="$_BACKEND_PORT"
elif echo "$BACKEND_URL" | grep -q '^https'; then
    BACKEND_PORT=443
else
    BACKEND_PORT=80
fi

# ── Parse PIPELINE_URL ───────────────────────────────────────────────────────
# Default: http://localhost:8001
PIPELINE_URL="${PIPELINE_URL:-http://localhost:8001}"

PIPELINE_HOST=$(echo "$PIPELINE_URL" | sed -E 's~^https?://~~' | sed -E 's~(:[0-9]+)?(/.*)?$~~')
_PIPELINE_PORT=$(echo "$PIPELINE_URL" | grep -oE ':[0-9]+' | tr -d ':')
if [ -n "$_PIPELINE_PORT" ]; then
    PIPELINE_PORT="$_PIPELINE_PORT"
elif echo "$PIPELINE_URL" | grep -q '^https'; then
    PIPELINE_PORT=443
else
    PIPELINE_PORT=8001
fi

export BACKEND_HOST BACKEND_PORT PIPELINE_HOST PIPELINE_PORT

echo "[entrypoint] backend  → ${BACKEND_HOST}:${BACKEND_PORT}"
echo "[entrypoint] pipeline → ${PIPELINE_HOST}:${PIPELINE_PORT}"

# ── Generate lighttpd config from template ───────────────────────────────────
envsubst '${BACKEND_HOST} ${BACKEND_PORT} ${PIPELINE_HOST} ${PIPELINE_PORT}' \
    < /etc/lighttpd/lighttpd.conf.template \
    > /etc/lighttpd/lighttpd.conf

exec lighttpd -D -f /etc/lighttpd/lighttpd.conf
