#!/usr/bin/env bash
#
# SelfDriveSim launcher.
#
# Checks Redis, applies migrations, collects static files, prints the URLs for
# the PC and the phone, then serves ASGI (HTTP + WebSocket) on 0.0.0.0:8000.
#
# Usage:
#   ./scripts/run.sh              # from anywhere
#   PORT=9000 ./scripts/run.sh    # different port
#
set -euo pipefail

# --- locations ---------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# --- configuration -----------------------------------------------------------
PORT="${PORT:-8000}"
REDIS_HOST="${REDIS_HOST:-127.0.0.1}"
REDIS_PORT="${REDIS_PORT:-6379}"

# --- interpreter: prefer the local virtualenv --------------------------------
if [ -x ".venv/bin/python" ]; then
  PY=".venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PY="python3"
else
  PY="python"
fi

echo "==> Using interpreter: $PY ($($PY --version 2>&1))"

# --- 1. Redis must answer ----------------------------------------------------
echo "==> Checking Redis at ${REDIS_HOST}:${REDIS_PORT} …"
if command -v redis-cli >/dev/null 2>&1; then
  if ! redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" ping 2>/dev/null | grep -q PONG; then
    cat >&2 <<EOF

ERROR: Redis did not answer at ${REDIS_HOST}:${REDIS_PORT}.

Start it with:
  sudo systemctl enable --now redis-server      # systemd systems
  docker run -d -p ${REDIS_PORT}:6379 redis:7   # or Docker

Or point REDIS_HOST / REDIS_PORT at a running instance.

EOF
    exit 1
  fi
else
  echo "    redis-cli not found; skipping the ping check (install redis-tools to enable it)."
fi
echo "    Redis is up."

# --- 2. database -------------------------------------------------------------
echo "==> Applying migrations …"
"$PY" manage.py migrate --noinput

# --- 3. static files ---------------------------------------------------------
echo "==> Collecting static files …"
"$PY" manage.py collectstatic --noinput

# --- 4. URLs -----------------------------------------------------------------
LAN_IPS="$(hostname -I 2>/dev/null || true)"
[ -z "$LAN_IPS" ] && LAN_IPS="$(ip -o -4 addr show scope global 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | tr '\n' ' ' || true)"

echo
echo "    ┌─────────────────────────────────────────────────────────────┐"
echo "    │  PC (renderer, fullscreen):  http://localhost:${PORT}/          │"
for ip in $LAN_IPS; do
  echo "    │  PC over LAN:                http://${ip}:${PORT}/"
done
echo "    │  Phone (needs HTTPS, see below):                            │"
echo "    │    ngrok http ${PORT}   →  https://…/dashboard/                 │"
echo "    └─────────────────────────────────────────────────────────────┘"
echo
echo "    Keyboard fallback drives the car when no phone command arrives."
echo

# --- 5. serve ----------------------------------------------------------------
echo "==> Starting Daphne on 0.0.0.0:${PORT} (Ctrl+C to stop) …"
exec "$PY" -m daphne -b 0.0.0.0 -p "$PORT" selfdriving_sim.asgi:application
