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
TLS_PORT="${TLS_PORT:-8443}"
SIMULATION_TLS_PORT="$TLS_PORT"
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

# First non-loopback address: the one the phone has to reach on the same Wi-Fi.
LAN_IP=""
for ip in $LAN_IPS; do
  case "$ip" in
    127.*) continue ;;
  esac
  LAN_IP="$ip"
  break
done
PHONE_HOST="${LAN_IP:-<LAN-IP>}"

echo
echo "    PC (renderer, fullscreen):  http://localhost:${PORT}/"
echo "    Phone dashboard:            https://${PHONE_HOST}:${TLS_PORT}/dashboard/"
echo
echo "    Phone setup - scan the QR code shown on the monitor page, same Wi-Fi,"
echo "      then accept the one-time self-signed certificate warning:"
echo "        Android Chrome:  Advanced -> Proceed to <IP>"
echo "        iOS Safari:      Show Details -> visit this website"
echo "    HTTPS on port ${TLS_PORT} is what makes the camera work on Android and iOS."
echo "    Guide: docs/guide.fa.md (fa) - docs/guide.en.md (en)"
echo
echo "    Keyboard fallback drives the car when no phone command arrives."
echo

# --- 5. serve ----------------------------------------------------------------
echo "==> Serving on 0.0.0.0:${PORT} (http) and 0.0.0.0:${TLS_PORT} (https) - Ctrl+C to stop"
exec "$PY" manage.py runsecure "$PORT"
