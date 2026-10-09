# SelfDriveSim server

Django and Django Channels server for the closed-loop self-driving simulator. A phone captures the rendered scene with its camera, runs the DAVE-2-GRU model in the browser, and sends driving commands over WebSocket. The server relays those commands to the Three.js renderer on the PC.

This README covers phase 1: the server skeleton. It includes ASGI with Daphne, a Redis channel layer, and two WebSocket endpoints that relay commands.

## Requirements

- Ubuntu 22.04 or later, or Windows with WSL2 running Ubuntu
- Python 3.11
- A Redis server

The TensorFlow packages in `requirements.txt` install on Linux and macOS only. Windows users should work inside WSL2.

Ubuntu 22.04 ships Python 3.10 and Ubuntu 24.04 ships Python 3.12. To get Python 3.11, use the deadsnakes PPA: `sudo add-apt-repository ppa:deadsnakes/ppa && sudo apt install python3.11 python3.11-venv`.

## Setup

Run all commands from this directory (`selfdriving_sim/`).

### 1. Install and start Redis

```bash
sudo apt update
sudo apt install -y redis-server
sudo systemctl enable --now redis-server
redis-cli ping   # expected output: PONG
```

### 2. Create a virtual environment and install dependencies

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Create the database

```bash
python manage.py migrate
```

### 4. Optional: local configuration

Create a `.env` file in this directory to override the defaults. Real environment variables take precedence over the file.

| Variable | Default | Purpose |
|---|---|---|
| `DJANGO_SECRET_KEY` | development placeholder | Django secret key. Set a real value outside development. |
| `REDIS_HOST` | `127.0.0.1` | Redis host for the channel layer. |
| `REDIS_PORT` | `6379` | Redis port for the channel layer. |
| `DJANGO_CSRF_TRUSTED_ORIGINS` | empty | Extra origins, comma-separated, allowed to send unsafe requests. |

## Running the server

```bash
daphne -b 0.0.0.0 -p 8000 selfdriving_sim.asgi:application
```

| Path | Description |
|---|---|
| `GET /health/` | Returns `{"status": "ok", "phase": 1}`. |
| `/admin/` | Django admin site. |
| `/ws/control/` | WebSocket endpoint for the phone (see below). |
| `/ws/render/` | WebSocket endpoint for the Three.js page (see below). |

The root path `/` returns 404 until the scene page is added in phase 2.

## WebSocket protocol

Both endpoints join the `simulation_render` channel group. Messages flow from `/ws/control/` to `/ws/render/` through Redis. Several Daphne processes can share one Redis instance.

### Control frame (phone to server)

```json
{"steering": 0.25, "throttle": 0.6, "brake": 0.0, "turn_signal": 1}
```

The server validates each frame and clips values into range before it broadcasts:

| Field | Valid range | Out-of-range values |
|---|---|---|
| `steering` | -1 to 1 | Clipped |
| `throttle` | 0 to 1 | Clipped |
| `brake` | 0 to 1 | Clipped |
| `turn_signal` | 0 (off), 1 (left), 2 (right) | Clipped to 0 to 2, then rounded to the nearest integer |

Malformed JSON, missing fields, and non-numeric values are refused. The sender receives `{"type": "error", "message": "..."}`. Nothing is broadcast, and the connection stays open.

### Render frame (server to browser)

The render endpoint sends each valid command to the browser as a flat JSON object with the same four fields:

```json
{"steering": 0.25, "throttle": 0.6, "brake": 0.0, "turn_signal": 1}
```

## Security notes

- `DEBUG = True` and `ALLOWED_HOSTS = ["*"]` are development settings. Change them before deploying.
- With `ALLOWED_HOSTS = ["*"]`, the WebSocket origin check accepts every origin, and handshakes without an `Origin` header are accepted too. Restrict `ALLOWED_HOSTS` before deploying so that the origin check applies.
- Redis has no authentication by default. Keep it bound to `127.0.0.1` or to a trusted network.

## Tests

```bash
pytest
```

The tests use an in-memory channel layer, so they do not need Redis. `simulation/tests/test_redis_channel_layer.py` runs only when a Redis server is reachable at `REDIS_HOST:REDIS_PORT`. Otherwise pytest skips it, and the skip reason appears in the summary.

## Project layout

```
selfdriving_sim/
├── manage.py
├── pytest.ini               pytest and pytest-asyncio settings
├── requirements.txt
├── conftest.py              shared fixtures (in-memory channel layer)
├── selfdriving_sim/         project package: settings, ASGI/WSGI entry points, root URLs
├── simulation/              scene app: consumers, routing, health view, tests
├── client/                  mobile client app (phase 3)
└── models/                  model download, extension and conversion (phase 4)
```
