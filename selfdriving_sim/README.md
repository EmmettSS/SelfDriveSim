# selfdriving_sim

Closed-loop end-to-end driving simulator: Django Channels + Three.js + TensorFlow.js.

## Setup (Ubuntu, Python 3.11)

```bash
sudo apt update && sudo apt install -y redis-server
sudo systemctl enable --now redis-server
python3.11 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python manage.py migrate
daphne -b 0.0.0.0 -p 8000 selfdriving_sim.asgi:application
```

- Health check: `curl http://localhost:8000/health/`
- WebSockets: `/ws/control/` (phone → server), `/ws/render/` (server → renderer)
- Tests: `pytest` (uses an in-memory channel layer; Redis not needed)
