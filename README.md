# SelfDriveSim

A closed-loop, end-to-end self-driving simulator that runs entirely on hardware you already own: a PC, a phone, and a Wi-Fi network.

A Three.js scene on the PC renders a first-person three-lane road with random oncoming traffic, full screen. A phone mounted on a stand in front of the monitor watches that screen through its rear camera. In the phone's browser — with no app install and no server-side GPU — a DAVE-2-GRU convolutional-recurrent network runs on TensorFlow.js, analyses the last three camera frames, and produces four driving commands: `steering` (−1…1), `throttle` (0…1), `brake` (0…1) and `turn_signal` (0 off / 1 left / 2 right). The commands travel over a WebSocket through Django Channels, and the car in the scene applies them with a simple kinematic model: it accelerates, brakes, coasts, steers, blinks its turn signals at 3 Hz, grinds against an invisible road wall, and crashes into traffic. Speed telemetry and collision/wall hazards flow back to the phone, closing the loop.

Nothing leaves the local network, and camera frames never leave the phone.

## Requirements

- **Python 3.11**, **Redis 6+**
- **A modern browser** — Chrome on the PC and an up-to-date Chrome on the *Android* phone
- **Node.js 20+** *(optional)* — only for the frontend test suite and rebuilding the Tailwind stylesheet
- For the model pipeline only: Linux/macOS (TensorFlow has no Windows wheels; use WSL2 on Windows)

## Quick start

```bash
cd selfdriving_sim
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver 0.0.0.0:8000     # or: ./scripts/run.sh
```

1. **PC** — open `http://localhost:8000/` and press F11 for fullscreen.
2. **Phone (Android Chrome, same Wi-Fi)** — find the PC's LAN address with `hostname -I` (Linux/macOS) or `ipconfig` (Windows), then enable the one-time flag `chrome://flags/#unsafely-treat-insecure-origin-as-secure` → **Enabled** → add `http://<PC-LAN-IP>:8000` → **Relaunch**.
3. Open `http://<PC-LAN-IP>:8000/dashboard/` on the phone, tap **Allow** for the camera, then **شروع خودران**. iOS has no such flag — use the local-HTTPS route in the guides instead.

## Documentation

| Document | Contents |
|---|---|
| [`selfdriving_sim/docs/guide.fa.md`](selfdriving_sim/docs/guide.fa.md) | راهنمای کامل فارسی: نصب و اجرا، اتصال گوشی با فلگ Chrome، معماری و پروتکل WebSocket، فیزیک صحنه، مدل DAVE-2-GRU و پایپ‌لاین تبدیل، عیب‌یابی، تست‌ها و نتایج |
| [`selfdriving_sim/docs/guide.en.md`](selfdriving_sim/docs/guide.en.md) | The same guide in English |

## License

Copyright © 2026 EmmettSS. All rights reserved.

This project is **not** open-source software. The repository is public so the work can be read and reviewed; copying, redistributing or reusing the code requires prior written permission from the author. Third-party components keep their own licenses (the Vazirmatn font, Three.js and TensorFlow.js — see the `vendor` READMEs and `client/static/client/fonts/OFL.txt`).

---

Designed & developed by **Emmett Software Development Group**.
