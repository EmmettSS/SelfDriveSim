# SelfDriveSim

A closed-loop, end-to-end self-driving simulator that runs entirely on hardware you already own: a PC, a phone, and a Wi-Fi network.

A Three.js scene on the PC renders a first-person three-lane road with random oncoming traffic, full screen. A phone mounted on a stand in front of the monitor watches that screen through its rear camera. In the phone's browser — with no app install and no server-side GPU — a DAVE-2-GRU convolutional-recurrent network runs on TensorFlow.js, analyses the last three camera frames, and produces four driving commands: `steering` (−1…1), `throttle` (0…1), `brake` (0…1) and `turn_signal` (0 off / 1 left / 2 right). The commands travel over a WebSocket through Django Channels, and the car in the scene applies them with a simple kinematic model: it accelerates, brakes, coasts, steers, blinks its turn signals at 3 Hz, grinds against invisible road walls, and crashes into traffic. Speed telemetry and collision/wall hazards flow back to the phone, closing the loop.

The result is a self-contained research and demo platform for end-to-end driving policies: swap the model, change the scene, and measure how the policy behaves — nothing leaves the local network, and camera frames never leave the phone.

```
 ┌───────────────────────── PC ─────────────────────────┐      ┌────────────────── Phone ──────────────────┐
 │                                                      │      │                                           │
 │  Three.js first-person scene  ←── control_update ──┐ │      │  rear camera (160×120 frames)             │
 │   • ego physics (speed/heading)                   │ │      │        ↓ 3-frame window                   │
 │   • traffic, collisions, walls                    │ │      │  DAVE-2-GRU in TensorFlow.js (browser)    │
 │   • HUD: speed, heading, signals, counters        │ │      │        ↓ 20 Hz                            │
 │        ↓ telemetry 4 Hz, hazards                  │ │      │  steering · throttle · brake · turn_signal│
 └──────────────┬────────────────────────────────────┼─┘      └───────────────┬───────────────────────────┘
                │              ┌─────────────────────┘                        │ control frames
                │              │ /ws/render/                                  │ /ws/control/
                └──────────────┤ Django Channels + Daphne (ASGI) ←────────────┘
                               │   validation · clipping · group fan-out
                               └───────────────┬─────────────────────────────
                                               │
                                        [( Redis channel layer )]
```

## Documentation

| Document | Contents |
|---|---|
| [`selfdriving_sim/README.md`](selfdriving_sim/README.md) | Server setup, WebSocket protocol, scene details, tests |
| [`selfdriving_sim/docs/architecture.md`](selfdriving_sim/docs/architecture.md) | Architecture diagram, layers, WebSocket data flow, life of a frame (Persian) |
| [`selfdriving_sim/docs/troubleshooting.md`](selfdriving_sim/docs/troubleshooting.md) | Redis, camera/HTTPS, WebSocket, model load, off-road, latency (Persian) |
| [`selfdriving_sim/docs/results.md`](selfdriving_sim/docs/results.md) | Measured results, closed-loop test protocol, challenges (Persian) |
| [`selfdriving_sim/docs/phase-5-closed-loop.md`](selfdriving_sim/docs/phase-5-closed-loop.md) | Phase 5 report: physics, first-person view, hazards (Persian) |
| [`selfdriving_sim/docs/phase-4-model.md`](selfdriving_sim/docs/phase-4-model.md) | Model download, extension, conversion, parity (Persian) |
| [`selfdriving_sim/docs/phase-3-mobile.md`](selfdriving_sim/docs/phase-3-mobile.md) | Android + ngrok dashboard guide (Persian) |
| [`selfdriving_sim/models/README.md`](selfdriving_sim/models/README.md) | Model pipeline: pins, contract, conversion, parity |

## Requirements

- **Python 3.11+** (tested on 3.11)
- **Redis 6+** — the Channels layer that fans frames between WebSocket consumers
- **A modern browser** — Chrome/Edge (PC and phone), Firefox, Safari 16+
- **Node.js 20+** *(optional)* — only for the frontend test suite and rebuilding the Tailwind stylesheet
- For the ML pipeline only: Linux/macOS (TensorFlow has no Windows wheels; use WSL2 on Windows)

## Quick start

```bash
# Linux / macOS — checks Redis, migrates, collects static files, starts Daphne,
# and prints the URLs for the PC and the phone:
./scripts/run.sh

# Windows:
scripts\run.bat
```

Manual equivalent (from `selfdriving_sim/`):

```bash
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver 0.0.0.0:8000
```

Then:

1. **PC** — open `http://localhost:8000/` and press F11 for fullscreen. The HUD must show «وصل» (connected).
2. **Phone** — `getUserMedia` needs a secure context, so the phone connects over **HTTPS**. The simplest path is an ngrok tunnel next to the server:
   ```bash
   ngrok http 8000
   ```
   Open `https://<your-tunnel>/dashboard/` on the phone, grant camera access, and mount the phone so its rear camera sees the whole monitor.
3. Press **شروع خودران** (start self-driving) on the phone. Within a second the car on the monitor starts driving the road it sees.

No phone connected? The scene falls back to the keyboard after 0.5 s without commands: `↑/↓` throttle & brake, `←/→` steer.

## Repository layout

```
SelfDriveSim/
├── README.md                    this file
└── selfdriving_sim/
    ├── manage.py
    ├── requirements.txt
    ├── scripts/                 run.sh / run.bat launcher
    ├── selfdriving_sim/         settings, ASGI entry, root URLs
    ├── simulation/              the scene app: page, consumers, scene.js, tests
    ├── client/                  the phone app: dashboard, camera + inference.js,
    │                            converted DAVE-2-GRU model, vendored TF.js, tests
    ├── models/                  checkpoint download → extension → TF.js conversion
    └── docs/                    phase reports, architecture, troubleshooting, results
```

A deeper tour lives in [`selfdriving_sim/README.md`](selfdriving_sim/README.md).

## The control loop, frame by frame

Every ~50 ms on the phone:

1. `drawImage` copies the rear-camera video into a hidden 160×120 canvas (the model's native input size).
2. The canvas pixels join a ring buffer; the model consumes the **3 newest frames**, raw 0–255 values (the network normalises internally).
3. The GRU forward pass produces the four heads; the phone clamps them to the protocol ranges.
4. One JSON frame with only those four numbers is sent to `/ws/control/`. Camera images never leave the device.

On the server, per frame:

5. `ControlConsumer` validates and clips the payload (`sanitize_command`), then publishes it to the `simulation_render` group.
6. `RenderConsumer` forwards it to the PC page as a `control_update` frame.

Every rendered frame on the PC (`dt = min(clock.getDelta(), 0.1)`):

7. `applyControls` stores the command; fresh commands (≤ 0.5 s old) override the keyboard.
8. Longitudinal step — brake beats throttle; released pedals coast at 2 m/s²; speed is clipped to 0…20 m/s.
9. Heading step — `steering × 1.5 rad/s × min(1, speed/5)`, active above 0.5 m/s; position integrates `sin/cos(heading) × speed × dt`.
10. The invisible wall clamps `|x| ≤ 9`, scrubs speed while grinding, and emits a `wall` hazard; `Box3` overlap with traffic emits a `collision` hazard, flashes the screen red, and resets the car to the centre lane.
11. Turn-signal lamps blink with `blinkerPhase += dt × 3`.
12. The camera rides the front bumper, looking 30 m ahead — exactly the view the network was trained to drive from.
13. Four times a second the scene sends `{type: "telemetry", speed_kmh}` back to the phone; hazards are relayed immediately.

## The DAVE-2-GRU model

- **Source:** [`maxim-igenbergs/dave2-gru`](https://huggingface.co/maxim-igenbergs/dave2-gru) on Hugging Face (checkpoint `genroads`, pinned revision).
- **Architecture:** a DAVE-2 style CNN applied per-frame through `TimeDistributed`, followed by a 128-unit GRU and dense heads — a closed-loop variant of the classic NVIDIA end-to-end driving network.
- **Input:** `(batch, 3, 120, 160, 3)` — three consecutive RGB frames at 120×160, raw 0–255 pixels (normalisation is folded into the first convolution).
- **Outputs:** `steering` and `throttle` as linear regressions, clamped by the phone; `brake` and `turn_signal` heads are wired but neutral (their training data had no such labels) — the released checkpoint drives with throttle and steering only.
- **On-device conversion:** the Keras checkpoint is flattened, extended to four heads, and converted to TensorFlow.js float16 (≈ 5.6 MB, under the 10 MB budget), then served as a Django static file. A numerical parity harness compares the browser model against Keras. See [`models/README.md`](selfdriving_sim/models/README.md).

## Tests

```bash
cd selfdriving_sim
pytest                        # Django/Channels/model pipeline (in-memory layer; Redis tests skip without Redis)
npm ci && npm test            # Node suite: scene physics, inference, GRU op, model contract
npm run test:browser          # Playwright, needs `npx playwright install chromium`
```

## Limitations and future work

- The `brake` and `turn_signal` heads ship neutral; they need labelled data and fine-tuning before they drive.
- Sim-to-real gap: the phone watches an emitted screen — brightness, viewing angle, refresh-rate banding and glare all reach the network; the closed-loop test in [`docs/results.md`](selfdriving_sim/docs/results.md) includes a setup checklist.
- One straight, endless road; oncoming traffic only; no intersections, traffic lights, pedestrians or lateral road curvature.
- Kinematic bicycle-less physics (no tyre slip, no weight transfer), tuned for readable closed-loop behaviour rather than realism.
- The phone infers at ~20 Hz over Wi-Fi/ngrok; latency is dominated by the tunnel, not the model (see troubleshooting).
- Future: fine-tune the brake/turn heads on simulator recordings, curved roads and richer traffic, dataset capture from the scene itself, and a latency instrumentation overlay.

## License

Copyright © 2026 EmmettSS. All rights reserved.

This project is **not** open-source software. The repository is public so the work can be read and reviewed; copying, redistributing or reusing the code requires prior written permission from the author. Third-party components keep their own licenses (see `selfdriving_sim/client/static/client/fonts/OFL.txt` for the Vazirmatn font and the vendor READMEs for Three.js and TensorFlow.js).
