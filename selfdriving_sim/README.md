# SelfDriveSim server

Django and Django Channels server for the closed-loop self-driving simulator. A phone captures the rendered scene with its camera, runs the DAVE-2-GRU model in the browser, and sends driving commands over WebSocket. The server relays those commands to the Three.js renderer on the PC, where they drive the car through a kinematic model; speed telemetry and collision/wall hazards flow back to the phone.

This README covers the complete system: the Django/Channels server, the first-person Three.js scene at `/`, the mobile camera dashboard at `/dashboard/`, the DAVE-2-GRU model that runs in the phone's browser with TensorFlow.js, and the closed-loop physics that turns commands into motion.

**[راهنمای فارسی فاز ۵: اعمال فرمان‌ها در شبیه‌ساز، تست حلقه‌بسته و چک‌لیست پذیرش](docs/phase-5-closed-loop.md)**

**[معماری کامل سیستم](docs/architecture.md)** · **[عیب‌یابی](docs/troubleshooting.md)** · **[نتایج و سنجش‌ها](docs/results.md)**

**[راهنمای فارسی فاز ۴: مدل DAVE-2-GRU، دانلود، تبدیل و چک‌لیست پذیرش](docs/phase-4-model.md)**

**[راهنمای فارسی فاز ۳: اندروید، ngrok، داشبورد و چک‌لیست تست](docs/phase-3-mobile.md)**

## Requirements

- Ubuntu 22.04 or later, or Windows with WSL2 running Ubuntu
- Python 3.11
- A Redis server
- A current Android Chrome browser for the mobile dashboard
- Node 20+ when rebuilding Tailwind styles, running the frontend tests or the model parity check

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
| `SIMULATION_CONTROL_LOG_LEVEL` | `INFO` | Console logging for validated controls; `WARNING` silences the 20 Hz stream. |
| `DJANGO_ALLOW_IFRAME_PREVIEW` | unset | Set to `1` only for an embedded development preview; disables X-Frame-Options while `DEBUG` is true. Leave unset for ngrok. |

## Running the server

The easiest path is the launcher, which checks Redis, migrates, collects static files and prints the URLs:

```bash
./scripts/run.sh      # Linux/macOS
scripts\run.bat       # Windows
```

Manual equivalent:

```bash
python manage.py runserver 0.0.0.0:8000
```

Daphne replaces the `runserver` command, so this serves HTTP, WebSocket and the static files of the scene.

| Path | Description |
|---|---|
| `GET /` | The Three.js scene. |
| `GET /dashboard/` | The RTL mobile camera dashboard, which runs the model in the browser. |
| `GET /health/` | Returns `{"status": "ok", "phase": 6}`. |
| `/admin/` | Django admin site. |
| `/ws/control/` | WebSocket endpoint for the phone (see below). |
| `/ws/render/` | WebSocket endpoint for the Three.js page (see below). |

### Serving the static files

The scene loads `/static/simulation/js/scene.js`, so the static files have to be served.

- `python manage.py runserver` wraps the application in `ASGIStaticFilesHandler` while `DEBUG` is `True`, and serves them.
- The standalone `daphne -b 0.0.0.0 -p 8000 selfdriving_sim.asgi:application` command serves HTTP and WebSocket but **not** the static files. Use `runserver` in development, or run `python manage.py collectstatic` and let a web server serve `staticfiles/` next to Daphne.

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

The render endpoint sends each valid command to the browser as a JSON object with a `type` field and the four control fields:

```json
{"type": "control_update", "steering": 0.25, "throttle": 0.6, "brake": 0.0, "turn_signal": 1}
```

The scene ignores frames whose `type` is not `control_update`, and drops a `control_update` frame whose fields are not numbers.

### Telemetry frame (scene to phone)

Four times a second the scene reports the car's speed back through the same group:

```json
{"type": "telemetry", "speed_kmh": 47}
```

The speed is clipped to 0–400 km/h; the phone shows it next to the throttle and brake indicators. Commands are never included.

### Hazard frame (scene to phone)

When the car hits traffic or grinds the invisible wall, the scene raises a hazard. The server relays only the two known kinds and rejects anything else:

```json
{"type": "hazard", "hazard": "collision"}
```

(`"wall"` for grinding the invisible road edge.) The dashboard shows a red warning banner for four seconds — «برخورد با دیوار — مسیر را عوض کن» — and the monitor flashes red or orange, so the phone camera sees the event as well.

## The scene

`GET /` renders `simulation/templates/simulation/index.html`, which loads `simulation/static/simulation/js/scene.js` as an ES module.

### What is on screen

A **first-person view from the front bumper** of the player car — the same viewpoint the DAVE-2 model drives from. The car body itself is invisible; the camera sits at eye height above the hood and looks 30 units ahead, level with the horizon.

- A straight, **endless** three-lane road: 20 units wide, dashed lane dividers, solid edge lines and guard rails. The road group snaps forward in whole dash periods, so the drive never reaches an end.
- Trees and lamp posts on both sides, recycled one span ahead once the car passes them. The shadow-casting sun follows the car.
- Traffic cars in red, yellow, green and orange, spawned 150–200 units **ahead** in a random lane every 2.5–4.5 seconds, driving towards the car at 9–21 units per second, at most 20 at once, with a 25-unit same-lane gap. They are recycled 15 units behind the car.
- Yellow turn-signal lamps on the hood corners and a HUD overlay.
- Fog and a directional light with shadows.

### The ego car physics

The car owns `{x, z, speed, heading}` and drives towards `-z`. Per frame with `dt = min(clock.getDelta(), 0.1)`:

| Step | Rule |
|---|---|
| Longitudinal | `brake > 0.1` → `speed -= 20·brake·dt`; else `throttle > 0.1` → `speed += 8·throttle·dt`; else coast at `2 m/s²`. Clipped to `0…20 m/s`. |
| Heading | only above `0.5 m/s`: `heading += steering · 1.5 · min(1, speed/5) · dt` |
| Position | `x += sin(heading)·speed·dt`, `z -= cos(heading)·speed·dt` |
| Wall | `\|x\| ≤ 9` is clamped; grinding scrubs 15 m/s² off the speed and raises a `wall` hazard |
| Collision | `Box3` overlap with traffic → red flash, full stop, counter, 1 s freeze, then a reset to the centre lane with 2 s immunity |
| Turn signals | `blinkerPhase += dt·3`, colour toggles on `floor(phase) % 2` — three state changes per second |

`STEERING_SIGN = 1` maps the model's positive steering onto a right turn on screen; flipping it inverts the whole mapping with a one-line change (see `docs/phase-5-closed-loop.md`).

### HUD

| Element | Shows |
|---|---|
| `#ws-status` | `وصل` in green when `/ws/render/` is open, `قطع` in red when it is closed, and the attempt number while reconnecting. |
| `#speed-value` | Speed of the car in km/h. |
| `#heading-value` | Heading in degrees, `(-180, 180]`. |
| `#turn-left` / `#turn-right` | Turn arrows, blinking in phase with the 3D lamps. |
| `#collision-count` | Traffic collisions this session (red above zero). |
| `#wall-count` | Wall-grind events this session (red above zero). |
| `#fps-value` | Smoothed frame rate. |
| `#hazard-flash` | Full-screen flash: red «برخورد!» on a collision, orange «برخورد با دیوار — مسیر را عوض کن» on the wall — visible to the phone camera by design. |

### Manual controls

The keyboard drives the car **whenever no fresh phone command has arrived for 0.5 seconds**:

| Keys | Effect |
|---|---|
| `↑` / `W` | Throttle |
| `↓` / `S` | Brake |
| `←` `→` / `A` `D` | Steering |

Phone commands override the keyboard as long as they keep arriving (~20 Hz). `window.applyControls(command)` is the entry point the render socket feeds.

### WebSocket client in the page

The page connects to `ws://<host>/ws/render/` (`wss:` when the page is served over HTTPS). A closed socket is retried with an exponential backoff of 0.5, 1, 2, 4, 8 seconds capped at 15 seconds, for at most 10 attempts. After that the HUD stays red until the page is reloaded.

### Vendored Three.js

Three.js r160 (`three@0.160.1`, `build/three.module.min.js`) is committed under `simulation/static/simulation/js/vendor/`, so the page works on a LAN with no internet access. See the README in that directory for the hash and the update steps. The UMD bundles that r160 still ships are deprecated and were removed in r161, which is why the module build is used.

- `DEBUG = True` and `ALLOWED_HOSTS = ["*"]` are development settings. Change them before deploying.
- With `ALLOWED_HOSTS = ["*"]`, the WebSocket origin check accepts every origin, and handshakes without an `Origin` header are accepted too. Restrict `ALLOWED_HOSTS` before deploying so that the origin check applies.
- Redis has no authentication by default. Keep it bound to `127.0.0.1` or to a trusted network.

## Mobile dashboard (phases 3–4)

Open `/dashboard/` on the phone over **HTTPS**. For the selected Android/ngrok setup, run `ngrok http 8000` on the same PC as Django and open its HTTPS URL with `/dashboard/` appended. Open the simulator at `/` on the PC. See the [full mobile guide](docs/phase-3-mobile.md) for permissions, ngrok troubleshooting, lifecycle behavior and the real-device acceptance checklist, and the [model guide](docs/phase-4-model.md) for phase 4.

The dashboard uses a rear-camera preference, a hidden 160×120 processing canvas (the model's own frame size), a visible steering wheel, left/right turn arrows, throttle/brake indicators and start/stop controls. After camera permission succeeds it opens `/ws/control/`, using WSS when the page is HTTPS. No frames are sent until Start is pressed. The 50 ms loop runs the model on the three newest frames and sends only the four control fields; images remain on the phone. Inbound telemetry updates the speed readout; inbound hazard frames raise a red warning banner for four seconds (`parseHazard` accepts only `collision` and `wall`).

FPS measures successful browser sends, not delivery acknowledgements. Speed is the scene car's km/h, forwarded from `/ws/render/` to `/ws/control/` as `{type: "telemetry", speed_kmh}`. Received commands drive the car through the scene physics (see *The scene* above). Stop sends one best-effort zero-throttle/full-brake request. Network loss, a camera failure or a hidden page stops inference; reconnecting never restarts it automatically. Do not treat this as a guaranteed emergency stop.

Tailwind CSS and Vazirmatn are served locally with no CDN or runtime npm dependency. The compiled stylesheet is committed for direct Django use. To change it:

```bash
npm ci
npm run build:css
# Optional, in a separate terminal:
npm run watch:css
```

Edit `client/styles/dashboard.css`, not the compiled `client/static/client/css/dashboard.css`. Rebuild after changing template or JavaScript class names. Font provenance and the OFL license are in `client/static/client/fonts/`.

## The model (phase 4)

The phone runs DAVE-2-GRU itself. TensorFlow.js 4.22.0 (`@tensorflow/tfjs`, `dist/tf.min.js`) is committed under `client/static/client/js/vendor/` next to the app, the same way Three.js is vendored for the scene, so the dashboard needs no CDN and no internet access. The converted model lives in `client/static/client/models/dave2/` and is served as a static file; the dashboard element carries its URL in `data-model-url`.

Producing that model from the Hugging Face checkpoint:

```bash
python models/download_model.py    # fetch, or verify the file you copied into models/data/
python models/extend_model.py      # four heads, flattened graph, numerical checks
python models/convert.py           # TensorFlow.js with float16 weights
```

`models/README.md` documents the checkpoint's source and digests, the input and output contract, what the extension changes and why, and the parity check that compares the browser model with Keras. The dashboard loads the model before the camera, shows `MODEL LOADING` while it does, and falls back to the phase-3 Mock when the model is missing or fails to load. Only four numbers leave the phone; frames never do.

## Tests

```bash
pytest
npm ci
npm test
# Or, without installing build/test tooling:
node --test "simulation/tests/js/*.test.mjs" "client/tests/js/*.test.mjs"
```

`pytest` covers the health endpoint, scene and dashboard templates, local static assets, control logging, and both WebSocket consumers. The tests use an in-memory channel layer, so they do not need Redis. `simulation/tests/test_redis_channel_layer.py` runs only when a Redis server is reachable at `REDIS_HOST:REDIS_PORT`. Otherwise pytest skips it, and the skip reason appears in the summary.

The scene Node tests cover the pure helpers of the scene: spawn scheduling, the traffic pool limit, control sanitising, the speed and steering model, the dash scrolling and the reconnect backoff. They need Node 18 or newer and no npm packages, because `scene.js` imports the vendored Three.js module and only touches the DOM inside `boot()`.

The mobile Node tests also cover camera permission and cleanup, Mock ranges, the send-rate counter, indicators, start/stop, WSS URLs, socket reconnection, late media results and backpressure using deterministic fakes. The model tests add the frame window, tensor handling and cleanup, the model contract, the clamped command, the dashboard's model path and its fallback to Mock, and — once `models/convert.py` has run — the converted model itself and its size budget. `client/tests/js/reset-after-gru.test.mjs` checks the browser GRU against Keras reference values in `client/tests/js/fixtures/gru_fixture.json`, which `models/tests/make_gru_fixture.py` regenerates.

`pytest models` covers the model pipeline without touching the network: the pinned digests, both checkpoint loading paths, the normalization measurement, the graph restructuring, the four heads, the conversion command and budget, and an end-to-end parity run that converts a stand-in and compares it with TensorFlow.js through Node.

For browser integration tests, activate the Python virtual environment and start Redis, then:

```bash
npx playwright install --with-deps chromium
npm run test:browser
```

Playwright uses a synthetic camera and the real Django/Redis control-to-render path. It covers permission recovery, lifecycle, reconnect and portrait/landscape layouts. It does **not** replace a physical Android/ngrok camera test in front of a real monitor.

## Project layout

```
selfdriving_sim/
├── manage.py
├── pytest.ini               pytest and pytest-asyncio settings
├── requirements.txt
├── scripts/                 run.sh / run.bat launcher (Redis check, migrate, collectstatic, daphne)
├── conftest.py              shared fixtures (in-memory channel layer)
├── selfdriving_sim/         project package: settings, ASGI/WSGI entry points, root URLs
├── simulation/              scene app
│   ├── consumers.py         ControlConsumer and RenderConsumer
│   ├── routing.py           WebSocket routes
│   ├── views.py             scene page and health endpoint
│   ├── templates/simulation/index.html      HUD with heading, signals and counters
│   ├── static/simulation/js/scene.js        first-person scene: physics, traffic, collisions, hazards
│   ├── static/simulation/js/vendor/three.module.min.js
│   └── tests/               pytest suite, plus js/ for the Node tests
├── client/                  mobile client app (phase 3)
│   ├── views.py, urls.py     DashboardView at /dashboard/
│   ├── templates/client/dashboard.html
│   ├── styles/dashboard.css Tailwind source
│   ├── static/client/       compiled CSS, inference.js, vendored tf.min.js and the
│   │                        converted model in models/dave2/, plus the local font
│   └── tests/               Django, Node and Playwright tests
├── package.json             CSS build and frontend test commands
├── playwright.config.js     Chromium integration configuration
├── models/                  model pipeline (phase 4)
│   ├── dave2_config.py      pins, input/output contract, paths and size budget
│   ├── download_model.py    fetch from Hugging Face, or verify a hand-copied file
│   ├── checkpoint.py        load a checkpoint, whichever Python version wrote it
│   ├── architecture.py      rebuild a topology from the JSON inside the file
│   ├── extend_model.py      flatten the graph, fold the normalization, add four heads
│   ├── convert.py           tensorflowjs_converter with float16 weights
│   ├── parity/              Keras reference and the TensorFlow.js comparison
│   ├── data/                checkpoints and generated files, ignored by Git
│   └── tests/
├── docs/architecture.md     architecture diagram, layers, frame lifecycle (Persian)
├── docs/troubleshooting.md  Redis, camera/HTTPS, WebSocket, model, off-road, latency
├── docs/results.md          measured results, closed-loop protocol, challenges
├── docs/phase-3-mobile.md   Android/ngrok guide and acceptance checklist
└── docs/phase-4-model.md    model guide, technical decisions and acceptance checklist
```
