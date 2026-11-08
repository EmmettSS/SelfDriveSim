/**
 * Phone-side camera, DAVE-2-GRU inference and Mock control pipeline.
 *
 * Only four numeric controls leave the phone; camera frames stay on its canvas.
 * The model runs in the browser with TensorFlow.js (window.tf, vendored next to
 * this file). If the model does not load, the dashboard uses mockInference.
 *
 * The input contract is the one the checkpoint was trained on and that
 * models/dave2_config.py documents: SEQUENCE_LENGTH consecutive RGB frames of
 * FRAME_WIDTH x FRAME_HEIGHT, as raw 0..255 pixel values. The network normalizes
 * them itself, so the phone must not scale them (see preprocessFrame).
 */

import { withResetAfterGRU } from "./reset-after-gru.js";
import {
  MonitorTracker,
  SAMPLE_COLUMNS,
  SAMPLE_INTERVAL_MS,
  SAMPLE_ROWS,
  coverMapping,
  detectScreenRect,
  findMonitorFromTap,
  luminanceGrid,
  refineScreenEdges,
} from "./monitor.js";

export const SEND_INTERVAL_MS = 50;
/** How long the hazard warning banner stays on the dashboard. */
export const HAZARD_BANNER_MS = 4000;
/** Model input: SEQUENCE_LENGTH consecutive RGB frames of 160 × 120. */
export const FRAME_WIDTH = 160;
export const FRAME_HEIGHT = 120;
export const SEQUENCE_LENGTH = 3;
/**
 * Gap between the frames of one window, in ticks. The training pipeline slides a
 * window of three consecutive frames over the recorded stream, so the gap is 1.
 */
export const FRAME_STRIDE = 1;
/** Served by Django; the dashboard element's data-model-url overrides it. */
export const MODEL_URL = "/static/client/models/dave2/model.json";
export const MODEL_LOAD_TIMEOUT_MS = 20000;
export const MODEL_OUTPUTS = Object.freeze(["steering", "throttle", "brake", "turn_signal"]);
export const TURN_SIGNAL_CLASSES = 3;
export const MAX_RECONNECT_ATTEMPTS = 10;
export const CONNECT_TIMEOUT_MS = 10000;
export const MAX_BUFFERED_BYTES = 64 * 1024;
export const IDLE_COMMAND = Object.freeze({ steering: 0, throttle: 0, brake: 0, turn_signal: 0 });
export const STOP_COMMAND = Object.freeze({ steering: 0, throttle: 0, brake: 1, turn_signal: 0 });

/** A reproducible, bounded signal, independent of the camera's frame rate. */
export function mockInference(elapsedSeconds = 0) {
  const t = Number.isFinite(elapsedSeconds) ? Math.max(0, elapsedSeconds) : 0;
  const turnSequence = [0, 1, 0, 2];
  return {
    steering: Number((Math.sin(t * 0.7) * 0.65).toFixed(3)),
    throttle: Number((0.45 + Math.sin(t * 0.45) * 0.25).toFixed(3)),
    brake: 0,
    turn_signal: turnSequence[Math.floor(t / 3) % turnSequence.length],
  };
}

/** Use the page's host, including its port; a secure page needs WSS. */
export function controlSocketUrl(location) {
  return `${location.protocol === "https:" ? "wss:" : "ws:"}//${location.host}/ws/control/`;
}

/** Attempts are one-based; retries stop instead of polling indefinitely. */
export function nextReconnectDelayMs(attempt) {
  if (!Number.isInteger(attempt) || attempt < 1 || attempt > MAX_RECONNECT_ATTEMPTS) return null;
  return Math.min(500 * 2 ** (attempt - 1), 15000);
}

export function applyTelemetry(payload) {
  if (payload === null || typeof payload !== "object" || payload.type !== "telemetry") return null;
  const value = payload.speed_kmh;
  if (typeof value !== "number" || !Number.isFinite(value)) return null;
  return Math.max(0, Math.round(value));
}

/** Persian warning the dashboard shows for each hazard kind. */
export const HAZARD_MESSAGES = Object.freeze({
  collision: "⚠ برخورد! خودرو به لاین وسط برمی‌گردد",
  wall: "⚠ برخورد با دیوار — مسیر را عوض کن",
});

/**
 * Extract a validated hazard kind from a server frame, or null. The scene
 * raises these on a traffic collision and on grinding the invisible wall, so
 * the dashboard can warn the driving model to change its path.
 */
export function parseHazard(payload) {
  if (payload === null || typeof payload !== "object" || payload.type !== "hazard") return null;
  const kind = payload.hazard;
  return typeof kind === "string" && kind in HAZARD_MESSAGES ? kind : null;
}

export function cameraErrorMessage(error) {
  const messages = {
    NotAllowedError: "اجازهٔ دوربین داده نشد. در تنظیمات سایت در Chrome، دوربین را مجاز کنید و دوباره تلاش کنید. صفحه را مستقیماً در مرورگر باز کنید، نه داخل برنامه‌ای دیگر.",
    PermissionDeniedError: "دسترسی دوربین مسدود است. اجازهٔ دوربین را در تنظیمات سایت فعال کنید.",
    NotFoundError: "دوربینی پیدا نشد. اتصال یا تنظیمات دوربین گوشی را بررسی کنید.",
    DevicesNotFoundError: "دوربینی پیدا نشد. اتصال یا تنظیمات دوربین گوشی را بررسی کنید.",
    NotReadableError: "دوربین در دسترس نیست؛ برنامه‌های دیگری را که از دوربین استفاده می‌کنند ببندید و دوباره تلاش کنید.",
    TrackStartError: "برنامهٔ دیگری از دوربین استفاده می‌کند. آن را ببندید و دوباره تلاش کنید.",
    OverconstrainedError: "دوربین با تنظیمات درخواستی سازگار نیست. مرورگر را به‌روز کنید یا دوربین دیگری امتحان کنید.",
    SecurityError: "مرورگر دسترسی به دوربین را محدود کرده است. آدرس HTTPS را مستقیماً و بدون iframe باز کنید؛ روی iOS هشدار گواهی را با Show Details و گزینهٔ visit this website تأیید کنید.",
    AbortError: "راه‌اندازی دوربین کامل نشد. دوباره تلاش کنید.",
  };
  return messages[error?.name] ?? "راه‌اندازی دوربین ناموفق بود. صفحه را روی HTTPS (پورت امن runsecure) باز کنید و مجوز دوربین را بدهید.";
}

/** Count successful socket.send calls, not timer ticks or camera frames. */
export class SendRateCounter {
  constructor(now = 0) { this.reset(now); }

  reset(now) {
    this.windowStart = now;
    this.count = 0;
    this.value = 0;
  }

  sample(now, sent) {
    if (sent) this.count += 1;
    const elapsed = now - this.windowStart;
    if (elapsed >= 1000) {
      this.value = Math.round(this.count * 1000 / elapsed);
      this.count = 0;
      this.windowStart = now;
    }
    return this.value;
  }
}

/**
 * Keeps the frames one inference needs. With SEQUENCE_LENGTH 3 and FRAME_STRIDE 1
 * the window spans three ticks and holds the frames t-2, t-1 and t. Frames that
 * leave the window go to the injected dispose callback.
 */
export class FrameWindow {
  constructor({ length = SEQUENCE_LENGTH, stride = FRAME_STRIDE, dispose = () => {} } = {}) {
    this.length = length;
    this.stride = stride;
    this.span = (length - 1) * stride + 1;
    this.dispose = dispose;
    this.frames = [];
  }

  get ready() {
    return this.frames.length >= this.span;
  }

  push(frame) {
    this.frames.push(frame);
    while (this.frames.length > this.span) this.dispose(this.frames.shift());
  }

  /** Returns the frames for one inference, oldest first. The window still owns them. */
  sequence() {
    if (!this.ready) throw new Error("Frame window is not full yet");
    const start = this.frames.length - this.span;
    return Array.from({ length: this.length }, (_, i) => this.frames[start + i * this.stride]);
  }

  clear() {
    for (const frame of this.frames) this.dispose(frame);
    this.frames = [];
  }
}

/**
 * Reads the canvas as RGB float values in 0..255. The caller owns the tensor.
 *
 * No division by 255 here: the checkpoint was trained on raw pixel values and
 * normalized them in its first layer. models/extend_model.py folds that
 * normalization into the first convolution, so the converted model still expects
 * 0..255 and dividing again would darken every frame to a 255th of its value.
 */
export function preprocessFrame(tf, canvas) {
  return tf.tidy(() => {
    if (typeof HTMLCanvasElement !== "undefined" && canvas instanceof HTMLCanvasElement) {
      const ctx = canvas.getContext?.("2d");
      if (ctx && typeof ctx.getImageData === "function") {
        const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        return tf.browser.fromPixels(imgData, 3).toFloat();
      }
    }
    return tf.browser.fromPixels(canvas, 3).toFloat();
  });
}

function tensorsOf(value) {
  if (value == null) return [];
  return Array.isArray(value) ? value : [value];
}

/** Checks a loaded model against the input and output contract of models/dave2_config.py. */
export function validateModelContract(model) {
  const expected = [null, SEQUENCE_LENGTH, FRAME_HEIGHT, FRAME_WIDTH, 3];
  const inputs = model.inputs ?? [];
  if (inputs.length !== 1 || JSON.stringify(inputs[0].shape) !== JSON.stringify(expected)) {
    const found = inputs.map((input) => JSON.stringify(input.shape)).join(", ");
    throw new Error(`Model input must be ${JSON.stringify(expected)}, found ${found || "none"}`);
  }
  const names = model.outputNames ?? [];
  const missing = MODEL_OUTPUTS.filter((name) => !names.includes(name));
  if (missing.length > 0) throw new Error(`Model has no output named ${missing.join(", ")}`);
}

/**
 * Loads the converted model and checks its contract. `source` is a URL or an IO
 * handler (tests load from memory). One prediction on zeros then compiles the
 * WebGL shaders, so the first real frame is not delayed.
 */
export async function loadModel(tf, source = MODEL_URL) {
  const model = await tf.loadLayersModel(withResetAfterGRU(tf, source));
  try {
    validateModelContract(model);
    warmUp(tf, model);
  } catch (error) {
    model.dispose();
    throw error;
  }
  return model;
}

/** Runs one prediction on zeros, in the shape the model itself declares. */
export function warmUp(tf, model) {
  const shape = model.inputs[0].shape.map((size) => (size === null || size === undefined ? 1 : size));
  tf.tidy(() => model.predict(tf.zeros(shape)));
}

function namedOutputs(model, predicted) {
  const tensors = tensorsOf(predicted);
  const named = {};
  model.outputNames.forEach((name, index) => { named[name] = tensors[index]; });
  for (const name of MODEL_OUTPUTS) {
    if (!named[name]) throw new Error(`Model did not return ${name}`);
  }
  return named;
}

const roundTo3 = (value) => Number(value.toFixed(3));
const clamp = (value, low, high) => Math.min(high, Math.max(low, value));

/**
 * Turns the four heads into the protocol command. The steering and throttle heads
 * are clamped to their ranges, and the turn signal is the most probable class.
 */
export function toCommand({ steering, throttle, brake, turnProbabilities }) {
  const probabilities = Array.from(turnProbabilities);
  const values = [steering, throttle, brake, ...probabilities];
  if (probabilities.length !== TURN_SIGNAL_CLASSES || !values.every(Number.isFinite)) {
    throw new Error("Model returned a non-finite or incomplete output");
  }
  return {
    steering: roundTo3(clamp(steering, -1, 1)),
    throttle: roundTo3(clamp(throttle, 0, 1)),
    brake: roundTo3(clamp(brake, 0, 1)),
    turn_signal: probabilities.indexOf(Math.max(...probabilities)),
  };
}

/**
 * One inference on SEQUENCE_LENGTH frames, oldest first. The frame tensors stay
 * with the caller. This function disposes only the tensors it creates.
 */
export async function realInference(tf, model, frames) {
  // tidy disposes the intermediate stacked tensor and keeps only the batched input.
  const input = tf.tidy(() => tf.stack(frames).expandDims(0));
  let predicted = null;
  try {
    predicted = model.predict(input);
    const named = namedOutputs(model, predicted);
    const [steering, throttle, brake, turn] = await Promise.all(
      MODEL_OUTPUTS.map((name) => named[name].data()),
    );
    return toCommand({
      steering: steering[0],
      throttle: throttle[0],
      brake: brake[0],
      turnProbabilities: turn,
    });
  } finally {
    input.dispose();
    for (const tensor of tensorsOf(predicted)) tensor.dispose();
  }
}

const ELEMENT_IDS = [
  "camera-feed", "process-canvas", "camera-prompt", "camera-title", "camera-message",
  "camera-status", "camera-resolution", "camera-dot", "btn-camera", "btn-reconnect",
  "ws-status", "speed-value", "fps-value", "btn-start", "btn-stop", "run-dot", "run-status",
  "status-message", "indicator-left", "indicator-right", "indicator-throttle",
  "indicator-brake", "throttle-value", "brake-value", "throttle-fill", "brake-fill",
  "throttle-meter", "brake-meter", "steering-wheel", "steering-value", "packet-count",
  "mode-badge", "hazard-banner", "hazard-banner-text",
  "screen-canvas", "screen-overlay", "btn-screen-lock", "lock-status", "camera-window",
  "dashboard", "dashboard-shell", "btn-view-mode", "btn-cockpit-toggle", "btn-snap-edges", "btn-trim-minus", "btn-trim-plus", "btn-manual-crop",
  "manual-crop-layer", "manual-crop-box", "btn-manual-crop-done", "btn-manual-crop-reset",
];

/**
 * Builds the sampler that reads the camera into monitor.js's luminance grid.
 *
 * The hidden 64x48 canvas does the downscale; environments without
 * getImageData (test fakes, strict CSP) leave the sampler disabled and the
 * dashboard falls back to full-frame capture.
 */
function createScreenSampler(sampleCanvas) {
  const context = sampleCanvas?.getContext?.("2d");
  if (!context || typeof context.drawImage !== "function" || typeof context.getImageData !== "function") {
    return null;
  }
  return (video) => {
    if (!video?.videoWidth || !video?.videoHeight) return null;
    context.drawImage(video, 0, 0, SAMPLE_COLUMNS, SAMPLE_ROWS);
    const pixels = context.getImageData(0, 0, SAMPLE_COLUMNS, SAMPLE_ROWS);
    return detectScreenRect(luminanceGrid(pixels.data, SAMPLE_COLUMNS, SAMPLE_ROWS));
  };
}

/**
 * Owns all streams, sockets and timers so retry/stop/navigation cannot duplicate them.
 * `runtime` injects TensorFlow.js and the inference functions; tests pass fakes.
 * Without window.tf, or when the model fails to load, the dashboard uses Mock.
 */
export class DashboardController {
  constructor({
    window: browserWindow,
    document: browserDocument,
    modelUrl = MODEL_URL,
    runtime = {},
    logger = console,
  }) {
    this.window = browserWindow;
    this.document = browserDocument;
    this.logger = logger;
    this.modelUrl = modelUrl;
    this.runtime = {
      tf: browserWindow.tf ?? null,
      loadModel,
      preprocess: preprocessFrame,
      infer: realInference,
      ...runtime,
    };
    this.ui = Object.fromEntries(ELEMENT_IDS.map((id) => [id, browserDocument.getElementById(id)]));
    this.video = this.ui["camera-feed"];
    this.canvas = this.ui["process-canvas"];
    if (this.canvas) {
      this.canvas.removeAttribute?.("hidden");
      if (this.canvas.style) {
        this.canvas.style.position = "fixed";
        this.canvas.style.left = "-9999px";
        this.canvas.style.top = "-9999px";
        this.canvas.style.width = "160px";
        this.canvas.style.height = "120px";
        this.canvas.style.opacity = "0.01";
        this.canvas.style.pointerEvents = "none";
        this.canvas.style.zIndex = "-1000";
      }
    }
    this.context = this.canvas.getContext("2d", { alpha: false });
    // drawImage performs the resize to the model size; high quality reduces aliasing.
    if (this.context) this.context.imageSmoothingQuality = "high";
    this.stream = null;
    this.cameraReady = false;
    this.cameraTask = null;
    this.cameraGeneration = 0;
    // Monitor auto-lock: detect the bright screen, border it, crop to it.
    this.screenLockEnabled = true;
    this.screenTimer = null;
    this.tracker = new MonitorTracker();
    this.hasManualCrop = false;
    this.manualCropRect = null;
    try {
      const saved = this.window?.localStorage?.getItem?.("selfdrivesim_manual_rect");
      if (saved) {
        const parsed = JSON.parse(saved);
        if (parsed && typeof parsed.x0 === "number" && parsed.x1 > parsed.x0) {
          this.manualCropRect = parsed;
          this.hasManualCrop = true;
          this.tracker.rect = parsed;
          this.tracker.locked = true;
          this.tracker.stableCount = this.tracker.settling;
        }
      }
    } catch {}
    this.manualCropMode = false;
    this.activeDragHandle = null;
    this.dragStartPointer = null;
    this.dragStartRect = null;
    this.lastFrameStreamTime = 0;
    this.streamCanvas = null;
    this.streamCtx = null;
    this.latestMobileObstacles = [];
    this.lastMobileObstacleTime = 0;
    this.mobileObstacleScanPending = false;
    this.lastMobileScanTime = 0;
    this.screenSampler = createScreenSampler(this.ui["screen-canvas"]);
    this.cocoDetector = null;
    this.cocoLoading = false;
    this.screenSamplingPending = false;
    this.aimMode = false;
    this.bezelTrim = 0.015;
    this.latestGrid = null;
    this.wsControl = null;
    this.reconnectAttempt = 0;
    this.reconnectTimer = null;
    this.connectTimer = null;
    this.loopTimer = null;
    this.running = false;
    this.suspended = false;
    this.destroyed = false;
    this.packetCount = 0;
    this.rate = new SendRateCounter();
    // Model state: idle | loading | ready | failed. Mock runs unless the model is ready.
    this.mode = "mock";
    this.modelState = "idle";
    this.modelError = null;
    this.model = null;
    this.modelTask = null;
    this.modelTimer = null;
    this.modelGeneration = 0;
    this.frames = new FrameWindow({ dispose: (tensor) => tensor?.dispose?.() });
    this.inferencePending = false;
    this.runGeneration = 0;

    this.handlers = {
      start: () => this.startInferenceLoop(),
      stop: () => this.stopInferenceLoop(),
      camera: () => void this.init(),
      reconnect: () => {
        this.reconnectAttempt = 0;
        this.connectControlWebSocket();
      },
      visibility: () => {
        if (this.document.hidden) this.suspend();
        else void this.init();
      },
      pagehide: () => this.suspend(),
      pageshow: (event) => { if (event.persisted) void this.init(); },
      videoerror: () => { if (this.cameraReady) this.handleCameraLoss(); },
      screenLock: () => this.setScreenLock(!this.screenLockEnabled),
      manualCrop: () => this.toggleManualCrop(),
      manualCropDone: () => this.saveAndCloseManualCrop(),
      manualCropReset: () => this.resetManualCrop(),
      viewMode: () => this.toggleAimMode(),
      snapEdges: () => this.snapEdges(),
      trimMinus: () => this.adjustBezelTrim(0.015),
      trimPlus: () => this.adjustBezelTrim(-0.015),
      pointerDownOverlay: (e) => this.handleOverlayPointerDown(e),
      pointerMoveOverlay: (e) => this.handleOverlayPointerMove(e),
      pointerUpOverlay: (e) => this.handleOverlayPointerUp(e),
      screenTap: (event) => this.handleScreenTap(event),
    };
    this.ui["btn-start"].addEventListener("click", this.handlers.start);
    this.ui["btn-stop"].addEventListener("click", this.handlers.stop);
    this.ui["btn-camera"].addEventListener("click", this.handlers.camera);
    this.ui["btn-reconnect"].addEventListener("click", this.handlers.reconnect);
    this.ui["btn-screen-lock"].addEventListener("click", this.handlers.screenLock);
    this.ui["btn-manual-crop"]?.addEventListener?.("click", this.handlers.manualCrop);
    this.ui["btn-manual-crop-done"]?.addEventListener?.("click", this.handlers.manualCropDone);
    this.ui["btn-manual-crop-reset"]?.addEventListener?.("click", this.handlers.manualCropReset);
    this.ui["btn-view-mode"]?.addEventListener?.("click", this.handlers.viewMode);
    this.ui["btn-cockpit-toggle"]?.addEventListener?.("click", this.handlers.viewMode);
    this.ui["btn-snap-edges"]?.addEventListener?.("click", this.handlers.snapEdges);
    this.ui["btn-trim-minus"]?.addEventListener?.("click", this.handlers.trimMinus);
    this.ui["btn-trim-plus"]?.addEventListener?.("click", this.handlers.trimPlus);
    this.ui["screen-overlay"]?.addEventListener?.("click", this.handlers.screenTap);
    this.ui["screen-overlay"]?.addEventListener?.("pointerdown", this.handlers.pointerDownOverlay);
    this.ui["screen-overlay"]?.addEventListener?.("pointermove", this.handlers.pointerMoveOverlay);
    this.ui["screen-overlay"]?.addEventListener?.("pointerup", this.handlers.pointerUpOverlay);
    this.ui["screen-overlay"]?.addEventListener?.("pointercancel", this.handlers.pointerUpOverlay);

    const cropLayer = this.ui["manual-crop-layer"];
    if (cropLayer?.addEventListener) {
      cropLayer.addEventListener("pointerdown", (e) => this.handleLayerPointerDown(e));
      cropLayer.addEventListener("pointermove", (e) => this.handleOverlayPointerMove(e));
      cropLayer.addEventListener("pointerup", (e) => this.handleOverlayPointerUp(e));
      cropLayer.addEventListener("pointercancel", (e) => this.handleOverlayPointerUp(e));
    }
    this.video.addEventListener("error", this.handlers.videoerror);
    this.document.addEventListener("visibilitychange", this.handlers.visibility);
    this.window.addEventListener("pagehide", this.handlers.pagehide);
    this.window.addEventListener("pageshow", this.handlers.pageshow);
    this.updateButtons();
  }

  now() { return this.window.performance.now(); }
  setText(id, text) { this.ui[id].textContent = String(text); }
  setStatus(message) { this.setText("status-message", message); }

  setSpeed(kmh) {
    this.setText("speed-value", kmh);
    this.ui["speed-value"].setAttribute("aria-label", `سرعت شبیه‌ساز ${kmh} کیلومتر بر ساعت`);
  }

  clearSpeed() {
    this.setText("speed-value", "—");
    this.ui["speed-value"].setAttribute("aria-label", "سرعت هنوز دریافت نمی‌شود");
  }

  /**
   * Show the hazard banner the simulator raises on a collision or a wall
   * grind. The banner hides itself after HAZARD_BANNER_MS; overlapping
   * hazards restart the timer instead of stacking.
   */
  showHazard(kind) {
    const message = HAZARD_MESSAGES[kind];
    if (message === undefined) return;
    this.setText("hazard-banner-text", message);
    this.ui["hazard-banner"].classList.add("is-visible");
    this.window.clearTimeout(this.hazardTimer);
    this.hazardTimer = this.window.setTimeout(() => {
      this.hazardTimer = null;
      this.ui["hazard-banner"].classList.remove("is-visible");
    }, HAZARD_BANNER_MS);
  }

  hideHazard() {
    this.window.clearTimeout(this.hazardTimer);
    this.hazardTimer = null;
    this.ui["hazard-banner"].classList.remove("is-visible");
  }

  setSocketStatus(text, color) {
    this.setText("ws-status", text);
    for (const name of ["red", "green", "pending"]) {
      this.ui["ws-status"].classList.toggle(name, name === color);
    }
    this.updateButtons();
  }

  updateButtons() {
    const canStart = this.cameraReady && this.wsControl?.readyState === 1
      && !this.suspended && !this.destroyed && !this.document.hidden
      && this.modelState !== "loading";
    this.ui["btn-start"].disabled = !canStart || this.running;
    this.ui["btn-stop"].disabled = !this.running;
  }

  /**
   * Starts loading the model once. A failed load is not final: the next init()
   * (for example from the camera button) tries again.
   */
  ensureModel() {
    if (this.destroyed || this.modelState === "ready") return Promise.resolve(this.modelState === "ready");
    if (this.modelTask) return this.modelTask;
    if (!this.runtime.tf) {
      this.modelState = "failed";
      this.modelError = "TensorFlow.js did not load";
      this.logger.warn("[Model] TensorFlow.js is not loaded; using Mock");
      this.updateModelStatus();
      return Promise.resolve(false);
    }

    const generation = ++this.modelGeneration;
    this.modelState = "loading";
    this.modelError = null;
    this.updateModelStatus();
    // The async wrapper turns a synchronous throw into a rejection.
    const pending = (async () => this.runtime.loadModel(this.runtime.tf, this.modelUrl))();
    const timeout = new Promise((_, reject) => {
      this.modelTimer = this.window.setTimeout(
        () => reject(new Error(`no response after ${MODEL_LOAD_TIMEOUT_MS} ms`)),
        MODEL_LOAD_TIMEOUT_MS,
      );
    });
    this.modelTask = Promise.race([pending, timeout])
      .then((model) => {
        if (generation !== this.modelGeneration || this.destroyed) {
          model?.dispose?.();
          return false;
        }
        this.model = model;
        this.modelState = "ready";
        this.logger.info("[Model] Loaded", { url: this.modelUrl });
        this.setStatus("مدل DAVE-2-GRU بارگذاری شد. پس از اتصال دوربین و ارتباط، شروع را بزنید.");
        return true;
      }, (error) => {
        // A load that finishes after its timeout must not keep its tensors alive.
        pending.then((late) => late?.dispose?.(), () => {});
        if (generation === this.modelGeneration) {
          this.modelState = "failed";
          this.modelError = error?.message ?? String(error);
          this.logger.warn("[Model] Load failed; using Mock", this.modelError);
          this.setStatus("مدل بارگذاری نشد؛ شروع خودران فرمان‌های Mock می‌فرستد.");
        }
        return false;
      })
      .finally(() => {
        this.window.clearTimeout(this.modelTimer);
        this.modelTimer = null;
        this.modelTask = null;
        this.updateModelStatus();
      });
    return this.modelTask;
  }

  updateModelStatus() {
    const badges = { idle: "MOCK MODE", loading: "MODEL LOADING", ready: "DAVE-2 GRU", failed: "MOCK MODE" };
    this.setText("mode-badge", badges[this.modelState] ?? "MOCK MODE");
    this.updateButtons();
  }

  showCameraPrompt(title, message, canRetry = true) {
    this.ui["camera-prompt"].hidden = false;
    this.setText("camera-title", title);
    this.setText("camera-message", message);
    this.ui["btn-camera"].disabled = !canRetry;
    this.setText("btn-camera", canRetry ? "فعال‌سازی دوبارهٔ دوربین" : "در انتظار دسترسی…");
  }

  /**
   * Loads COCO-SSD object detection model in the background to recognize
   * physical computer monitors ('tv', 'laptop') directly from the camera feed.
   */
  ensureCocoDetector() {
    if (this.destroyed || this.cocoDetector || this.cocoLoading) return;
    if (typeof this.window.cocoSsd?.load !== "function") return;
    this.cocoLoading = true;
    this.window.cocoSsd.load({
      modelUrl: "/static/client/models/coco-ssd/model.json"
    }).then((model) => {
      if (this.destroyed) {
        model?.dispose?.();
        return;
      }
      this.cocoDetector = model;
      this.cocoLoading = false;
      this.logger.info("[MonitorDetector] AI Monitor Detection Model ready");
      if (this.cameraReady && this.screenLockEnabled && !this.screenTimer) {
        this.startScreenTracking();
      }
    }).catch((err) => {
      this.cocoLoading = false;
      this.logger.warn("[MonitorDetector] Fallback to CV detector:", err);
    });
  }

  async init() {
    if (this.destroyed || this.document.hidden) return false;
    this.suspended = false;
    // The model and AI monitor detector load in parallel with the camera permission request.
    void this.ensureModel();
    void this.ensureCocoDetector();
    if (await this.startCamera()) {
      if (!this.suspended && !this.destroyed && !this.document.hidden) {
        this.reconnectAttempt = 0;
        this.connectControlWebSocket();
        return true;
      }
    }
    return false;
  }

  startCamera() {
    if (this.destroyed || this.suspended || this.document.hidden) return Promise.resolve(false);
    if (this.cameraReady) return Promise.resolve(true);
    if (this.cameraTask) return this.cameraTask;
    const generation = ++this.cameraGeneration;
    const task = this.requestCamera(generation);
    this.cameraTask = task;
    // A late permission result must not overwrite a newer attempt's state.
    void task.finally(() => { if (this.cameraTask === task) this.cameraTask = null; });
    return task;
  }

  async requestCamera(generation) {
    this.showCameraPrompt("چشم‌های شبیه‌ساز", "برای دیدن مانیتور، دسترسی به دوربین عقب را تأیید کنید.", false);
    if (!this.window.isSecureContext) {
      this.showCameraPrompt("اتصال امن لازم است", "دوربین فقط روی محتوای امن (HTTPS یا localhost) کار می‌کند. سرور را با دستور python manage.py runsecure اجرا کنید و از QR روی مانیتور آدرس را باز کنید؛ روی iOS ابتدا هشدار گواهی را با Show Details و visit this website تأیید کنید. گام‌به‌گام در docs/guide.fa.md آمده است.");
      this.setText("camera-status", "HTTPS لازم است");
      this.setSocketStatus("منتظر دوربین", "red");
      return false;
    }
    if (!this.window.navigator.mediaDevices?.getUserMedia) {
      this.showCameraPrompt("دوربین پشتیبانی نمی‌شود", "از نسخهٔ به‌روز Chrome اندروید روی یک منبع امن (HTTPS یا فلگ Chrome) استفاده کنید.");
      this.setSocketStatus("منتظر دوربین", "red");
      return false;
    }

    let stream;
    try {
      try {
        stream = await this.window.navigator.mediaDevices.getUserMedia({
          audio: false,
          video: { facingMode: { ideal: "environment" }, width: { ideal: 1280 }, height: { ideal: 720 } },
        });
      } catch {
        // Fallback to basic video constraint if ideal environment camera is unavailable
        stream = await this.window.navigator.mediaDevices.getUserMedia({
          audio: false,
          video: true,
        });
      }
      if (generation !== this.cameraGeneration || this.suspended || this.destroyed || this.document.hidden) {
        stream.getTracks().forEach((track) => track.stop());
        return false;
      }
      this.stream = stream;
      this.video.muted = true;
      this.video.defaultMuted = true;
      this.video.playsInline = true;
      this.video.setAttribute("playsinline", "");
      this.video.setAttribute("webkit-playsinline", "");
      this.video.srcObject = stream;
      await this.video.play();
      if (generation !== this.cameraGeneration || this.suspended || this.destroyed || this.document.hidden) {
        stream.getTracks().forEach((track) => track.stop());
        return false;
      }
      this.cameraReady = true;
      this.ui["camera-prompt"].hidden = true;
      this.ui["camera-dot"].classList.add("active");
      this.setText("camera-status", "تصویر زنده");
      this.setText("camera-resolution", `${this.video.videoWidth} × ${this.video.videoHeight}`);
      for (const track of stream.getVideoTracks()) {
        track.addEventListener("ended", () => {
          if (generation === this.cameraGeneration) this.handleCameraLoss();
        }, { once: true });
      }
      this.updateButtons();
      this.startScreenTracking();
      this.updateLockStatus();
      return true;
    } catch (error) {
      stream?.getTracks().forEach((track) => track.stop());
      if (generation !== this.cameraGeneration) return false;
      this.stream = null;
      this.video.srcObject = null;
      this.cameraReady = false;
      this.showCameraPrompt("دوربین آماده نیست", cameraErrorMessage(error), true);
      this.setText("camera-status", "دوربین غیرفعال");
      this.setText("run-status", "منتظر دوربین");
      this.setSocketStatus("منتظر دوربین", "red");
      this.setStatus("تا فعال شدن دوربین، هیچ اتصالی برای ارسال فرمان ساخته نمی‌شود.");
      return false;
    }
  }

  releaseCamera() {
    this.cameraGeneration += 1;
    this.cameraTask = null;
    this.cameraReady = false;
    this.stream?.getTracks().forEach((track) => track.stop());
    this.stream = null;
    this.video.pause();
    this.video.srcObject = null;
    this.ui["camera-dot"].classList.remove("active");
    this.setText("camera-status", "دوربین غیرفعال");
    this.setText("camera-resolution", "CAM 01");
    this.stopScreenTracking();
    this.clearScreenOverlay();
    this.updateLockStatus();
    this.updateButtons();
  }

  handleCameraLoss() {
    this.stopInferenceLoop("دوربین قطع شد؛ ارسال متوقف شد.");
    this.disconnectControlWebSocket();
    this.releaseCamera();
    this.setSocketStatus("قطع", "red");
    this.showCameraPrompt("تصویر دوربین قطع شد", "دوربین را دوباره فعال کنید. ادامهٔ ارسال به شروع دستی نیاز دارد.");
  }

  /* --- Monitor auto-lock: detect the screen, border it, crop to it ------ */

  /**
   * Periodic monitor sampling from the live camera.
   *
   * Each sample feeds one raw detection (or miss) into the tracker, then
   * syncs the focus border on the overlay and the lock status line. The
   * sampler is null in environments without `getImageData` (test fakes,
   * strict CSP), in which case tracking simply never starts.
   */
  startScreenTracking() {
    if ((!this.screenSampler && !this.cocoDetector) || !this.screenLockEnabled) return;
    this.stopScreenTracking();
    void this.screenUpdate();
    this.screenTimer = this.window.setInterval(() => void this.screenUpdate(), SAMPLE_INTERVAL_MS);
  }

  /** Stops the sampling interval; the overlay freezes in its last state. */
  stopScreenTracking() {
    if (this.screenTimer === null) return;
    this.window.clearInterval(this.screenTimer);
    this.screenTimer = null;
  }

  sampleCameraGrid() {
    const canvas = this.ui["screen-canvas"];
    const context = canvas?.getContext?.("2d");
    if (!context || typeof context.drawImage !== "function" || typeof context.getImageData !== "function") {
      return null;
    }
    if (!this.video?.videoWidth || !this.video?.videoHeight) return null;
    context.drawImage(this.video, 0, 0, SAMPLE_COLUMNS, SAMPLE_ROWS);
    const pixels = context.getImageData(0, 0, SAMPLE_COLUMNS, SAMPLE_ROWS);
    this.latestGrid = luminanceGrid(pixels.data, SAMPLE_COLUMNS, SAMPLE_ROWS);
    return this.latestGrid;
  }

  /** One detection sample; a miss keeps the previous smoothed estimate. */
  async screenUpdate() {
    if (!this.cameraReady || (!this.screenSampler && !this.cocoDetector) || !this.screenLockEnabled) return;
    if (this.screenSamplingPending) return;
    this.screenSamplingPending = true;

    try {
      let raw = null;
      const grid = this.sampleCameraGrid();

      // 1. Deep Learning Monitor Detection: detects physical 'tv', 'laptop', 'monitor'
      if (this.cocoDetector && this.video?.videoWidth && this.video?.videoHeight) {
        try {
          const preds = await this.cocoDetector.detect(this.video, 8, 0.2);
          // 1a. Find detected keyboards to enforce lower boundary
          const keyboards = preds.filter((p) => p.class === "keyboard");
          let keyboardTopNorm = 1.0;
          if (keyboards.length > 0) {
            keyboards.sort((a, b) => a.bbox[1] - b.bbox[1]);
            keyboardTopNorm = keyboards[0].bbox[1] / this.video.videoHeight;
          }

          // 1b. Prefer dedicated tv / monitor over laptop (which includes keyboard)
          const dedicatedMonitors = preds.filter(
            (p) => (p.class === "tv" || p.class === "monitor") && p.score >= 0.25
          );
          const laptops = preds.filter((p) => p.class === "laptop" && p.score >= 0.25);
          const monitors = dedicatedMonitors.length > 0 ? dedicatedMonitors : laptops;

          if (monitors.length > 0) {
            monitors.sort((a, b) => b.score - a.score);
            const bbox = monitors[0].bbox;
            const isLaptop = monitors[0].class === "laptop";
            // In a laptop bounding box, the screen panel is only the upper ~58% (bottom 42% is the keyboard)
            const effectiveH = isLaptop ? bbox[3] * 0.58 : bbox[3];

            const rawBox = {
              x0: Math.max(0, bbox[0] / this.video.videoWidth),
              y0: Math.max(0, bbox[1] / this.video.videoHeight),
              x1: Math.min(1, (bbox[0] + bbox[2]) / this.video.videoWidth),
              y1: Math.min(1, (bbox[1] + effectiveH) / this.video.videoHeight),
            };

            // If a keyboard was found, ensure the monitor box stays strictly above it
            if (keyboardTopNorm < 0.98 && rawBox.y1 > keyboardTopNorm) {
              rawBox.y1 = Math.max(rawBox.y0 + 0.15, keyboardTopNorm - 0.02);
            }

            // Active bezel refinement snaps box past TV stand, bottom chin, keyboard valley, and outer bezels
            raw = grid ? refineScreenEdges(grid, rawBox, { maxSnapFraction: 0.45, insetPercent: this.bezelTrim }) : rawBox;
          }
        } catch {
          // Fall back to CV detector
        }
      }

      // 2. High-precision CV fallback
      if (!raw) {
        if (grid) {
          const cvRect = detectScreenRect(grid);
          raw = cvRect ? refineScreenEdges(grid, cvRect, { insetPercent: this.bezelTrim }) : null;
        } else if (this.screenSampler) {
          raw = this.screenSampler(this.video);
        }
      }

      if (this.manualCropMode || this.hasManualCrop) {
        // While user is in manual crop mode or has calibrated manual crop, preserve user's manual points and do not overwrite with auto-Otsu
        this.drawScreenOverlay();
        this.updateLockStatus();
        if (!this.running && this.wsControl?.readyState === 1) {
          this.captureFrame();
        }
        return;
      }

      this.tracker.observe(raw);
      this.drawScreenOverlay();
      this.updateLockStatus();
      if (!this.running && this.wsControl?.readyState === 1) {
        this.captureFrame();
      }
    } finally {
      this.screenSamplingPending = false;
    }
  }

  /**
   * Toggles Aim / Full View Mode: eliminates all UI occlusion over the live camera feed
   * so the entire PC monitor, borders, and camera alignment are 100% visible.
   */
  toggleAimMode(forced) {
    this.aimMode = typeof forced === "boolean" ? forced : !this.aimMode;
    const shell = this.ui["dashboard-shell"] || this.document.querySelector(".dashboard-shell") || this.document.getElementById("dashboard");
    const btnView = this.ui["btn-view-mode"];
    const btnCockpit = this.ui["btn-cockpit-toggle"];

    if (this.aimMode) {
      shell?.classList?.add?.("aim-mode");
      btnView?.setAttribute?.("aria-pressed", "true");
      if (btnView) btnView.textContent = "🚘 خروج از دید کامل";
      if (btnCockpit) btnCockpit.textContent = "🚘 خروج از دید کامل";
    } else {
      shell?.classList?.remove?.("aim-mode");
      btnView?.setAttribute?.("aria-pressed", "false");
      if (btnView) btnView.textContent = "⛶ دید کامل مانیتور";
      if (btnCockpit) btnCockpit.textContent = "⛶ دید کامل مانیتور";
    }
  }

  /**
   * Manual instant edge snapping to lock borders directly onto monitor perimeter.
   */
  snapEdges() {
    const grid = this.sampleCameraGrid() || this.latestGrid;
    if (!grid) return;
    const current = this.tracker.rect;
    if (current) {
      this.tracker.rect = refineScreenEdges(grid, current, { maxSnapFraction: 0.32, insetPercent: this.bezelTrim });
      this.tracker.stableCount = this.tracker.settling;
      this.tracker.locked = true;
      this.drawScreenOverlay();
      this.updateLockStatus();
    } else {
      const detected = detectScreenRect(grid);
      if (detected) {
        this.tracker.rect = refineScreenEdges(grid, detected, { insetPercent: this.bezelTrim });
        this.tracker.stableCount = this.tracker.settling;
        this.tracker.locked = true;
        this.drawScreenOverlay();
        this.updateLockStatus();
      }
    }
  }

  /**
   * Fine-tunes bezel margin trimming (positive delta contracts box inward, negative expands).
   */
  adjustBezelTrim(delta) {
    this.bezelTrim = Math.max(0, Math.min(0.08, (this.bezelTrim || 0.015) + delta));
    if (this.tracker.rect) {
      const r = this.tracker.rect;
      const rw = r.x1 - r.x0;
      const rh = r.y1 - r.y0;
      const dx = rw * (delta * 0.5);
      const dy = rh * (delta * 0.5);
      if (r.x1 - dx > r.x0 + dx + 0.05 && r.y1 - dy > r.y0 + dy + 0.05) {
        this.tracker.rect = {
          x0: Math.max(0, r.x0 + dx),
          y0: Math.max(0, r.y0 + dy),
          x1: Math.min(1, r.x1 - dx),
          y1: Math.min(1, r.y1 - dy),
        };
        this.drawScreenOverlay();
      }
    }
  }

  /**
   * Toggles Manual Crop Mode: opens an interactive high-contrast overlay with 6 large,
   * numbered, touch-friendly control handles (1 to 6) allowing the user to precisely
   * adjust the monitor crop borders on any mobile touchscreen.
   */
  toggleManualCrop(forced) {
    const next = typeof forced === "boolean" ? forced : !this.manualCropMode;
    if (this.manualCropMode === next) return;
    this.manualCropMode = next;
    const btn = this.ui["btn-manual-crop"];
    const layer = this.ui["manual-crop-layer"];
    const overlay = this.ui["screen-overlay"];

    if (this.manualCropMode) {
      btn?.setAttribute?.("aria-pressed", "true");
      if (btn) {
        btn.textContent = "✋ کادر دستی: روشن";
        btn.style.color = "#0b1012";
        btn.style.backgroundColor = "#f5d764";
        btn.style.borderColor = "#f5d764";
        btn.style.fontWeight = "700";
      }
      if (layer) {
        layer.removeAttribute("hidden");
        layer.hidden = false;
        layer.style.display = "block";
      }
      if (overlay) {
        overlay.style.pointerEvents = "auto";
        overlay.style.touchAction = "none";
        overlay.style.zIndex = "50";
      }

      // If no valid crop exists yet, initialize a centered default box
      if (!this.tracker.rect) {
        this.tracker.rect = { x0: 0.15, y0: 0.15, x1: 0.85, y1: 0.85 };
      }
      this.manualCropRect = { ...this.tracker.rect };
      this.hasManualCrop = true;
      this.tracker.locked = true;
      this.tracker.stableCount = this.tracker.settling;
      this.syncManualCropDOM();
      if (this.window?.requestAnimationFrame) {
        this.window.requestAnimationFrame(() => this.syncManualCropDOM());
      }
      this.drawScreenOverlay();
      this.updateLockStatus();
    } else {
      btn?.setAttribute?.("aria-pressed", "false");
      if (btn) {
        btn.textContent = "✋ کادر دستی (۶ نقطه)";
        btn.style.color = "";
        btn.style.backgroundColor = "";
        btn.style.borderColor = "";
        btn.style.fontWeight = "";
      }
      if (layer) {
        layer.setAttribute("hidden", "");
        layer.style.display = "none";
      }
      if (overlay) {
        overlay.style.pointerEvents = "none";
        overlay.style.touchAction = "";
        overlay.style.zIndex = "1";
      }
      this.activeDragHandle = null;
      this.dragStartPointer = null;
      this.dragStartRect = null;
      this.drawScreenOverlay();
      this.updateLockStatus();
    }
  }

  saveAndCloseManualCrop() {
    if (this.tracker.rect) {
      this.manualCropRect = { ...this.tracker.rect };
      this.hasManualCrop = true;
      try {
        this.window?.localStorage?.setItem?.("selfdrivesim_manual_rect", JSON.stringify(this.manualCropRect));
      } catch {}
    }
    this.toggleManualCrop(false);
    if (!this.running && this.wsControl?.readyState === 1) {
      this.captureFrame();
    }
  }

  resetManualCrop() {
    this.tracker.rect = { x0: 0.15, y0: 0.15, x1: 0.85, y1: 0.85 };
    this.manualCropRect = { ...this.tracker.rect };
    this.hasManualCrop = true;
    this.tracker.locked = true;
    this.tracker.stableCount = this.tracker.settling;
    try {
      this.window?.localStorage?.setItem?.("selfdrivesim_manual_rect", JSON.stringify(this.manualCropRect));
    } catch {}
    this.syncManualCropDOM();
    this.drawScreenOverlay();
    this.updateLockStatus();
    if (!this.running && this.wsControl?.readyState === 1) {
      this.captureFrame();
    }
  }

  syncManualCropDOM() {
    const layer = this.ui["manual-crop-layer"];
    const box = this.ui["manual-crop-box"];
    if (!layer || !box || layer.hidden) return;

    const rect = this.tracker.rect || { x0: 0.15, y0: 0.15, x1: 0.85, y1: 0.85 };
    const w = layer.clientWidth || this.window?.innerWidth || 360;
    const h = layer.clientHeight || this.window?.innerHeight || 640;
    const vw = this.video?.videoWidth || w || 640;
    const vh = this.video?.videoHeight || h || 480;
    const { scale, offsetX, offsetY } = coverMapping(vw, vh, w, h);

    const px0 = rect.x0 * vw * scale + offsetX;
    const py0 = rect.y0 * vh * scale + offsetY;
    const px1 = rect.x1 * vw * scale + offsetX;
    const py1 = rect.y1 * vh * scale + offsetY;

    box.style.left = `${Math.max(0, Math.round(px0))}px`;
    box.style.top = `${Math.max(0, Math.round(py0))}px`;
    box.style.width = `${Math.max(40, Math.round(px1 - px0))}px`;
    box.style.height = `${Math.max(40, Math.round(py1 - py0))}px`;
  }

  handleLayerPointerDown(event) {
    if (!this.manualCropMode) return;
    const layer = this.ui["manual-crop-layer"];
    const box = this.ui["manual-crop-box"];
    if (!layer || !box) return;

    const handleEl = event.target?.closest?.(".crop-handle");
    const isBox = event.target === box || event.target?.closest?.(".manual-crop-box") === box;

    let handleId = null;
    if (handleEl) {
      handleId = handleEl.dataset.handle;
    } else if (isBox) {
      handleId = "inside";
    } else {
      return;
    }

    this.activeDragHandle = handleId;
    this.dragStartPointer = { px: event.clientX, py: event.clientY };
    this.dragStartRect = { ...(this.tracker.rect || { x0: 0.15, y0: 0.15, x1: 0.85, y1: 0.85 }) };
    try {
      layer.setPointerCapture?.(event.pointerId);
    } catch {}

    if (!this._windowDragBound && this.window?.addEventListener) {
      this._windowDragBound = true;
      this._windowPointerMoveHandler = (e) => this.handleOverlayPointerMove(e);
      this._windowPointerUpHandler = (e) => this.handleOverlayPointerUp(e);
      this.window.addEventListener("pointermove", this._windowPointerMoveHandler);
      this.window.addEventListener("pointerup", this._windowPointerUpHandler);
      this.window.addEventListener("pointercancel", this._windowPointerUpHandler);
    }
    event.preventDefault?.();
  }

  /**
   * Calculates the canvas-pixel locations of the 6 main control handles
   * (Top-Left, Top-Center, Top-Right, Bottom-Left, Bottom-Center, Bottom-Right)
   * plus side midpoints for smooth responsive touch adjustments.
   */
  getOverlayHandles(width, height) {
    if (!this.tracker?.rect) return null;
    const rect = this.tracker.rect;
    const vw = this.video?.videoWidth || width || 640;
    const vh = this.video?.videoHeight || height || 480;
    const { scale, offsetX, offsetY } = coverMapping(vw, vh, width, height);
    const x0 = rect.x0 * vw * scale + offsetX;
    const y0 = rect.y0 * vh * scale + offsetY;
    const x1 = rect.x1 * vw * scale + offsetX;
    const y1 = rect.y1 * vh * scale + offsetY;
    const midX = (x0 + x1) / 2;
    const midY = (y0 + y1) / 2;

    return [
      { id: "tl", index: 1, label: "TL", name: "بالا-چپ", x: x0, y: y0, cursor: "nwse-resize" },
      { id: "tc", index: 2, label: "TC", name: "بالا-وسط", x: midX, y: y0, cursor: "ns-resize" },
      { id: "tr", index: 3, label: "TR", name: "بالا-راست", x: x1, y: y0, cursor: "nesw-resize" },
      { id: "ml", index: null, label: "ML", name: "وسط-چپ", x: x0, y: midY, cursor: "ew-resize" },
      { id: "mr", index: null, label: "MR", name: "وسط-راست", x: x1, y: midY, cursor: "ew-resize" },
      { id: "bl", index: 4, label: "BL", name: "پایین-چپ", x: x0, y: y1, cursor: "nesw-resize" },
      { id: "bc", index: 5, label: "BC", name: "پایین-وسط", x: midX, y: y1, cursor: "ns-resize" },
      { id: "br", index: 6, label: "BR", name: "پایین-راست", x: x1, y: y1, cursor: "nwse-resize" },
    ];
  }

  handleOverlayPointerDown(event) {
    if (!this.manualCropMode) return;
    const canvas = this.ui["screen-overlay"];
    const bbox = canvas?.getBoundingClientRect?.();
    const width = canvas?.clientWidth || canvas?.width || 0;
    const height = canvas?.clientHeight || canvas?.height || 0;
    if (!bbox || width < 10 || height < 10) return;

    const px = event.clientX - bbox.left;
    const py = event.clientY - bbox.top;
    const handles = this.getOverlayHandles(width, height);
    if (!handles) return;

    let hitHandle = null;
    let minDist = 34;
    for (const h of handles) {
      const dist = Math.hypot(px - h.x, py - h.y);
      if (dist < minDist) {
        minDist = dist;
        hitHandle = h.id;
      }
    }

    if (hitHandle) {
      this.activeDragHandle = hitHandle;
    } else {
      const rect = this.tracker.rect;
      const vw = this.video?.videoWidth || width || 640;
      const vh = this.video?.videoHeight || height || 480;
      const { scale, offsetX, offsetY } = coverMapping(vw, vh, width, height);
      const x0 = rect.x0 * vw * scale + offsetX;
      const y0 = rect.y0 * vh * scale + offsetY;
      const x1 = rect.x1 * vw * scale + offsetX;
      const y1 = rect.y1 * vh * scale + offsetY;
      if (px >= x0 && px <= x1 && py >= y0 && py <= y1) {
        this.activeDragHandle = "inside";
      } else {
        return;
      }
    }

    this.dragStartPointer = { px, py };
    this.dragStartRect = { ...this.tracker.rect };
    canvas.setPointerCapture?.(event.pointerId);
    event.preventDefault?.();
  }

  handleOverlayPointerMove(event) {
    if (!this.manualCropMode) return;
    const canvas = this.ui["screen-overlay"];
    const layer = this.ui["manual-crop-layer"];
    const width = canvas?.clientWidth || layer?.clientWidth || this.window?.innerWidth || 640;
    const height = canvas?.clientHeight || layer?.clientHeight || this.window?.innerHeight || 480;

    const px = event.clientX;
    const py = event.clientY;

    if (!this.activeDragHandle) {
      const handles = this.getOverlayHandles(width, height);
      if (handles && canvas) {
        const hover = handles.find((h) => Math.hypot(px - h.x, py - h.y) < 26);
        if (hover) {
          canvas.style.cursor = hover.cursor;
          return;
        }
      }
      if (canvas) canvas.style.cursor = "crosshair";
      return;
    }

    const vw = this.video?.videoWidth || width || 640;
    const vh = this.video?.videoHeight || height || 480;
    const { scale } = coverMapping(vw, vh, width, height);
    if (!scale) return;

    const dxNorm = (px - this.dragStartPointer.px) / (vw * scale);
    const dyNorm = (py - this.dragStartPointer.py) / (vh * scale);

    const s = this.dragStartRect;
    let x0 = s.x0;
    let y0 = s.y0;
    let x1 = s.x1;
    let y1 = s.y1;
    const minSpan = 0.05;

    switch (this.activeDragHandle) {
      case "tl":
        x0 = Math.max(0, Math.min(s.x1 - minSpan, s.x0 + dxNorm));
        y0 = Math.max(0, Math.min(s.y1 - minSpan, s.y0 + dyNorm));
        break;
      case "tc":
        y0 = Math.max(0, Math.min(s.y1 - minSpan, s.y0 + dyNorm));
        break;
      case "tr":
        x1 = Math.min(1, Math.max(s.x0 + minSpan, s.x1 + dxNorm));
        y0 = Math.max(0, Math.min(s.y1 - minSpan, s.y0 + dyNorm));
        break;
      case "ml":
        x0 = Math.max(0, Math.min(s.x1 - minSpan, s.x0 + dxNorm));
        break;
      case "mr":
        x1 = Math.min(1, Math.max(s.x0 + minSpan, s.x1 + dxNorm));
        break;
      case "bl":
        x0 = Math.max(0, Math.min(s.x1 - minSpan, s.x0 + dxNorm));
        y1 = Math.min(1, Math.max(s.y0 + minSpan, s.y1 + dyNorm));
        break;
      case "bc":
        y1 = Math.min(1, Math.max(s.y0 + minSpan, s.y1 + dyNorm));
        break;
      case "br":
        x1 = Math.min(1, Math.max(s.x0 + minSpan, s.x1 + dxNorm));
        y1 = Math.min(1, Math.max(s.y0 + minSpan, s.y1 + dyNorm));
        break;
      case "inside": {
        const w = s.x1 - s.x0;
        const h = s.y1 - s.y0;
        x0 = Math.max(0, Math.min(1 - w, s.x0 + dxNorm));
        y0 = Math.max(0, Math.min(1 - h, s.y0 + dyNorm));
        x1 = x0 + w;
        y1 = y0 + h;
        break;
      }
    }

    this.tracker.rect = {
      x0: Number(x0.toFixed(4)),
      y0: Number(y0.toFixed(4)),
      x1: Number(x1.toFixed(4)),
      y1: Number(y1.toFixed(4)),
    };
    this.manualCropRect = { ...this.tracker.rect };
    this.hasManualCrop = true;
    this.tracker.locked = true;
    this.tracker.stableCount = this.tracker.settling;
    this.syncManualCropDOM();
    this.drawScreenOverlay();
    if (!this.running && this.wsControl?.readyState === 1) {
      this.captureFrame();
    }
    event.preventDefault?.();
  }

  handleOverlayPointerUp(event) {
    if (this._windowDragBound && this.window?.removeEventListener) {
      this._windowDragBound = false;
      if (this._windowPointerMoveHandler) this.window.removeEventListener("pointermove", this._windowPointerMoveHandler);
      if (this._windowPointerUpHandler) {
        this.window.removeEventListener("pointerup", this._windowPointerUpHandler);
        this.window.removeEventListener("pointercancel", this._windowPointerUpHandler);
      }
    }
    if (this.activeDragHandle) {
      this.activeDragHandle = null;
      this.dragStartPointer = null;
      this.dragStartRect = null;
      try {
        this.ui["manual-crop-layer"]?.releasePointerCapture?.(event.pointerId);
        this.ui["screen-overlay"]?.releasePointerCapture?.(event.pointerId);
      } catch {}
      this.syncManualCropDOM();
      this.drawScreenOverlay();
      this.updateLockStatus();
    }
  }

  /**
   * Tapping directly on the monitor in viewfinder locks boundaries instantly.
   */
  handleScreenTap(event) {
    if (!this.cameraReady || !this.screenLockEnabled || this.manualCropMode) return;
    const canvas = this.ui["screen-overlay"];
    const rect = canvas?.getBoundingClientRect?.();
    if (!rect || rect.width < 10 || rect.height < 10) return;

    const clickX = event.clientX - rect.left;
    const clickY = event.clientY - rect.top;
    const { scale, offsetX, offsetY } = coverMapping(this.video.videoWidth, this.video.videoHeight, rect.width, rect.height);
    const normX = (clickX - offsetX) / (this.video.videoWidth * scale);
    const normY = (clickY - offsetY) / (this.video.videoHeight * scale);

    const grid = this.sampleCameraGrid() || this.latestGrid;
    if (normX >= 0 && normX <= 1 && normY >= 0 && normY <= 1 && grid) {
      const snapped = findMonitorFromTap(grid, normX, normY);
      if (snapped) {
        this.tracker.rect = snapped;
        this.tracker.stableCount = this.tracker.settling;
        this.tracker.locked = true;
        this.drawScreenOverlay();
        this.updateLockStatus();
      }
    }
  }

  /**
   * The user's on/off switch for the whole monitor-lock feature.
   *
   * Off: tracking stops, the estimate is forgotten, the overlay clears and
   * `captureFrame` falls back to the full frame. On: tracking restarts from
   * scratch (fresh detection of the monitor), as the phone may have moved.
   */
  setScreenLock(enabled) {
    this.screenLockEnabled = Boolean(enabled);
    this.ui["btn-screen-lock"].setAttribute("aria-pressed", String(this.screenLockEnabled));
    this.setText("btn-screen-lock", this.screenLockEnabled ? "قفل خودکار: روشن" : "قفل خودکار: خاموش");
    if (!this.screenLockEnabled) {
      this.hasManualCrop = false;
      this.manualCropRect = null;
      this.stopScreenTracking();
      this.tracker.reset();
      this.clearScreenOverlay();
    } else if (this.cameraReady) {
      this.startScreenTracking();
    }
    this.updateLockStatus();
  }

  /**
   * Draws the focus frame on the full-viewport overlay.
   *
   * Video coordinates are mapped exactly how the browser maps them for the
   * fixed, cover-fitted `<video>`: shared scale plus centered offsets.
   * All pixels outside the detected monitor are blacked out/masked so only the
   * real monitor is visible and active for the driver, with a sleek live HUD
   * targeting border around the monitor.
   */
  drawScreenOverlay() {
    const canvas = this.ui["screen-overlay"];
    const context = canvas?.getContext?.("2d");
    const rect = this.manualCropRect || this.tracker.cropRect() || (this.tracker.rect && (this.hasManualCrop || this.manualCropMode || (this.screenLockEnabled && this.tracker.stableCount > 0)) ? this.tracker.rect : null);
    const width = canvas?.clientWidth ?? 0;
    const height = canvas?.clientHeight ?? 0;
    // Fakes expose a recorder context without the drawing verbs, and the
    // canvas measures 0x0 until the layout has run.
    if (!context || typeof context.clearRect !== "function" || typeof context.strokeRect !== "function") return;
    if (!rect || !this.video.videoWidth || !this.video.videoHeight || width < 2 || height < 2) return;

    // Sync internal canvas buffer size to match client display size 1:1 (eliminates stretching and distortion)
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }

    const { scale, offsetX, offsetY } = coverMapping(this.video.videoWidth, this.video.videoHeight, width, height);
    const x0 = rect.x0 * this.video.videoWidth * scale + offsetX;
    const y0 = rect.y0 * this.video.videoHeight * scale + offsetY;
    const x1 = rect.x1 * this.video.videoWidth * scale + offsetX;
    const y1 = rect.y1 * this.video.videoHeight * scale + offsetY;
    const rw = x1 - x0;
    const rh = y1 - y0;

    context.clearRect(0, 0, width, height);

    // 1. Remove all camera pixels outside the detected monitor:
    // Solid cockpit mask removes room walls, ceiling, desk, keyboard, etc.
    context.fillStyle = "rgba(7, 12, 15, 0.95)";
    context.fillRect(0, 0, width, height);

    // 2. Cut out clear monitor window
    context.clearRect(x0, y0, rw, rh);

    // 3. Live glowing HUD border around the monitor
    const isLocked = this.tracker.locked;
    const primaryColor = isLocked ? "#f5d764" : "#38bdf8";

    // Bounding rectangle
    context.strokeStyle = primaryColor;
    context.lineWidth = 2;
    context.strokeRect(x0, y0, rw, rh);

    // L-shaped high-tech corner brackets
    if (typeof context.beginPath === "function" && typeof context.moveTo === "function") {
      const cornerLen = Math.max(10, Math.min(26, rw * 0.12, rh * 0.12));
      context.lineWidth = 3.5;
      context.strokeStyle = primaryColor;
      context.beginPath();
      // Top-left
      context.moveTo(x0, y0 + cornerLen); context.lineTo(x0, y0); context.lineTo(x0 + cornerLen, y0);
      // Top-right
      context.moveTo(x1 - cornerLen, y0); context.lineTo(x1, y0); context.lineTo(x1, y0 + cornerLen);
      // Bottom-right
      context.moveTo(x1 - cornerLen, y1); context.lineTo(x1, y1); context.lineTo(x1, y1 - cornerLen);
      // Bottom-left
      context.moveTo(x0 + cornerLen, y1); context.lineTo(x0, y1); context.lineTo(x0, y1 - cornerLen);
      context.stroke();

      // Center crosshair inside monitor
      const cx = x0 + rw / 2;
      const cy = y0 + rh / 2;
      const ch = 8;
      context.lineWidth = 1;
      context.strokeStyle = isLocked ? "rgba(245, 215, 100, 0.45)" : "rgba(56, 189, 248, 0.45)";
      context.beginPath();
      context.moveTo(cx - ch, cy); context.lineTo(cx + ch, cy);
      context.moveTo(cx, cy - ch); context.lineTo(cx, cy + ch);
      context.stroke();
    }

    // 4. If in manual crop mode, draw interactive 6-point handles on the border
    if (this.manualCropMode) {
      const handles = this.getOverlayHandles(width, height);
      if (handles && typeof context.beginPath === "function" && typeof context.arc === "function") {
        for (const h of handles) {
          const isCorner = ["tl", "tr", "bl", "br"].includes(h.id);
          const isEdgeCenter = ["tc", "bc"].includes(h.id);
          const isActive = this.activeDragHandle === h.id;
          const outerRadius = isActive ? 18 : (isCorner ? 15 : (isEdgeCenter ? 13 : 11));
          const innerRadius = isActive ? 11 : (isCorner ? 8.5 : (isEdgeCenter ? 7.5 : 6.5));

          // Outer halo
          context.beginPath();
          context.arc(h.x, h.y, outerRadius, 0, Math.PI * 2);
          context.fillStyle = isActive ? "rgba(245, 215, 100, 0.45)" : "rgba(245, 215, 100, 0.22)";
          context.fill();
          context.strokeStyle = "#f5d764";
          context.lineWidth = 1.5;
          context.stroke();

          // Inner solid knob
          context.beginPath();
          context.arc(h.x, h.y, innerRadius, 0, Math.PI * 2);
          context.fillStyle = isActive ? "#ffffff" : "#f5d764";
          context.fill();
          context.strokeStyle = "#070c0f";
          context.lineWidth = 2;
          context.stroke();

          // Handle index tag (1..6)
          if (h.index && typeof context.fillText === "function") {
            context.font = "bold 9px ui-monospace, sans-serif";
            context.fillStyle = "#070c0f";
            context.textAlign = "center";
            context.textBaseline = "middle";
            context.fillText(String(h.index), h.x, h.y);
          }
        }
        if (typeof context.textAlign !== "undefined") {
          context.textAlign = "start";
          context.textBaseline = "alphabetic";
        }
      }
    }

    // Live status badge above top edge
    if (typeof context.measureText === "function" && typeof context.fillText === "function") {
      context.font = "600 11px ui-monospace, monospace";
      const badgeText = this.manualCropMode
        ? "✋ کادر دستی (۶ نقطه کنترل) | MANUAL CROP"
        : (isLocked ? "● MONITOR LOCKED ✓" : "◌ ACQUIRING MONITOR...");
      const tw = context.measureText(badgeText).width;
      const bw = tw + 14;
      const bh = 20;
      const bx = Math.max(8, Math.min(x0 + 4, width - bw - 8));
      const by = y0 >= bh + 4 ? y0 - bh - 2 : y0 + 4;
      context.fillStyle = "rgba(7, 12, 15, 0.9)";
      context.fillRect(bx, by, bw, bh);
      context.strokeStyle = primaryColor;
      context.lineWidth = 1;
      context.strokeRect(bx, by, bw, bh);
      context.fillStyle = primaryColor;
      context.fillText(badgeText, bx + 7, by + 14);
    }
  }

  /** Clears the overlay entirely (lock off, camera gone). */
  clearScreenOverlay() {
    const canvas = this.ui["screen-overlay"];
    const context = canvas?.getContext?.("2d");
    if (context && typeof context.clearRect === "function") {
      const width = canvas?.clientWidth || canvas?.width || 0;
      const height = canvas?.clientHeight || canvas?.height || 0;
      context.clearRect(0, 0, width, height);
    }
  }

  /** The one-line status under the camera meta row. */
  updateLockStatus() {
    const status = this.ui["lock-status"];
    const cameraWindow = this.ui["camera-window"];
    if (!status) return;
    if (this.manualCropMode) {
      status.textContent = "تنظیم دستی کادر (۶ نقطه) · کادر روی مانیتور قفل شد";
      cameraWindow?.classList?.add?.("monitor-locked");
    } else if (!this.screenLockEnabled) {
      status.textContent = "قفل خودکار خاموش";
      cameraWindow?.classList?.remove?.("monitor-locked");
    } else if (this.tracker.locked && this.tracker.rect) {
      status.textContent = "مانیتور قفل شد ✓ · لبه‌ها منطبق شدند";
      cameraWindow?.classList?.add?.("monitor-locked");
    } else if (this.tracker.rect && this.tracker.stableCount > 0) {
      status.textContent = "مانیتور پیدا شد · تثبیت…";
      cameraWindow?.classList?.remove?.("monitor-locked");
    } else if (this.cameraReady) {
      status.textContent = "در جست‌وجوی مانیتور…";
      cameraWindow?.classList?.remove?.("monitor-locked");
    } else {
      status.textContent = "";
      cameraWindow?.classList?.remove?.("monitor-locked");
    }
  }

  connectControlWebSocket() {
    if (!this.cameraReady || this.suspended || this.destroyed || this.document.hidden) return;
    if (this.wsControl && this.wsControl.readyState < 2) return;
    this.window.clearTimeout(this.reconnectTimer);
    this.reconnectTimer = null;
    this.ui["btn-reconnect"].hidden = true;
    this.setSocketStatus("در حال اتصال…", "pending");

    let socket;
    try {
      socket = new this.window.WebSocket(controlSocketUrl(this.window.location));
      this.wsControl = socket;
    } catch {
      this.wsControl = null;
      this.scheduleReconnect();
      return;
    }

    // Stale socket events cannot close or update a replacement connection.
    socket.onopen = () => {
      if (socket !== this.wsControl) return;
      this.window.clearTimeout(this.connectTimer);
      this.connectTimer = null;
      this.reconnectAttempt = 0;
      this.setSocketStatus("متصل", "green");
      this.setText("run-status", "آمادهٔ شروع");
      this.setStatus("دوربین و ارتباط آماده‌اند. برای ارسال فرمان، شروع خودران را بزنید.");
    };
    socket.onclose = () => {
      if (socket !== this.wsControl) return;
      this.window.clearTimeout(this.connectTimer);
      this.connectTimer = null;
      this.wsControl = null;
      this.clearSpeed();
      this.stopInferenceLoop("ارتباط قطع شد؛ ارسال متوقف است. پس از اتصال دوباره، شروع را بزنید.", { sendStop: false });
      this.scheduleReconnect();
    };
    socket.onerror = () => {
      if (socket !== this.wsControl) return;
      this.setSocketStatus("خطای اتصال", "red");
      socket.close();
    };
    socket.onmessage = (event) => {
      if (socket !== this.wsControl) return;
      try {
        const reply = JSON.parse(event.data);
        if (reply?.type === "error") {
          this.stopInferenceLoop("سرور فرمان را نپذیرفت. قرارداد پیام و لاگ سرور را بررسی کنید.", { sendStop: false });
          return;
        }
        const speed = applyTelemetry(reply);
        if (speed !== null) this.setSpeed(speed);
        const hazard = parseHazard(reply);
        if (hazard !== null) this.showHazard(hazard);
      } catch {
        // Unknown/non-JSON server frames are not driving commands.
      }
    };
    this.connectTimer = this.window.setTimeout(() => {
      if (socket === this.wsControl && socket.readyState === 0) socket.close();
    }, CONNECT_TIMEOUT_MS);
    this.updateButtons();
  }

  scheduleReconnect() {
    this.setSocketStatus("قطع", "red");
    if (!this.cameraReady || this.suspended || this.destroyed || this.document.hidden) return;
    this.reconnectAttempt += 1;
    const delay = nextReconnectDelayMs(this.reconnectAttempt);
    if (delay === null) {
      this.ui["btn-reconnect"].hidden = false;
      this.setSocketStatus("اتصال ناموفق", "red");
      this.setStatus("تلاش‌های اتصال تمام شد. سرور و شبکه را بررسی کنید و «اتصال مجدد» را بزنید.");
      return;
    }
    this.setSocketStatus(`تلاش مجدد ${this.reconnectAttempt}`, "red");
    this.reconnectTimer = this.window.setTimeout(() => {
      this.reconnectTimer = null;
      this.connectControlWebSocket();
    }, delay);
  }

  disconnectControlWebSocket() {
    this.window.clearTimeout(this.reconnectTimer);
    this.window.clearTimeout(this.connectTimer);
    this.reconnectTimer = null;
    this.connectTimer = null;
    const socket = this.wsControl;
    this.wsControl = null;
    if (socket) {
      socket.onopen = socket.onclose = socket.onerror = socket.onmessage = null;
      if (socket.readyState < 2) socket.close(1000, "Dashboard inactive");
    }
    this.ui["btn-reconnect"].hidden = true;
    this.updateButtons();
  }

  /** Draws the video into the model-sized canvas, cropped to the locked monitor.
   *
   * While the tracker is locked and the auto lock is on, only the monitor
   * region reaches the model — ceiling, desk and glare fall outside the crop.
   * The source rectangle is given in normalized coordinates, so any camera
   * resolution works without recomputing anything.
   */
  captureFrame() {
    if (!this.cameraReady || this.video.readyState < 2 || !this.video.videoWidth || !this.video.videoHeight) return false;
    if (!this.context) throw new Error("Canvas 2D is unavailable");
    const crop = (this.manualCropMode || this.hasManualCrop)
      ? (this.manualCropRect || this.tracker.rect)
      : (this.screenLockEnabled ? (this.tracker.cropRect() || (this.tracker.locked && this.tracker.rect ? this.tracker.rect : null)) : null);
    const width = this.video.videoWidth;
    const height = this.video.videoHeight;
    if (crop && Number.isFinite(crop.x0) && Number.isFinite(crop.x1) && crop.x1 > crop.x0 && crop.y1 > crop.y0) {
      const sx = Math.max(0, Math.min(width - 10, crop.x0 * width));
      const sy = Math.max(0, Math.min(height - 10, crop.y0 * height));
      const sw = Math.max(10, Math.min(width - sx, (crop.x1 - crop.x0) * width));
      const sh = Math.max(10, Math.min(height - sy, (crop.y1 - crop.y0) * height));
      this.context.drawImage(this.video, sx, sy, sw, sh, 0, 0, FRAME_WIDTH, FRAME_HEIGHT);

      // Dedicated high-resolution (320x240) perception canvas for crisp streaming and AI detection
      if (!this.streamCanvas && typeof this.document?.createElement === "function") {
        this.streamCanvas = this.document.createElement("canvas");
        this.streamCanvas.width = 320;
        this.streamCanvas.height = 240;
        this.streamCtx = this.streamCanvas.getContext("2d", { willReadFrequently: true });
      }
      if (this.streamCtx) {
        this.streamCtx.drawImage(this.video, sx, sy, sw, sh, 0, 0, 320, 240);
      }
    } else {
      this.context.drawImage(this.video, 0, 0, FRAME_WIDTH, FRAME_HEIGHT);
      if (!this.streamCanvas && typeof this.document?.createElement === "function") {
        this.streamCanvas = this.document.createElement("canvas");
        this.streamCanvas.width = 320;
        this.streamCanvas.height = 240;
        this.streamCtx = this.streamCanvas.getContext("2d", { willReadFrequently: true });
      }
      if (this.streamCtx) {
        this.streamCtx.drawImage(this.video, 0, 0, 320, 240);
      }
    }

    // Stream cropped camera feed from the mobile phone to the simulator via WebSocket (live browser only)
    if (typeof HTMLVideoElement !== "undefined" && this.video instanceof HTMLVideoElement) {
      const now = this.now();
      if (!this.lastFrameStreamTime || now - this.lastFrameStreamTime >= 100) {
        // ONLY send preview frame if the control socket has ZERO buffered data
        // This ensures camera frames never compete with, delay, or backpressure driving controls
        if (this.wsControl?.readyState === 1 && (this.wsControl.bufferedAmount || 0) === 0) {
          this.lastFrameStreamTime = now;
          try {
            const streamTarget = this.streamCanvas || this.canvas;
            if (typeof streamTarget?.toDataURL === "function") {
              const dataUrl = streamTarget.toDataURL("image/jpeg", 0.60);
              const payload = { type: "camera_frame", image: dataUrl };
              if (Array.isArray(this.latestMobileObstacles) && this.latestMobileObstacles.length > 0) {
                payload.obstacles = this.latestMobileObstacles;
              }
              this.wsControl.send(JSON.stringify(payload));
            }
          } catch {}
        }
      }
    }

    // Safety check: detect if camera lens is covered or dark (finger, tape, or physical occlusion)
    // Physical blockage yields avgLum < 4.5. Require 3 consecutive dark frames (~150ms) to avoid false trips
    // when cropping to asphalt or navigating lighting changes.
    if (this.context && typeof this.context.getImageData === "function") {
      try {
        const imgData = this.context.getImageData(0, 0, FRAME_WIDTH, FRAME_HEIGHT).data;
        let sumLum = 0;
        let count = 0;
        for (let i = 0; i < imgData.length; i += 32) {
          sumLum += 0.299 * imgData[i] + 0.587 * imgData[i + 1] + 0.114 * imgData[i + 2];
          count++;
        }
        const avgLum = count > 0 ? (sumLum / count) : 255;
        if (avgLum < 4.5) {
          this.darkFrameCount = (this.darkFrameCount || 0) + 1;
        } else {
          this.darkFrameCount = 0;
        }
        this.isCameraBlocked = this.darkFrameCount >= 3;
      } catch {
        this.isCameraBlocked = false;
      }
    } else {
      this.isCameraBlocked = false;
    }

    return true;
  }

  sendCommand(output) {
    const socket = this.wsControl;
    if (socket?.readyState !== 1 || socket.bufferedAmount > MAX_BUFFERED_BYTES) return false;
    try {
      socket.send(JSON.stringify(output));
      this.packetCount += 1;
      this.setText("packet-count", this.packetCount);
      return true;
    } catch {
      return false;
    }
  }

  startInferenceLoop() {
    if (this.running || !this.cameraReady || this.wsControl?.readyState !== 1
      || this.suspended || this.destroyed || this.document.hidden) return false;
    this.running = true;
    this.mode = this.modelState === "ready" ? "model" : "mock";
    this.startedAt = this.now();
    this.rate.reset(this.startedAt);
    this.setText("fps-value", "0");
    this.ui["run-dot"].classList.add("active");
    if (this.mode === "model") {
      this.setText("run-status", "در حال ارسال · DAVE-2-GRU");
      this.setStatus("فرمان‌های مدل ارسال می‌شوند. تا پر شدن پنجرهٔ فریم‌ها، فرمان خنثی ارسال می‌شود.");
    } else {
      this.setText("run-status", "در حال ارسال · Mock");
      this.setStatus("فرمان‌های آزمایشی ارسال می‌شوند؛ مدل در دسترس نیست.");
    }
    this.loopTimer = this.window.setInterval(() => this.inferenceTick(), SEND_INTERVAL_MS);
    this.updateButtons();
    return true;
  }

  inferenceTick() {
    if (!this.running) return;
    if (this.wsControl?.readyState !== 1) {
      this.stopInferenceLoop("ارتباط قطع شد؛ ارسال متوقف است.", { sendStop: false });
      return;
    }
    if (this.wsControl.bufferedAmount > MAX_BUFFERED_BYTES) {
      this.stopInferenceLoop("شبکه کند است؛ ارسال برای جلوگیری از انباشته شدن فرمان‌ها متوقف شد.", { sendStop: false });
      this.wsControl.close(4000, "Control backpressure");
      return;
    }
    const now = this.now();
    let captured;
    try {
      captured = this.captureFrame();
    } catch {
      this.handleCameraLoss();
      this.setStatus("ثبت تصویر دوربین ناموفق بود. دوربین را دوباره فعال کنید.");
      return;
    }
    if (!captured) {
      this.setText("fps-value", this.rate.sample(now, false));
      return;
    }
    if (this.isCameraBlocked) {
      // Safety Interlock: Lens is covered/blocked -> Emergency stop immediately!
      const emergencyStopCmd = { steering: 0, throttle: 0, brake: 1, turn_signal: 0 };
      this.dispatchCommand(emergencyStopCmd, now);
      this.setStatus("⚠ لنز دوربین مسدود است (تصویر تاریک) — توقف اضطراری خودرو");
      return;
    }

    // Change 3: On-device mobile obstacle scanning directly on the phone
    if (this.cocoDetector && (!this.lastMobileScanTime || now - this.lastMobileScanTime >= 200)) {
      this.lastMobileScanTime = now;
      void this.scanMobileObstacles();
    }

    if (this.mode === "model") {
      this.modelTick(now);
      return;
    }
    this.dispatchCommand(mockInference((now - this.startedAt) / 1000), now);
  }

  /** Sends one command and updates the HUD. A failed send stops the loop. */
  dispatchCommand(output, now) {
    let finalOutput = output;
    if (Array.isArray(this.latestMobileObstacles) && this.latestMobileObstacles.length > 0) {
      const nearestAhead = this.latestMobileObstacles.find((o) => o.distanceM < 42 && o.laneIdx === 1);
      if (nearestAhead) {
        finalOutput = { ...output };
        finalOutput.throttle = Math.min(finalOutput.throttle, 0.35);
        if (nearestAhead.distanceM < 16) {
          finalOutput.brake = Math.max(finalOutput.brake, 0.75);
          finalOutput.throttle = 0;
        }
        if (Math.abs(finalOutput.steering) < 0.25) {
          finalOutput.steering = -0.45; // Evade to left lane
          finalOutput.turn_signal = 1;
        }
        this.setStatus(`⚠️ هوش مصنوعی گوشی: مانع در ${nearestAhead.distanceM} متری (مسیر روبرو) — مانور فرار`);
      }
    }
    if (!this.sendCommand(finalOutput)) {
      this.stopInferenceLoop("ارسال فرمان ناموفق بود؛ اتصال را بررسی کنید.", { sendStop: false });
      this.wsControl?.close(4000, "Control send failed");
      return false;
    }
    this.updateIndicators(finalOutput);
    this.setText("fps-value", this.rate.sample(now, true));
    return true;
  }

  /**
   * Scans the high-resolution cropped road canvas for vehicles directly on the mobile phone,
   * avoiding network lag and providing zero-delay obstacle detections.
   */
  async scanMobileObstacles() {
    if (!this.running || !this.cocoDetector || this.mobileObstacleScanPending) return;
    const target = this.streamCanvas || this.canvas;
    if (!target) return;
    this.mobileObstacleScanPending = true;
    try {
      const preds = await this.cocoDetector.detect(target, 5, 0.28);
      const valid = new Set(["car", "truck", "bus"]);
      const detected = [];
      const targetW = target.width || 320;
      const targetH = target.height || 240;
      const scaleX = 160 / targetW;
      const scaleY = 120 / targetH;

      for (const p of preds) {
        const cls = (p.class || "").toLowerCase();
        if (valid.has(cls) && (p.score || 0) >= 0.28) {
          const [px, py, pw, ph] = p.bbox;
          const bx = px * scaleX;
          const by = py * scaleY;
          const bw = pw * scaleX;
          const bh = ph * scaleY;
          if (bw < 3 || bh < 3) continue;

          const yBottom = by + bh;
          const dy = Math.max(yBottom - 59.5, 0.8);
          const distGround = 100 / dy;
          const distWidth = (1.8 * 140) / Math.max(bw, 1);
          const distM = Math.max(1, Math.min(95, Math.round(distGround * 0.60 + distWidth * 0.40)));

          const centerX = bx + bw / 2;
          let laneIdx = 1;
          if (centerX < 66) laneIdx = 0;
          else if (centerX > 94) laneIdx = 2;

          detected.push({
            source: "mobile_ai",
            class: cls,
            score: +(p.score).toFixed(2),
            distanceM: distM,
            laneIdx,
            laneName: ["چپ", "وسط", "راست"][laneIdx],
            bbox: [Math.round(bx), Math.round(by), Math.round(bw), Math.round(bh)],
          });
        }
      }
      this.latestMobileObstacles = detected;
      this.lastMobileObstacleTime = this.now();
    } catch {
      // Ignore detector errors
    } finally {
      this.mobileObstacleScanPending = false;
    }
  }

  /**
   * Model mode. Every tick adds the newest frame to the window. Until the window is
   * full, the neutral command is sent. After that, one inference runs at a time, and
   * ticks that arrive during it are skipped, so stale commands never queue up.
   */
  modelTick(now) {
    const { tf, preprocess, infer } = this.runtime;
    let sequence;
    try {
      this.frames.push(preprocess(tf, this.canvas));
      if (!this.frames.ready) {
        this.dispatchCommand(IDLE_COMMAND, now);
        return;
      }
      if (this.inferencePending) return;
      sequence = this.frames.sequence();
    } catch (error) {
      this.logger.error("[Model] Frame preprocessing failed", error);
      this.stopInferenceLoop("پردازش تصویر برای مدل ناموفق بود؛ ارسال متوقف شد.");
      return;
    }

    this.inferencePending = true;
    const generation = this.runGeneration;
    // The async wrapper turns a synchronous throw into a rejection.
    (async () => infer(tf, this.model, sequence))()
      .then((output) => {
        if (generation === this.runGeneration) this.dispatchCommand(output, this.now());
      }, (error) => {
        if (generation !== this.runGeneration) return;
        this.logger.error("[Model] Inference failed", error);
        this.stopInferenceLoop("اجرای مدل ناموفق بود؛ ارسال متوقف شد. برای تلاش دوباره، شروع را بزنید.");
      })
      .finally(() => {
        if (generation === this.runGeneration) this.inferencePending = false;
      });
  }

  stopInferenceLoop(message = null, { sendStop = true } = {}) {
    const wasRunning = this.running;
    this.running = false;
    // Results of an inference still in flight belong to the stopped run and are dropped.
    this.runGeneration += 1;
    this.inferencePending = false;
    this.frames.clear();
    this.window.clearInterval(this.loopTimer);
    this.loopTimer = null;
    // Best effort only: a socket send is not an acknowledgement from the renderer.
    const stopSent = wasRunning && sendStop && this.sendCommand(STOP_COMMAND);
    this.updateIndicators(stopSent ? STOP_COMMAND : IDLE_COMMAND);
    this.rate.reset(this.now());
    this.setText("fps-value", "0");
    this.ui["run-dot"].classList.remove("active");
    this.setText("run-status", "ارسال متوقف");
    let status = message ?? (stopSent
      ? "ارسال متوقف شد؛ گاز صفر و ترمز کامل درخواست شد (بدون تأیید دریافت)."
      : "حلقهٔ ارسال متوقف است.");
    if (wasRunning && !stopSent) status += " رسیدن فرمان توقف تضمین نمی‌شود.";
    this.setStatus(status);
    this.updateButtons();
  }

  updateIndicators(output) {
    const leftSignal = output.turn_signal === 1 || (output.turn_signal === 0 && output.steering <= -0.15);
    const rightSignal = output.turn_signal === 2 || (output.turn_signal === 0 && output.steering >= 0.15);
    for (const [side, active, label] of [["left", leftSignal, "چپ"], ["right", rightSignal, "راست"]]) {
      this.ui[`indicator-${side}`].classList.toggle("active", active);
      this.ui[`indicator-${side}`].setAttribute("aria-label", `راهنمای ${label} ${active ? "روشن" : "خاموش"}`);
    }
    for (const pedal of ["throttle", "brake"]) {
      const value = Math.round(output[pedal] * 100);
      this.ui[`indicator-${pedal}`].classList.toggle("active", output[pedal] > 0);
      this.setText(`${pedal}-value`, `${value}%`);
      this.ui[`${pedal}-fill`].style.height = `${value}%`;
      this.ui[`${pedal}-meter`].setAttribute("aria-valuenow", String(value));
    }
    const steering = `${output.steering >= 0 ? "+" : ""}${output.steering.toFixed(2)}`;
    this.setText("steering-value", steering);
    this.ui["steering-wheel"].style.setProperty("--steering-angle", `${output.steering * 120}deg`);
    this.ui["steering-wheel"].setAttribute("aria-label", `فرمان ${steering} در بازهٔ منفی یک تا یک`);
  }

  suspend() {
    this.suspended = true;
    this.hideHazard();
    this.stopInferenceLoop("صفحه غیرفعال شد؛ برای ادامه پس از بازگشت، دوباره شروع را بزنید.");
    this.disconnectControlWebSocket();
    this.releaseCamera();
    this.setSocketStatus("متوقف", "red");
    this.showCameraPrompt("دوربین متوقف است", "با بازگشت به صفحه، دوربین و ارتباط دوباره آماده می‌شوند؛ ارسال خودکار شروع نمی‌شود.");
  }

  destroy() {
    this.suspend();
    this.destroyed = true;
    this.modelGeneration += 1;
    this.model?.dispose?.();
    this.model = null;
    this.cocoDetector?.dispose?.();
    this.cocoDetector = null;
    this.streamCanvas = null;
    this.streamCtx = null;
    this.latestMobileObstacles = [];
    this.ui["btn-start"].removeEventListener("click", this.handlers.start);
    this.ui["btn-stop"].removeEventListener("click", this.handlers.stop);
    this.ui["btn-camera"].removeEventListener("click", this.handlers.camera);
    this.ui["btn-reconnect"].removeEventListener("click", this.handlers.reconnect);
    this.ui["btn-screen-lock"].removeEventListener("click", this.handlers.screenLock);
    this.ui["btn-manual-crop"]?.removeEventListener?.("click", this.handlers.manualCrop);
    this.ui["screen-overlay"]?.removeEventListener?.("pointerdown", this.handlers.pointerDownOverlay);
    this.ui["screen-overlay"]?.removeEventListener?.("pointermove", this.handlers.pointerMoveOverlay);
    this.ui["screen-overlay"]?.removeEventListener?.("pointerup", this.handlers.pointerUpOverlay);
    this.ui["screen-overlay"]?.removeEventListener?.("pointercancel", this.handlers.pointerUpOverlay);
    this.ui["screen-overlay"]?.removeEventListener?.("click", this.handlers.screenTap);
    this.video.removeEventListener("error", this.handlers.videoerror);
    this.document.removeEventListener("visibilitychange", this.handlers.visibility);
    this.window.removeEventListener("pagehide", this.handlers.pagehide);
    this.window.removeEventListener("pageshow", this.handlers.pageshow);
  }
}

if (typeof document !== "undefined") {
  const dashboardElement = document.getElementById("dashboard") || document.getElementById("dashboard-shell");
  if (dashboardElement) {
    const dashboard = new DashboardController({
      window,
      document,
      modelUrl: dashboardElement.dataset.modelUrl || MODEL_URL,
    });
    if (typeof window !== "undefined") {
      window.dashboardApp = dashboard;
    }
    void dashboard.init();
  }
}
