/**
 * Phone-side camera and Mock control pipeline.
 *
 * Only four numeric controls leave the phone; camera frames stay on its canvas.
 * Replace mockInference at the inference boundary in phase 4. Do not substitute
 * a model's preprocessing or output contract until that model is inspected.
 */

export const SEND_INTERVAL_MS = 50;
export const FRAME_WIDTH = 200;
export const FRAME_HEIGHT = 66;
export const MAX_RECONNECT_ATTEMPTS = 10;
export const CONNECT_TIMEOUT_MS = 10000;
export const MAX_BUFFERED_BYTES = 1024;
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

/** Use the page's host, including its port; HTTPS tunnels require WSS. */
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

export function cameraErrorMessage(error) {
  const messages = {
    NotAllowedError: "اجازهٔ دوربین داده نشد. در تنظیمات سایت در Chrome، دوربین را مجاز کنید و دوباره تلاش کنید. صفحه را مستقیماً در مرورگر باز کنید، نه داخل برنامه‌ای دیگر.",
    PermissionDeniedError: "دسترسی دوربین مسدود است. اجازهٔ دوربین را در تنظیمات سایت فعال کنید.",
    NotFoundError: "دوربینی پیدا نشد. اتصال یا تنظیمات دوربین گوشی را بررسی کنید.",
    DevicesNotFoundError: "دوربینی پیدا نشد. اتصال یا تنظیمات دوربین گوشی را بررسی کنید.",
    NotReadableError: "دوربین در دسترس نیست؛ برنامه‌های دیگری را که از دوربین استفاده می‌کنند ببندید و دوباره تلاش کنید.",
    TrackStartError: "برنامهٔ دیگری از دوربین استفاده می‌کند. آن را ببندید و دوباره تلاش کنید.",
    OverconstrainedError: "دوربین با تنظیمات درخواستی سازگار نیست. مرورگر را به‌روز کنید یا دوربین دیگری امتحان کنید.",
    SecurityError: "مرورگر دسترسی به دوربین را محدود کرده است. آدرس HTTPS را مستقیماً در Chrome باز کنید.",
    AbortError: "راه‌اندازی دوربین کامل نشد. دوباره تلاش کنید.",
  };
  return messages[error?.name] ?? "راه‌اندازی دوربین ناموفق بود. مجوز دوربین و اتصال امن HTTPS را بررسی و دوباره تلاش کنید.";
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

const ELEMENT_IDS = [
  "camera-feed", "process-canvas", "camera-prompt", "camera-title", "camera-message",
  "camera-status", "camera-resolution", "camera-dot", "btn-camera", "btn-reconnect",
  "ws-status", "speed-value", "fps-value", "btn-start", "btn-stop", "run-dot", "run-status",
  "status-message", "indicator-left", "indicator-right", "indicator-throttle",
  "indicator-brake", "throttle-value", "brake-value", "throttle-fill", "brake-fill",
  "throttle-meter", "brake-meter", "steering-wheel", "steering-value", "packet-count",
];

/** Own all streams, sockets and timers so retry/stop/navigation cannot duplicate them. */
export class DashboardController {
  constructor({ window: browserWindow, document: browserDocument }) {
    this.window = browserWindow;
    this.document = browserDocument;
    this.ui = Object.fromEntries(ELEMENT_IDS.map((id) => [id, browserDocument.getElementById(id)]));
    this.video = this.ui["camera-feed"];
    this.canvas = this.ui["process-canvas"];
    this.context = this.canvas.getContext("2d", { alpha: false });
    this.stream = null;
    this.cameraReady = false;
    this.cameraTask = null;
    this.cameraGeneration = 0;
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
    };
    this.ui["btn-start"].addEventListener("click", this.handlers.start);
    this.ui["btn-stop"].addEventListener("click", this.handlers.stop);
    this.ui["btn-camera"].addEventListener("click", this.handlers.camera);
    this.ui["btn-reconnect"].addEventListener("click", this.handlers.reconnect);
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

  setSocketStatus(text, color) {
    this.setText("ws-status", text);
    for (const name of ["red", "green", "pending"]) {
      this.ui["ws-status"].classList.toggle(name, name === color);
    }
    this.updateButtons();
  }

  updateButtons() {
    const canStart = this.cameraReady && this.wsControl?.readyState === 1
      && !this.suspended && !this.destroyed && !this.document.hidden;
    this.ui["btn-start"].disabled = !canStart || this.running;
    this.ui["btn-stop"].disabled = !this.running;
  }

  showCameraPrompt(title, message, canRetry = true) {
    this.ui["camera-prompt"].hidden = false;
    this.setText("camera-title", title);
    this.setText("camera-message", message);
    this.ui["btn-camera"].disabled = !canRetry;
    this.setText("btn-camera", canRetry ? "فعال‌سازی دوبارهٔ دوربین" : "در انتظار دسترسی…");
  }

  async init() {
    if (this.destroyed || this.document.hidden) return false;
    this.suspended = false;
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
      this.showCameraPrompt("اتصال امن لازم است", "دوربین روی IP محلی با HTTP فعال نمی‌شود. آدرس HTTPS تونل ngrok را همراه /dashboard/ مستقیماً در Chrome باز کنید.");
      this.setText("camera-status", "HTTPS لازم است");
      this.setSocketStatus("منتظر دوربین", "red");
      return false;
    }
    if (!this.window.navigator.mediaDevices?.getUserMedia) {
      this.showCameraPrompt("دوربین پشتیبانی نمی‌شود", "از نسخهٔ به‌روز Chrome اندروید و یک آدرس HTTPS استفاده کنید.");
      this.setSocketStatus("منتظر دوربین", "red");
      return false;
    }

    let stream;
    try {
      stream = await this.window.navigator.mediaDevices.getUserMedia({
        audio: false,
        video: { facingMode: { ideal: "environment" }, width: { ideal: 1280 }, height: { ideal: 720 } },
      });
      if (generation !== this.cameraGeneration || this.suspended || this.destroyed || this.document.hidden) {
        stream.getTracks().forEach((track) => track.stop());
        return false;
      }
      this.stream = stream;
      this.video.srcObject = stream;
      this.video.muted = true;
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
      return true;
    } catch (error) {
      stream?.getTracks().forEach((track) => track.stop());
      if (generation !== this.cameraGeneration) return false;
      this.stream = null;
      this.video.srcObject = null;
      this.cameraReady = false;
      this.showCameraPrompt("دوربین آماده نیست", cameraErrorMessage(error));
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
    this.updateButtons();
  }

  handleCameraLoss() {
    this.stopInferenceLoop("دوربین قطع شد؛ ارسال متوقف شد.");
    this.disconnectControlWebSocket();
    this.releaseCamera();
    this.setSocketStatus("قطع", "red");
    this.showCameraPrompt("تصویر دوربین قطع شد", "دوربین را دوباره فعال کنید. ادامهٔ ارسال به شروع دستی نیاز دارد.");
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
      this.setStatus("دوربین و ارتباط آماده‌اند. برای ارسال Mock، شروع خودران را بزنید.");
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
      this.setStatus("تلاش‌های اتصال تمام شد. سرور و ngrok را بررسی کنید و «اتصال مجدد» را بزنید.");
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

  /** Resize only for now; phase 4 supplies the model-specific tensor pipeline. */
  captureFrame() {
    if (!this.cameraReady || this.video.readyState < 2 || !this.video.videoWidth || !this.video.videoHeight) return false;
    if (!this.context) throw new Error("Canvas 2D is unavailable");
    this.context.drawImage(this.video, 0, 0, FRAME_WIDTH, FRAME_HEIGHT);
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
    this.startedAt = this.now();
    this.rate.reset(this.startedAt);
    this.setText("fps-value", "0");
    this.ui["run-dot"].classList.add("active");
    this.setText("run-status", "در حال ارسال · Mock");
    this.setStatus("فرمان‌های آزمایشی ارسال می‌شوند؛ مدل واقعی در این فاز اجرا نمی‌شود.");
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
    const output = mockInference((now - this.startedAt) / 1000);
    if (!this.sendCommand(output)) {
      this.stopInferenceLoop("ارسال فرمان ناموفق بود؛ اتصال را بررسی کنید.", { sendStop: false });
      this.wsControl?.close(4000, "Control send failed");
      return;
    }
    this.updateIndicators(output);
    this.setText("fps-value", this.rate.sample(now, true));
  }

  stopInferenceLoop(message = null, { sendStop = true } = {}) {
    const wasRunning = this.running;
    this.running = false;
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
    for (const [side, signal, label] of [["left", 1, "چپ"], ["right", 2, "راست"]]) {
      const active = output.turn_signal === signal;
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
    this.stopInferenceLoop("صفحه غیرفعال شد؛ برای ادامه پس از بازگشت، دوباره شروع را بزنید.");
    this.disconnectControlWebSocket();
    this.releaseCamera();
    this.setSocketStatus("متوقف", "red");
    this.showCameraPrompt("دوربین متوقف است", "با بازگشت به صفحه، دوربین و ارتباط دوباره آماده می‌شوند؛ ارسال خودکار شروع نمی‌شود.");
  }

  destroy() {
    this.suspend();
    this.destroyed = true;
    this.ui["btn-start"].removeEventListener("click", this.handlers.start);
    this.ui["btn-stop"].removeEventListener("click", this.handlers.stop);
    this.ui["btn-camera"].removeEventListener("click", this.handlers.camera);
    this.ui["btn-reconnect"].removeEventListener("click", this.handlers.reconnect);
    this.video.removeEventListener("error", this.handlers.videoerror);
    this.document.removeEventListener("visibilitychange", this.handlers.visibility);
    this.window.removeEventListener("pagehide", this.handlers.pagehide);
    this.window.removeEventListener("pageshow", this.handlers.pageshow);
  }
}

if (typeof document !== "undefined" && document.getElementById("dashboard")) {
  const dashboard = new DashboardController({ window, document });
  void dashboard.init();
}
