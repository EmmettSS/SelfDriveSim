/** Run with npm test, or node --test client/tests/js/*.test.mjs. */
import assert from "node:assert/strict";
import test from "node:test";

import {
  CONNECT_TIMEOUT_MS,
  DashboardController,
  FRAME_HEIGHT,
  FRAME_WIDTH,
  MAX_BUFFERED_BYTES,
  MAX_RECONNECT_ATTEMPTS,
  SEND_INTERVAL_MS,
  STOP_COMMAND,
  SendRateCounter,
  cameraErrorMessage,
  controlSocketUrl,
  mockInference,
  nextReconnectDelayMs,
} from "../../static/client/js/inference.js";

class FakeClock {
  now = 0;
  nextId = 0;
  tasks = new Map();
  add(fn, delay, repeat = false) {
    const id = ++this.nextId;
    this.tasks.set(id, { fn, delay, repeat, at: this.now + delay });
    return id;
  }
  clear(id) { this.tasks.delete(id); }
  advance(ms) {
    const target = this.now + ms;
    while (true) {
      const next = [...this.tasks.entries()].sort((a, b) => a[1].at - b[1].at)[0];
      if (!next || next[1].at > target) break;
      const [id, task] = next;
      this.now = task.at;
      if (task.repeat) task.at += task.delay;
      else this.tasks.delete(id);
      task.fn();
    }
    this.now = target;
  }
  intervals() { return [...this.tasks.values()].filter((task) => task.repeat); }
}

class FakeClassList {
  names = new Set();
  add(name) { this.names.add(name); }
  remove(name) { this.names.delete(name); }
  contains(name) { return this.names.has(name); }
  toggle(name, on = !this.contains(name)) {
    if (on) this.add(name);
    else this.remove(name);
    return on;
  }
}

class FakeElement extends EventTarget {
  textContent = "";
  disabled = false;
  hidden = false;
  srcObject = null;
  readyState = 4;
  videoWidth = 1280;
  videoHeight = 720;
  classList = new FakeClassList();
  attributes = {};
  draws = [];
  style = { setProperty(name, value) { this[name] = value; } };
  getContext() { return { drawImage: (...args) => this.draws.push(args) }; }
  setAttribute(name, value) { this.attributes[name] = value; }
  async play() {}
  pause() {}
}

class FakeDocument extends EventTarget {
  hidden = false;
  elements = new Map();
  getElementById(id) {
    if (!this.elements.has(id)) this.elements.set(id, new FakeElement());
    return this.elements.get(id);
  }
}

class FakeTrack extends EventTarget {
  stopped = false;
  stop() { this.stopped = true; }
}

class FakeStream {
  tracks = [new FakeTrack()];
  getTracks() { return this.tracks; }
  getVideoTracks() { return this.tracks; }
}

class FakeSocket {
  readyState = 0;
  bufferedAmount = 0;
  sent = [];
  failSend = false;
  constructor(url) { this.url = url; }
  open() { this.readyState = 1; this.onopen?.(); }
  close(code = 1000) { this.readyState = 3; this.closeCode = code; this.onclose?.({ code }); }
  send(data) {
    if (this.failSend || this.readyState !== 1) throw new Error("socket unavailable");
    this.sent.push(JSON.parse(data));
  }
  message(data) { this.onmessage?.({ data }); }
}

function harness({ secure = true, getUserMedia } = {}) {
  const clock = new FakeClock();
  const document = new FakeDocument();
  const window = new EventTarget();
  const sockets = [];
  const mediaCalls = [];
  const stream = new FakeStream();
  Object.assign(window, {
    isSecureContext: secure,
    performance: { now: () => clock.now },
    location: { protocol: "https:", host: "sim.ngrok-free.app" },
    navigator: { mediaDevices: { getUserMedia: async (constraints) => {
      mediaCalls.push(constraints);
      return getUserMedia ? getUserMedia(constraints) : stream;
    } } },
    WebSocket: class extends FakeSocket {
      constructor(url) { super(url); sockets.push(this); }
    },
    setInterval: (fn, ms) => clock.add(fn, ms, true),
    clearInterval: (id) => clock.clear(id),
    setTimeout: (fn, ms) => clock.add(fn, ms),
    clearTimeout: (id) => clock.clear(id),
  });
  const controller = new DashboardController({ window, document });
  return { controller, window, document, clock, sockets, mediaCalls, stream, ui: controller.ui };
}

async function connected(options) {
  const h = harness(options);
  assert.equal(await h.controller.init(), true);
  h.sockets[0].open();
  return h;
}

test("Mock signals stay within the four-field protocol over a full minute", () => {
  for (let tick = 0; tick < 1200; tick += 1) {
    const output = mockInference(tick * 0.05);
    assert.deepEqual(Object.keys(output), ["steering", "throttle", "brake", "turn_signal"]);
    assert.ok(Number.isFinite(output.steering) && Math.abs(output.steering) <= 1);
    assert.ok(output.throttle >= 0 && output.throttle <= 1);
    assert.equal(output.brake, 0);
    assert.ok([0, 1, 2].includes(output.turn_signal));
  }
  assert.deepEqual(mockInference(3), mockInference(3));
  assert.notDeepEqual(mockInference(1), mockInference(2));
});

test("turn signals cycle off, left, off, right without overlapping", () => {
  assert.deepEqual([0, 3, 6, 9, 12].map((t) => mockInference(t).turn_signal), [0, 1, 0, 2, 0]);
  for (const time of [NaN, Infinity, -Infinity, -1]) {
    assert.deepEqual(mockInference(time), mockInference(0));
  }
});

test("WebSocket URLs use the current host, port and secure scheme", () => {
  assert.equal(controlSocketUrl({ protocol: "http:", host: "192.168.1.2:8000" }), "ws://192.168.1.2:8000/ws/control/");
  assert.equal(controlSocketUrl({ protocol: "https:", host: "example.ngrok-free.app" }), "wss://example.ngrok-free.app/ws/control/");
  assert.equal(controlSocketUrl({ protocol: "https:", host: "[::1]:8443" }), "wss://[::1]:8443/ws/control/");
});

test("backoff doubles from 500 ms, caps at 15 s and ends after ten retries", () => {
  assert.deepEqual(Array.from({ length: 10 }, (_, i) => nextReconnectDelayMs(i + 1)), [500, 1000, 2000, 4000, 8000, 15000, 15000, 15000, 15000, 15000]);
  for (const attempt of [0, -1, 11, 0.5, NaN, Infinity]) assert.equal(nextReconnectDelayMs(attempt), null);
});

test("camera errors provide permission, busy-camera and missing-camera guidance", () => {
  assert.match(cameraErrorMessage({ name: "NotAllowedError" }), /Chrome/);
  assert.match(cameraErrorMessage({ name: "NotReadableError" }), /برنامه/);
  assert.match(cameraErrorMessage({ name: "NotFoundError" }), /پیدا نشد/);
  assert.match(cameraErrorMessage(null), /HTTPS/);
});

test("FPS counts successful sends over elapsed time, not timer frequency", () => {
  const counter = new SendRateCounter();
  for (let i = 1; i <= 20; i += 1) counter.sample(i * 50, i % 2 === 0);
  assert.equal(counter.value, 10);
  assert.equal(counter.sample(2000, false), 0);
  counter.reset(2000);
  for (let i = 1; i <= 20; i += 1) counter.sample(2000 + i * 50, true);
  assert.equal(counter.value, 20);
  counter.reset(5000);
  assert.equal(counter.sample(7000, true), 1);
});

test("init requests the rear camera first, then WSS, but never starts sending", async () => {
  const h = harness();
  assert.equal(h.ui["btn-start"].disabled, true);
  assert.equal(h.sockets.length, 0);
  await h.controller.init();
  assert.deepEqual(h.mediaCalls[0], {
    audio: false,
    video: { facingMode: { ideal: "environment" }, width: { ideal: 1280 }, height: { ideal: 720 } },
  });
  assert.equal(h.ui["camera-feed"].srcObject, h.stream);
  assert.equal(h.ui["camera-prompt"].hidden, true);
  assert.equal(h.ui["btn-start"].disabled, true, "handshake still pending");
  assert.equal(h.sockets[0].url, "wss://sim.ngrok-free.app/ws/control/");
  h.sockets[0].open();
  assert.equal(h.ui["btn-start"].disabled, false);
  assert.equal(h.ui["ws-status"].classList.contains("green"), true);
  h.clock.advance(2000);
  assert.equal(h.sockets[0].sent.length, 0);
});

test("insecure HTTP and unsupported media do not create a socket", async () => {
  const insecure = harness({ secure: false });
  assert.equal(await insecure.controller.init(), false);
  assert.equal(insecure.mediaCalls.length, 0);
  assert.equal(insecure.sockets.length, 0);
  assert.match(insecure.ui["camera-message"].textContent, /HTTPS/);
  const unsupported = harness();
  unsupported.window.navigator.mediaDevices = undefined;
  assert.equal(await unsupported.controller.init(), false);
  assert.equal(unsupported.sockets.length, 0);
});

test("denied permissions are recoverable and do not open WebSocket early", async () => {
  let deny = true;
  const stream = new FakeStream();
  const h = harness({ getUserMedia: () => {
    if (deny) throw Object.assign(new Error("denied"), { name: "NotAllowedError" });
    return stream;
  } });
  assert.equal(await h.controller.init(), false);
  assert.equal(h.sockets.length, 0);
  assert.equal(h.ui["btn-start"].disabled, true);
  assert.equal(h.ui["btn-camera"].disabled, false);
  assert.match(h.ui["camera-message"].textContent, /اجازه/);
  deny = false;
  assert.equal(await h.controller.init(), true);
  assert.equal(h.sockets.length, 1);
});

test("duplicate init calls share a pending camera request and one socket", async () => {
  let resolve;
  const h = harness({ getUserMedia: () => new Promise((done) => { resolve = done; }) });
  const first = h.controller.init();
  const second = h.controller.init();
  assert.equal(h.mediaCalls.length, 1);
  resolve(h.stream);
  await Promise.all([first, second]);
  assert.equal(h.sockets.length, 1);
});

test("a late camera permission result after suspension is released", async () => {
  let resolve;
  const h = harness({ getUserMedia: () => new Promise((done) => { resolve = done; }) });
  const pending = h.controller.init();
  h.controller.suspend();
  resolve(h.stream);
  assert.equal(await pending, false);
  assert.equal(h.stream.tracks[0].stopped, true);
  assert.equal(h.sockets.length, 0);
  assert.equal(h.controller.cameraReady, false);
});

test("a failed video.play releases every acquired camera track", async () => {
  const h = harness();
  h.ui["camera-feed"].play = async () => { throw new Error("autoplay failed"); };
  assert.equal(await h.controller.init(), false);
  assert.equal(h.stream.tracks[0].stopped, true);
  assert.equal(h.ui["camera-feed"].srcObject, null);
  assert.equal(h.sockets.length, 0);
});

test("captureFrame waits for video data and draws exactly 200 × 66", async () => {
  const h = await connected();
  h.ui["camera-feed"].readyState = 1;
  assert.equal(h.controller.captureFrame(), false);
  h.ui["camera-feed"].readyState = 4;
  assert.equal(h.controller.captureFrame(), true);
  assert.deepEqual(h.ui["process-canvas"].draws[0], [h.ui["camera-feed"], 0, 0, FRAME_WIDTH, FRAME_HEIGHT]);
});

test("Start is guarded, idempotent and schedules exactly one 50 ms loop", async () => {
  const h = harness();
  assert.equal(h.controller.startInferenceLoop(), false);
  await h.controller.init();
  assert.equal(h.controller.startInferenceLoop(), false);
  h.sockets[0].open();
  assert.equal(h.controller.startInferenceLoop(), true);
  assert.equal(h.controller.startInferenceLoop(), false);
  assert.equal(h.clock.intervals().length, 1);
  assert.equal(h.clock.intervals()[0].delay, SEND_INTERVAL_MS);
  h.clock.advance(1000);
  assert.equal(h.sockets[0].sent.length, 20);
  assert.equal(h.ui["fps-value"].textContent, "20");
  assert.equal(h.ui["packet-count"].textContent, "20");
  assert.equal(h.ui["btn-start"].disabled, true);
  assert.equal(h.ui["btn-stop"].disabled, false);
});

test("Stop clears the loop, sends one final brake request and is idempotent", async () => {
  const h = await connected();
  h.controller.startInferenceLoop();
  h.clock.advance(1000);
  h.controller.stopInferenceLoop();
  assert.deepEqual(h.sockets[0].sent.at(-1), STOP_COMMAND);
  assert.equal(h.sockets[0].sent.length, 21);
  assert.equal(h.clock.intervals().length, 0);
  assert.equal(h.ui["brake-value"].textContent, "100%");
  assert.equal(h.ui["fps-value"].textContent, "0");
  h.controller.stopInferenceLoop();
  h.clock.advance(5000);
  assert.equal(h.sockets[0].sent.length, 21);
  assert.equal(h.ui["btn-start"].disabled, false);
  assert.equal(h.ui["btn-stop"].disabled, true);
});

test("HUD rotates the steering wheel, fills pedals, and switches each arrow", async () => {
  const h = await connected();
  h.controller.updateIndicators({ steering: -0.5, throttle: 0.61, brake: 0, turn_signal: 1 });
  assert.equal(h.ui["steering-wheel"].style["--steering-angle"], "-60deg");
  assert.equal(h.ui["steering-value"].textContent, "-0.50");
  assert.equal(h.ui["throttle-fill"].style.height, "61%");
  assert.equal(h.ui["throttle-meter"].attributes["aria-valuenow"], "61");
  assert.equal(h.ui["indicator-throttle"].classList.contains("active"), true);
  assert.equal(h.ui["indicator-brake"].classList.contains("active"), false);
  assert.equal(h.ui["indicator-left"].classList.contains("active"), true);
  assert.equal(h.ui["indicator-right"].classList.contains("active"), false);
  h.controller.updateIndicators({ steering: 0.5, throttle: 0, brake: 1, turn_signal: 2 });
  assert.equal(h.ui["indicator-left"].classList.contains("active"), false);
  assert.equal(h.ui["indicator-right"].classList.contains("active"), true);
  assert.equal(h.ui["indicator-throttle"].classList.contains("active"), false);
  assert.equal(h.ui["indicator-brake"].classList.contains("active"), true);
});

test("reconnect pauses inference, uses backoff and requires a manual restart", async () => {
  const h = await connected();
  h.controller.startInferenceLoop();
  h.clock.advance(50);
  h.sockets[0].close(1006);
  assert.equal(h.controller.running, false);
  assert.equal(h.ui["fps-value"].textContent, "0");
  assert.equal(h.ui["ws-status"].classList.contains("red"), true);
  h.clock.advance(499);
  assert.equal(h.sockets.length, 1);
  h.clock.advance(1);
  assert.equal(h.sockets.length, 2);
  h.sockets[1].open();
  h.clock.advance(1000);
  assert.equal(h.sockets[1].sent.length, 0);
  assert.equal(h.clock.intervals().length, 0);
  assert.equal(h.ui["btn-start"].disabled, false);
  h.controller.startInferenceLoop();
  h.clock.advance(50);
  assert.equal(h.sockets[1].sent.length, 1);
});

test("retry exhaustion presents a manual reconnect and no remaining timer", async () => {
  const h = await connected();
  h.sockets[0].close(1006);
  for (let attempt = 1; attempt <= MAX_RECONNECT_ATTEMPTS; attempt += 1) {
    h.clock.advance(nextReconnectDelayMs(attempt));
    h.sockets.at(-1).close(1006);
  }
  assert.equal(h.sockets.length, 11);
  assert.equal(h.ui["btn-reconnect"].hidden, false);
  assert.equal(h.clock.tasks.size, 0);
  h.ui["btn-reconnect"].dispatchEvent(new Event("click"));
  assert.equal(h.sockets.length, 12);
  h.sockets.at(-1).open();
  assert.equal(h.ui["btn-reconnect"].hidden, true);
  assert.equal(h.controller.reconnectAttempt, 0);
});

test("a hanging handshake times out, and a socket error retries once", async () => {
  const h = harness();
  await h.controller.init();
  h.clock.advance(CONNECT_TIMEOUT_MS);
  assert.equal(h.sockets[0].readyState, 3);
  assert.equal(h.controller.reconnectAttempt, 1);
  h.clock.advance(500);
  h.sockets[1].onerror();
  assert.equal(h.controller.reconnectAttempt, 2);
  assert.equal(h.clock.tasks.size, 1);
});

test("stale socket callbacks cannot corrupt a replacement connection", async () => {
  const h = await connected();
  const staleClose = h.sockets[0].onclose;
  const staleOpen = h.sockets[0].onopen;
  h.sockets[0].close(1006);
  h.clock.advance(500);
  h.sockets[1].open();
  staleClose();
  staleOpen();
  assert.equal(h.controller.wsControl, h.sockets[1]);
  assert.equal(h.ui["ws-status"].classList.contains("green"), true);
  assert.equal(h.clock.tasks.size, 0);
});

test("backpressure stops the loop instead of queuing stale driving commands", async () => {
  const h = await connected();
  h.controller.startInferenceLoop();
  h.sockets[0].bufferedAmount = MAX_BUFFERED_BYTES + 1;
  h.clock.advance(50);
  assert.equal(h.sockets[0].sent.length, 0);
  assert.equal(h.sockets[0].closeCode, 4000);
  assert.equal(h.controller.running, false);
  assert.equal(h.clock.intervals().length, 0);
});

test("send exceptions stop inference and close the failed socket", async () => {
  const h = await connected();
  h.sockets[0].failSend = true;
  h.controller.startInferenceLoop();
  h.clock.advance(50);
  assert.equal(h.controller.running, false);
  assert.equal(h.sockets[0].readyState, 3);
  assert.equal(h.sockets[0].sent.length, 0);
});

test("server rejection stops the loop; invalid JSON is ignored", async () => {
  const h = await connected();
  h.controller.startInferenceLoop();
  h.sockets[0].message("not JSON");
  assert.equal(h.controller.running, true);
  h.sockets[0].message('{"type":"error","message":"bad command"}');
  assert.equal(h.controller.running, false);
  assert.match(h.ui["status-message"].textContent, /سرور/);
  assert.equal(h.sockets[0].sent.length, 0);
});

test("unready video does not send commands or inflate FPS", async () => {
  const h = await connected();
  h.ui["camera-feed"].readyState = 1;
  h.controller.startInferenceLoop();
  h.clock.advance(1000);
  assert.equal(h.sockets[0].sent.length, 0);
  assert.equal(h.ui["fps-value"].textContent, "0");
});

test("capture failures release the camera, socket and loop", async () => {
  const h = await connected();
  h.controller.context.drawImage = () => { throw new Error("capture failed"); };
  h.controller.startInferenceLoop();
  h.clock.advance(50);
  assert.equal(h.controller.cameraReady, false);
  assert.equal(h.controller.running, false);
  assert.equal(h.controller.wsControl, null);
  assert.equal(h.stream.tracks[0].stopped, true);
  assert.equal(h.clock.tasks.size, 0);
});

test("an ended camera track requests a stop then disconnects without retrying", async () => {
  const h = await connected();
  h.controller.startInferenceLoop();
  h.stream.tracks[0].dispatchEvent(new Event("ended"));
  assert.deepEqual(h.sockets[0].sent.at(-1), STOP_COMMAND);
  assert.equal(h.controller.running, false);
  assert.equal(h.ui["camera-prompt"].hidden, false);
  assert.equal(h.clock.tasks.size, 0);
});

test("hidden/pagehide cleanup releases resources and prevents background starts", async () => {
  const h = await connected();
  h.controller.startInferenceLoop();
  h.clock.advance(50);
  h.document.hidden = true;
  h.document.dispatchEvent(new Event("visibilitychange"));
  assert.deepEqual(h.sockets[0].sent.at(-1), STOP_COMMAND);
  assert.equal(h.controller.running, false);
  assert.equal(h.stream.tracks[0].stopped, true);
  assert.equal(h.ui["camera-feed"].srcObject, null);
  assert.equal(h.clock.tasks.size, 0);
  assert.equal(h.controller.startInferenceLoop(), false);
  assert.equal(await h.controller.init(), false);
  h.window.dispatchEvent(new Event("pagehide"));
  assert.equal(h.sockets[0].sent.length, 2);
});

test("visibility restoration reacquires camera/connection without resuming inference", async () => {
  const h = await connected();
  h.controller.startInferenceLoop();
  h.document.hidden = true;
  h.document.dispatchEvent(new Event("visibilitychange"));
  h.document.hidden = false;
  await h.controller.init();
  h.sockets[1].open();
  assert.equal(h.controller.cameraReady, true);
  assert.equal(h.controller.running, false);
  assert.equal(h.ui["btn-start"].disabled, false);
  assert.equal(h.mediaCalls.length, 2);
});

test("destroy cancels pending reconnect and removes event handlers", async () => {
  const h = await connected();
  h.sockets[0].close(1006);
  h.controller.destroy();
  h.clock.advance(60000);
  h.ui["btn-camera"].dispatchEvent(new Event("click"));
  h.document.dispatchEvent(new Event("visibilitychange"));
  assert.equal(h.sockets.length, 1);
  assert.equal(h.mediaCalls.length, 1);
  assert.equal(h.clock.tasks.size, 0);
  assert.equal(await h.controller.init(), false);
});

test("an older camera result cannot replace a successful post-suspension stream", async () => {
  let resolveOld;
  let calls = 0;
  const oldStream = new FakeStream();
  const newStream = new FakeStream();
  const h = harness({ getUserMedia: () => {
    calls += 1;
    return calls === 1 ? new Promise((resolve) => { resolveOld = resolve; }) : newStream;
  } });
  const oldInit = h.controller.init();
  h.controller.suspend();
  assert.equal(await h.controller.init(), true);
  resolveOld(oldStream);
  assert.equal(await oldInit, false);
  assert.equal(oldStream.tracks[0].stopped, true);
  assert.equal(newStream.tracks[0].stopped, false);
  assert.equal(h.controller.stream, newStream);
  assert.equal(h.ui["camera-feed"].srcObject, newStream);
  assert.equal(h.sockets.length, 1);
});
