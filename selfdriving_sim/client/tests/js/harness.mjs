/**
 * Shared fakes for the dashboard tests: a DOM-like document, a WebSocket, a camera
 * stream and a clock that only moves when a test advances it.
 */
import assert from "node:assert/strict";

import { DashboardController } from "../../static/client/js/inference.js";

/** Swallows log lines so test output stays readable. */
export const silentLogger = Object.freeze({ info() {}, warn() {}, error() {} });

export class FakeClock {
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

export class FakeClassList {
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

export class FakeElement extends EventTarget {
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

export class FakeDocument extends EventTarget {
  hidden = false;
  elements = new Map();
  getElementById(id) {
    if (!this.elements.has(id)) this.elements.set(id, new FakeElement());
    return this.elements.get(id);
  }
}

export class FakeTrack extends EventTarget {
  stopped = false;
  stop() { this.stopped = true; }
}

export class FakeStream {
  tracks = [new FakeTrack()];
  getTracks() { return this.tracks; }
  getVideoTracks() { return this.tracks; }
}

export class FakeSocket {
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

export function harness({ secure = true, getUserMedia, runtime, logger = silentLogger } = {}) {
  const clock = new FakeClock();
  const document = new FakeDocument();
  const window = new EventTarget();
  const sockets = [];
  const mediaCalls = [];
  const stream = new FakeStream();
  Object.assign(window, {
    isSecureContext: secure,
    performance: { now: () => clock.now },
    location: { protocol: "https:", host: "sim.local" },
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
  const controller = new DashboardController({ window, document, runtime, logger });
  return { controller, window, document, clock, sockets, mediaCalls, stream, logger, ui: controller.ui };
}

export async function connected(options) {
  const h = harness(options);
  assert.equal(await h.controller.init(), true);
  h.sockets[0].open();
  return h;
}
