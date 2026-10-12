/**
 * DAVE-2-GRU on the phone: frame window, tensor pipeline, dashboard model path and
 * the converted model in static/client/models/dave2/. Run with npm test.
 */
import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { readFile, readdir, stat } from "node:fs/promises";
import test from "node:test";

import * as tf from "@tensorflow/tfjs";

import {
  FRAME_HEIGHT,
  FRAME_STRIDE,
  FRAME_WIDTH,
  FrameWindow,
  IDLE_COMMAND,
  MODEL_LOAD_TIMEOUT_MS,
  MODEL_OUTPUTS,
  SEND_INTERVAL_MS,
  SEQUENCE_LENGTH,
  STOP_COMMAND,
  loadModel,
  mockInference,
  preprocessFrame,
  realInference,
  toCommand,
  validateModelContract,
} from "../../static/client/js/inference.js";
import { harness } from "./harness.mjs";

const CONVERTED_DIR = new URL("../../static/client/models/dave2/", import.meta.url);
const WEB_MODEL_BUDGET_BYTES = 10_000_000;
/** Ticks before the window holds SEQUENCE_LENGTH frames, one every FRAME_STRIDE ticks. */
const WARM_UP_TICKS = (SEQUENCE_LENGTH - 1) * FRAME_STRIDE;
/** The converted model is produced by models/convert.py and committed to the repository. */
const CONVERTED_MODEL_MISSING = "run python models/convert.py to produce the converted model";
const hasConvertedModel = existsSync(new URL("model.json", CONVERTED_DIR));

/** Lets promise chains started by a tick finish before the test reads the HUD. */
const settle = () => new Promise((resolve) => setImmediate(resolve));

/** A stand-in for a frame tensor that records how often it was disposed. */
function trackedFrames() {
  const frames = [];
  const make = () => {
    const frame = { disposed: false, dispose() { this.disposed = true; } };
    frames.push(frame);
    return frame;
  };
  return { frames, make, disposedCount: () => frames.filter((frame) => frame.disposed).length };
}

/** Builds a model with the four named heads on a small input, for contract tests. */
function contractModel({ inputShape = [SEQUENCE_LENGTH, FRAME_HEIGHT, FRAME_WIDTH, 3], names = MODEL_OUTPUTS } = {}) {
  const input = tf.input({ shape: inputShape });
  const flat = tf.layers.flatten().apply(input);
  const outputs = names.map((name, index) => tf.layers.dense({
    units: index === 3 ? 3 : 1,
    name,
    activation: "linear",
  }).apply(flat));
  return tf.model({ inputs: input, outputs });
}

/** A model whose four heads return fixed values for any input. */
function constantHeadModel(values) {
  const inputShape = [SEQUENCE_LENGTH, 4, 4, 3];
  const flatSize = inputShape.reduce((product, dim) => product * dim, 1);
  const input = tf.input({ shape: inputShape });
  const flat = tf.layers.flatten().apply(input);
  const outputs = Object.entries(values).map(([name, bias]) => {
    const units = bias.length;
    const layer = tf.layers.dense({ units, name, activation: name === "turn_signal" ? "softmax" : "linear" });
    const output = layer.apply(flat);
    layer.setWeights([tf.zeros([flatSize, units]), tf.tensor1d(bias)]);
    return output;
  });
  return tf.model({ inputs: input, outputs });
}

/** Loads the converted model from disk through the loader the phone uses. */
async function loadConvertedModel() {
  const json = JSON.parse(await readFile(new URL("model.json", CONVERTED_DIR), "utf8"));
  const groups = json.weightsManifest;
  const shards = await Promise.all(groups.flatMap((group) => group.paths)
    .map((path) => readFile(new URL(path, CONVERTED_DIR))));
  const bytes = Buffer.concat(shards);
  const weightData = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  return loadModel(tf, tf.io.fromMemory({
    modelTopology: json.modelTopology,
    weightSpecs: groups.flatMap((group) => group.weights),
    weightData,
  }));
}

test("FrameWindow feeds the three newest frames and disposes frames that leave the span", () => {
  const disposed = [];
  const frames = new FrameWindow({ dispose: (frame) => disposed.push(frame) });
  for (let frame = 0; frame < 2; frame += 1) frames.push(frame);
  assert.equal(frames.ready, false);
  frames.push(2);
  assert.equal(frames.ready, true);
  assert.deepEqual(frames.sequence(), [0, 1, 2]);
  frames.push(3);
  assert.deepEqual(frames.sequence(), [1, 2, 3]);
  assert.deepEqual(disposed, [0]);
  frames.clear();
  assert.deepEqual(disposed, [0, 1, 2, 3]);
  assert.equal(frames.ready, false);
  assert.throws(() => frames.sequence(), /not full/);
});

test("FrameWindow honours a stride wider than one tick", () => {
  const frames = new FrameWindow({ stride: 2 });
  for (let frame = 0; frame < 5; frame += 1) frames.push(frame);
  assert.equal(frames.span, 5);
  assert.deepEqual(frames.sequence(), [0, 2, 4]);
});

test("preprocessFrame keeps the raw 0..255 RGB values and leaves no intermediate tensors", () => {
  const before = tf.memory().numTensors;
  // Pixel data is RGBA, four bytes per pixel, and must be a Uint8Array for fromPixels.
  const pixels = { data: new Uint8Array([0, 255, 51, 255, 255, 0, 0, 255]), width: 2, height: 1 };
  const frame = preprocessFrame(tf, pixels);
  assert.deepEqual(frame.shape, [1, 2, 3]);
  // The checkpoint normalizes its input itself, so the alpha channel is dropped
  // and nothing is divided here.
  const expected = [0, 255, 51, 255, 0, 0];
  frame.dataSync().forEach((value, index) => assert.ok(Math.abs(value - expected[index]) < 1e-6));
  assert.equal(tf.memory().numTensors, before + 1, "only the returned tensor is alive");
  frame.dispose();
  assert.equal(tf.memory().numTensors, before);
});

test("validateModelContract accepts the four-head contract and rejects other layouts", () => {
  const model = contractModel();
  assert.doesNotThrow(() => validateModelContract(model));
  model.dispose();
  const wrongInput = contractModel({ inputShape: [SEQUENCE_LENGTH, 66, 200, 3] });
  assert.throws(() => validateModelContract(wrongInput), /Model input must be/);
  wrongInput.dispose();
  const missingHead = contractModel({ names: ["steering", "throttle", "brake", "gear"] });
  assert.throws(() => validateModelContract(missingHead), /turn_signal/);
  missingHead.dispose();
});

test("toCommand clamps and rounds the heads and takes the most probable turn signal", () => {
  assert.deepEqual(
    toCommand({ steering: 1.7, throttle: -0.2, brake: 0.0004, turnProbabilities: [0.1, 0.2, 0.7] }),
    { steering: 1, throttle: 0, brake: 0, turn_signal: 2 },
  );
  assert.deepEqual(
    toCommand({ steering: -0.12345, throttle: 0.5, brake: 1, turnProbabilities: new Float32Array([0.9, 0.05, 0.05]) }),
    { steering: -0.123, throttle: 0.5, brake: 1, turn_signal: 0 },
  );
  assert.throws(() => toCommand({ steering: NaN, throttle: 0, brake: 0, turnProbabilities: [1, 0, 0] }), /non-finite/);
  assert.throws(() => toCommand({ steering: 0, throttle: 0, brake: 0, turnProbabilities: [1, 0] }), /incomplete/);
});

test("realInference returns a clamped command and disposes every tensor it creates", async () => {
  const model = constantHeadModel({
    steering: [2.5],
    throttle: [-0.4],
    brake: [0.0004],
    turn_signal: [0, 3, 1],
  });
  const frames = Array.from({ length: SEQUENCE_LENGTH }, () => tf.zeros([4, 4, 3]));
  const before = tf.memory().numTensors;
  const command = await realInference(tf, model, frames);
  assert.deepEqual(command, { steering: 1, throttle: 0, brake: 0, turn_signal: 1 });
  assert.equal(tf.memory().numTensors, before, "no tensor created by realInference survives");
  frames.forEach((frame) => frame.dispose());
  model.dispose();
});

test("the model replaces Mock after loading: neutral warm-up, then model commands", async () => {
  const modelCommand = { steering: 0.25, throttle: 0.5, brake: 0, turn_signal: 2 };
  const logs = [];
  const logger = {
    info: (...args) => logs.push(["info", ...args]),
    warn: (...args) => logs.push(["warn", ...args]),
    error: (...args) => logs.push(["error", ...args]),
  };
  const frames = trackedFrames();
  const model = { dispose() { this.disposed = true; } };
  const h = harness({
    logger,
    runtime: {
      tf: {},
      loadModel: async () => model,
      preprocess: () => frames.make(),
      infer: async () => modelCommand,
    },
  });
  assert.equal(await h.controller.init(), true);
  await settle();
  assert.equal(h.ui["mode-badge"].textContent, "DAVE-2 GRU");
  assert.ok(logs.some(([level, message]) => level === "info" && message === "[Model] Loaded"));

  h.sockets[0].open();
  assert.equal(h.controller.startInferenceLoop(), true);
  assert.equal(h.ui["run-status"].textContent, "در حال ارسال · DAVE-2-GRU");
  for (let tick = 0; tick < 8; tick += 1) {
    h.clock.advance(SEND_INTERVAL_MS);
    await settle();
  }
  const sent = h.sockets[0].sent;
  assert.equal(sent.length, 8);
  assert.deepEqual(sent.slice(0, WARM_UP_TICKS), Array(WARM_UP_TICKS).fill(IDLE_COMMAND));
  assert.deepEqual(sent.slice(WARM_UP_TICKS), Array(sent.length - WARM_UP_TICKS).fill(modelCommand));
  assert.equal(frames.frames.length, 8);
  assert.equal(frames.disposedCount(), 5, "only frames that left the 3-frame window are disposed");

  h.controller.stopInferenceLoop();
  assert.deepEqual(h.sockets[0].sent.at(-1), STOP_COMMAND);
  assert.equal(frames.disposedCount(), 8, "stopping releases the frames still in the window");
});

test("a model that fails to load falls back to Mock and says so", async () => {
  const warnings = [];
  const h = harness({
    logger: { info() {}, warn: (...args) => warnings.push(args), error() {} },
    runtime: {
      tf: {},
      loadModel: async () => { throw new Error("404 model.json"); },
      preprocess: () => ({ dispose() {} }),
      infer: async () => { throw new Error("must not run"); },
    },
  });
  assert.equal(await h.controller.init(), true);
  await settle();
  assert.equal(h.ui["mode-badge"].textContent, "MOCK MODE");
  assert.ok(warnings.some(([message]) => message === "[Model] Load failed; using Mock"));
  h.sockets[0].open();
  h.controller.startInferenceLoop();
  assert.equal(h.ui["run-status"].textContent, "در حال ارسال · Mock");
  h.clock.advance(150);
  assert.deepEqual(h.sockets[0].sent[2], mockInference(0.15));
});

test("a model that never answers is abandoned after the load timeout", async () => {
  const h = harness({
    runtime: { tf: {}, loadModel: () => new Promise(() => {}), preprocess: () => ({}), infer: async () => ({}) },
  });
  assert.equal(await h.controller.init(), true);
  assert.equal(h.ui["mode-badge"].textContent, "MODEL LOADING");
  h.sockets[0].open();
  assert.equal(h.ui["btn-start"].disabled, true, "Start waits for the model");
  h.clock.advance(MODEL_LOAD_TIMEOUT_MS);
  await settle();
  assert.equal(h.ui["mode-badge"].textContent, "MOCK MODE");
  assert.equal(h.ui["btn-start"].disabled, false);
});

test("a model that arrives after its timeout is disposed instead of used", async () => {
  let resolveLoad;
  const late = { disposed: false, dispose() { this.disposed = true; } };
  const h = harness({
    runtime: {
      tf: {},
      loadModel: () => new Promise((resolve) => { resolveLoad = resolve; }),
      preprocess: () => ({}),
      infer: async () => ({}),
    },
  });
  await h.controller.init();
  h.clock.advance(MODEL_LOAD_TIMEOUT_MS);
  await settle();
  resolveLoad(late);
  await settle();
  assert.equal(late.disposed, true);
  assert.equal(h.controller.model, null);
  assert.equal(h.ui["mode-badge"].textContent, "MOCK MODE");
});

test("one inference runs at a time; ticks during a slow inference send nothing", async () => {
  let calls = 0;
  const h = harness({
    runtime: {
      tf: {},
      loadModel: async () => ({ dispose() {} }),
      preprocess: () => ({ dispose() {} }),
      infer: () => { calls += 1; return new Promise(() => {}); },
    },
  });
  await h.controller.init();
  await settle();
  h.sockets[0].open();
  h.controller.startInferenceLoop();
  for (let tick = 0; tick < 10; tick += 1) {
    h.clock.advance(SEND_INTERVAL_MS);
    await settle();
  }
  assert.equal(calls, 1);
  assert.equal(h.sockets[0].sent.length, WARM_UP_TICKS, "neutral warm-up commands only, then nothing while pending");
});

test("an inference failure stops the loop with a stop request and never sends Mock", async () => {
  const errors = [];
  const h = harness({
    logger: { info() {}, warn() {}, error: (...args) => errors.push(args) },
    runtime: {
      tf: {},
      loadModel: async () => ({ dispose() {} }),
      preprocess: () => ({ dispose() {} }),
      infer: async () => { throw new Error("WebGL context lost"); },
    },
  });
  await h.controller.init();
  await settle();
  h.sockets[0].open();
  h.controller.startInferenceLoop();
  h.clock.advance(SEND_INTERVAL_MS * 5);
  await settle();
  assert.equal(h.controller.running, false);
  assert.match(h.ui["status-message"].textContent, /اجرای مدل/);
  assert.equal(errors.length, 1);
  assert.deepEqual(h.sockets[0].sent, [...Array(WARM_UP_TICKS).fill(IDLE_COMMAND), STOP_COMMAND]);
});

test("an inference that finishes after Stop is dropped", async () => {
  let finish;
  const h = harness({
    runtime: {
      tf: {},
      loadModel: async () => ({ dispose() {} }),
      preprocess: () => ({ dispose() {} }),
      infer: () => new Promise((resolve) => { finish = resolve; }),
    },
  });
  await h.controller.init();
  await settle();
  h.sockets[0].open();
  h.controller.startInferenceLoop();
  h.clock.advance(SEND_INTERVAL_MS * 5);
  await settle();
  h.controller.stopInferenceLoop();
  const sentAtStop = h.sockets[0].sent.length;
  finish({ steering: 0.9, throttle: 1, brake: 0, turn_signal: 0 });
  await settle();
  assert.equal(h.sockets[0].sent.length, sentAtStop);
  assert.deepEqual(h.sockets[0].sent.at(-1), STOP_COMMAND);
});

test("destroy disposes the model and every frame still held", async () => {
  const frames = trackedFrames();
  const model = { disposed: false, dispose() { this.disposed = true; } };
  const h = harness({
    runtime: {
      tf: {},
      loadModel: async () => model,
      preprocess: () => frames.make(),
      infer: async () => ({ steering: 0, throttle: 0, brake: 0, turn_signal: 0 }),
    },
  });
  await h.controller.init();
  await settle();
  h.sockets[0].open();
  h.controller.startInferenceLoop();
  h.clock.advance(SEND_INTERVAL_MS * 3);
  await settle();
  h.controller.destroy();
  assert.equal(model.disposed, true);
  assert.equal(h.controller.model, null);
  assert.equal(frames.disposedCount(), frames.frames.length);
});

test("without window.tf the dashboard runs Mock and shows it", async () => {
  const h = harness();
  assert.equal(await h.controller.init(), true);
  await settle();
  assert.equal(h.ui["mode-badge"].textContent, "MOCK MODE");
  h.sockets[0].open();
  h.controller.startInferenceLoop();
  h.clock.advance(SEND_INTERVAL_MS);
  assert.deepEqual(h.sockets[0].sent[0], mockInference(0.05));
});

test("the converted model loads from static files and satisfies the four-head contract", { skip: !hasConvertedModel && CONVERTED_MODEL_MISSING }, async () => {
  const model = await loadConvertedModel();
  try {
    validateModelContract(model);
    const frames = Array.from({ length: SEQUENCE_LENGTH }, () => tf.zeros([FRAME_HEIGHT, FRAME_WIDTH, 3]));
    const command = await realInference(tf, model, frames);
    frames.forEach((frame) => frame.dispose());
    assert.deepEqual(Object.keys(command), ["steering", "throttle", "brake", "turn_signal"]);
    assert.ok(Math.abs(command.steering) <= 1);
    assert.ok(command.throttle >= 0 && command.throttle <= 1);
    assert.ok([0, 1, 2].includes(command.turn_signal));
  } finally {
    model.dispose();
  }
});

test("the converted model folder is committed, has model.json, and stays under 10 MB", { skip: !hasConvertedModel && CONVERTED_MODEL_MISSING }, async () => {
  const names = await readdir(CONVERTED_DIR);
  assert.ok(names.includes("model.json"));
  let total = 0;
  for (const name of names) total += (await stat(new URL(name, CONVERTED_DIR))).size;
  assert.ok(total < WEB_MODEL_BUDGET_BYTES, `converted model is ${total} bytes`);
});
