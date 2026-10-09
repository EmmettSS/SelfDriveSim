/**
 * Node tests for the pure helpers of the Three.js scene.
 *
 * Run from this directory (`selfdriving_sim/`):
 *
 *     node --test "simulation/tests/js/*.test.mjs"
 *
 * `scene.js` imports the vendored ES module build of Three.js and only touches
 * the DOM inside `boot()`, so the module loads in Node and the scheduling,
 * control, speed and reconnect logic can be checked directly.
 */

import assert from "node:assert/strict";
import test from "node:test";

import {
  ACCELERATION_MS2,
  COASTING_RESISTANCE_MS2,
  EGO_LATERAL_LIMIT,
  LANE_CENTRES,
  MAX_RECONNECT_ATTEMPTS,
  MAX_SPEED_MS,
  MAX_TRAFFIC_CARS,
  RECONNECT_BASE_DELAY_MS,
  RECONNECT_MAX_DELAY_MS,
  SPAWN_DELAY_MAX_S,
  SPAWN_DELAY_MIN_S,
  SPAWN_Z_MAX,
  SPAWN_Z_MIN,
  TRAFFIC_COLOURS,
  TRAFFIC_SPEED_MAX,
  TRAFFIC_SPEED_MIN,
  canSpawnTraffic,
  clamp,
  dashScrollOffset,
  nextReconnectDelayMs,
  normalizeCommand,
  pickLane,
  pickSpawnDelay,
  pickSpawnZ,
  pickTrafficColour,
  pickTrafficSpeed,
  smoothFrameRate,
  speedToKmh,
  telemetryFrame,
  stepLateralPosition,
  stepSpeed,
} from "../../static/simulation/js/scene.js";

/** Deterministic replacement for Math.random. */
const atZero = () => 0;
const atOne = () => 1;
const atHalf = () => 0.5;

test("clamp keeps values inside the inclusive bounds", () => {
  assert.equal(clamp(0.4, 0, 1), 0.4);
  assert.equal(clamp(-3, 0, 1), 0);
  assert.equal(clamp(3, 0, 1), 1);
});

test("spawn delay stays between 2.5 and 4.5 seconds", () => {
  assert.equal(pickSpawnDelay(atZero), SPAWN_DELAY_MIN_S);
  assert.equal(pickSpawnDelay(atOne), SPAWN_DELAY_MAX_S);
  assert.equal(pickSpawnDelay(atHalf), 3.5);

  for (let i = 0; i < 500; i += 1) {
    const delay = pickSpawnDelay();
    assert.ok(delay >= SPAWN_DELAY_MIN_S && delay <= SPAWN_DELAY_MAX_S, `delay ${delay}`);
  }
});

test("traffic lanes are picked from the three lane centres", () => {
  assert.equal(pickLane(atZero), LANE_CENTRES[0]);
  // Math.random never returns 1; the guard keeps such a value inside the array.
  assert.equal(pickLane(atOne), LANE_CENTRES[0]);
  assert.equal(pickLane(atHalf), LANE_CENTRES[1]);

  const seen = new Set();
  for (let i = 0; i < 500; i += 1) {
    const lane = pickLane();
    assert.ok(LANE_CENTRES.includes(lane), `lane ${lane}`);
    seen.add(lane);
  }
  assert.equal(seen.size, LANE_CENTRES.length);
});

test("traffic speed is the specified per frame range converted to units per second", () => {
  assert.equal(TRAFFIC_SPEED_MIN, 9);
  assert.equal(TRAFFIC_SPEED_MAX, 21);
  assert.equal(pickTrafficSpeed(atZero), TRAFFIC_SPEED_MIN);
  assert.equal(pickTrafficSpeed(atOne), TRAFFIC_SPEED_MAX);
});

test("traffic spawns between z -200 and z -150", () => {
  assert.equal(pickSpawnZ(atZero), SPAWN_Z_MIN);
  assert.equal(pickSpawnZ(atOne), SPAWN_Z_MAX);
  assert.ok(pickSpawnZ(atHalf) > SPAWN_Z_MIN && pickSpawnZ(atHalf) < SPAWN_Z_MAX);
});

test("traffic colours come from the palette", () => {
  assert.equal(pickTrafficColour(atZero), TRAFFIC_COLOURS[0]);
  for (let i = 0; i < 200; i += 1) {
    assert.ok(TRAFFIC_COLOURS.includes(pickTrafficColour()));
  }
});

test("the traffic pool stops at 20 cars", () => {
  assert.ok(canSpawnTraffic(0));
  assert.ok(canSpawnTraffic(MAX_TRAFFIC_CARS - 1));
  assert.equal(canSpawnTraffic(MAX_TRAFFIC_CARS), false);
  assert.equal(canSpawnTraffic(MAX_TRAFFIC_CARS + 5), false);
});

test("reconnect delays double, saturate and stop after ten attempts", () => {
  assert.equal(nextReconnectDelayMs(1), RECONNECT_BASE_DELAY_MS);
  assert.equal(nextReconnectDelayMs(2), 1000);
  assert.equal(nextReconnectDelayMs(3), 2000);
  assert.equal(nextReconnectDelayMs(4), 4000);
  assert.equal(nextReconnectDelayMs(5), 8000);
  assert.equal(nextReconnectDelayMs(6), RECONNECT_MAX_DELAY_MS);
  assert.equal(nextReconnectDelayMs(MAX_RECONNECT_ATTEMPTS), RECONNECT_MAX_DELAY_MS);
  // The budget is used up, so there is no further attempt.
  assert.equal(nextReconnectDelayMs(MAX_RECONNECT_ATTEMPTS + 1), null);
  assert.equal(nextReconnectDelayMs(0), null);
});

test("normalizeCommand clips values the way the server does", () => {
  assert.deepEqual(
    normalizeCommand({ steering: 0.25, throttle: 0.6, brake: 0, turn_signal: 1 }),
    { steering: 0.25, throttle: 0.6, brake: 0, turn_signal: 1 },
  );
  assert.deepEqual(
    normalizeCommand({ steering: 5, throttle: 2, brake: -1, turn_signal: 7 }),
    { steering: 1, throttle: 1, brake: 0, turn_signal: 2 },
  );
  // 1.4 rounds up, 1.5 rounds to 2 like the server's round().
  assert.equal(normalizeCommand({ steering: 0, throttle: 0, brake: 0, turn_signal: 1.4 }).turn_signal, 1);
  assert.equal(normalizeCommand({ steering: 0, throttle: 0, brake: 0, turn_signal: 1.5 }).turn_signal, 2);
});

test("normalizeCommand rejects payloads the server would reject", () => {
  assert.equal(normalizeCommand(null), null);
  assert.equal(normalizeCommand("steering"), null);
  assert.equal(normalizeCommand([{ steering: 0 }]), null);
  assert.equal(normalizeCommand({ steering: 0.1 }), null, "missing fields");
  assert.equal(normalizeCommand({ steering: 0, throttle: 0, brake: 0 }), null);
  assert.equal(
    normalizeCommand({ steering: "0.1", throttle: 0, brake: 0, turn_signal: 0 }),
    null,
    "a numeric string is not a number",
  );
  assert.equal(
    normalizeCommand({ steering: true, throttle: 0, brake: 0, turn_signal: 0 }),
    null,
    "booleans are refused",
  );
  assert.equal(
    normalizeCommand({ steering: NaN, throttle: 0, brake: 0, turn_signal: 0 }),
    null,
    "NaN is refused",
  );
});

test("throttle accelerates up to the top speed and brake stops at zero", () => {
  let speed = 0;
  for (let i = 0; i < 600; i += 1) {
    speed = stepSpeed(speed, 1, 0, 1 / 60);
  }
  assert.equal(speed, MAX_SPEED_MS);
  assert.ok(ACCELERATION_MS2 > 0);

  // Full brake from the top speed brings the car to a standstill.
  let braking = MAX_SPEED_MS;
  for (let i = 0; i < 600; i += 1) {
    braking = stepSpeed(braking, 0, 1, 1 / 60);
  }
  assert.equal(braking, 0);

  // Coasting slows the car down until it stands still, without ever reversing.
  let coasting = 10;
  for (let i = 0; i < 3600; i += 1) {
    coasting = stepSpeed(coasting, 0, 0, 1 / 60);
  }
  assert.equal(coasting, 0);

  // One second of full throttle adds the acceleration minus the resistance.
  assert.ok(
    Math.abs(stepSpeed(0, 1, 0, 1) - (ACCELERATION_MS2 - COASTING_RESISTANCE_MS2)) < 0.001,
  );
  // A stationary car with no throttle stays put.
  assert.equal(stepSpeed(0, 0, 1, 1), 0);
});

test("steering moves the car sideways and stops at the road edge", () => {
  assert.equal(stepLateralPosition(0, 1, 1), 7);
  assert.equal(stepLateralPosition(0, -1, 1), -7);
  assert.equal(stepLateralPosition(0, 0, 1), 0);
  // Full lock held for a long time stays on the road.
  assert.equal(stepLateralPosition(0, 1, 60), EGO_LATERAL_LIMIT);
  assert.equal(stepLateralPosition(0, -1, 60), -EGO_LATERAL_LIMIT);
});

test("speed is reported in km/h", () => {
  assert.equal(speedToKmh(0), 0);
  assert.equal(speedToKmh(10), 36);
  assert.equal(Math.round(speedToKmh(MAX_SPEED_MS)), 108);
});

test("lane markings scroll inside one dash period", () => {
  assert.equal(dashScrollOffset(0), 0);
  assert.equal(dashScrollOffset(6), 0, "the pattern repeats every dash plus gap");
  assert.equal(dashScrollOffset(7), 1);
  assert.equal(dashScrollOffset(-1), 5, "a negative distance wraps forward");
  for (let distance = 0; distance < 100; distance += 0.7) {
    const offset = dashScrollOffset(distance);
    assert.ok(offset >= 0 && offset < 6, `offset ${offset}`);
  }
});

test("telemetry frames carry only a clipped integer km/h reading", () => {
  assert.deepEqual(telemetryFrame(10), { type: "telemetry", speed_kmh: 36 });
  assert.equal(telemetryFrame(-4).speed_kmh, 0);
});

test("the frame rate estimate converges on the measured interval", () => {
  let rate = 60;
  for (let i = 0; i < 200; i += 1) {
    rate = smoothFrameRate(rate, 1 / 30);
  }
  assert.ok(Math.abs(rate - 30) < 0.01, `rate ${rate}`);
  // A zero delta must not divide by zero.
  assert.equal(smoothFrameRate(45, 0), 45);
});
