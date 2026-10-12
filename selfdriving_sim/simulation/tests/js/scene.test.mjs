/**
 * Node tests for the pure helpers of the Three.js scene.
 *
 * Run from this directory (`selfdriving_sim/`):
 *
 *     node --test "simulation/tests/js/*.test.mjs"
 *
 * `scene.js` imports the vendored ES module build of Three.js and only touches
 * the DOM inside `boot()`, so the module loads in Node and the phase-5 physics
 * (speed, heading, position), the wall, the blinker timing, the collision box
 * maths, the camera pose and the reconnect logic can be checked directly.
 */

import assert from "node:assert/strict";
import test from "node:test";

import {
  ACCELERATION_MS2,
  BLINKER_PHASE_RATE,
  BRAKE_DECELERATION_MS2,
  CAMERA_BUMPER_M,
  CAMERA_EYE_HEIGHT,
  CAMERA_LOOK_AHEAD_M,
  CAR_HALF_X,
  CAR_HALF_Z,
  COASTING_FRICTION_MS2,
  COLLISION_BROAD_X,
  COLLISION_BROAD_Z,
  DESPAWN_BEHIND_M,
  HAZARD_KINDS,
  HEADING_MIN_SPEED_MS,
  HEADING_SPEED_REF_MS,
  LANE_CENTRES,
  MAX_RECONNECT_ATTEMPTS,
  MAX_SPEED_MS,
  MAX_TRAFFIC_CARS,
  MIN_LANE_GAP_M,
  RECONNECT_BASE_DELAY_MS,
  RECONNECT_MAX_DELAY_MS,
  ROAD_WALL_X,
  SPAWN_DELAY_MAX_S,
  SPAWN_DELAY_MIN_S,
  SPAWN_GAP_MAX_M,
  SPAWN_GAP_MIN_M,
  STEER_RATE,
  STEERING_SIGN,
  TRAFFIC_COLOURS,
  TRAFFIC_SPEED_MAX,
  TRAFFIC_SPEED_MIN,
  blinkerIsLit,
  cameraPose,
  canSpawnTraffic,
  carsCollide,
  clamp,
  clampToWall,
  hazardFrame,
  headingDegrees,
  nextReconnectDelayMs,
  normalizeCommand,
  pickLane,
  pickSpawnDelay,
  pickSpawnGap,
  pickTrafficColour,
  pickTrafficSpeed,
  smoothFrameRate,
  speedToKmh,
  stepHeading,
  stepLongitudinalSpeed,
  stepWorldPosition,
  telemetryFrame,
  trafficSpawnZ,
} from "../../static/simulation/js/scene.js";

/** Deterministic replacement for Math.random. */
const atZero = () => 0;
const atOne = () => 1;
const atHalf = () => 0.5;

/* ------------------------------- basics --------------------------------- */

test("clamp keeps values inside the inclusive bounds", () => {
  assert.equal(clamp(0.4, 0, 1), 0.4);
  assert.equal(clamp(-3, 0, 1), 0);
  assert.equal(clamp(3, 0, 1), 1);
});

/* --------------------------- longitudinal model -------------------------- */

test("the longitudinal constants match the phase-5 spec values", () => {
  assert.equal(MAX_SPEED_MS, 20);
  assert.equal(ACCELERATION_MS2, 8);
  assert.equal(BRAKE_DECELERATION_MS2, 20);
  assert.equal(COASTING_FRICTION_MS2, 2);
});

test("brake above 0.1 decelerates by braking * brake", () => {
  assert.equal(stepLongitudinalSpeed(20, 0, 1, 0.1), 18);
  assert.equal(stepLongitudinalSpeed(20, 0, 0.5, 0.1), 19);
  // Brake wins over throttle when both pedals are pressed.
  assert.equal(stepLongitudinalSpeed(10, 1, 1, 0.1), 8);
});

test("throttle above 0.1 accelerates by acceleration * throttle", () => {
  assert.equal(stepLongitudinalSpeed(0, 1, 0, 0.1), 0.8);
  assert.equal(stepLongitudinalSpeed(0, 0.5, 0, 0.1), 0.4);
});

test("released pedals coast down by the constant friction", () => {
  assert.equal(stepLongitudinalSpeed(10, 0, 0, 0.5), 9);
});

test("speed is clipped to 0..maxSpeed", () => {
  assert.equal(stepLongitudinalSpeed(0.5, 0, 0, 10), 0);
  assert.equal(stepLongitudinalSpeed(MAX_SPEED_MS, 1, 0, 10), MAX_SPEED_MS);
  assert.equal(stepLongitudinalSpeed(0, 0, 1, 100), 0);
});

/* ------------------------------ heading model ---------------------------- */

test("the steering constants match the phase-5 spec values", () => {
  assert.equal(STEER_RATE, 1.5);
  assert.equal(HEADING_MIN_SPEED_MS, 0.5);
  assert.equal(HEADING_SPEED_REF_MS, 5);
  assert.equal(STEERING_SIGN, 1);
});

test("the heading only reacts while the car rolls faster than 0.5 m/s", () => {
  assert.equal(stepHeading(0, 1, 0.5, 1), 0);
  assert.equal(stepHeading(0, 1, 0.49, 1), 0);
  assert.ok(stepHeading(0, 1, 0.51, 1) > 0);
});

test("steering authority ramps in with min(1, speed / 5)", () => {
  const dt = 1;
  assert.equal(stepHeading(0, 1, 5, dt), STEER_RATE); // full lock from 5 m/s up
  assert.equal(stepHeading(0, 1, 2.5, dt), STEER_RATE / 2);
  assert.equal(stepHeading(0, 1, 20, dt), STEER_RATE);
});

test("negative steering turns the heading negative", () => {
  assert.equal(stepHeading(0, -1, 10, 1), -STEER_RATE);
  assert.equal(stepHeading(-STEER_RATE, -1, 10, 1), -2 * STEER_RATE);
});

/* ----------------------------- position model ---------------------------- */

test("heading 0 drives the car towards -z (forward = -z)", () => {
  const next = stepWorldPosition(2, 50, 0, 10, 0.1);
  assert.equal(next.x, 2);
  assert.equal(next.z, 49);
});

test("positive heading increases x, which is screen-right", () => {
  const next = stepWorldPosition(0, 0, Math.PI / 2, 10, 1);
  assert.ok(Math.abs(next.x - 10) < 1e-12);
  assert.ok(Math.abs(next.z) < 1e-12);
});

test("heading PI drives towards +z and -PI/2 towards -x", () => {
  const back = stepWorldPosition(0, 0, Math.PI, 10, 1);
  assert.ok(Math.abs(back.z - 10) < 1e-12);
  const left = stepWorldPosition(0, 0, -Math.PI / 2, 10, 1);
  assert.ok(Math.abs(left.x + 10) < 1e-12);
});

/* --------------------------------- wall ---------------------------------- */

test("the invisible wall clamps the lateral position at |x| = 9", () => {
  assert.equal(ROAD_WALL_X, 9);
  assert.equal(clampToWall(9), 9);
  assert.equal(clampToWall(9.4), 9);
  assert.equal(clampToWall(-12), -9);
  assert.equal(clampToWall(0), 0);
});

/* ------------------------------- blinkers -------------------------------- */

test("the blinker phase toggles three times per second", () => {
  assert.equal(BLINKER_PHASE_RATE, 3);
  // On during [0, 1), off during [1, 2), on again at 2.
  assert.equal(blinkerIsLit(0), true);
  assert.equal(blinkerIsLit(0.99), true);
  assert.equal(blinkerIsLit(1), false);
  assert.equal(blinkerIsLit(1.5), false);
  assert.equal(blinkerIsLit(2), true);
});

test("accumulating dt shows three state changes per simulated second", () => {
  let phase = 0;
  let changes = 0;
  let previous = blinkerIsLit(phase);
  for (let i = 0; i < 6; i += 1) {
    phase += (1 / 3) * BLINKER_PHASE_RATE; // (1/3)*3 === 1 exactly, no drift
    const lit = blinkerIsLit(phase);
    if (lit !== previous) {
      changes += 1;
      previous = lit;
    }
  }
  assert.equal(changes, 6); // two seconds of blink at three toggles per second
});

/* ------------------------------- camera ---------------------------------- */

test("the camera rides on the front bumper and looks ahead along -z", () => {
  const pose = cameraPose(3, -7, 0);
  assert.equal(pose.position.x, 3);
  assert.equal(pose.position.y, CAMERA_EYE_HEIGHT);
  assert.equal(pose.position.z, -7 - CAMERA_BUMPER_M);
  assert.equal(pose.lookAt.x, 3);
  assert.equal(pose.lookAt.y, CAMERA_EYE_HEIGHT);
  assert.equal(pose.lookAt.z, -7 - CAMERA_LOOK_AHEAD_M);
});

test("the camera pose rotates with the heading", () => {
  const pose = cameraPose(0, 0, Math.PI / 2);
  // Facing (sin h, -cos h) = (1, 0): the bumper sits at +x, looking at +x.
  assert.ok(Math.abs(pose.position.x - CAMERA_BUMPER_M) < 1e-12);
  assert.ok(Math.abs(pose.position.z) < 1e-12);
  assert.ok(Math.abs(pose.lookAt.x - CAMERA_LOOK_AHEAD_M) < 1e-12);
});

/* ------------------------------ collisions ------------------------------- */

test("two cars overlap when both axes are within the summed half extents", () => {
  assert.equal(CAR_HALF_X, 1);
  assert.equal(CAR_HALF_Z, 2.2);
  assert.equal(carsCollide(0, 0, 0, 0), true); // same spot
  assert.equal(carsCollide(0, 0, 1.5, 0), true); // lateral overlap
  assert.equal(carsCollide(0, 0, 0, 4.0), true); // nose to tail
  assert.equal(carsCollide(0, 0, 2.01, 0), false); // 1 cm gap sideways
  assert.equal(carsCollide(0, 0, 0, 4.41), false); // 1 cm gap lengthwise
  assert.equal(carsCollide(0, 0, 5, 0), false);
});

test("the broad phase reaches slightly past two touching car boxes", () => {
  assert.ok(COLLISION_BROAD_X > 2 * CAR_HALF_X);
  assert.ok(COLLISION_BROAD_Z > 2 * CAR_HALF_Z);
});

/* ------------------------------- heading HUD ----------------------------- */

test("heading degrees are normalised into (-180, 180]", () => {
  assert.equal(headingDegrees(0), 0);
  assert.equal(headingDegrees(Math.PI / 2), 90);
  assert.equal(headingDegrees(-Math.PI / 2), -90);
  assert.equal(headingDegrees(Math.PI), 180);
  assert.equal(headingDegrees(-Math.PI), 180);
  assert.equal(headingDegrees(2 * Math.PI), 0);
  assert.equal(headingDegrees(Math.PI + 0.1), -180 + 5.729577951308233);
});

/* -------------------------------- traffic -------------------------------- */

test("spawn delay stays between 2.5 and 4.5 seconds", () => {
  assert.equal(pickSpawnDelay(atZero), SPAWN_DELAY_MIN_S);
  assert.equal(pickSpawnDelay(atOne), SPAWN_DELAY_MAX_S);
});

test("traffic lanes are picked from the three lane centres", () => {
  for (let i = 0; i < 20; i += 1) {
    assert.ok(LANE_CENTRES.includes(pickLane(Math.random)));
  }
  // floor(random * 3) % 3: random 0 -> lane -4, 1/3 -> 0, 2/3 -> +4, 1 -> -4.
  assert.equal(pickLane(atZero), LANE_CENTRES[0]);
  assert.equal(pickLane(() => 1 / 3), LANE_CENTRES[1]);
  assert.equal(pickLane(() => 2 / 3), LANE_CENTRES[2]);
  assert.equal(pickLane(atOne), LANE_CENTRES[0]);
});

test("traffic speed is the specified per frame range converted to units per second", () => {
  assert.equal(pickTrafficSpeed(atZero), TRAFFIC_SPEED_MIN);
  assert.equal(pickTrafficSpeed(atOne), TRAFFIC_SPEED_MAX);
});

test("traffic spawns 150 to 200 units ahead of the ego car", () => {
  assert.equal(pickSpawnGap(atZero), SPAWN_GAP_MIN_M);
  assert.equal(pickSpawnGap(atOne), SPAWN_GAP_MAX_M);
  assert.equal(trafficSpawnZ(-40, 150), -190);
  assert.equal(trafficSpawnZ(10, 200), -190);
});

test("traffic colours come from the palette", () => {
  for (let i = 0; i < 20; i += 1) {
    assert.ok(TRAFFIC_COLOURS.includes(pickTrafficColour(Math.random)));
  }
});

test("the traffic pool stops at 20 cars", () => {
  assert.equal(canSpawnTraffic(0), true);
  assert.equal(canSpawnTraffic(19), true);
  assert.equal(canSpawnTraffic(20), false);
  assert.equal(MAX_TRAFFIC_CARS, 20);
  assert.ok(MIN_LANE_GAP_M > 0);
  assert.ok(DESPAWN_BEHIND_M > 0);
});

/* -------------------------------- hazards -------------------------------- */

test("hazard frames carry only a known hazard kind", () => {
  assert.deepEqual(hazardFrame("collision"), { type: "hazard", hazard: "collision" });
  assert.deepEqual(hazardFrame("wall"), { type: "hazard", hazard: "wall" });
  assert.deepEqual(HAZARD_KINDS, ["collision", "wall"]);
});

test("hazard frames reject unknown kinds", () => {
  for (const kind of ["offroad", "", "COLLISION", "traffic", null, undefined, 3]) {
    assert.equal(hazardFrame(kind), null);
  }
});

/* ------------------------- control normalisation ------------------------- */

test("normalizeCommand clips values the way the server does", () => {
  assert.deepEqual(
    normalizeCommand({ steering: 5, throttle: -1, brake: 0.5, turn_signal: 1 }),
    { steering: 1, throttle: 0, brake: 0.5, turn_signal: 1 },
  );
  assert.deepEqual(
    normalizeCommand({ steering: 0.123456, throttle: 1, brake: 0, turn_signal: 2.7 }),
    { steering: 0.123456, throttle: 1, brake: 0, turn_signal: 2 },
  );
});

test("normalizeCommand rejects payloads the server would reject", () => {
  for (const payload of [
    null,
    "text",
    42,
    [],
    {},
    { steering: "0", throttle: 0, brake: 0, turn_signal: 0 },
    { steering: 0, brake: 0, turn_signal: 0 },
    { steering: 0, throttle: true, brake: 0, turn_signal: 0 },
    { steering: NaN, throttle: 0, brake: 0, turn_signal: 0 },
    { steering: 0, throttle: 0, brake: Infinity, turn_signal: 0 },
  ]) {
    assert.equal(normalizeCommand(payload), null);
  }
});

/* ------------------------------- telemetry ------------------------------- */

test("speed is reported in km/h", () => {
  assert.equal(speedToKmh(0), 0);
  assert.equal(speedToKmh(10), 36);
  assert.equal(speedToKmh(20), 72);
});

test("telemetry frames carry only a clipped integer km/h reading", () => {
  assert.deepEqual(telemetryFrame(7.4), { type: "telemetry", speed_kmh: 27 });
  assert.deepEqual(telemetryFrame(-5), { type: "telemetry", speed_kmh: 0 });
  assert.deepEqual(telemetryFrame(999), { type: "telemetry", speed_kmh: 400 });
});

/* ------------------------------ frame rate ------------------------------- */

test("the frame rate estimate converges on the measured interval", () => {
  assert.equal(smoothFrameRate(60, 0), 60);
  // alpha = dt / window, so the estimate moves by (target - previous) * alpha.
  assert.ok(Math.abs(smoothFrameRate(0, 1 / 120) - 2) < 1e-9);
  assert.ok(Math.abs(smoothFrameRate(30, 1 / 60) - 31) < 1e-9);
  assert.ok(Math.abs(smoothFrameRate(30, 1 / 60, 1 / 30) - 45) < 1e-9);
});

/* ------------------------------- reconnect ------------------------------- */

test("reconnect delays double from 500 ms, cap at 15 s and stop after ten tries", () => {
  assert.deepEqual(
    Array.from({ length: 10 }, (_, i) => nextReconnectDelayMs(i + 1)),
    [500, 1000, 2000, 4000, 8000, 15000, 15000, 15000, 15000, 15000],
  );
  assert.equal(MAX_RECONNECT_ATTEMPTS, 10);
  assert.equal(RECONNECT_BASE_DELAY_MS, 500);
  assert.equal(RECONNECT_MAX_DELAY_MS, 15000);
  for (const attempt of [0, -1, 11, 0.5, NaN, Infinity]) {
    assert.equal(nextReconnectDelayMs(attempt), null);
  }
});
