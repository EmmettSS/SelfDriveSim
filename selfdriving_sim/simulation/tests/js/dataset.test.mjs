import test from "node:test";
import assert from "node:assert/strict";

import {
  BASE_DOMAIN_PROFILE,
  createDomainProfile,
  createSeededRandom,
} from "../../static/simulation/js/domain-randomization.js";
import {
  createExpertDriver,
  nearestTrafficGap,
} from "../../static/simulation/js/expert-driver.js";

const emptyState = { x: 0, z: 0, speed: 12, heading: 0, time: 1 };
const car = (x, gap, active = true) => ({
  position: { x, z: -gap },
  userData: { active },
});
const traffic = (...cars) => ({ cars });

function constantRandom(value) {
  return () => value;
}

test("seeded random streams and appearance profiles are repeatable", () => {
  const first = createSeededRandom(42);
  const second = createSeededRandom(42);
  assert.deepEqual(Array.from({ length: 20 }, first), Array.from({ length: 20 }, second));
  assert.deepEqual(createDomainProfile(createSeededRandom(91)), createDomainProfile(createSeededRandom(91)));
  assert.notDeepEqual(createDomainProfile(createSeededRandom(91)), createDomainProfile(createSeededRandom(92)));
});

test("domain profiles stay within camera and appearance bounds", () => {
  const profile = createDomainProfile(createSeededRandom(2026));
  assert.equal(profile.texturedSurfaces, true);
  assert.ok(profile.fogFar > profile.fogNear + 100);
  assert.ok(profile.cameraFov >= 54 && profile.cameraFov <= 70);
  assert.ok(profile.cameraEyeHeight >= 0.63 && profile.cameraEyeHeight <= 0.91);
  assert.ok(Math.abs(profile.cameraPitchRadians) <= 0.035);
  assert.ok(Math.abs(profile.cameraYawRadians) <= 0.022);
  assert.ok(Math.abs(profile.cameraRollRadians) <= 0.018);
  assert.equal(BASE_DOMAIN_PROFILE.texturedSurfaces, false);
});

test("traffic gap ignores inactive cars and vehicles already passed", () => {
  assert.equal(nearestTrafficGap(emptyState, traffic(
    car(0, 35),
    car(0, 15, false),
    car(0, -4),
    car(4, 10),
  ), 1), 35);
  assert.equal(nearestTrafficGap(emptyState, traffic(car(-4, 40)), 1), Infinity);
});

test("a clear left lane is signalled before steering begins", () => {
  const driver = createExpertDriver({ random: constantRandom(0.5), cruiseSpeed: 13 });
  const oncoming = traffic(car(0, 80), car(4, 40));
  const first = driver.next(emptyState, oncoming);
  assert.equal(first.turn_signal, 1);
  assert.equal(first.steering, 0);
  assert.equal(first.brake, 0);

  const later = driver.next({ ...emptyState, time: 1.5 }, oncoming);
  assert.equal(later.turn_signal, 1);
  assert.ok(later.steering < 0);
});

test("a clear right lane uses the right signal class", () => {
  const driver = createExpertDriver({ random: constantRandom(0.5), cruiseSpeed: 13 });
  const command = driver.next(emptyState, traffic(car(0, 80), car(-4, 35)));
  assert.equal(command.turn_signal, 2);
});

test("the policy brakes when all adjacent lanes are blocked", () => {
  const driver = createExpertDriver({ random: constantRandom(0.5), cruiseSpeed: 13 });
  const command = driver.next(emptyState, traffic(car(0, 40), car(-4, 60), car(4, 55)));
  assert.equal(command.turn_signal, 0);
  assert.equal(command.throttle, 0);
  assert.ok(command.brake > 0);
});

test("traffic in another lane does not request a turn or brake", () => {
  const driver = createExpertDriver({ random: constantRandom(0.5), cruiseSpeed: 13 });
  const command = driver.next(emptyState, traffic(car(4, 20)));
  assert.equal(command.turn_signal, 0);
  assert.equal(command.brake, 0);
});
