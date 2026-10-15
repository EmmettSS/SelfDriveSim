import { LANE_CENTRES } from "./scene.js";

const clamp = (value, low, high) => Math.max(low, Math.min(high, value));

function nearestLaneIndex(x) {
  let nearest = 0;
  for (let index = 1; index < LANE_CENTRES.length; index += 1) {
    if (Math.abs(LANE_CENTRES[index] - x) < Math.abs(LANE_CENTRES[nearest] - x)) {
      nearest = index;
    }
  }
  return nearest;
}

function gapToCar(state, car) {
  return state.z - car.position.z;
}

/** Return the nearest active oncoming vehicle in one lane, if any. */
export function nearestTrafficGap(state, traffic, laneIndex) {
  const laneX = LANE_CENTRES[laneIndex];
  let nearest = Infinity;
  for (const car of traffic?.cars ?? []) {
    if (!car.userData?.active || Math.abs(car.position.x - laneX) > 1.4) continue;
    const gap = gapToCar(state, car);
    if (gap > 0 && gap < nearest) nearest = gap;
  }
  return nearest;
}

/** Build a deterministic policy that follows a lane and avoids oncoming traffic. */
export function createExpertDriver({
  random = Math.random,
  cruiseSpeed = 13,
  decisionDistance = 105,
  safeLaneDistance = 92,
  brakeDistance = 72,
  signalLeadSeconds = 0.45,
} = {}) {
  let targetLaneIndex = 1;
  let maneuverStartedAt = null;
  let maneuverSignal = 0;
  const laneOffset = (random() - 0.5) * 0.7;

  function clearManeuver() {
    targetLaneIndex = null;
    maneuverStartedAt = null;
    maneuverSignal = 0;
  }

  function beginManeuver(state, currentLaneIndex, nextLaneIndex) {
    targetLaneIndex = nextLaneIndex;
    maneuverStartedAt = state.time;
    maneuverSignal = nextLaneIndex < currentLaneIndex ? 1 : 2;
  }

  function chooseSafeAdjacentLane(state, traffic, currentLaneIndex) {
    const candidates = [currentLaneIndex - 1, currentLaneIndex + 1]
      .filter((index) => index >= 0 && index < LANE_CENTRES.length);
    if (random() < 0.5) candidates.reverse();

    let best = null;
    for (const index of candidates) {
      const gap = nearestTrafficGap(state, traffic, index);
      if (gap >= safeLaneDistance && (best === null || gap > best.gap)) {
        best = { index, gap };
      }
    }
    return best?.index ?? null;
  }

  function steeringFor(state, laneIndex) {
    const targetX = LANE_CENTRES[laneIndex] + laneOffset;
    const lateralError = targetX - state.x;
    return clamp(lateralError * 0.48 - state.heading * 1.25, -0.72, 0.72);
  }

  return {
    next(state, traffic) {
      const currentLaneIndex = nearestLaneIndex(state.x);
      if (targetLaneIndex === null) targetLaneIndex = currentLaneIndex;

      if (
        maneuverStartedAt !== null &&
        Math.abs(state.x - (LANE_CENTRES[targetLaneIndex] + laneOffset)) < 0.35 &&
        Math.abs(state.heading) < 0.055
      ) {
        clearManeuver();
        targetLaneIndex = currentLaneIndex;
      }

      if (maneuverStartedAt === null) {
        targetLaneIndex = currentLaneIndex;
        const currentGap = nearestTrafficGap(state, traffic, currentLaneIndex);
        if (currentGap <= decisionDistance) {
          const nextLaneIndex = chooseSafeAdjacentLane(state, traffic, currentLaneIndex);
          if (nextLaneIndex !== null) {
            beginManeuver(state, currentLaneIndex, nextLaneIndex);
          }
        }
      }

      const currentGap = nearestTrafficGap(state, traffic, currentLaneIndex);
      const noSafeLane = maneuverStartedAt === null && currentGap < brakeDistance;
      const brake = noSafeLane
        ? clamp((brakeDistance - currentGap + 8) / 28, 0.2, 1)
        : 0;
      const throttle = brake > 0
        ? 0
        : state.speed < cruiseSpeed - 0.35
          ? clamp(0.14 + (cruiseSpeed - state.speed) * 0.05, 0.14, 0.55)
          : 0;

      const waitingForSignal = maneuverStartedAt !== null
        && state.time - maneuverStartedAt < signalLeadSeconds;
      const steering = waitingForSignal
        ? 0
        : steeringFor(state, targetLaneIndex ?? currentLaneIndex);

      return {
        steering,
        throttle,
        brake,
        turn_signal: maneuverStartedAt === null ? 0 : maneuverSignal,
      };
    },
    get laneOffset() {
      return laneOffset;
    },
    reset(laneIndex = 1) {
      targetLaneIndex = laneIndex;
      maneuverStartedAt = null;
      maneuverSignal = 0;
    },
  };
}
