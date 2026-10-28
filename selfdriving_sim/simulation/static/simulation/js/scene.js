/**
 * Three.js first-person driving scene for the closed-loop simulator.
 *
 * The closed loop is real: the ego car owns a heading-based state
 * ``{x, z, speed, heading}`` and drives through the world, while the road,
 * the markings and the scenery follow it so the drive never ends. The camera
 * sits on the front bumper (first-person view), which is exactly the image
 * the DAVE-2-GRU model on the phone expects.
 *
 * The module is split in five parts:
 *
 *   1. Tuning constants (the released v1.0 values).
 *   2. Pure helpers: the longitudinal/heading/position steps, the wall and
 *      collision maths, blinker timing and the camera pose. They never touch
 *      the DOM or WebGL, so ``simulation/tests/js/scene.test.mjs`` can import
 *      and exercise them under Node.
 *   3. Scene construction: endless road, lane markings, scenery and traffic.
 *   4. The simulation object that owns the state and the animation loop.
 *   5. The browser bootstrap: HUD, keyboard fallback, WebSocket client.
 *
 * The bootstrap only runs when a document exists, which keeps the module
 * importable in a headless test runner.
 */

import * as THREE from "./vendor/three.module.min.js";

/* ------------------------------------------------------------------------- *
 * 1. Tuning constants
 * ------------------------------------------------------------------------- */

/** Reference frame rate of the per-frame speeds given in the specification. */
export const FRAME_RATE_HZ = 60;

/** Lateral position of the centre of each driving lane. */
export const LANE_CENTRES = [-4, 0, 4];

/** Paint colours of the traffic cars. */
export const TRAFFIC_COLOURS = [0xe53935, 0xfdd835, 0x43a047, 0xfb8c00];

/** Paint colour of the player car (kept for the physics body and tests). */
export const EGO_COLOUR = 0x2196f3;

/** Upper bound for the number of traffic cars that exist at the same time. */
export const MAX_TRAFFIC_CARS = 20;

/** Bounds for the pause between two traffic spawns, in seconds. */
export const SPAWN_DELAY_MIN_S = 2.5;
export const SPAWN_DELAY_MAX_S = 4.5;

/** Traffic oncoming speed in units per second: 0.15 to 0.35 per frame at 60 Hz. */
export const TRAFFIC_SPEED_MIN = 0.15 * FRAME_RATE_HZ;
export const TRAFFIC_SPEED_MAX = 0.35 * FRAME_RATE_HZ;

/**
 * Traffic spawns this far *ahead* of the ego car (the car drives towards -z,
 * so "ahead" is a smaller z). A spawn is skipped when another car of the same
 * lane is closer than MIN_LANE_GAP_M.
 */
export const SPAWN_GAP_MIN_M = 150;
export const SPAWN_GAP_MAX_M = 200;
export const MIN_LANE_GAP_M = 25;

/** A traffic car is recycled once it is this far *behind* the ego car. */
export const DESPAWN_BEHIND_M = 15;

/** Road geometry. The plane covers 400 units and follows the ego car. */
export const ROAD_WIDTH = 20;
export const ROAD_LENGTH = 400;
export const ROAD_Z_CENTRE = -170;
export const ROAD_COLOUR = 0x2a2a2a;

/** Lane marking pattern, in world units. Its sum is the seamless scroll unit. */
export const DASH_SIZE = 3;
export const DASH_GAP = 3;

/**
 * Longitudinal model of the player car — the released v1.0 values:
 * maxSpeed 20, acceleration 8, braking 20, friction 2.
 */
export const MAX_SPEED_MS = 20;
export const ACCELERATION_MS2 = 8;
export const BRAKE_DECELERATION_MS2 = 20;
export const COASTING_FRICTION_MS2 = 2;

/**
 * Lateral model. ``STEER_RATE`` is the spec's steeringRate (1.5). The heading
 * only reacts above HEADING_MIN_SPEED_MS and is scaled by min(1, speed / 5).
 */
export const STEER_RATE = 1.5;
export const HEADING_MIN_SPEED_MS = 0.5;
export const HEADING_SPEED_REF_MS = 5;

/**
 * DAVE-2 steering convention: negative = left, positive = right. The scene
 * drives towards -z, where world +x is on the *right* of the screen, so the
 * spec integration ``x += sin(heading) * speed * dt`` maps positive steering
 * onto a right turn. ``STEERING_SIGN`` exists so a real-world closed-loop test
 * can flip the whole mapping with a one-line change if the trained model ever
 * turns out to use the opposite convention.
 */
export const STEERING_SIGN = 1;

/**
 * Invisible wall at |x| = 9 (the spec's off-road threshold). The user chose a
 * hard wall over a visual penalty: the car is clamped here, grinding against
 * the wall scrubs speed, and a hazard warning is sent to the phone.
 */
export const ROAD_WALL_X = 9;
/** Constant deceleration applied while the car grinds along the wall. */
export const WALL_SCRUB_MS2 = 15;
/** Minimum seconds between two wall-hit events (counter + hazard message). */
export const WALL_EVENT_COOLDOWN_S = 1;

/**
 * Turn signals blink per the spec formula ``blinkerPhase += dt * 3`` with the
 * colour toggled on ``floor(phase) % 2`` — three state changes per second.
 */
export const BLINKER_PHASE_RATE = 3;
export const BLINKER_ON_COLOUR = 0xffeb3b;
export const BLINKER_OFF_COLOUR = 0x333333;
export const BLINKER_SIZE = { x: 0.3, y: 0.15, z: 0.15 };
/** Blinker position on the first-person camera (hood corners, camera space). */
export const BLINKER_CAMERA_OFFSET = { x: 0.9, y: -0.45, z: -1.4 };

/**
 * Collision handling (user decision): red flash, full stop, a short freeze,
 * then the car resets to the centre of the middle lane with a short immunity
 * so overlapping traffic cannot instantly re-trigger.
 */
export const COLLISION_FREEZE_S = 1.0;
export const COLLISION_IMMUNITY_S = 2.0;

/** Half extents of the axis-aligned collision box of one car (body 1.8 x 4.0). */
export const CAR_HALF_X = 1.0;
export const CAR_HALF_Z = 2.2;

/** Broad phase: cars further apart than this in one axis cannot collide. */
export const COLLISION_BROAD_X = 2 * CAR_HALF_X + 0.5;
export const COLLISION_BROAD_Z = 2 * CAR_HALF_Z + 0.5;

/**
 * First-person camera, mounted on the front bumper. The eye sits slightly
 * above the hood and looks 30 units ahead at the same height, so the horizon
 * stays level like in the training images.
 */
export const CAMERA_FOV = 60;
export const CAMERA_EYE_HEIGHT = 0.75;
export const CAMERA_BUMPER_M = 2.2;
export const CAMERA_LOOK_AHEAD_M = 30;

/**
 * A WebSocket command stays authoritative for this long. Without fresh phone
 * commands the keyboard fallback drives the car again, so the scene stays
 * usable when no phone is connected.
 */
export const COMMAND_FRESH_S = 1.5;

/** Reconnect policy of the render socket. */
export const MAX_RECONNECT_ATTEMPTS = 10;
export const RECONNECT_BASE_DELAY_MS = 500;
export const RECONNECT_MAX_DELAY_MS = 15000;

/** Field table shared by the client side check and the HUD. */
const CONTROL_FIELDS = [
  ["steering", -1, 1, false],
  ["throttle", 0, 1, false],
  ["brake", 0, 1, false],
  ["turn_signal", 0, 2, true],
];

/** Hazards the scene may report to the phone. Mirrors the server allow-list. */
export const HAZARD_KINDS = Object.freeze(["collision", "wall"]);

/** Road geometry presets: standard narrow highway (3.4m), compact (2.8m), or wide (original 20m). */
export const ROAD_PRESETS = Object.freeze({
  narrow: {
    key: "narrow",
    name: "باریک و استاندارد (۳.۴ متر)",
    laneWidth: 3.4,
    roadWidth: 10.8,
    wallX: 5.0,
    laneCentres: [-3.4, 0, 3.4],
    dividerXs: [-1.7, 1.7],
  },
  veryNarrow: {
    key: "veryNarrow",
    name: "بسیار باریک و فشرده (۲.۸ متر)",
    laneWidth: 2.8,
    roadWidth: 8.8,
    wallX: 4.0,
    laneCentres: [-2.8, 0, 2.8],
    dividerXs: [-1.4, 1.4],
  },
  wide: {
    key: "wide",
    name: "عریض اولیه (بزرگراه ۲۰ متر)",
    laneWidth: 4.0,
    roadWidth: 20.0,
    wallX: 9.0,
    laneCentres: [-4.0, 0, 4.0],
    dividerXs: [-2.0, 2.0],
  },
});

/** Longitudinal lookahead horizon for tactical corridor evaluation (meters). */
export const ADAS_LOOKAHEAD_HORIZON_M = 80;
/** Distance threshold below which a lane is considered obstructed (meters). */
export const ADAS_OBSTACLE_DANGER_M = 65;
/** Minimum clearance ahead in the center lane during startup or reset (meters). */
export const STARTUP_RUNWAY_CLEARANCE_M = 110;

/** Traffic density presets: dense (heavy & rapid), medium, or sparse (original). */
export const TRAFFIC_PRESETS = Object.freeze({
  dense: {
    key: "dense",
    name: "ترافیک سنگین و پرتردد",
    spawnDelayMin: 0.8,
    spawnDelayMax: 1.6,
    spawnGapMin: 55,
    spawnGapMax: 110,
    initialCount: 5,
  },
  medium: {
    key: "medium",
    name: "ترافیک متوسط",
    spawnDelayMin: 1.5,
    spawnDelayMax: 2.5,
    spawnGapMin: 90,
    spawnGapMax: 150,
    initialCount: 3,
  },
  sparse: {
    key: "sparse",
    name: "ترافیک خلوت اولیه",
    spawnDelayMin: 2.5,
    spawnDelayMax: 4.5,
    spawnGapMin: 150,
    spawnGapMax: 200,
    initialCount: 0,
  },
});

/* ------------------------------------------------------------------------- *
 * 2. Pure helpers
 * ------------------------------------------------------------------------- */

/** Clamp ``value`` into the inclusive range ``low`` .. ``high``. */
export function clamp(value, low, high) {
  return Math.min(Math.max(value, low), high);
}

export function randomBetween(low, high, random = Math.random) {
  return low + random() * (high - low);
}

/** Seconds to wait before the next traffic car is spawned. */
export function pickSpawnDelay(random = Math.random, minS = SPAWN_DELAY_MIN_S, maxS = SPAWN_DELAY_MAX_S) {
  return randomBetween(minS, maxS, random);
}

/** x position of the lane a traffic car is spawned in. */
export function pickLane(random = Math.random, laneCentres = LANE_CENTRES) {
  const index = Math.floor(random() * laneCentres.length) % laneCentres.length;
  return laneCentres[index];
}

/** Closing speed of a traffic car, in units per second. */
export function pickTrafficSpeed(random = Math.random) {
  return randomBetween(TRAFFIC_SPEED_MIN, TRAFFIC_SPEED_MAX, random);
}

/** Distance ahead of the ego car where the next traffic car is spawned. */
export function pickSpawnGap(random = Math.random, minM = SPAWN_GAP_MIN_M, maxM = SPAWN_GAP_MAX_M) {
  return randomBetween(minM, maxM, random);
}

/** World z of a traffic car spawned `gap` units ahead of `egoZ`. */
export function trafficSpawnZ(egoZ, gap) {
  return egoZ - gap;
}

/** Palette index the next car body will be painted with. */
export function pickTrafficColour(random = Math.random) {
  const index = Math.floor(random() * TRAFFIC_COLOURS.length) % TRAFFIC_COLOURS.length;
  return TRAFFIC_COLOURS[index];
}
/** Whether the pool has room for one more traffic car. */
export function canSpawnTraffic(activeCount) {
  return activeCount < MAX_TRAFFIC_CARS;
}

/**
 * Check if a lane at longitudinal position z is occupied by an active traffic car.
 */
export function isLaneBlockedAtZ(cars, laneX, z, margin = 24, laneTol = 0.8) {
  if (!Array.isArray(cars)) return false;
  for (const car of cars) {
    if (!car || !car.userData?.active) continue;
    if (Math.abs(car.position.x - laneX) < laneTol && Math.abs(car.position.z - z) < margin) {
      return true;
    }
  }
  return false;
}

/**
 * Count how many lanes in laneCentres are blocked at longitudinal coordinate z.
 */
export function countBlockedLanesAtZ(cars, laneCentres, z, margin = 24, laneTol = 0.8) {
  if (!Array.isArray(cars) || !Array.isArray(laneCentres)) return 0;
  let count = 0;
  for (const laneX of laneCentres) {
    if (isLaneBlockedAtZ(cars, laneX, z, margin, laneTol)) {
      count++;
    }
  }
  return count;
}

/**
 * Tactical corridor and obstacle assessment.
 * Evaluates forward clearance in each lane, selects the safest navigable corridor,
 * and computes lateral guidance steering and turn signal commands.
 *
 * @param {number} egoX - Current ego lateral position.
 * @param {number} egoZ - Current ego longitudinal position (driving towards -z).
 * @param {number} egoHeading - Current heading in radians (0 = straight down -z).
 * @param {Array} cars - List of traffic cars ({ position: {x, z}, userData: { active } }).
 * @param {number[]} laneCentres - Array of lane X coordinates, typically [-3.4, 0, 3.4].
 * @param {number} dangerDist - Threshold below which a lane is blocked (default 45m).
 * @param {number} horizon - Max lookahead distance (default 80m).
 * @returns {Object} Corridor evaluation result.
 */
/**
 * Evaluates tactical lane maneuver, steering, and turn signals given lane clearances.
 * Used by both ground-truth simulation radar and pure visual computer vision detection.
 *
 * @param {number} egoX - Current lateral x coordinate of ego vehicle.
 * @param {number} egoHeading - Current yaw heading in radians.
 * @param {number[]} clearances - Array of forward clearance distances (meters) per lane.
 * @param {number[]} laneCentres - Lateral centers of the lanes.
 * @param {number} dangerDist - Threshold below which a lane is blocked (default 45m).
 * @returns {Object} Corridor evaluation result.
 */
export function evaluateCorridorManeuver(
  egoX,
  egoHeading,
  clearances,
  laneCentres = LANE_CENTRES,
  dangerDist = ADAS_OBSTACLE_DANGER_M,
  lockedTargetLane = null,
  prevSteering = 0,
) {
  const numLanes = laneCentres.length;

  // Find current ego lane
  let currentLaneIdx = 0;
  let minEgoDiff = Infinity;
  for (let i = 0; i < numLanes; i++) {
    const diff = Math.abs(egoX - laneCentres[i]);
    if (diff < minEgoDiff) {
      minEgoDiff = diff;
      currentLaneIdx = i;
    }
  }

  const currentClearance = clearances[currentLaneIdx];
  const isCurrentBlocked = currentClearance < dangerDist;

  let targetLaneIdx = currentLaneIdx;
  let shouldAvoid = false;
  let nextLockedLane = lockedTargetLane;

  // 1. Maintain commitment to active locked target lane (hysteresis against rapid swaying)
  if (lockedTargetLane !== null) {
    const targetClearance = clearances[lockedTargetLane];
    const reachedTarget = Math.abs(egoX - laneCentres[lockedTargetLane]) < 0.45 && Math.abs(egoHeading) < 0.12;

    // As long as the chosen target lane has adequate clearance (>= 15m), STAY COMMITTED!
    if (targetClearance >= 15) {
      targetLaneIdx = lockedTargetLane;
      shouldAvoid = true;
      // If arrived and aligned in target lane, and current lane is clear ahead, unlock
      if (reachedTarget && !isCurrentBlocked) {
        nextLockedLane = null;
      }
    } else {
      // Emergency: target lane became severely blocked, break lock to re-evaluate
      nextLockedLane = null;
    }
  }

  // 2. If current lane is blocked and no valid lock exists, select best escape lane and LOCK it
  if (isCurrentBlocked && nextLockedLane === null) {
    shouldAvoid = true;
    if (currentLaneIdx === 1) {
      // In center lane: evaluate Left (0) and Right (2)
      const leftClear = clearances[0] >= dangerDist;
      const rightClear = clearances[2] >= dangerDist;
      if (leftClear && !rightClear) {
        targetLaneIdx = 0;
      } else if (!leftClear && rightClear) {
        targetLaneIdx = 2;
      } else if (leftClear && rightClear) {
        // Both open: prefer Left (standard overtaking lane)
        targetLaneIdx = 0;
      } else {
        // Neither meets full dangerDist: pick the side with greater clearance
        targetLaneIdx = clearances[0] >= clearances[2] ? 0 : 2;
      }
    } else if (currentLaneIdx === 2) {
      targetLaneIdx = clearances[1] >= dangerDist ? 1 : (clearances[0] >= dangerDist ? 0 : 1);
    } else {
      targetLaneIdx = clearances[1] >= dangerDist ? 1 : (clearances[2] >= dangerDist ? 2 : 1);
    }
    nextLockedLane = targetLaneIdx;
  } else if (!isCurrentBlocked && nextLockedLane === null) {
    // Current lane is safe and clear. Cruising in current lane without unnecessary lane jumps
    targetLaneIdx = currentLaneIdx;
  }

  const targetX = laneCentres[targetLaneIdx];
  const lateralError = targetX - egoX;

  // Crisp approach angle (~13.8 degrees) so the entire vehicle and rear bumper quickly clear the lane
  const maxApproachAngle = 0.24;
  const desiredHeading = clamp(lateralError * 0.12, -maxApproachAngle, maxApproachAngle);
  const headingDiff = desiredHeading - egoHeading;

  // Responsive lateral steering with balanced smoothing (prevents hitting rear of car during lane shifts)
  const rawSteering = clamp(headingDiff * 1.8, -0.48, 0.48);
  const tacticalSteering = clamp(prevSteering * 0.50 + rawSteering * 0.50, -0.48, 0.48);

  const isAligned = Math.abs(lateralError) < 0.35 && Math.abs(egoHeading) < 0.08;
  const isManeuvering = shouldAvoid || targetLaneIdx !== currentLaneIdx || !isAligned;

  let turnSignal = 0;
  if (targetLaneIdx < currentLaneIdx || (targetLaneIdx === 0 && !isAligned)) {
    turnSignal = 1; // Left turn signal
  } else if (targetLaneIdx > currentLaneIdx || (targetLaneIdx === 2 && !isAligned)) {
    turnSignal = 2; // Right turn signal
  }

  return {
    clearances,
    currentLaneIdx,
    targetLaneIdx,
    targetX,
    shouldAvoid,
    isAligned,
    isManeuvering,
    tacticalSteering,
    turnSignal,
    lockedTargetLane: nextLockedLane,
  };
}

export function assessCorridors(
  egoX,
  egoZ,
  egoHeading,
  cars,
  laneCentres = LANE_CENTRES,
  dangerDist = ADAS_OBSTACLE_DANGER_M,
  horizon = ADAS_LOOKAHEAD_HORIZON_M,
  lockedTargetLane = null,
  prevSteering = 0,
) {
  const numLanes = laneCentres.length;
  const clearances = new Array(numLanes).fill(horizon);

  if (Array.isArray(cars)) {
    for (const car of cars) {
      if (!car || !car.userData?.active) continue;
      // Driving towards -z: car is ahead or alongside (within collision half-length CAR_HALF_Z * 2 = 4.4m)
      const forwardDist = egoZ - car.position.z;
      if (forwardDist < -4.5 || forwardDist > horizon) continue;

      let closestIdx = 0;
      let minDiff = Infinity;
      for (let i = 0; i < numLanes; i++) {
        const diff = Math.abs(car.position.x - laneCentres[i]);
        if (diff < minDiff) {
          minDiff = diff;
          closestIdx = i;
        }
      }
      const effectiveDist = Math.max(0, forwardDist);
      if (effectiveDist < clearances[closestIdx]) {
        clearances[closestIdx] = effectiveDist;
      }
    }
  }

  return evaluateCorridorManeuver(
    egoX,
    egoHeading,
    clearances,
    laneCentres,
    dangerDist,
    lockedTargetLane,
    prevSteering,
  );
}

/**
 * Assesses forward lane corridors and computes tactical maneuvers based solely on
 * pure visual computer vision detections from the camera feed (zero sim memory access).
 *
 * @param {number} egoX - Current lateral x coordinate of ego vehicle.
 * @param {number} egoHeading - Current yaw heading in radians.
 * @param {Array<Object>} visualObstacles - Array of objects with { distanceM, laneIdx }.
 * @param {number[]} laneCentres - Lateral centers of the lanes.
 * @param {number} dangerDist - Threshold below which a lane is blocked (default 45m).
 * @param {number} horizon - Max lookahead distance (default 80m).
 * @param {number|null} lockedTargetLane - Currently committed evasion lane index.
 * @param {number} prevSteering - Previous steering angle for low-pass smoothing.
 * @returns {Object} Corridor evaluation result.
 */
export function assessCorridorsFromVisualDetections(
  egoX,
  egoHeading,
  visualObstacles,
  laneCentres = LANE_CENTRES,
  dangerDist = ADAS_OBSTACLE_DANGER_M,
  horizon = ADAS_LOOKAHEAD_HORIZON_M,
  lockedTargetLane = null,
  prevSteering = 0,
  blindSpotHolds = null,
  currentTime = null,
) {
  const numLanes = laneCentres.length;
  const clearances = new Array(numLanes).fill(horizon);

  if (Array.isArray(visualObstacles)) {
    for (const obs of visualObstacles) {
      if (!obs || obs.distanceM === undefined || obs.laneIdx === undefined) continue;
      const dist = obs.distanceM;
      if (dist < 0 || dist > horizon) continue;
      const idx = obs.laneIdx;
      if (idx >= 0 && idx < numLanes) {
        if (dist < clearances[idx]) {
          clearances[idx] = Math.max(0, dist);
        }
        // If an oncoming vehicle is within 8m in lane idx, record a short blind spot hold
        // (0.5s is sufficient at 35 m/s closing speed to pass behind the rear bumper)
        if (Array.isArray(blindSpotHolds) && typeof currentTime === "number" && dist <= 8) {
          blindSpotHolds[idx] = Math.max(blindSpotHolds[idx] || 0, currentTime + 0.5);
        }
      }
    }
  }

  // Active blind spot memory: keep lane marked as blocked while an oncoming vehicle passes alongside
  if (Array.isArray(blindSpotHolds) && typeof currentTime === "number") {
    for (let i = 0; i < numLanes; i++) {
      if (currentTime < (blindSpotHolds[i] || 0)) {
        clearances[i] = Math.min(clearances[i], 3);
      }
    }
  }

  const result = evaluateCorridorManeuver(
    egoX,
    egoHeading,
    clearances,
    laneCentres,
    dangerDist,
    lockedTargetLane,
    prevSteering,
  );
  result.source = "vision";
  return result;
}

// Reusable offscreen canvas for high-performance visual obstacle detection
let _detectOffscreen = null;

/**
 * Pure Computer Vision Vehicle & Obstacle Detector.
 * Operates strictly on raw pixel buffers from the camera feed.
 * Zero simulation memory reading, zero leakage of 3D object positions.
 *
 * @param {HTMLCanvasElement|ImageData} source - Source canvas or raw ImageData (160x120 or scaled).
 * @param {number} egoX - Current lateral position of ego vehicle for optical compensation.
 * @param {number} egoHeading - Current yaw angle of ego vehicle for optical compensation.
 * @param {number[]} laneCentres - Lateral centers of lanes for perspective calibration.
 * @returns {Array<Object>} Array of detected obstacles sorted by distance.
 */
export function detectVisualObstacles(source, egoX = 0, egoHeading = 0, laneCentres = LANE_CENTRES) {
  if (!source) return [];

  let data = null;
  let imgWidth = 160;
  let imgHeight = 120;

  if (source.data && source.width && source.height) {
    // Passed raw ImageData directly
    data = source.data;
    imgWidth = source.width;
    imgHeight = source.height;
  } else if (typeof document !== "undefined" && typeof document.createElement === "function") {
    // Passed canvas: downscale / copy to 160x120 offscreen buffer
    if (!_detectOffscreen) {
      _detectOffscreen = document.createElement("canvas");
      _detectOffscreen.width = 160;
      _detectOffscreen.height = 120;
    }
    const ctx = _detectOffscreen.getContext("2d", { willReadFrequently: true });
    if (!ctx) return [];
    ctx.drawImage(source, 0, 0, 160, 120);
    const imgData = ctx.getImageData(0, 0, 160, 120);
    data = imgData.data;
  } else {
    return [];
  }

  // Exact 3D pinhole road projection:
  // Horizon at y = 59.5. Valid road detection range is y = 61 to 85 (Z ~ 4m..67m).
  // Road width 10.2m (half 5.1m) maps to halfRoad = 4.15 * dy pixels at row y.
  const grid = new Uint8Array(160 * 120);
  for (let y = 61; y <= 85; y++) {
    const dy = y - 59.5;
    const halfRoad = Math.round(4.15 * dy);
    const roadMinX = Math.max(0, 80 - halfRoad);
    const roadMaxX = Math.min(159, 80 + halfRoad);

    for (let x = roadMinX; x <= roadMaxX; x++) {
      const i = (y * 160 + x) * 4;
      const r = data[i], g = data[i + 1], b = data[i + 2];

      // 1. Asphalt road: neutral dark gray (r, g, b close together, moderate lum)
      const maxDiff = Math.max(Math.abs(r - g), Math.abs(g - b), Math.abs(r - b));
      const lum = 0.299 * r + 0.587 * g + 0.114 * b;
      const isAsphalt = maxDiff <= 25 && lum >= 10 && lum <= 110;

      // 2. Road lane markings (white dashed dividers, solid edge lines, warm yellow dashes)
      const isLaneMarking = (lum > 85 && maxDiff <= 35) || (r > 80 && g > 75 && b < 65);

      if (!isAsphalt && !isLaneMarking) {
        // 3. Genuine vehicle body or lighting:
        // Must have distinct color saturation (chroma >= 25) with sufficient luminance,
        // or dedicated high-intensity emissive headlamp / red taillight:
        const maxC = Math.max(r, g, b);
        const minC = Math.min(r, g, b);
        const chroma = maxC - minC;
        const isSaturatedBody = chroma >= 25 && lum >= 20;
        const isHeadlight = (r > 195 && g > 185 && b > 140);
        const isTaillight = (r > 160 && g < 60 && b < 60);

        if (isSaturatedBody || isHeadlight || isTaillight) {
          grid[y * 160 + x] = 1;
        }
      }
    }
  }

  // 8-way connected components with 2px horizontal bridging for headlights/windshields
  const visited = new Uint8Array(160 * 120);
  const rawBoxes = [];
  const neighbors = [
    [-1, 0], [1, 0], [0, -1], [0, 1],
    [-1, -1], [-1, 1], [1, -1], [1, 1],
    [-2, 0], [2, 0]
  ];

  for (let y = 61; y <= 85; y++) {
    for (let x = 0; x < 160; x++) {
      const idx = y * 160 + x;
      if (grid[idx] === 1 && !visited[idx]) {
        let bx1 = x, bx2 = x, by1 = y, by2 = y, count = 0;
        const q = [idx];
        visited[idx] = 1;
        while (q.length > 0) {
          const cur = q.pop();
          const cy = Math.floor(cur / 160);
          const cx = cur % 160;
          count++;
          if (cx < bx1) bx1 = cx;
          if (cx > bx2) bx2 = cx;
          if (cy < by1) by1 = cy;
          if (cy > by2) by2 = cy;

          for (const [dx, dy] of neighbors) {
            const nx = cx + dx;
            const ny = cy + dy;
            if (nx >= 0 && nx < 160 && ny >= 61 && ny <= 85) {
              const nIdx = ny * 160 + nx;
              if (grid[nIdx] === 1 && !visited[nIdx]) {
                visited[nIdx] = 1;
                q.push(nIdx);
              }
            }
          }
        }

        const bw = bx2 - bx1 + 1;
        const bh = by2 - by1 + 1;
        const dy = Math.max(by2 - 59.5, 1);
        const distM = Math.max(1, Math.round(100 / dy));
        const aspect = bw / bh;
        const area = bw * bh;
        const density = count / area;

        // Geometric vehicle filters to eliminate highway markings & artifacts:
        // 1. Reject thin vertical stripes (lane markings, divider lines, guardrail posts): aspect < 0.45 or (bw <= 2 && bh >= 3)
        // 2. Reject wide horizontal bands (horizon, road seams, glare stripes): aspect > 4.5 or (bh <= 2 && bw >= 12)
        // 3. Density: a vehicle is a solid 2D shape (density >= 0.40)
        // 4. Minimum pixel count and dimensions:
        const isSkinnyStripe = aspect < 0.45 || (bw <= 2 && bh >= 3);
        const isWideStripe = aspect > 4.5 || (bh <= 2 && bw >= 12);
        const isSolidObject = density >= 0.40;
        const isRightSize = distM <= 32
          ? (count >= 8 && bw >= 3 && bh >= 2)
          : (count >= 6 && bw >= 3 && bh >= 2);

        if (!isSkinnyStripe && !isWideStripe && isSolidObject && isRightSize && by2 >= 61) {
          const centerX = (bx1 + bx2) / 2;
          const thetaCam = (centerX - 80) / 48;
          const thetaWorld = thetaCam + egoHeading;
          const worldX = egoX + distM * thetaWorld;

          // Road lateral boundary filter:
          // Ignore detections outside paved road width (guardrails, shoulder, grass)
          if (Math.abs(worldX) > 4.8) {
            continue;
          }

          // Lane divider line rejection:
          // Dashed lines lie at X = -1.7 and X = +1.7. If a narrow blob (bw <= 4) is directly on the divider, reject it
          if (bw <= 4 && (Math.abs(worldX - (-1.7)) < 0.22 || Math.abs(worldX - 1.7) < 0.22)) {
            continue;
          }

          let laneIdx = 1;
          if (worldX < -1.7) {
            laneIdx = 0;
          } else if (worldX > 1.7) {
            laneIdx = 2;
          }

          rawBoxes.push({
            laneIdx,
            laneName: ["چپ", "وسط", "راست"][laneIdx],
            distanceM: distM,
            worldX,
            bbox: [bx1, by1, bw, bh],
            pixelCount: count,
          });
        }
      }
    }
  }

  // Merge overlapping or close bounding boxes in the same lane (< 6m longitudinal distance)
  const merged = [];
  for (const b of rawBoxes) {
    const existing = merged.find(m => m.laneIdx === b.laneIdx && Math.abs(m.distanceM - b.distanceM) <= 6);
    if (existing) {
      const x1 = Math.min(existing.bbox[0], b.bbox[0]);
      const y1 = Math.min(existing.bbox[1], b.bbox[1]);
      const x2 = Math.max(existing.bbox[0] + existing.bbox[2], b.bbox[0] + b.bbox[2]);
      const y2 = Math.max(existing.bbox[1] + existing.bbox[3], b.bbox[1] + b.bbox[3]);
      existing.bbox = [x1, y1, x2 - x1, y2 - y1];
      existing.distanceM = Math.min(existing.distanceM, b.distanceM);
      existing.pixelCount += b.pixelCount;
    } else {
      merged.push({ ...b });
    }
  }

  merged.sort((a, b) => a.distanceM - b.distanceM);
  return merged;
}

/**
 * Renders visual obstacle detection overlays (bounding boxes, distance and lane tags)
 * onto a 2D rendering context.
 *
 * @param {CanvasRenderingContext2D} ctx - 2D context to draw overlays onto.
 * @param {Array<Object>} obstacles - Detected obstacles.
 * @param {number} targetLaneIdx - Currently targeted corridor lane index (0, 1, or 2).
 * @param {number} width - Canvas width in pixels (default 160).
 * @param {number} height - Canvas height in pixels (default 120).
 */
export function drawVisualDetectionOverlays(ctx, obstacles, targetLaneIdx = 1, width = 160, height = 120) {
  if (!ctx || !Array.isArray(obstacles) || obstacles.length === 0) return;
  ctx.save();
  for (const obs of obstacles) {
    const [x, y, w, h] = obs.bbox;
    const isDanger = obs.distanceM < 30;
    const color = isDanger ? "#f59e0b" : "#22c55e";

    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.strokeRect(x, y, w, h);

    // Corner reticles for high-tech HUD look
    const corner = Math.min(Math.max(Math.floor(w / 3), 2), 4);
    ctx.lineWidth = 2.0;
    // Top-left
    ctx.beginPath();
    ctx.moveTo(x, y + corner);
    ctx.lineTo(x, y);
    ctx.lineTo(x + corner, y);
    ctx.stroke();
    // Top-right
    ctx.beginPath();
    ctx.moveTo(x + w - corner, y);
    ctx.lineTo(x + w, y);
    ctx.lineTo(x + w, y + corner);
    ctx.stroke();

    // Distance and lane label
    const laneTag = obs.laneIdx === 0 ? "چپ" : (obs.laneIdx === 1 ? "وسط" : "راست");
    const tagText = `${obs.distanceM}m | ${laneTag}`;
    ctx.font = "bold 8.5px Consolas, monospace";
    const textW = ctx.measureText(tagText).width;
    const tagY = Math.max(y - 12, 9);
    const tagX = Math.max(Math.min(x, width - textW - 5), 2);

    ctx.fillStyle = "rgba(15, 23, 42, 0.85)";
    ctx.fillRect(tagX - 2, tagY - 2, textW + 4, 11);
    ctx.strokeStyle = color;
    ctx.lineWidth = 0.8;
    ctx.strokeRect(tagX - 2, tagY - 2, textW + 4, 11);

    ctx.fillStyle = color;
    ctx.fillText(tagText, tagX, tagY + 7);
  }
  ctx.restore();
}

/**
 * Delay before the ``attempt``-th reconnect, or ``null`` when the budget of
 * attempts is used up. The delay doubles every attempt and saturates. Only
 * whole attempts are accepted, matching the phone side helper.
 */
export function nextReconnectDelayMs(attempt) {
  if (!Number.isInteger(attempt) || attempt < 1 || attempt > MAX_RECONNECT_ATTEMPTS) {
    return null;
  }
  const delay = RECONNECT_BASE_DELAY_MS * 2 ** (attempt - 1);
  return Math.min(delay, RECONNECT_MAX_DELAY_MS);
}

/**
 * Mirror of the server side ``sanitize_command``: clip every field into its
 * valid range and return ``null`` when a field is missing or not a number.
 *
 * Keeping the two implementations in step means the renderer never has to
 * defend itself against a value the server let through, and a frame the server
 * would reject is dropped here as well.
 */
export function normalizeCommand(payload) {
  if (payload === null || typeof payload !== "object" || Array.isArray(payload)) {
    return null;
  }
  const command = {};
  for (const [field, low, high, isInteger] of CONTROL_FIELDS) {
    const value = payload[field];
    // typeof true is "boolean", so booleans are refused like on the server.
    if (typeof value !== "number" || !Number.isFinite(value)) {
      return null;
    }
    const clipped = clamp(value, low, high);
    command[field] = isInteger ? Math.round(clipped) : clipped;
  }
  return command;
}

/**
 * One longitudinal step of the player car, exactly as the released physics
 * spells it out: brake wins over throttle, released pedals leave coasting
 * friction,
 * and the result is clipped to 0..maxSpeed. Returns the speed in m/s.
 */
export function stepLongitudinalSpeed(speed, throttle, brake, dt) {
  let next;
  if (brake > 0.1) {
    next = speed - BRAKE_DECELERATION_MS2 * brake * dt;
  } else if (throttle > 0.1) {
    next = speed + ACCELERATION_MS2 * throttle * dt;
  } else {
    next = speed - COASTING_FRICTION_MS2 * dt;
  }
  return clamp(next, 0, MAX_SPEED_MS);
}

/**
 * One heading step. The car only yaws while rolling faster than
 * HEADING_MIN_SPEED_MS, and the steering authority ramps in with speed
 * (min(1, speed / HEADING_SPEED_REF_MS)) so the car cannot spin on the spot.
 * Returns the new heading in radians.
 */
export function stepHeading(heading, steering, speed, dt) {
  if (speed <= HEADING_MIN_SPEED_MS) {
    return heading;
  }
  const authority = Math.min(1, speed / HEADING_SPEED_REF_MS);
  return heading + STEERING_SIGN * steering * STEER_RATE * authority * dt;
}

/**
 * One position step. Forward is -z (the camera looks down -z, so world +x is
 * the *right* side of the screen): steering positive -> heading positive ->
 * x increases -> the car turns right on screen, matching DAVE-2.
 * Returns the new {x, z}.
 */
export function stepWorldPosition(x, z, heading, speed, dt) {
  return {
    x: x + Math.sin(heading) * speed * dt,
    z: z - Math.cos(heading) * speed * dt,
  };
}

/** The invisible wall clamps the lateral position to |x| = wallX. */
export function clampToWall(x, wallX = ROAD_WALL_X) {
  return clamp(x, -wallX, wallX);
}

/** Heading in radians, normalised to degrees in the range (-180, 180]. */
export function headingDegrees(heading) {
  const deg = (heading * 180) / Math.PI;
  return deg - 360 * Math.ceil(deg / 360 - 0.5);
}

/**
 * Whether the blinker lamps are lit at ``phase``. The phase advances by
 * BLINKER_PHASE_RATE per second, so the lamp toggles three times a second.
 */
export function blinkerIsLit(phase) {
  return Math.floor(phase) % 2 === 0;
}

/**
 * First-person camera pose for an ego car at (x, z) with ``heading``. The
 * camera sits on the front bumper (CAMERA_BUMPER_M ahead of the centre along
 * the facing direction) and looks CAMERA_LOOK_AHEAD_M ahead at eye height.
 * Facing direction for heading h is (sin h, -cos h).
 */
export function cameraPose(x, z, heading) {
  const sinH = Math.sin(heading);
  const cosH = Math.cos(heading);
  return {
    position: {
      x: x + CAMERA_BUMPER_M * sinH,
      y: CAMERA_EYE_HEIGHT,
      z: z - CAMERA_BUMPER_M * cosH,
    },
    lookAt: {
      x: x + CAMERA_LOOK_AHEAD_M * sinH,
      y: CAMERA_EYE_HEIGHT,
      z: z - CAMERA_LOOK_AHEAD_M * cosH,
    },
  };
}

/**
 * Axis-aligned collision test in closed form. The simulation uses the same
 * expression through THREE.Box3.set(min, max) + intersectsBox — this helper
 * mirrors it exactly (per axis: a.max >= b.min && a.min <= b.max) so the
 * Node tests can pin the behaviour without WebGL.
 */
export function carsCollide(egoX, egoZ, carX, carZ) {
  return (
    Math.abs(egoX - carX) <= CAR_HALF_X + CAR_HALF_X &&
    Math.abs(egoZ - carZ) <= CAR_HALF_Z + CAR_HALF_Z
  );
}

/** Telemetry frame sent from the scene to the phone. Commands are never included. */
export function telemetryFrame(speedMs) {
  const speedKmh = clamp(Math.round(speedToKmh(speedMs)), 0, 400);
  return { type: "telemetry", speed_kmh: speedKmh };
}

/**
 * Hazard frame sent when the car collides or grinds the wall. The server
 * relays it to the phone dashboard, where it shows a warning banner; on the
 * monitor the matching screen flash is visible to the phone camera as well.
 * Unknown kinds return ``null`` so a typo cannot send garbage.
 */
export function hazardFrame(kind) {
  if (!HAZARD_KINDS.includes(kind)) {
    return null;
  }
  return { type: "hazard", hazard: kind };
}

/** Convert m/s into km/h. */
export function speedToKmh(speed) {
  return speed * 3.6;
}

/** Frame rate smoothed over ``windowSeconds``, in frames per second. */
export function smoothFrameRate(previous, dt, windowSeconds = 0.5) {
  if (dt <= 0) {
    return previous;
  }
  const alpha = clamp(dt / windowSeconds, 0, 1);
  return previous + (1 / dt - previous) * alpha;
}

/* ------------------------------------------------------------------------- *
 * 3. Scene construction
 * ------------------------------------------------------------------------- */

const SKY_COLOUR = 0xbcd4e6;
const FOG_NEAR = 90;
const FOG_FAR = 300;
const GROUND_COLOUR = 0x4a6b3a;
const LANE_DIVIDER_X = [-2, 2];
const SCENERY_SPACING = 22;
/** Scenery stays within this distance behind the car before it is recycled. */
const SCENERY_KEEP_BEHIND_M = 25;

/** Hard ceiling on drawing-buffer oversampling; above it GPUs start to crawl. */
export const MAX_PIXEL_RATIO = 2;

/**
 * The pixel ratio the renderer should rasterize at, clamped to sane bounds.
 *
 * Re-derived on every resize: when the user zooms the browser or drags the
 * window onto a second monitor, ``devicePixelRatio`` changes silently — a
 * ratio captured once at startup leaves the canvas blurry or over-crisp.
 *
 * @param {number} devicePixelRatio The window's current device pixel ratio.
 * @returns {number} A positive ratio of at most {@link MAX_PIXEL_RATIO}.
 */
export function choosePixelRatio(devicePixelRatio) {
  const raw = Number(devicePixelRatio) || 1;
  return Math.min(Math.max(raw, 1), MAX_PIXEL_RATIO);
}

/**
 * The CSS-pixel box the 3D scene must exactly fill.
 *
 * The stylesheet locks the canvas to the document viewport
 * (``position: fixed; inset: 0``), so the canvas's own layout box
 * (``clientWidth``/``clientHeight``) is the ground truth of "the screen":
 * it already accounts for browser zoom, snap layouts, scrollbars, docked
 * devtools and fullscreen transitions. ``window.innerWidth/innerHeight`` are
 * only the fallback for a canvas that is not laid out yet. Dimensions never
 * drop below 1 so WebGL never sees a zero-size framebuffer.
 *
 * @param {{clientWidth?: number, clientHeight?: number}|null} canvas The scene canvas.
 * @param {{innerWidth?: number, innerHeight?: number}} [windowLike] Viewport to fall back to.
 * @returns {{width: number, height: number}} CSS pixel dimensions.
 */
export function sceneViewport(canvas, windowLike) {
  const viewport = windowLike ?? (typeof window === "undefined" ? {} : window);
  const width = canvas?.clientWidth ?? 0;
  const height = canvas?.clientHeight ?? 0;
  if (width > 0 && height > 0) {
    return { width, height };
  }
  return {
    width: Math.max(1, viewport.innerWidth ?? 1),
    height: Math.max(1, viewport.innerHeight ?? 1),
  };
}

function createRenderer(canvas) {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true });
  const { width, height } = sceneViewport(canvas);
  renderer.setPixelRatio(choosePixelRatio(window.devicePixelRatio));
  // The third argument keeps the stylesheet owner of the CSS size —
  // position:fixed + inset:0 + 100%/100% does the fitting, never inline px.
  renderer.setSize(width, height, false);
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  return renderer;
}

function createScene() {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(SKY_COLOUR);
  // The fog hides the far end of the road, where traffic appears.
  scene.fog = new THREE.Fog(SKY_COLOUR, FOG_NEAR, FOG_FAR);
  return scene;
}

function createCamera(canvas) {
  const { width, height } = sceneViewport(canvas);
  const camera = new THREE.PerspectiveCamera(
    CAMERA_FOV,
    width / height,
    0.1,
    600,
  );
  // The first frame of update() snaps this to the bumper pose.
  const pose = cameraPose(0, 0, 0);
  camera.position.set(pose.position.x, pose.position.y, pose.position.z);
  camera.lookAt(pose.lookAt.x, pose.lookAt.y, pose.lookAt.z);
  return camera;
}

function createLights(scene) {
  scene.add(new THREE.AmbientLight(0xffffff, 0.55));

  const sun = new THREE.DirectionalLight(0xfff4e0, 1.1);
  sun.castShadow = true;
  sun.shadow.mapSize.set(1024, 1024);
  // The shadow volume only has to cover the road around the player car.
  sun.shadow.camera.left = -30;
  sun.shadow.camera.right = 30;
  sun.shadow.camera.top = 40;
  sun.shadow.camera.bottom = -70;
  sun.shadow.camera.near = 1;
  sun.shadow.camera.far = 130;
  scene.add(sun);
  // The target is steered to the ego car every frame, so the light direction
  // stays constant while the shadow frustum follows the endless drive.
  scene.add(sun.target);
  return sun;
}

function createRoadRoot(scene) {
  const root = new THREE.Group();

  const ground = new THREE.Mesh(
    new THREE.PlaneGeometry(240, ROAD_LENGTH + 60),
    new THREE.MeshStandardMaterial({ color: GROUND_COLOUR, roughness: 1 }),
  );
  ground.rotation.x = -Math.PI / 2;
  ground.position.set(0, -0.02, ROAD_Z_CENTRE);
  ground.receiveShadow = true;
  root.add(ground);

  const road = new THREE.Mesh(
    new THREE.PlaneGeometry(ROAD_WIDTH, ROAD_LENGTH),
    new THREE.MeshStandardMaterial({ color: ROAD_COLOUR, roughness: 0.95 }),
  );
  road.rotation.x = -Math.PI / 2;
  road.position.set(0, 0, ROAD_Z_CENTRE);
  road.receiveShadow = true;
  root.add(road);

  const lineSpan = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(0, 0.02, ROAD_Z_CENTRE - ROAD_LENGTH / 2),
    new THREE.Vector3(0, 0.02, ROAD_Z_CENTRE + ROAD_LENGTH / 2),
  ]);

  const dashMaterial = new THREE.LineDashedMaterial({
    color: 0xffffff,
    dashSize: DASH_SIZE,
    gapSize: DASH_GAP,
  });
  for (const x of LANE_DIVIDER_X) {
    const line = new THREE.Line(lineSpan, dashMaterial);
    line.position.x = x;
    line.computeLineDistances();
    root.add(line);
  }

  const edgeMaterial = new THREE.LineBasicMaterial({ color: 0xffffff });
  const halfWidth = ROAD_WIDTH / 2 - 0.5;
  for (const x of [-halfWidth, halfWidth]) {
    const line = new THREE.Line(lineSpan, edgeMaterial);
    line.position.x = x;
    root.add(line);
  }

  const railMaterial = new THREE.MeshStandardMaterial({
    color: 0x9e9e9e,
    roughness: 0.5,
    metalness: 0.6,
  });
  const railGeometry = new THREE.BoxGeometry(0.2, 0.5, ROAD_LENGTH);
  for (const x of [-(ROAD_WIDTH / 2 + 0.4), ROAD_WIDTH / 2 + 0.4]) {
    const rail = new THREE.Mesh(railGeometry, railMaterial);
    rail.position.set(x, 0.45, ROAD_Z_CENTRE);
    rail.castShadow = true;
    root.add(rail);
  }

  scene.add(root);
  return root;
}

/**
 * Trees and lamp posts beside the road. They are static world objects: when
 * the car has driven past one by SCENERY_KEEP_BEHIND_M it is moved one span
 * ahead, so the decoration is endless too.
 */
function createScenery(scene) {
  const group = new THREE.Group();

  const trunkGeometry = new THREE.CylinderGeometry(0.22, 0.3, 2.2, 6);
  const trunkMaterial = new THREE.MeshStandardMaterial({ color: 0x5d4037, roughness: 1 });
  const crownGeometry = new THREE.ConeGeometry(1.7, 4.4, 7);
  const crownMaterial = new THREE.MeshStandardMaterial({ color: 0x2e7d32, roughness: 1 });
  const poleGeometry = new THREE.CylinderGeometry(0.12, 0.16, 7, 8);
  const poleMaterial = new THREE.MeshStandardMaterial({ color: 0x616161, roughness: 0.7 });
  const headGeometry = new THREE.BoxGeometry(0.5, 0.22, 1.2);
  const headMaterial = new THREE.MeshStandardMaterial({
    color: 0x37474f,
    emissive: 0xfff2c4,
    emissiveIntensity: 0.25,
  });

  const span = 360; // scenery window: span behind the car + KEEP_BEHIND ahead
  const startZ = -span + SCENERY_KEEP_BEHIND_M;

  for (let z = startZ; z < SCENERY_KEEP_BEHIND_M; z += SCENERY_SPACING) {
    for (const side of [-1, 1]) {
      // Trees are jittered so the two rows do not look like a corridor.
      const tree = new THREE.Group();
      const trunk = new THREE.Mesh(trunkGeometry, trunkMaterial);
      trunk.position.y = 1.1;
      trunk.castShadow = true;
      const crown = new THREE.Mesh(crownGeometry, crownMaterial);
      crown.position.y = 3.9;
      crown.castShadow = true;
      tree.add(trunk, crown);
      tree.position.set(
        side * (14 + Math.abs((z * 7) % 9)),
        0,
        z + Math.abs((z * 3) % SCENERY_SPACING),
      );
      group.add(tree);
    }
  }

  for (let z = startZ; z < SCENERY_KEEP_BEHIND_M; z += SCENERY_SPACING * 2) {
    for (const side of [-1, 1]) {
      const lamp = new THREE.Group();
      const pole = new THREE.Mesh(poleGeometry, poleMaterial);
      pole.position.y = 3.5;
      pole.castShadow = true;
      const head = new THREE.Mesh(headGeometry, headMaterial);
      head.position.set(-side * 0.5, 6.9, 0);
      lamp.add(pole, head);
      lamp.position.set(side * 11, 0, z);
      group.add(lamp);
    }
  }

  scene.add(group);
  group.userData.span = span;
  return group;
}

/**
 * Build cars out of boxes and cylinders. Geometries and the shared materials
 * are created once, so a pooled car costs four extra meshes and no GPU upload.
 * A built car faces -z.
 */
function createCarFactory() {
  const geometries = {
    body: new THREE.BoxGeometry(1.8, 1.2, 4),
    cabin: new THREE.BoxGeometry(1.5, 0.7, 2),
    glass: new THREE.BoxGeometry(1.42, 0.62, 0.12),
    wheel: new THREE.CylinderGeometry(0.36, 0.36, 0.28, 16),
    lamp: new THREE.BoxGeometry(0.34, 0.16, 0.08),
  };
  // Turn the wheel so that its axis points along x.
  geometries.wheel.rotateZ(Math.PI / 2);

  const paint = new Map();
  const cabinMaterial = new THREE.MeshStandardMaterial({ color: 0x263238, roughness: 0.6 });
  const glassMaterial = new THREE.MeshStandardMaterial({
    color: 0x1b2733,
    roughness: 0.15,
    metalness: 0.5,
  });
  const tyreMaterial = new THREE.MeshStandardMaterial({ color: 0x14161a, roughness: 0.95 });
  const headMaterial = new THREE.MeshStandardMaterial({
    color: 0xfff6d8,
    emissive: 0xfff0c0,
    emissiveIntensity: 0.7,
  });
  const tailMaterial = new THREE.MeshStandardMaterial({
    color: 0x8e1a1a,
    emissive: 0xff2020,
    emissiveIntensity: 0.5,
  });

  function bodyMaterial(colour) {
    if (!paint.has(colour)) {
      paint.set(colour, new THREE.MeshStandardMaterial({ color: colour, roughness: 0.35 }));
    }
    return paint.get(colour);
  }

  /**
   * Return a car group whose origin sits on the road. The body keeps the
   * specified centre at y = 0.6 above that origin, the wheels touch the
   * ground, and the car faces -z.
   */
  function buildCar(colour) {
    const car = new THREE.Group();

    const body = new THREE.Mesh(geometries.body, bodyMaterial(colour));
    body.position.y = 0.6;
    body.castShadow = true;
    car.add(body);

    const cabin = new THREE.Mesh(geometries.cabin, cabinMaterial);
    cabin.position.set(0, 1.55, 0.2);
    cabin.castShadow = true;
    car.add(cabin);

    const windshield = new THREE.Mesh(geometries.glass, glassMaterial);
    windshield.position.set(0, 1.55, -0.55);
    car.add(windshield);

    const wheelGeometry = geometries.wheel;
    for (const [x, z] of [
      [-0.85, -1.3],
      [0.85, -1.3],
      [-0.85, 1.3],
      [0.85, 1.3],
    ]) {
      const wheel = new THREE.Mesh(wheelGeometry, tyreMaterial);
      wheel.position.set(x, 0.36, z);
      car.add(wheel);
    }

    for (const side of [-1, 1]) {
      const head = new THREE.Mesh(geometries.lamp, headMaterial);
      head.position.set(side * 0.6, 0.75, -2.02);
      car.add(head);
      const tail = new THREE.Mesh(geometries.lamp, tailMaterial);
      tail.position.set(side * 0.6, 0.75, 2.02);
      car.add(tail);
    }

    car.userData.body = body;
    return car;
  }

  return { buildCar, bodyMaterial };
}

/* ------------------------------------------------------------------------- *
 * 4. The simulation
 * ------------------------------------------------------------------------- */

/**
 * Create the scene, the cars and the state, and drive them from the frame
 * loop: physics, blinkers, traffic, collisions, HUD, render.
 *
 * ``hooks.hud`` receives the HUD readouts, ``hooks.onTelemetry`` the periodic
 * speed frame and ``hooks.onHazard`` every collision/wall event.
 */
function createSimulation(canvas, hooks) {
  const { hud, onTelemetry, onHazard } = hooks;

  const renderer = createRenderer(canvas);
  const scene = createScene();
  // Blinker lamps are camera children, so the camera must be part of the graph.
  const camera = createCamera();
  scene.add(camera);
  const sun = createLights(scene);
  const roadRoot = createRoadRoot(scene);
  const scenery = createScenery(scene);
  const { buildCar, bodyMaterial } = createCarFactory();

  /**
   * The state of the ego car, in the units of the released physics. The car
   * drives towards -z; heading is in radians, 0 = straight ahead.
   */
  const state = {
    x: 0,
    z: 0,
    speed: 0, // m/s
    heading: 0,
    time: 0, // simulation clock, seconds
    turnSignal: 0, // 0 off, 1 left, 2 right
    blinkerPhase: 0,
    collisions: 0,
    wallHits: 0,
    frozenUntil: 0, // post-collision pause
    immuneUntil: 0, // no collision checks during the reset window
    pendingReset: false, // return to the centre lane once the freeze ends
    lastWallEventAt: -Infinity,
    controls: { steering: 0, throttle: 0, brake: 0, turn_signal: 0 }, // keyboard
  };

  /** The latest phone and browser autopilot commands with simulated and real wall arrival times. */
  const external = {
    command: null,
    at: -Infinity,
    wallTime: 0,
    browserCommand: null,
    browserAt: -Infinity,
    browserWallTime: 0,
  };

  /**
   * The ego mesh is the physics body. The camera rides on its front bumper,
   * so the body itself stays invisible — the driver must not see a hood-less
   * car floating in front of the lens.
   */
  const egoCar = buildCar(EGO_COLOUR);
  egoCar.visible = false;
  scene.add(egoCar);

  /** Reused collision boxes; set() avoids a per-frame allocation. */
  const egoBox = new THREE.Box3();
  const trafficBox = new THREE.Box3();

  /** Axis-aligned car box centred at (x, z), padded y range for the 3D test. */
  function setCarBox(box, x, z) {
    box.min.set(x - CAR_HALF_X, 0.1, z - CAR_HALF_Z);
    box.max.set(x + CAR_HALF_X, 1.5, z + CAR_HALF_Z);
    return box;
  }

  /**
   * Two small yellow hood-corner lamps, parented to the camera so they stay
   * in view: in first person the car body is invisible, and this is how the
   * 3 Hz blink stays visible to the human *and* to the phone camera.
   * Each lamp swaps between an "on" and an "off" material.
   */
  function createBlinkers() {
    const geometry = new THREE.BoxGeometry(BLINKER_SIZE.x, BLINKER_SIZE.y, BLINKER_SIZE.z);
    const onMaterial = new THREE.MeshStandardMaterial({
      color: BLINKER_ON_COLOUR,
      emissive: BLINKER_ON_COLOUR,
      emissiveIntensity: 0.9,
      roughness: 0.4,
    });
    const offMaterial = new THREE.MeshStandardMaterial({
      color: BLINKER_OFF_COLOUR,
      roughness: 0.6,
    });
    const left = new THREE.Mesh(geometry, offMaterial);
    left.position.set(-BLINKER_CAMERA_OFFSET.x, BLINKER_CAMERA_OFFSET.y, BLINKER_CAMERA_OFFSET.z);
    left.userData.signal = 1; // turn_signal class: 1 = left
    const right = new THREE.Mesh(geometry, offMaterial);
    right.position.set(BLINKER_CAMERA_OFFSET.x, BLINKER_CAMERA_OFFSET.y, BLINKER_CAMERA_OFFSET.z);
    right.userData.signal = 2; // turn_signal class: 2 = right
    camera.add(left, right);
    return [
      { mesh: left, onMaterial, offMaterial },
      { mesh: right, onMaterial, offMaterial },
    ];
  }

  const blinkers = createBlinkers();

  let activeRoadPreset = ROAD_PRESETS.narrow;
  let activeTrafficPreset = TRAFFIC_PRESETS.dense;

  state.wallX = activeRoadPreset.wallX;

  const traffic = {
    cars: [], // every pooled car, active or not
    timer: pickSpawnDelay(Math.random, activeTrafficPreset.spawnDelayMin, activeTrafficPreset.spawnDelayMax),
    laneCentres: activeRoadPreset.laneCentres,
    spawnDelayMin: activeTrafficPreset.spawnDelayMin,
    spawnDelayMax: activeTrafficPreset.spawnDelayMax,
    spawnGapMin: activeTrafficPreset.spawnGapMin,
    spawnGapMax: activeTrafficPreset.spawnGapMax,
  };

  function updateRoadVisuals(preset) {
    activeRoadPreset = preset;
    state.wallX = preset.wallX;
    traffic.laneCentres = preset.laneCentres;

    const { road, dashLines, edgeLines, rails } = roadRoot.userData;
    if (road) {
      road.scale.x = preset.roadWidth / ROAD_WIDTH;
    }
    if (dashLines && dashLines.length >= 2) {
      dashLines[0].position.x = preset.dividerXs[0];
      dashLines[1].position.x = preset.dividerXs[1];
    }
    if (edgeLines && edgeLines.length >= 2) {
      const edgeX = preset.roadWidth / 2 - 0.5;
      edgeLines[0].position.x = -edgeX;
      edgeLines[1].position.x = edgeX;
    }
    if (rails && rails.length >= 2) {
      const railX = preset.roadWidth / 2 + 0.4;
      rails[0].position.x = -railX;
      rails[1].position.x = railX;
    }
  }

  function preseedTraffic() {
    for (const car of traffic.cars) {
      releaseTrafficCar(car);
    }
    const count = activeTrafficPreset.initialCount || 0;
    if (count <= 0) return;

    const laneL = traffic.laneCentres[0];
    const laneC = traffic.laneCentres[1];
    const laneR = traffic.laneCentres[2];

    // Realistic highway scenario:
    // 1. Ego starts in Center lane (spawn lane). Center runway is guaranteed clear for >= 10s.
    // 2. Right and Left lanes have traffic far ahead.
    const initialSlots = [
      { lane: laneR, dist: 120 },
      { lane: laneR, dist: 210 },
      { lane: laneR, dist: 310 },
    ];

    for (let i = 0; i < Math.min(count, initialSlots.length); i++) {
      const slot = initialSlots[i];
      spawnTrafficCar(slot.lane, state.z - slot.dist);
    }
  }

  /** Take an inactive car from the pool, or build one if there is room. */
  function acquireTrafficCar() {
    for (const car of traffic.cars) {
      if (!car.userData.active) {
        return car;
      }
    }
    if (!canSpawnTraffic(traffic.cars.length)) {
      return null;
    }
    // The paint colour is assigned on every spawn, so the initial one only has
    // to be a valid material.
    const car = buildCar(TRAFFIC_COLOURS[0]);
    car.userData.active = false;
    scene.add(car);
    traffic.cars.push(car);
    return car;
  }

  /**
   * Place a traffic car in a random lane, SPAWN_GAP ahead of the ego car,
   * driving towards it (+z, so the mesh faces +z via rotation.y = PI).
   * Ensures at least one lane is ALWAYS guaranteed unobstructed across the road.
   */
  function spawnTrafficCar(customLane = null, customZ = null) {
    const car = acquireTrafficCar();
    if (car === null) {
      return false;
    }
    const laneCentres = traffic.laneCentres || LANE_CENTRES;
    const candidateZ = customZ !== null ? customZ : trafficSpawnZ(state.z, pickSpawnGap(Math.random, traffic.spawnGapMin, traffic.spawnGapMax));

    // Determine viable lanes at candidateZ:
    // 1. Must have no car in the same lane within MIN_LANE_GAP_M
    // 2. Must keep at least ONE lane guaranteed open across the road:
    //    countBlockedLanesAtZ must remain < laneCentres.length - 1
    let candidateLanes = customLane !== null ? [customLane] : laneCentres;
    // User requirement: No cars spawn on the spawn lane (center lane) for the first 10 seconds
    if (state.time < 10.0 && customLane === null) {
      const spawnLaneX = laneCentres[1];
      const nonSpawn = candidateLanes.filter(lx => Math.abs(lx - spawnLaneX) > 0.5);
      if (nonSpawn.length > 0) {
        candidateLanes = nonSpawn;
      }
    }
    const allowedLanes = [];

    for (const laneX of candidateLanes) {
      // Check collision/stacking in candidate lane
      let laneBlocked = false;
      for (const other of traffic.cars) {
        if (other !== car && other.userData.active) {
          if (Math.abs(other.position.x - laneX) < 0.6 && Math.abs(other.position.z - candidateZ) < MIN_LANE_GAP_M) {
            laneBlocked = true;
            break;
          }
        }
      }
      if (laneBlocked) continue;

      // Check multi-lane blockage constraint:
      // Guarantee at least TWO open lanes in any 50m longitudinal slice so ego always has clear escape routes
      if (customLane === null) {
        const blockedNow = countBlockedLanesAtZ(traffic.cars, laneCentres, candidateZ, 50);
        if (blockedNow >= 1) {
          continue;
        }
      }
      allowedLanes.push(laneX);
    }

    if (allowedLanes.length === 0) {
      return false;
    }

    const x = allowedLanes[Math.floor(Math.random() * allowedLanes.length)];
    car.userData.speed = pickTrafficSpeed();
    car.userData.body.material = bodyMaterial(pickTrafficColour());
    car.position.set(x, 0, candidateZ);
    car.rotation.y = Math.PI;
    car.userData.active = true;
    car.visible = true;
    return true;
  }

  function releaseTrafficCar(car) {
    car.userData.active = false;
    car.visible = false;
  }

  function updateTraffic(dt) {
    for (const car of traffic.cars) {
      if (!car.userData.active) {
        continue;
      }
      car.position.z += car.userData.speed * dt; // towards the ego car
      if (car.position.z > state.z + DESPAWN_BEHIND_M) {
        releaseTrafficCar(car);
      }
    }
    traffic.timer -= dt;
    if (traffic.timer <= 0) {
      traffic.timer = pickSpawnDelay(Math.random, traffic.spawnDelayMin, traffic.spawnDelayMax);
      spawnTrafficCar();
    }
  }

  const settings = {
    drivingMode: "camera",
    isBrowserAutopilotActive: false,
    biasCorrection: true,
    adasAvoidance: true,
    obstacleSource: "vision", // "vision" (pure CV) | "ground_truth" (sim memory) | "off"
    steeringGain: 1.5,
    invertSteering: false,
    onRestart: null,
  };

  const corridorState = {
    lockedTargetLane: null,
    prevSteering: 0,
    blindSpotHolds: [0, 0, 0],
  };

  let latestVisualObstacles = [];
  let detectTimer = 0;
  const visualTracker = [
    { hits: 0, distanceM: 80, lastSeen: 0 },
    { hits: 0, distanceM: 80, lastSeen: 0 },
    { hits: 0, distanceM: 80, lastSeen: 0 },
  ];

  const PARKED_CONTROLS = Object.freeze({ steering: 0, throttle: 0, brake: 1, turn_signal: 0 });

  /**
   * Determines whether the dynamic simulation world should run.
   * Gated strictly so that the simulation does NOT run on its own:
   * 1. In "camera" mode: waits until phone dashboard connects and clicks "شروع خودران" (fresh commands arrive).
   * 2. In "sim" mode: waits until in-browser Autopilot "شروع خودران" is clicked.
   * 3. In "manual" mode: keyboard controls drive.
   * When inactive, ego remains parked and oncoming traffic stays frozen in place.
   */
  function isSimActive() {
    if (settings.drivingMode === "camera") {
      if (external.command === null) return false;
      // If the dashboard explicitly sent a stop command (throttle == 0 && brake >= 0.8), park
      if (external.command.throttle === 0 && external.command.brake >= 0.8) {
        return false;
      }
      const nowWall = (typeof performance !== "undefined" ? performance.now() : Date.now()) / 1000;
      return (state.time - external.at <= COMMAND_FRESH_S) ||
             (Boolean(external.wallTime) && nowWall - external.wallTime <= COMMAND_FRESH_S);
    }
    if (settings.drivingMode === "sim") {
      return Boolean(settings.isBrowserAutopilotActive);
    }
    if (settings.drivingMode === "manual") {
      return true;
    }
    return false;
  }

  /**
   * Phone commands are authoritative for COMMAND_FRESH_S; afterwards the
   * keyboard fallback drives again. This keeps manual testing possible while
   * the closed loop stays uninterrupted at the phone's 20 Hz send rate.
   * Also applies tactical corridor avoidance when ADAS is enabled.
   */
  function effectiveControls() {
    let ctrl = PARKED_CONTROLS;

    if (settings.drivingMode === "camera") {
      // In camera mode: driving ONLY occurs when fresh commands arrive from the phone dashboard!
      const nowWall = (typeof performance !== "undefined" ? performance.now() : Date.now()) / 1000;
      const isFresh = external.command !== null && (
        (state.time - external.at <= COMMAND_FRESH_S) ||
        (Boolean(external.wallTime) && nowWall - external.wallTime <= COMMAND_FRESH_S)
      );
      // If no fresh command, or phone commanded brake / stop / covered camera: full stop immediately!
      if (!isFresh || (external.command.throttle === 0 && external.command.brake >= 0.8)) {
        return PARKED_CONTROLS;
      }

      ctrl = {
        steering: external.command.steering,
        // Ensure adequate cruising throttle so the car has full steering authority and doesn't stall
        throttle: external.command.throttle > 0.05 ? Math.max(external.command.throttle, 0.42) : external.command.throttle,
        brake: external.command.brake,
        turn_signal: external.command.turn_signal,
      };

      // In camera mode: ADAS corridor evaluation provides smart evasion & lane changing
      if (settings.adasAvoidance && settings.obstacleSource !== "off" && (ctrl.throttle > 0 || state.speed > 0.5)) {
        let corridor = null;
        if (settings.obstacleSource === "ground_truth" && traffic) {
          corridor = assessCorridors(
            state.x,
            state.z,
            state.heading,
            traffic.cars,
            traffic.laneCentres,
            ADAS_OBSTACLE_DANGER_M,
            ADAS_LOOKAHEAD_HORIZON_M,
            corridorState.lockedTargetLane,
            corridorState.prevSteering,
          );
        } else {
          corridor = assessCorridorsFromVisualDetections(
            state.x,
            state.heading,
            latestVisualObstacles,
            traffic?.laneCentres || LANE_CENTRES,
            ADAS_OBSTACLE_DANGER_M,
            ADAS_LOOKAHEAD_HORIZON_M,
            corridorState.lockedTargetLane,
            corridorState.prevSteering,
            corridorState.blindSpotHolds,
            state.time,
          );
        }
        if (corridor) {
          corridorState.lockedTargetLane = corridor.lockedTargetLane;
          corridorState.prevSteering = corridor.tacticalSteering;

          if (corridor.isManeuvering) {
            const currClear = corridor.clearances ? corridor.clearances[corridor.currentLaneIdx] : 80;
            const tgtClear = corridor.clearances ? corridor.clearances[corridor.targetLaneIdx] : 80;
            // Only hard brake if ego lane is critically blocked (< 6m) AND no escape lane is available (tgtClear < 8m)
            const isEmergencyBrake = currClear < 6 && tgtClear < 8;
            const needSlowDown = currClear < 14 && tgtClear < 10;
            const tacticalThrottle = isEmergencyBrake ? 0 : (needSlowDown ? 0.25 : Math.max(ctrl.throttle, 0.45));
            const tacticalBrake = isEmergencyBrake ? 0.85 : (needSlowDown ? 0.20 : 0);
            return {
              steering: corridor.tacticalSteering,
              throttle: tacticalThrottle,
              brake: tacticalBrake,
              turn_signal: corridor.turnSignal,
              corridor,
            };
          } else {
            // Cruising clear ahead: phone camera steers, or apply smooth lane centering if biasCorrection is enabled
            let cameraSteering = ctrl.steering;
            if (settings.biasCorrection) {
              const laneCenter = (traffic?.laneCentres || LANE_CENTRES)[corridor.currentLaneIdx] ?? 0;
              const drift = state.x - laneCenter;
              cameraSteering = clamp(ctrl.steering - drift * 0.15 - state.heading * 0.40, -1, 1);
            }
            return {
              steering: cameraSteering,
              throttle: ctrl.throttle,
              brake: ctrl.brake,
              turn_signal: ctrl.turn_signal,
              corridor,
            };
          }
        }
      }
      return ctrl;
    }

    if (settings.drivingMode === "sim") {
      // In sim mode: driving ONLY occurs when in-browser autopilot is toggled ON
      const nowWall = (typeof performance !== "undefined" ? performance.now() : Date.now()) / 1000;
      const isFresh = external.browserCommand !== null && (
        (state.time - external.browserAt <= COMMAND_FRESH_S) ||
        (Boolean(external.browserWallTime) && nowWall - external.browserWallTime <= COMMAND_FRESH_S)
      );
      if (settings.isBrowserAutopilotActive && isFresh) {
        ctrl = external.browserCommand;
      } else if (settings.isBrowserAutopilotActive) {
        ctrl = { steering: 0, throttle: 0.45, brake: 0, turn_signal: 0 };
      } else {
        ctrl = PARKED_CONTROLS;
      }
    } else if (settings.drivingMode === "manual") {
      ctrl = state.controls;
    } else {
      ctrl = PARKED_CONTROLS;
    }

    if (settings.adasAvoidance && settings.obstacleSource !== "off" && (ctrl.throttle > 0 || state.speed > 0.5)) {
      let corridor = null;
      if (settings.obstacleSource === "ground_truth" && traffic) {
        corridor = assessCorridors(
          state.x,
          state.z,
          state.heading,
          traffic.cars,
          traffic.laneCentres,
          ADAS_OBSTACLE_DANGER_M,
          ADAS_LOOKAHEAD_HORIZON_M,
          corridorState.lockedTargetLane,
          corridorState.prevSteering,
        );
      } else {
        corridor = assessCorridorsFromVisualDetections(
          state.x,
          state.heading,
          latestVisualObstacles,
          traffic?.laneCentres || LANE_CENTRES,
          ADAS_OBSTACLE_DANGER_M,
          ADAS_LOOKAHEAD_HORIZON_M,
          corridorState.lockedTargetLane,
          corridorState.prevSteering,
          corridorState.blindSpotHolds,
          state.time,
        );
      }
      if (corridor) {
        corridorState.lockedTargetLane = corridor.lockedTargetLane;
        corridorState.prevSteering = corridor.tacticalSteering;
        if (corridor.isManeuvering) {
          const currClear = corridor.clearances ? corridor.clearances[corridor.currentLaneIdx] : 80;
          const tgtClear = corridor.clearances ? corridor.clearances[corridor.targetLaneIdx] : 80;
          const isEmergencyBrake = currClear < 6 && tgtClear < 8;
          const needSlowDown = currClear < 14 && tgtClear < 10;
          const tacticalThrottle = isEmergencyBrake ? 0 : (needSlowDown ? 0.25 : (corridor.shouldAvoid ? Math.max(ctrl.throttle, 0.45) : ctrl.throttle));
          const tacticalBrake = isEmergencyBrake ? 0.85 : (needSlowDown ? 0.20 : (corridor.shouldAvoid ? 0 : ctrl.brake));
          return {
            steering: corridor.tacticalSteering,
            throttle: tacticalThrottle,
            brake: tacticalBrake,
            turn_signal: corridor.turnSignal,
            corridor,
          };
        }
      }
    }
    return ctrl;
  }

  /**
   * Store a validated command and switch the turn signal target.
   * Called by ``window.applyControls`` in the bootstrap (for phone)
   * or by in-browser autopilot (with source="browser").
   * Applies bias correction and steering gain settings if enabled.
   */
  function receiveCommand(data, source = "phone") {
    const raw = normalizeCommand(data);
    if (raw === null) {
      return false;
    }

    let steer = raw.steering;
    if (settings.biasCorrection) {
      steer -= 0.080;
    }
    if (settings.steeringGain && settings.steeringGain !== 1.0) {
      steer *= settings.steeringGain;
    }
    if (settings.invertSteering) {
      steer = -steer;
    }
    steer = clamp(steer, -1, 1);

    const command = {
      steering: steer,
      throttle: raw.throttle,
      brake: raw.brake,
      turn_signal: raw.turn_signal,
    };

    const nowWall = (typeof performance !== "undefined" ? performance.now() : Date.now()) / 1000;
    if (source === "browser") {
      external.browserCommand = command;
      external.browserAt = state.time;
      external.browserWallTime = nowWall;
    } else {
      external.command = command;
      external.at = state.time;
      external.wallTime = nowWall;
    }
    updateTurnSignalLights(command.turn_signal);
    return true;
  }

  /** Set the signal class the blinker animation follows. */
  function updateTurnSignalLights(turnSignal) {
    state.turnSignal = turnSignal;
  }

  /**
   * One physics step of the ego car. The order follows the brief: speed,
   * then heading (it needs the new speed), then position (it needs the new
   * heading), then the wall, then the mesh. During the post-collision freeze
   * the car stands still and resets to the centre lane when the freeze ends.
   */
  function updateEgoPhysics(dt) {
    if (state.pendingReset && state.time >= state.frozenUntil) {
      state.x = 0;
      state.heading = 0;
      state.speed = 0;
      state.turnSignal = 0;
      state.pendingReset = false;
    }

    if (state.time < state.frozenUntil) {
      syncEgoMesh();
      return;
    }

    const controls = effectiveControls();
    const { steering, throttle, brake } = controls;

    // Synchronize blinker turn signals:
    // If the model/ADAS explicit turn signal head commands 1 or 2, use it;
    // otherwise if steering is actively turning (|steering| >= 0.15), activate the blinker!
    if (controls.turn_signal !== undefined && controls.turn_signal !== 0) {
      state.turnSignal = controls.turn_signal;
    } else if (steering <= -0.15) {
      state.turnSignal = 1; // Left blinker
    } else if (steering >= 0.15) {
      state.turnSignal = 2; // Right blinker
    } else if (Math.abs(steering) < 0.08) {
      state.turnSignal = 0; // Off
    }

    state.speed = stepLongitudinalSpeed(state.speed, throttle, brake, dt);
    state.heading = stepHeading(state.heading, steering, state.speed, dt);
    const position = stepWorldPosition(state.x, state.z, state.heading, state.speed, dt);
    state.x = position.x;
    state.z = position.z;

    // Invisible wall: clamp, scrub speed while grinding, and raise one
    // wall-hit event (HUD counter + phone hazard) per contact.
    const currentWallX = state.wallX || ROAD_WALL_X;
    if (Math.abs(state.x) > currentWallX) {
      state.x = clampToWall(state.x, currentWallX);
      state.speed = Math.max(0, state.speed - WALL_SCRUB_MS2 * dt);
      if (state.time - state.lastWallEventAt >= WALL_EVENT_COOLDOWN_S) {
        state.lastWallEventAt = state.time;
        state.wallHits += 1;
        onWallContact();
      }
    }

    syncEgoMesh();
  }

  function syncEgoMesh() {
    // The mesh faces -z at rotation.y = 0, so facing (sin h, -cos h) needs
    // rotation.y = -heading.
    egoCar.position.set(state.x, 0, state.z);
    egoCar.rotation.y = -state.heading;
  }

  /**
   * Toggle the hood-corner lamps: the phase advances 3 per second and the
   * colour flips on every integer step — the spec blink at "3 Hz".
   */
  function animateBlinkers(dt) {
    state.blinkerPhase += dt * BLINKER_PHASE_RATE;
    const lit = blinkerIsLit(state.blinkerPhase);
    for (const blinker of blinkers) {
      const active = state.turnSignal === blinker.mesh.userData.signal;
      blinker.mesh.material = active && lit ? blinker.onMaterial : blinker.offMaterial;
    }
  }

  /** Collision with traffic: stop, flash, count, warn, freeze, then reset. */
  function onCollision() {
    state.collisions += 1;
    state.speed = 0;
    state.frozenUntil = state.time + COLLISION_FREEZE_S;
    state.immuneUntil = state.frozenUntil + COLLISION_IMMUNITY_S;
    state.pendingReset = true;
    // A crash also mutes wall events for a moment, so the two flashes and
    // messages do not fire on top of each other.
    state.lastWallEventAt = state.time;
    hud.flash("collision");
    onHazard("collision");
  }

  /** Grinding the wall: orange flash, count, and a warning to the phone. */
  function onWallContact() {
    hud.flash("wall");
    onHazard("wall");
  }

  /**
   * Box test between the ego car and every nearby traffic car. The broad
   * phase keeps the per-frame cost at a couple of comparisons; the Box3 test
   * itself is the exact expression mirrored by the pure ``carsCollide``.
   */
  function checkCollision() {
    if (state.time < state.immuneUntil || state.time < state.frozenUntil) {
      return;
    }
    setCarBox(egoBox, state.x, state.z);
    for (const car of traffic.cars) {
      if (!car.userData.active) {
        continue;
      }
      if (Math.abs(car.position.x - state.x) > COLLISION_BROAD_X) {
        continue;
      }
      if (Math.abs(car.position.z - state.z) > COLLISION_BROAD_Z) {
        continue;
      }
      if (
        egoBox.intersectsBox(setCarBox(trafficBox, car.position.x, car.position.z))
      ) {
        onCollision();
        return;
      }
    }
  }

  let frameRate = 60;
  let hudTimer = 0;

  /** Numbers are refreshed at 4 Hz; the blinking arrows run every frame. */
  function updateHud(dt, active) {
    hud.setTurnSignal(
      state.turnSignal === 1 && blinkerIsLit(state.blinkerPhase),
      state.turnSignal === 2 && blinkerIsLit(state.blinkerPhase),
    );

    if (hud.setSimState) {
      if (active) {
        if (settings.drivingMode === "camera") {
          hud.setSimState("خودران گوشی فعال ▶", "active-phone");
        } else {
          hud.setSimState("خودران مدل فعال ▶", "active-sim");
        }
      } else {
        hud.setSimState("در انتظار شروع ⏸", "waiting");
      }
    }

    hudTimer += dt;
    if (hudTimer < 0.25) {
      return;
    }
    hudTimer = 0;
    hud.setSpeed(state.speed);
    hud.setHeading(headingDegrees(state.heading));
    hud.setCollisionCount(state.collisions);
    hud.setWallCount(state.wallHits);
    hud.setFrameRate(frameRate);
    onTelemetry(state.speed);
  }

  /**
   * Keep the road under the car: snap the road group to the nearest multiple
   * of the dash period. Because every marking repeats with that period, the
   * snap never moves a visible feature.
   */
  function updateEndlessRoad() {
    const period = DASH_SIZE + DASH_GAP;
    roadRoot.position.z = period * Math.round(state.z / period);
  }

  /** Recycle scenery that the car has driven past to one span ahead. */
  function updateScenery() {
    const span = scenery.userData.span;
    for (const item of scenery.children) {
      if (item.position.z > state.z + SCENERY_KEEP_BEHIND_M) {
        item.position.z -= span;
      } else if (item.position.z < state.z - span + SCENERY_KEEP_BEHIND_M) {
        item.position.z += span;
      }
    }
  }

  /** Steer the shadow-casting sun (and its target) along with the car. */
  function updateSun() {
    sun.position.set(state.x + 18, 30, state.z + 12);
    sun.target.position.set(state.x, 0, state.z);
    sun.target.updateMatrixWorld();
  }

  function updateCamera() {
    const pose = cameraPose(state.x, state.z, state.heading);
    camera.position.set(pose.position.x, pose.position.y, pose.position.z);
    camera.lookAt(pose.lookAt.x, pose.lookAt.y, pose.lookAt.z);
  }

  function update(dt) {
    frameRate = smoothFrameRate(frameRate, dt);

    const active = isSimActive();

    if (active) {
      // User requirement: Start going with 10 km/h (2.78 m/s) at the beginning
      if (state.speed < (10 / 3.6) && state.time <= 0.3) {
        state.speed = 10 / 3.6;
      }
      state.time += dt;
      updateEgoPhysics(dt);
      animateBlinkers(dt);
      updateTraffic(dt);
      updateEndlessRoad();
      updateScenery();
      updateSun();
      checkCollision();
    } else {
      // Simulation is waiting for dashboard start or sim start.
      // Vehicle remains parked and oncoming traffic stays frozen in place.
      state.speed = 0;
      syncEgoMesh();
    }

    updateCamera();
    updateHud(dt, active);

    renderer.render(scene, camera);

    detectTimer -= dt;
    if (detectTimer <= 0) {
      detectTimer = 0.08; // ~12.5 Hz pure computer vision scan
      if (settings.adasAvoidance && settings.obstacleSource !== "off") {
        const visualSource = (settings.drivingMode === "camera" && simulation?.phoneCameraReady && simulation?.phoneCameraImage && (performance.now() - (simulation.lastPhoneCameraTime || 0) < 5000))
          ? simulation.phoneCameraImage
          : canvas;
        const rawDetections = detectVisualObstacles(visualSource, state.x, state.heading, traffic?.laneCentres || LANE_CENTRES);

        const detectedLanes = new Set();
        for (const obs of rawDetections) {
          const l = obs.laneIdx;
          if (l >= 0 && l < 3) {
            detectedLanes.add(l);
            visualTracker[l].hits = Math.min((visualTracker[l].hits || 0) + 1, 5);
            visualTracker[l].distanceM = obs.distanceM;
            visualTracker[l].lastSeen = state.time;
          }
        }
        for (let l = 0; l < 3; l++) {
          if (!detectedLanes.has(l) && state.time - (visualTracker[l].lastSeen || 0) > 0.22) {
            visualTracker[l].hits = Math.max(0, (visualTracker[l].hits || 0) - 1);
          }
        }

        // Only confirm obstacles observed across at least 2 consecutive detection frames,
        // completely rejecting momentary highway line glitches, shadows, and reflections
        latestVisualObstacles = rawDetections.filter(obs => {
          const t = visualTracker[obs.laneIdx];
          return t && (t.hits >= 2 || obs.distanceM <= 12);
        });
      }
    }
  }

  /**
   * Refit the WebGL backbuffer and the camera frustum onto the screen.
   *
   * Everything is derived from the canvas's own layout box via
   * {@link sceneViewport}, so this is exact after every kind of geometry
   * change: window resize, browser zoom, snap layouts, a move to another
   * monitor with different DPI, or entering/exiting fullscreen. The pixel
   * ratio is re-read each time for the same reason.
   */
  function resize() {
    const { width, height } = sceneViewport(canvas);
    renderer.setPixelRatio(choosePixelRatio(window.devicePixelRatio));
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
  }

  /**
   * Reset simulation to starting state: ego car at center lane, zero speed,
   * heading straight, clear traffic, and zero collision/wall counters.
   */
  function restart() {
    state.x = 0;
    state.z = 0;
    state.speed = 0;
    state.heading = 0;
    state.time = 0;
    state.turnSignal = 0;
    state.blinkerPhase = 0;
    state.collisions = 0;
    state.wallHits = 0;
    state.frozenUntil = state.time + 0.5;
    state.immuneUntil = state.time + 1.5;
    state.pendingReset = false;
    state.lastWallEventAt = -Infinity;
    external.command = null;
    external.at = -Infinity;
    external.wallTime = 0;
    external.browserCommand = null;
    external.browserAt = -Infinity;
    external.browserWallTime = 0;
    settings.isBrowserAutopilotActive = false;
    latestVisualObstacles = [];
    corridorState.lockedTargetLane = null;
    corridorState.prevSteering = 0;
    corridorState.blindSpotHolds = [0, 0, 0];
    if (typeof settings.onRestart === "function") {
      settings.onRestart();
    }

    for (const car of traffic.cars) {
      car.userData.active = false;
      car.visible = false;
    }
    traffic.timer = pickSpawnDelay(Math.random, traffic.spawnDelayMin, traffic.spawnDelayMax);
    preseedTraffic();

    if (hud) {
      hud.setSpeed(0);
      hud.setHeading(0);
      hud.setCollisionCount(0);
      hud.setWallCount(0);
      hud.setTurnSignal(false, false);
      if (typeof hud.clearFlash === "function") {
        hud.clearFlash();
      }
      if (hud.setSimState) {
        hud.setSimState("در انتظار شروع ⏸", "waiting");
      }
    }
    syncEgoMesh();
    updateCamera();
    updateSun();
    updateEndlessRoad();
    for (const item of scenery.children) {
      if (item.userData.baseZ !== undefined) {
        item.position.z = item.userData.baseZ;
      }
    }
  }

  updateRoadVisuals(activeRoadPreset);
  preseedTraffic();

  return {
    state,
    traffic,
    settings,
    corridorState,
    external,
    get latestVisualObstacles() {
      return latestVisualObstacles;
    },
    set latestVisualObstacles(val) {
      latestVisualObstacles = val;
    },
    update,
    resize,
    restart,
    receiveCommand,
    effectiveControls,
    isSimActive,
    isPhoneActive() {
      if (external.command === null) return false;
      if (external.command.throttle === 0 && external.command.brake >= 0.8) return false;
      const nowWall = (typeof performance !== "undefined" ? performance.now() : Date.now()) / 1000;
      return (state.time - external.at <= COMMAND_FRESH_S) ||
             (Boolean(external.wallTime) && nowWall - external.wallTime <= COMMAND_FRESH_S);
    },
    setKeyboardControls(controls) {
      state.controls = controls;
    },
    spawnTrafficCar,
    setRoadPreset(key) {
      const preset = ROAD_PRESETS[key] || ROAD_PRESETS.narrow;
      updateRoadVisuals(preset);
      restart();
    },
    setTrafficPreset(key) {
      const preset = TRAFFIC_PRESETS[key] || TRAFFIC_PRESETS.dense;
      activeTrafficPreset = preset;
      traffic.spawnDelayMin = preset.spawnDelayMin;
      traffic.spawnDelayMax = preset.spawnDelayMax;
      traffic.spawnGapMin = preset.spawnGapMin;
      traffic.spawnGapMax = preset.spawnGapMax;
      restart();
    },
    scene,
    camera,
    renderer,
  };
}

/* ------------------------------------------------------------------------- *
 * 5. Browser bootstrap
 * ------------------------------------------------------------------------- */

function createHud() {
  const status = document.getElementById("ws-status");
  const speed = document.getElementById("speed-value");
  const frameRate = document.getElementById("fps-value");
  const heading = document.getElementById("heading-value");
  const collisionCount = document.getElementById("collision-count");
  const wallCount = document.getElementById("wall-count");
  const turnLeft = document.getElementById("turn-left");
  const turnRight = document.getElementById("turn-right");
  const flash = document.getElementById("hazard-flash");
  const flashText = document.getElementById("hazard-flash-text");
  const simBadge = document.getElementById("sim-state-badge");

  return {
    setSimState(text, stateClass) {
      if (simBadge) {
        simBadge.textContent = text;
        simBadge.className = `hud-value hud-state-badge is-${stateClass}`;
      }
    },
    setStatus(text, state) {
      status.textContent = text;
      status.className = `hud-value is-${state}`;
    },
    setSpeed(speedMs) {
      speed.textContent = String(Math.round(speedToKmh(speedMs)));
    },
    setFrameRate(value) {
      frameRate.textContent = String(Math.round(value));
    },
    setHeading(degrees) {
      heading.textContent = `${degrees > 0 ? "+" : ""}${Math.round(degrees)}°`;
    },
    setCollisionCount(count) {
      collisionCount.textContent = String(count);
      collisionCount.classList.toggle("is-hot", count > 0);
    },
    setWallCount(count) {
      wallCount.textContent = String(count);
      wallCount.classList.toggle("is-hot", count > 0);
    },
    /** Blinking arrows follow the exact lamp phase in the 3D scene. */
    setTurnSignal(leftLit, rightLit) {
      turnLeft.classList.toggle("is-lit", leftLit);
      turnRight.classList.toggle("is-lit", rightLit);
    },
    /**
     * Full-screen flash. Red for a collision, orange for the wall; the CSS
     * animation removes itself, and the text tells the phone camera (and the
     * human) what happened.
     */
    flash(kind) {
      const isCollision = kind === "collision";
      flashText.textContent = isCollision ? "برخورد!" : "برخورد با دیوار — مسیر را عوض کن";
      flash.classList.remove("is-collision", "is-wall", "is-running");
      // Force a reflow so restarting the animation works on consecutive hits.
      void flash.offsetWidth;
      flash.classList.add(isCollision ? "is-collision" : "is-wall", "is-running");
    },
    clearFlash() {
      if (flash) {
        flash.classList.remove("is-collision", "is-wall", "is-running");
      }
      if (flashText) {
        flashText.textContent = "";
      }
    },
  };
}

/** Read the keyboard into a control payload. Arrows and WASD do the same. */
function createKeyboardControls() {
  const pressed = new Set();

  function controlsFromKeys() {
    const left = pressed.has("ArrowLeft") || pressed.has("KeyA");
    const right = pressed.has("ArrowRight") || pressed.has("KeyD");
    return {
      steering: (right ? 1 : 0) - (left ? 1 : 0),
      throttle: pressed.has("ArrowUp") || pressed.has("KeyW") ? 1 : 0,
      brake: pressed.has("ArrowDown") || pressed.has("KeyS") ? 1 : 0,
      turn_signal: 0,
    };
  }

  /**
   * Report a control payload on every key event. Losing the window clears the
   * held keys, so the car does not keep driving when the tab is left.
   */
  return function attach(onChange) {
    const handle = (event) => {
      if (event.type === "keydown") {
        pressed.add(event.code);
        // Stop the arrows from scrolling the page while the car is steered.
        if (event.code.startsWith("Arrow")) {
          event.preventDefault();
        }
      } else {
        pressed.delete(event.code);
      }
      onChange(controlsFromKeys());
    };

    window.addEventListener("keydown", handle);
    window.addEventListener("keyup", handle);
    window.addEventListener("blur", () => {
      pressed.clear();
      onChange(controlsFromKeys());
    });
  };
}

/**
 * Connect to the render endpoint, report the state in the HUD and hand every
 * command to ``applyControls``. A closed socket is retried with an exponential
 * backoff for up to MAX_RECONNECT_ATTEMPTS attempts.
 */
function connectRenderSocket(hud) {
  let attempt = 0;
  let socket = null;

  function renderSocketUrl() {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    return `${protocol}//${window.location.host}/ws/render/`;
  }

  function connect() {
    hud.setStatus("در حال اتصال…", "pending");
    socket = new WebSocket(renderSocketUrl());

    socket.onopen = () => {
      attempt = 0;
      hud.setStatus("وصل", "connected");
    };

    socket.onmessage = (event) => {
      let message;
      try {
        message = JSON.parse(event.data);
      } catch {
        console.warn("[Render] dropped a frame that is not valid JSON");
        return;
      }
      if (message.type === "camera_frame" && message.image) {
        if (!simulation.phoneCameraImage) {
          simulation.phoneCameraImage = new Image();
          simulation.phoneCameraImage.onload = () => {
            simulation.phoneCameraReady = true;
          };
        }
        simulation.phoneCameraImage.src = message.image;
        simulation.lastPhoneCameraTime = performance.now();
        return;
      }
      if (message.type !== "control_update") {
        return;
      }
      const command = normalizeCommand(message);
      if (command === null) {
        console.warn("[Render] dropped an invalid command", message);
        return;
      }
      window.applyControls(command);
    };

    socket.onclose = () => {
      attempt += 1;
      const delay = nextReconnectDelayMs(attempt);
      if (delay === null) {
        hud.setStatus("قطع (تلاش ناموفق)", "disconnected");
        return;
      }
      hud.setStatus(`قطع (${attempt})`, "disconnected");
      setTimeout(connect, delay);
    };
  }

  connect();
  return {
    sendTelemetry(speedMs) {
      return sendFrame(telemetryFrame(speedMs));
    },
    sendHazard(kind) {
      const frame = hazardFrame(kind);
      if (frame === null) {
        return false;
      }
      return sendFrame(frame);
    },
  };

  /** One guarded socket write; a closed or congested socket drops the frame. */
  function sendFrame(frame) {
    if (socket?.readyState !== WebSocket.OPEN || socket.bufferedAmount > 1024) {
      return false;
    }
    try {
      socket.send(JSON.stringify(frame));
      return true;
    } catch {
      return false;
    }
  }
}

/**
 * Makes a floating panel draggable within the browser viewport using pointer events.
 */
function makeElementDraggable(element, handle) {
  let isDragging = false;
  let startX = 0, startY = 0;
  let initialLeft = 0, initialTop = 0;

  handle.addEventListener("pointerdown", (e) => {
    if (e.target.closest("button") || e.target.closest("a") || e.target.closest("input") || e.target.closest("select")) {
      return;
    }
    if (element.classList.contains("is-popped-out")) return;

    isDragging = true;
    element.classList.add("is-dragging");
    try {
      handle.setPointerCapture(e.pointerId);
    } catch {}

    const rect = element.getBoundingClientRect();
    startX = e.clientX;
    startY = e.clientY;
    initialLeft = rect.left;
    initialTop = rect.top;

    element.style.right = "auto";
    element.style.bottom = "auto";
    element.style.left = `${initialLeft}px`;
    element.style.top = `${initialTop}px`;
    e.preventDefault();
  });

  handle.addEventListener("pointermove", (e) => {
    if (!isDragging) return;
    const dx = e.clientX - startX;
    const dy = e.clientY - startY;

    const maxX = Math.max(0, window.innerWidth - element.offsetWidth);
    const maxY = Math.max(0, window.innerHeight - element.offsetHeight);

    const newLeft = Math.max(0, Math.min(maxX, initialLeft + dx));
    const newTop = Math.max(0, Math.min(maxY, initialTop + dy));

    element.style.left = `${newLeft}px`;
    element.style.top = `${newTop}px`;
  });

  const stopDrag = (e) => {
    if (!isDragging) return;
    isDragging = false;
    element.classList.remove("is-dragging");
    try {
      handle.releasePointerCapture(e.pointerId);
    } catch {}
  };

  handle.addEventListener("pointerup", stopDrag);
  handle.addEventListener("pointercancel", stopDrag);
}

/**
 * Manages popping out a panel into an independent OS browser window (exportable)
 * for multi-monitor setups or clearing view for phone cameras, and restoring it seamlessly.
 */
function setupPopout(element, popoutBtn) {
  let popoutWindow = null;
  const parentNode = element.parentNode;
  const nextNode = element.nextSibling;

  function restorePanel() {
    if (!element.classList.contains("is-popped-out")) return;
    element.classList.remove("is-popped-out");
    element.style.position = "";
    element.style.width = "";
    element.style.boxShadow = "";
    element.style.border = "";
    if (parentNode) {
      parentNode.insertBefore(element, nextNode);
    }
    if (popoutBtn) {
      popoutBtn.textContent = "⧉";
      popoutBtn.title = "باز کردن در پنجره مجزا (Pop-out to separate window)";
    }
    popoutWindow = null;
  }

  function openPopout() {
    if (popoutWindow && !popoutWindow.closed) {
      popoutWindow.focus();
      return;
    }

    const width = 440;
    const height = 740;
    const left = (window.screenX || window.screenLeft || 0) + Math.max(0, (window.outerWidth || 1000) - width - 20);
    const top = (window.screenY || window.screenTop || 0) + 30;

    popoutWindow = window.open(
      "",
      "AutopilotSettingsPopout",
      `width=${width},height=${height},left=${left},top=${top},resizable=yes,scrollbars=yes`
    );

    if (!popoutWindow) {
      alert("پنجره پاپ‌آپ توسط مرورگر مسدود شد. لطفاً اجازه باز شدن پنجره‌های پاپ‌آپ را بدهید.");
      return;
    }

    const doc = popoutWindow.document;
    doc.open();
    doc.write(`<!DOCTYPE html>
<html lang="fa" dir="rtl">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>تنظیمات خودران و شبیه‌ساز | SelfDriveSim</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/rastikerdar/vazirmatn@v33.003/Vazirmatn-font-face.css">
</head>
<body>
</body>
</html>`);
    doc.close();

    const styles = document.querySelectorAll("link[rel='stylesheet'], style");
    for (const styleNode of styles) {
      doc.head.appendChild(styleNode.cloneNode(true));
    }

    // Force independent window scrollability and responsiveness
    const popoutOverride = doc.createElement("style");
    popoutOverride.textContent = `
      *, *::before, *::after {
        box-sizing: border-box !important;
      }
      html {
        overflow-x: hidden !important;
        overflow-y: auto !important;
        height: 100% !important;
        background: #0d1117 !important;
      }
      body {
        margin: 0 !important;
        padding: 0 !important;
        height: auto !important;
        min-height: 100% !important;
        overflow-x: hidden !important;
        overflow-y: auto !important;
        background: #0d1117 !important;
        color: #e6edf3 !important;
        font-family: 'Vazirmatn', -apple-system, BlinkMacSystemFont, 'Segoe UI', Tahoma, sans-serif !important;
        direction: rtl !important;
      }
      #test-panel {
        position: static !important;
        box-sizing: border-box !important;
        width: 100% !important;
        max-width: 100% !important;
        inset: auto !important;
        background: transparent !important;
        box-shadow: none !important;
        border: none !important;
        padding: 12px 14px 40px 14px !important;
        margin: 0 !important;
      }
    `;
    doc.head.appendChild(popoutOverride);

    element.classList.add("is-popped-out");
    element.hidden = false;
    doc.body.appendChild(element);

    if (popoutBtn) {
      popoutBtn.textContent = "↩ بازگشت به صفحه اصلی";
      popoutBtn.title = "بازگرداندن پنل به صفحه اصلی شبیه‌ساز";
    }

    popoutWindow.addEventListener("beforeunload", restorePanel);
    popoutWindow.addEventListener("pagehide", restorePanel);
  }

  setInterval(() => {
    if (element.classList.contains("is-popped-out") && popoutWindow && popoutWindow.closed) {
      restorePanel();
    }
  }, 200);

  popoutBtn?.addEventListener("click", () => {
    if (element.classList.contains("is-popped-out")) {
      if (popoutWindow && !popoutWindow.closed) {
        popoutWindow.close();
      } else {
        restorePanel();
      }
    } else {
      openPopout();
    }
  });

  window.addEventListener("beforeunload", () => {
    if (popoutWindow && !popoutWindow.closed) {
      popoutWindow.close();
    }
  });
}

/**
 * In-browser Autopilot test panel: feeds the Three.js canvas directly to
 * DAVE-2 without camera or network latency, proving model steering in closed loop.
 */
async function initAutopilot(simulation, sceneCanvas) {
  const panel = document.getElementById("test-panel");
  const header = document.getElementById("test-panel-header");
  const toggleBtn = document.getElementById("btn-toggle-test-panel");
  const closeBtn = document.getElementById("test-panel-close");
  const popoutBtn = document.getElementById("btn-popout-panel");
  const minBtn = document.getElementById("btn-minimize-panel");
  const driveBadge = document.getElementById("active-drive-source");
  const btnSourceCamera = document.getElementById("btn-source-camera");
  const btnSourceSim = document.getElementById("btn-source-sim");
  const sourceIndicator = document.getElementById("visual-source-indicator");
  const phoneSection = document.getElementById("phone-camera-mode-section");
  const phoneStatusBadge = document.getElementById("phone-running-status");
  const simSection = document.getElementById("sim-autopilot-mode-section");
  const autopilotBtn = document.getElementById("btn-autopilot-toggle");
  const autopilotBtnText = document.getElementById("autopilot-btn-text");
  const invertCheck = document.getElementById("chk-invert-steering");
  const biasCheck = document.getElementById("chk-bias-correction");
  const adasCheck = document.getElementById("chk-adas-avoidance");
  const selAdasSource = document.getElementById("sel-adas-source");
  const corridorStatus = document.getElementById("adas-corridor-status");
  const gainRange = document.getElementById("range-steering-gain");
  const gainVal = document.getElementById("steering-gain-val");
  const restartBtn = document.getElementById("btn-test-restart");
  const statusLabel = document.getElementById("test-model-status");
  const fpsLabel = document.getElementById("autopilot-fps");
  const steerVal = document.getElementById("gauge-steering-val");
  const steerFill = document.getElementById("gauge-steering-fill");
  const throttleVal = document.getElementById("gauge-throttle-val");
  const throttleFill = document.getElementById("gauge-throttle-fill");
  const brakeVal = document.getElementById("gauge-brake-val");
  const previewCanvas = document.getElementById("autopilot-canvas");

  if (!panel || !previewCanvas) return;

  // 1. Draggability
  if (header) {
    makeElementDraggable(panel, header);
  }

  // 2. Pop-out exportable window
  if (popoutBtn) {
    setupPopout(panel, popoutBtn);
  }

  // 3. Minimize / collapse
  if (minBtn) {
    minBtn.addEventListener("click", () => {
      const isMin = panel.classList.toggle("is-minimized");
      minBtn.textContent = isMin ? "▢" : "−";
      minBtn.title = isMin ? "بزرگ کردن پنل" : "کوچک کردن پنل";
    });
  }

  // 3b. View Size & Expandability Controls (makes camera feed & sim view bigger and expandable)
  const previewContainer = document.getElementById("autopilot-preview-container");
  const btnSizeNormal = document.getElementById("btn-size-normal");
  const btnSizeLarge = document.getElementById("btn-size-large");
  const btnSizeExpand = document.getElementById("btn-size-expand");
  const btnOverlayExpand = document.getElementById("btn-overlay-expand");
  const expandIcon = document.getElementById("expand-icon");
  const expandText = document.getElementById("expand-text");

  function setPreviewSize(mode) {
    if (!previewContainer) return;
    btnSizeNormal?.classList.toggle("is-active", mode === "normal");
    btnSizeLarge?.classList.toggle("is-active", mode === "large");
    btnSizeExpand?.classList.toggle("is-active", mode === "expand");

    if (mode === "expand") {
      panel.classList.add("is-expanded");
      previewContainer.className = "test-panel-preview is-expanded";
      if (expandIcon) expandIcon.textContent = "↙";
      if (expandText) expandText.textContent = "عادی";
      btnOverlayExpand?.setAttribute("title", "بازگشت به اندازه استاندارد (Collapse View)");
    } else if (mode === "large") {
      panel.classList.remove("is-expanded");
      previewContainer.className = "test-panel-preview size-normal";
      if (panel.style.width && parseInt(panel.style.width, 10) < 480) {
        panel.style.width = "480px";
      }
      if (expandIcon) expandIcon.textContent = "⤢";
      if (expandText) expandText.textContent = "گسترش";
      btnOverlayExpand?.setAttribute("title", "گسترش به حالت تمام‌عرض (Expand View)");
    } else {
      panel.classList.remove("is-expanded");
      previewContainer.className = "test-panel-preview size-normal";
      if (expandIcon) expandIcon.textContent = "⤢";
      if (expandText) expandText.textContent = "بزرگ";
      btnOverlayExpand?.setAttribute("title", "بزرگ‌نمایی و گسترش تصویر (Expand View)");
    }
  }

  btnSizeNormal?.addEventListener("click", () => setPreviewSize("normal"));
  btnSizeLarge?.addEventListener("click", () => setPreviewSize("large"));
  btnSizeExpand?.addEventListener("click", () => setPreviewSize("expand"));
  btnOverlayExpand?.addEventListener("click", () => {
    const isCurrentlyExpanded = panel.classList.contains("is-expanded") || previewContainer?.classList.contains("is-expanded");
    setPreviewSize(isCurrentlyExpanded ? "normal" : "expand");
  });

  // 4. Settings synchronization to simulation (active for phone camera & autopilot)
  function syncSettingsToSim() {
    if (!simulation?.settings) return;
    simulation.settings.biasCorrection = biasCheck ? biasCheck.checked : true;
    simulation.settings.adasAvoidance = adasCheck ? adasCheck.checked : true;
    simulation.settings.obstacleSource = selAdasSource ? selAdasSource.value : "vision";
    simulation.settings.steeringGain = gainRange ? parseFloat(gainRange.value) : 1.5;
    simulation.settings.invertSteering = invertCheck ? invertCheck.checked : false;
  }
  syncSettingsToSim();

  biasCheck?.addEventListener("change", syncSettingsToSim);
  adasCheck?.addEventListener("change", syncSettingsToSim);
  selAdasSource?.addEventListener("change", syncSettingsToSim);
  invertCheck?.addEventListener("change", syncSettingsToSim);
  gainRange?.addEventListener("input", () => {
    if (gainVal) gainVal.textContent = `${parseFloat(gainRange.value).toFixed(1)}x`;
    syncSettingsToSim();
  });

  const previewCtx = previewCanvas.getContext("2d", { willReadFrequently: true });
  let model = null;
  let tf = window.tf || null;
  let isActive = false;
  let loopTimer = null;
  const frameQueue = []; // keeps max 3 tensors
  let lastTime = performance.now();
  let frameCount = 0;
  let fps = 0;
  let isPredicting = false;

  // 5. Driving Mode: Phone Camera (Dashboard) vs In-Browser Simulation Autopilot
  function setDrivingMode(mode) {
    simulation.settings.drivingMode = mode;
    if (mode === "sim") {
      btnSourceSim?.classList.add("is-active");
      btnSourceCamera?.classList.remove("is-active");
      if (sourceIndicator) {
        sourceIndicator.textContent = "🎮 مدل شبیه‌ساز";
        sourceIndicator.className = "source-status-pill is-sim";
      }
      if (phoneSection) phoneSection.style.display = "none";
      if (simSection) simSection.style.display = "block";
    } else {
      // Default: "camera" (Phone Dashboard controls driving)
      simulation.settings.drivingMode = "camera";
      simulation.settings.isBrowserAutopilotActive = false;
      isActive = false;
      if (autopilotBtn) {
        autopilotBtn.classList.remove("is-on");
        autopilotBtn.classList.add("is-off");
        if (autopilotBtnText) autopilotBtnText.textContent = "شروع خودران (DAVE-2)";
      }
      btnSourceCamera?.classList.add("is-active");
      btnSourceSim?.classList.remove("is-active");
      if (sourceIndicator) {
        sourceIndicator.textContent = "📱 دوربین گوشی";
        sourceIndicator.className = "source-status-pill is-camera";
      }
      if (phoneSection) phoneSection.style.display = "block";
      if (simSection) simSection.style.display = "none";
    }
  }

  btnSourceCamera?.addEventListener("click", () => setDrivingMode("camera"));
  btnSourceSim?.addEventListener("click", () => setDrivingMode("sim"));
  setDrivingMode("camera");

  function setPanelOpen(open) {
    panel.hidden = !open;
    toggleBtn?.setAttribute("aria-expanded", String(open));
  }
  toggleBtn?.addEventListener("click", () => setPanelOpen(panel.hidden));
  closeBtn?.addEventListener("click", () => setPanelOpen(false));
  restartBtn?.addEventListener("click", () => simulation.restart());

  // 6. Live Controller Source Badge & Gauge Telemetry Monitor (every 50ms)
  setInterval(() => {
    const isPhone = simulation?.isPhoneActive ? simulation.isPhoneActive() : false;
    const mode = simulation?.settings?.drivingMode || "camera";

    if (driveBadge) {
      if (mode === "camera") {
        if (isPhone) {
          driveBadge.className = "drive-source-badge is-phone";
          driveBadge.innerHTML = '<span>📱 منبع کنترل: <strong>خودران گوشی فعال است · در حال رانندگی</strong></span>';
        } else {
          driveBadge.className = "drive-source-badge";
          driveBadge.innerHTML = '<span>📱 منبع کنترل: <strong>دوربین گوشی (در انتظار فشردن «شروع» در موبایل)</strong></span>';
        }
      } else if (mode === "sim") {
        if (isActive) {
          driveBadge.className = "drive-source-badge is-browser";
          driveBadge.innerHTML = '<span>🤖 منبع کنترل: <strong>خودران مدل شبیه‌ساز (DAVE-2)</strong></span>';
        } else {
          driveBadge.className = "drive-source-badge";
          driveBadge.innerHTML = '<span>🎮 منبع کنترل: <strong>مدل شبیه‌ساز (در انتظار فشردن شروع خودران)</strong></span>';
        }
      }
    }

    if (phoneStatusBadge) {
      if (isPhone) {
        phoneStatusBadge.className = "phone-status-badge is-driving";
        phoneStatusBadge.innerHTML = '<span>✓ خودران گوشی فعال است · فرامین دریافت و اجرا می‌شوند</span>';
      } else {
        phoneStatusBadge.className = "phone-status-badge is-waiting";
        phoneStatusBadge.innerHTML = '<span>⏳ در انتظار فشردن دکمه «شروع خودران» در داشبورد گوشی...</span>';
      }
    }

    if (simulation?.effectiveControls) {
      const eff = simulation.effectiveControls();
      const st = eff.steering ?? 0;
      const th = eff.throttle ?? 0;
      const br = eff.brake ?? 0;

      if (steerVal) steerVal.textContent = (st >= 0 ? "+" : "") + st.toFixed(3);
      if (steerFill) {
        if (st >= 0) {
          steerFill.style.left = "50%";
          steerFill.style.width = `${Math.min(50, st * 50)}%`;
          steerFill.style.background = "#38bdf8";
        } else {
          const w = Math.min(50, Math.abs(st) * 50);
          steerFill.style.left = `${50 - w}%`;
          steerFill.style.width = `${w}%`;
          steerFill.style.background = "#f5d764";
        }
      }
      if (throttleVal) throttleVal.textContent = th.toFixed(3);
      if (throttleFill) {
        throttleFill.style.width = `${Math.min(100, th * 100)}%`;
      }
      if (brakeVal) brakeVal.textContent = br.toFixed(3);

      if (eff.corridor && corridorStatus) {
        const c = eff.corridor;
        const laneNames = ["چپ ◀", "وسط ■", "راست ▶"];
        const dL = Math.round(c.clearances[0]);
        const dC = Math.round(c.clearances[1]);
        const dR = Math.round(c.clearances[2]);
        if (c.shouldAvoid) {
          corridorStatus.textContent = `⚠ مانع! سبقت/تغییر مسیر به ${laneNames[c.targetLaneIdx]} | L:${dL}m C:${dC}m R:${dR}m`;
          corridorStatus.style.borderColor = "#f5d764";
          corridorStatus.style.color = "#f5d764";
        } else {
          corridorStatus.textContent = `مسیر هدف: ${laneNames[c.targetLaneIdx]} | L:${dL}m C:${dC}m R:${dR}m`;
          corridorStatus.style.borderColor = "rgba(56, 189, 248, 0.2)";
          corridorStatus.style.color = "#38bdf8";
        }
      }
    }
  }, 50);

  try {
    statusLabel.textContent = "در حال بارگذاری TFJS...";
    if (!tf && typeof window !== "undefined") {
      for (let i = 0; i < 30 && !window.tf; i++) {
        await new Promise((r) => setTimeout(r, 100));
      }
      tf = window.tf;
    }
    if (!tf) {
      statusLabel.textContent = "کتابخانه TFJS یافت نشد";
      return;
    }

    statusLabel.textContent = "در حال بارگذاری مدل DAVE-2...";
    const { loadModel, realInference, SEQUENCE_LENGTH, MODEL_URL } =
      await import("/static/client/js/inference.js");

    model = await loadModel(tf, MODEL_URL || "/static/client/models/dave2/model.json");
    statusLabel.textContent = "مدل DAVE-2 آماده است ✓";
    statusLabel.classList.add("is-ready");

    async function sampleAndInfer() {
      if (!model || isPredicting) return;
      if (panel.hidden && !isActive) return;

      isPredicting = true;
      try {
        const cw = sceneCanvas.width || sceneCanvas.clientWidth;
        const ch = sceneCanvas.height || sceneCanvas.clientHeight;
        if (!cw || !ch) return;

        const targetAspect = 160 / 120;
        const canvasAspect = cw / ch;
        let sx, sy, sw, sh;
        if (canvasAspect > targetAspect) {
          sw = ch * targetAspect;
          sh = ch;
          sx = (cw - sw) / 2;
          sy = 0;
        } else {
          sw = cw;
          sh = cw / targetAspect;
          sx = 0;
          sy = (ch - sh) / 2;
        }

        const isCameraMode = simulation?.settings?.drivingMode === "camera";
        const hasPhoneFeed = isCameraMode && simulation.phoneCameraReady && simulation.phoneCameraImage && (performance.now() - (simulation.lastPhoneCameraTime || 0) < 5000);

        if (hasPhoneFeed) {
          previewCtx.drawImage(simulation.phoneCameraImage, 0, 0, 160, 120);
        } else {
          previewCtx.drawImage(sceneCanvas, sx, sy, sw, sh, 0, 0, 160, 120);
        }

        const modeTag = document.getElementById("autopilot-mode-tag");
        if (modeTag) {
          if (hasPhoneFeed) {
            modeTag.textContent = "📱 تصویر دوربین زنده گوشی";
            modeTag.style.background = "rgba(34, 197, 94, 0.4)";
            modeTag.style.color = "#86efac";
          } else if (isCameraMode) {
            modeTag.textContent = "📱 دوربین گوشی (در انتظار فریم)";
            modeTag.style.background = "rgba(56, 189, 248, 0.35)";
            modeTag.style.color = "#bae6fd";
          } else {
            modeTag.textContent = "🎮 شبیه‌ساز (Sim Vision)";
            modeTag.style.background = "rgba(56, 189, 248, 0.35)";
            modeTag.style.color = "#bae6fd";
          }
        }

        const imgData = previewCtx.getImageData(0, 0, 160, 120);
        let visualObstacles;
        if (simulation?.settings?.drivingMode === "camera") {
          // In camera mode, use the temporally stabilized confirmed obstacles from the simulation loop
          visualObstacles = simulation.latestVisualObstacles || [];
        } else {
          visualObstacles = detectVisualObstacles(
            imgData,
            simulation.state.x,
            simulation.state.heading,
            simulation.traffic?.laneCentres || LANE_CENTRES,
          );
          simulation.latestVisualObstacles = visualObstacles;
        }

        const adasEnabled = adasCheck ? adasCheck.checked : true;
        const adasSource = selAdasSource ? selAdasSource.value : (simulation.settings.obstacleSource || "vision");
        let corridor = null;
        if (adasEnabled && adasSource !== "off") {
          if (adasSource === "ground_truth" && simulation?.traffic) {
            corridor = assessCorridors(
              simulation.state.x,
              simulation.state.z,
              simulation.state.heading,
              simulation.traffic.cars,
              simulation.traffic.laneCentres,
              ADAS_OBSTACLE_DANGER_M,
              ADAS_LOOKAHEAD_HORIZON_M,
              simulation.corridorState?.lockedTargetLane,
              simulation.corridorState?.prevSteering || 0,
              simulation.corridorState?.blindSpotHolds,
              simulation.state.time,
            );
          } else {
            corridor = assessCorridorsFromVisualDetections(
              simulation.state.x,
              simulation.state.heading,
              visualObstacles,
              simulation.traffic?.laneCentres || LANE_CENTRES,
              ADAS_OBSTACLE_DANGER_M,
              ADAS_LOOKAHEAD_HORIZON_M,
              simulation.corridorState?.lockedTargetLane,
              simulation.corridorState?.prevSteering || 0,
              simulation.corridorState?.blindSpotHolds,
              simulation.state.time,
            );
          }
          if (corridor && simulation.corridorState) {
            simulation.corridorState.lockedTargetLane = corridor.lockedTargetLane;
            simulation.corridorState.prevSteering = corridor.tacticalSteering;
          }
        }

        if (corridorStatus && corridor) {
          const targetName = corridor.targetLaneIdx === 0 ? "چپ ◀" : (corridor.targetLaneIdx === 1 ? "وسط ■" : "راست ▶");
          const [lDist, cDist, rDist] = corridor.clearances.map(c => `${Math.round(c)}m`);
          const modeLabel = adasSource === "vision" ? "👁️ بینایی تصویر" : "📡 رادار شبیه‌ساز";
          corridorStatus.textContent = `[${modeLabel}] مسیر: ${targetName} | L:${lDist} C:${cDist} R:${rDist}`;
          if (corridor.shouldAvoid) {
            corridorStatus.style.background = "rgba(239, 68, 68, 0.15)";
            corridorStatus.style.borderColor = "rgba(239, 68, 68, 0.4)";
            corridorStatus.style.color = "#f87171";
          } else {
            corridorStatus.style.background = "rgba(56, 189, 248, 0.08)";
            corridorStatus.style.borderColor = "rgba(56, 189, 248, 0.2)";
            corridorStatus.style.color = "#38bdf8";
          }
        }

        // Overlay visual detection bounding boxes on the camera preview
        drawVisualDetectionOverlays(previewCtx, visualObstacles, corridor?.targetLaneIdx ?? 1);

        frameCount++;
        const now = performance.now();
        if (now - lastTime >= 1000) {
          fps = Math.round((frameCount * 1000) / (now - lastTime));
          if (fpsLabel) fpsLabel.textContent = `${fps} FPS`;
          frameCount = 0;
          lastTime = now;
        }

        // Only proceed to DAVE-2 tensor inference if in Sim mode and Autopilot is active
        if (!isActive || simulation?.settings?.drivingMode !== "sim") return;

        const frameTensor = tf.tidy(() => tf.browser.fromPixels(imgData, 3).toFloat());

        frameQueue.push(frameTensor);
        while (frameQueue.length > SEQUENCE_LENGTH) {
          const oldest = frameQueue.shift();
          oldest.dispose();
        }

        if (frameQueue.length === SEQUENCE_LENGTH) {
          const command = await realInference(tf, model, frameQueue);

          let effectiveSteering = command.steering;
          const isBiasCorrected = biasCheck ? biasCheck.checked : true;
          const bias = isBiasCorrected ? 0.080 : 0.0;
          const gain = gainRange ? parseFloat(gainRange.value) : 1.5;

          effectiveSteering = (effectiveSteering - bias) * gain;
          if (invertCheck?.checked) {
            effectiveSteering = -effectiveSteering;
          }
          effectiveSteering = clamp(effectiveSteering, -1, 1);

          let finalSteering = effectiveSteering;
          let finalTurnSignal = command.turn_signal;

          if (corridor) {
            if (corridor.isManeuvering) {
              finalSteering = corridor.tacticalSteering;
              finalTurnSignal = corridor.turnSignal;
            } else {
              const laneCenter = (simulation.traffic?.laneCentres || LANE_CENTRES)[corridor.currentLaneIdx] ?? 0;
              const drift = simulation.state.x - laneCenter;
              finalSteering = clamp(effectiveSteering - drift * 0.20 - simulation.state.heading * 0.60, -1, 1);
            }
          }

          if (isActive) {
            // When browser autopilot is active, ensure cruising throttle (min 0.45)
            // unless high brake is commanded.
            const cruisingThrottle = command.brake > 0.4 ? 0 : Math.max(command.throttle, 0.45);
            simulation.receiveCommand({
              steering: finalSteering,
              throttle: cruisingThrottle,
              brake: command.brake > 0.4 ? command.brake : 0,
              turn_signal: finalTurnSignal,
            }, "browser");
          }
        }
      } catch (err) {
        console.error("[Autopilot] prediction error", err);
      } finally {
        isPredicting = false;
      }
    }

    loopTimer = setInterval(sampleAndInfer, 50);

    autopilotBtn?.addEventListener("click", () => {
      isActive = !isActive;
      simulation.settings.isBrowserAutopilotActive = isActive;
      if (isActive) {
        autopilotBtn.classList.remove("is-off");
        autopilotBtn.classList.add("is-on");
        if (autopilotBtnText) autopilotBtnText.textContent = "خودران فعال است (توقف ⏹)";
        statusLabel.textContent = "مدل شبیه‌ساز در حال هدایت خودرو است";
      } else {
        autopilotBtn.classList.remove("is-on");
        autopilotBtn.classList.add("is-off");
        if (autopilotBtnText) autopilotBtnText.textContent = "شروع خودران (DAVE-2)";
        statusLabel.textContent = "مدل شبیه‌ساز متوقف شد (ترمز کامل)";
        simulation.receiveCommand({ steering: 0, throttle: 0, brake: 1, turn_signal: 0 }, "browser");
      }
    });

  } catch (error) {
    console.error("[Autopilot] load failed", error);
    statusLabel.textContent = "خطا: " + (error.message || error);
  }
}

function boot() {
  const canvas = document.getElementById("scene-canvas");
  const hud = createHud();

  let simulation;
  try {
    simulation = createSimulation(canvas, {
      hud,
      onTelemetry: (speedMs) => renderLink.sendTelemetry(speedMs),
      onHazard: (kind) => renderLink.sendHazard(kind),
    });
    window.simulation = simulation;
  } catch (error) {
    console.error("[Scene] could not create the WebGL renderer", error);
    hud.setStatus("WebGL در دسترس نیست", "disconnected");
    return;
  }

  // The keyboard drives the car whenever no fresh phone command arrives, so
  // the scene stays testable without a phone on the dashboard.
  createKeyboardControls()((controls) => {
    simulation.setKeyboardControls(controls);
  });

  // The control entry point: every validated command from the phone lands
  // here, before any physics step reads it.
  window.applyControls = (data) => {
    simulation.receiveCommand(data);
  };

  // Expose restart function and attach restart button listeners
  window.restartSimulation = () => {
    simulation.restart();
  };
  for (const id of ["btn-restart-sim", "btn-restart-sim-legend"]) {
    const btn = document.getElementById(id);
    if (btn) {
      btn.addEventListener("click", () => {
        simulation.restart();
      });
    }
  }

  // Hook up road width and traffic density selectors
  const roadWidthSelect = document.getElementById("sel-road-width");
  roadWidthSelect?.addEventListener("change", (e) => {
    simulation.setRoadPreset(e.target.value);
  });
  const trafficDensitySelect = document.getElementById("sel-traffic-density");
  trafficDensitySelect?.addEventListener("change", (e) => {
    simulation.setTrafficPreset(e.target.value);
  });

  // Initialize direct closed-loop Autopilot test panel
  void initAutopilot(simulation, canvas);

  // Every geometry change that the plain window resize event can miss goes
  // through these: browser zoom (Ctrl+wheel), snap layouts, a window moved
  // between monitors (different DPI) and fullscreen transitions. The
  // ResizeObserver fires right after observe() too, so the first paint is
  // already exact; the stylesheet locks the canvas box, scene.js only reads it.
  window.addEventListener("resize", simulation.resize);
  if (typeof ResizeObserver !== "undefined") {
    new ResizeObserver(() => simulation.resize()).observe(canvas);
  }
  document.addEventListener("fullscreenchange", simulation.resize);

  const clock = new THREE.Clock();
  const renderLink = connectRenderSocket(hud);

  function loop() {
    // Ignore the tab-switch pause; everything else runs on real delta time.
    const dt = Math.min(clock.getDelta(), 0.1);
    simulation.update(dt);
    requestAnimationFrame(loop);
  }

  loop();
}

// Expose the control entry point before any command can arrive. Guarded so the
// module stays importable outside a browser, for the Node tests.
if (typeof window !== "undefined") {
  if (typeof document !== "undefined" && document.getElementById("scene-canvas") !== null) {
    boot();
  }
}
