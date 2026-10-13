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
export const COMMAND_FRESH_S = 0.5;

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

/* ------------------------------------------------------------------------- *
 * 2. Pure helpers
 * ------------------------------------------------------------------------- */

/** Clamp ``value`` into the inclusive range ``low`` .. ``high``. */
export function clamp(value, low, high) {
  return Math.min(Math.max(value, low), high);
}

/** Uniform random value in ``low`` .. ``high``. ``random`` is injectable. */
export function randomBetween(low, high, random = Math.random) {
  return low + (high - low) * random();
}

/** Seconds to wait before the next traffic car is spawned. */
export function pickSpawnDelay(random = Math.random) {
  return randomBetween(SPAWN_DELAY_MIN_S, SPAWN_DELAY_MAX_S, random);
}

/** x position of the lane a traffic car is spawned in. */
export function pickLane(random = Math.random) {
  const index = Math.floor(random() * LANE_CENTRES.length) % LANE_CENTRES.length;
  return LANE_CENTRES[index];
}

/** Closing speed of a traffic car, in units per second. */
export function pickTrafficSpeed(random = Math.random) {
  return randomBetween(TRAFFIC_SPEED_MIN, TRAFFIC_SPEED_MAX, random);
}

/** Distance ahead of the ego car where the next traffic car is spawned. */
export function pickSpawnGap(random = Math.random) {
  return randomBetween(SPAWN_GAP_MIN_M, SPAWN_GAP_MAX_M, random);
}

/** World z of a traffic car spawned ``gap`` units ahead of ``egoZ``. */
export function trafficSpawnZ(egoZ, gap) {
  return egoZ - gap;
}

/** Paint colour of a traffic car. */
export function pickTrafficColour(random = Math.random) {
  const index = Math.floor(random() * TRAFFIC_COLOURS.length) % TRAFFIC_COLOURS.length;
  return TRAFFIC_COLOURS[index];
}

/** Whether the pool has room for one more traffic car. */
export function canSpawnTraffic(activeCount) {
  return activeCount < MAX_TRAFFIC_CARS;
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

/** The invisible wall clamps the lateral position to |x| = ROAD_WALL_X. */
export function clampToWall(x) {
  return clamp(x, -ROAD_WALL_X, ROAD_WALL_X);
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

function createRenderer(canvas) {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(window.innerWidth, window.innerHeight);
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

function createCamera() {
  const camera = new THREE.PerspectiveCamera(
    CAMERA_FOV,
    window.innerWidth / window.innerHeight,
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

/**
 * Build every road-fixed object (ground, asphalt, markings, rails) into one
 * group. Driving towards -z eventually leaves the 400-unit plane behind, so
 * the simulation snaps the group forward in multiples of the dash period,
 * where the pattern maps onto itself and the jump is invisible.
 */
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

  /** The latest phone command and the simulation time it arrived. */
  const external = {
    command: null,
    at: -Infinity,
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

  const traffic = {
    cars: [], // every pooled car, active or not
    timer: pickSpawnDelay(),
  };

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
   */
  function spawnTrafficCar() {
    const car = acquireTrafficCar();
    if (car === null) {
      return false;
    }
    const x = pickLane();
    const z = trafficSpawnZ(state.z, pickSpawnGap());
    // Do not stack cars on top of each other in the same lane.
    for (const other of traffic.cars) {
      if (
        other !== car &&
        other.userData.active &&
        other.position.x === x &&
        Math.abs(other.position.z - z) < MIN_LANE_GAP_M
      ) {
        return false; // car stays inactive; the next timer tick tries again
      }
    }
    car.userData.speed = pickTrafficSpeed();
    car.userData.body.material = bodyMaterial(pickTrafficColour());
    car.position.set(x, 0, z);
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
      spawnTrafficCar();
      traffic.timer = pickSpawnDelay();
    }
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

  /**
   * Phone commands are authoritative for COMMAND_FRESH_S; afterwards the
   * keyboard fallback drives again. This keeps manual testing possible while
   * the closed loop stays uninterrupted at the phone's 20 Hz send rate.
   */
  function effectiveControls() {
    if (external.command !== null && state.time - external.at <= COMMAND_FRESH_S) {
      return external.command;
    }
    return state.controls;
  }

  /**
   * Store a validated phone command and switch the turn signal target.
   * Called by ``window.applyControls`` in the bootstrap.
   */
  function receiveCommand(data) {
    const command = normalizeCommand(data);
    if (command === null) {
      return false;
    }
    external.command = command;
    external.at = state.time;
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
      state.pendingReset = false;
    }

    if (state.time < state.frozenUntil) {
      syncEgoMesh();
      return;
    }

    const { steering, throttle, brake } = effectiveControls();

    state.speed = stepLongitudinalSpeed(state.speed, throttle, brake, dt);
    state.heading = stepHeading(state.heading, steering, state.speed, dt);
    const position = stepWorldPosition(state.x, state.z, state.heading, state.speed, dt);
    state.x = position.x;
    state.z = position.z;

    // Invisible wall: clamp, scrub speed while grinding, and raise one
    // wall-hit event (HUD counter + phone hazard) per contact.
    if (Math.abs(state.x) > ROAD_WALL_X) {
      state.x = clampToWall(state.x);
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
  function updateHud(dt) {
    hud.setTurnSignal(
      state.turnSignal === 1 && blinkerIsLit(state.blinkerPhase),
      state.turnSignal === 2 && blinkerIsLit(state.blinkerPhase),
    );

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

  function update(dt) {
    state.time += dt;
    frameRate = smoothFrameRate(frameRate, dt);

    updateEgoPhysics(dt);
    animateBlinkers(dt);
    updateTraffic(dt);
    updateEndlessRoad();
    updateScenery();
    updateSun();
    updateCamera();
    checkCollision();
    updateHud(dt);

    renderer.render(scene, camera);
  }

  function resize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  }

  return {
    state,
    traffic,
    update,
    resize,
    receiveCommand,
    setKeyboardControls(controls) {
      state.controls = controls;
    },
    spawnTrafficCar,
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

  return {
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

  window.addEventListener("resize", simulation.resize);

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
