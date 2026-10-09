/**
 * Three.js driving scene for the closed-loop simulator.
 *
 * The module is split in five parts:
 *
 *   1. Tuning constants.
 *   2. Pure helpers: spawn scheduling, control sanitising, the speed model and
 *      the reconnect timing. They never touch the DOM, so
 *      ``simulation/tests/js/scene.test.mjs`` can import and exercise them
 *      under Node.
 *   3. Scene construction: road, lane markings, scenery and cars.
 *   4. The simulation object that owns the state and the animation loop.
 *   5. The browser bootstrap: HUD, keyboard input and the WebSocket client.
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

/** Paint colour of the player car. */
export const EGO_COLOUR = 0x2196f3;

/** Upper bound for the number of traffic cars that exist at the same time. */
export const MAX_TRAFFIC_CARS = 20;

/** Bounds for the pause between two traffic spawns, in seconds. */
export const SPAWN_DELAY_MIN_S = 2.5;
export const SPAWN_DELAY_MAX_S = 4.5;

/** Traffic closing speed in units per second: 0.15 to 0.35 per frame at 60 Hz. */
export const TRAFFIC_SPEED_MIN = 0.15 * FRAME_RATE_HZ;
export const TRAFFIC_SPEED_MAX = 0.35 * FRAME_RATE_HZ;

/** z range a traffic car is spawned in, and the z at which it is recycled. */
export const SPAWN_Z_MIN = -200;
export const SPAWN_Z_MAX = -150;
export const DESPAWN_Z = 10;

/** Road geometry. The plane covers z = -370 .. 30 around ROAD_Z_CENTRE. */
export const ROAD_WIDTH = 20;
export const ROAD_LENGTH = 400;
export const ROAD_Z_CENTRE = -170;
export const ROAD_COLOUR = 0x2a2a2a;

/** Lane marking pattern, in world units. */
export const DASH_SIZE = 3;
export const DASH_GAP = 3;

/** Longitudinal model of the player car. */
export const MAX_SPEED_MS = 30; // ~108 km/h, enough for the HUD in this phase
export const ACCELERATION_MS2 = 6;
export const BRAKE_DECELERATION_MS2 = 12;
export const ROLLING_DRAG_MS2 = 0.8; // proportional to speed
export const COASTING_RESISTANCE_MS2 = 0.4; // constant, so coasting comes to rest

/** Lateral model of the player car. */
export const STEER_RATE = 7; // units per second at full lock
export const EGO_LATERAL_LIMIT = 8.5; // keep the car on the 20 unit wide road

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

/** z position a traffic car is spawned at. */
export function pickSpawnZ(random = Math.random) {
  return randomBetween(SPAWN_Z_MIN, SPAWN_Z_MAX, random);
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
 * attempts is used up. The delay doubles every attempt and saturates.
 */
export function nextReconnectDelayMs(attempt) {
  if (attempt < 1 || attempt > MAX_RECONNECT_ATTEMPTS) {
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

/** One longitudinal step of the player car. Returns the speed in m/s. */
export function stepSpeed(speed, throttle, brake, dt) {
  // A stationary car without throttle stays stationary.
  if (speed <= 0 && throttle <= 0) {
    return 0;
  }
  const drag = ROLLING_DRAG_MS2 * (speed / MAX_SPEED_MS) + COASTING_RESISTANCE_MS2;
  const acceleration = throttle * ACCELERATION_MS2 - brake * BRAKE_DECELERATION_MS2 - drag;
  return clamp(speed + acceleration * dt, 0, MAX_SPEED_MS);
}

/** One lateral step of the player car. Returns the new x position. */
export function stepLateralPosition(x, steering, dt) {
  return clamp(x + steering * STEER_RATE * dt, -EGO_LATERAL_LIMIT, EGO_LATERAL_LIMIT);
}

/** Convert m/s into km/h. */
export function speedToKmh(speed) {
  return speed * 3.6;
}

/** Speed frame sent from the scene to the phone. Commands are never included. */
export function telemetryFrame(speedMs) {
  const speedKmh = clamp(Math.round(speedToKmh(speedMs)), 0, 400);
  return { type: "telemetry", speed_kmh: speedKmh };
}

/**
 * Offset applied to the lane markings so that they appear to scroll with the
 * car. The offset stays inside one dash period, which keeps the pattern
 * seamless and the number small.
 */
export function dashScrollOffset(distance) {
  const period = DASH_SIZE + DASH_GAP;
  return ((distance % period) + period) % period;
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
const SCENERY_START_Z = -340;
const SCENERY_END_Z = 20;

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
    60,
    window.innerWidth / window.innerHeight,
    0.1,
    600,
  );
  camera.position.set(0, 3, 8);
  camera.lookAt(0, 0, -20);
  return camera;
}

function createLights(scene) {
  scene.add(new THREE.AmbientLight(0xffffff, 0.55));

  const sun = new THREE.DirectionalLight(0xfff4e0, 1.1);
  sun.position.set(18, 30, 12);
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
  return sun;
}

function createRoad(scene) {
  const ground = new THREE.Mesh(
    new THREE.PlaneGeometry(240, ROAD_LENGTH + 60),
    new THREE.MeshStandardMaterial({ color: GROUND_COLOUR, roughness: 1 }),
  );
  ground.rotation.x = -Math.PI / 2;
  ground.position.set(0, -0.02, ROAD_Z_CENTRE);
  ground.receiveShadow = true;
  scene.add(ground);

  const road = new THREE.Mesh(
    new THREE.PlaneGeometry(ROAD_WIDTH, ROAD_LENGTH),
    new THREE.MeshStandardMaterial({ color: ROAD_COLOUR, roughness: 0.95 }),
  );
  road.rotation.x = -Math.PI / 2;
  road.position.set(0, 0, ROAD_Z_CENTRE);
  road.receiveShadow = true;
  scene.add(road);
  return road;
}

function laneLineGeometry() {
  return new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(0, 0.02, ROAD_Z_CENTRE - ROAD_LENGTH / 2),
    new THREE.Vector3(0, 0.02, ROAD_Z_CENTRE + ROAD_LENGTH / 2),
  ]);
}

/** Dashed lane dividers. Translating them along z makes them scroll. */
function createLaneLines(scene) {
  const geometry = laneLineGeometry();
  const material = new THREE.LineDashedMaterial({
    color: 0xffffff,
    dashSize: DASH_SIZE,
    gapSize: DASH_GAP,
  });

  const lines = [];
  for (const x of LANE_DIVIDER_X) {
    const line = new THREE.Line(geometry, material);
    line.position.x = x;
    line.computeLineDistances();
    scene.add(line);
    lines.push(line);
  }
  return lines;
}

/** Solid lines at the outer edge of the road. */
function createEdgeLines(scene) {
  const geometry = laneLineGeometry();
  const material = new THREE.LineBasicMaterial({ color: 0xffffff });
  const halfWidth = ROAD_WIDTH / 2 - 0.5;
  for (const x of [-halfWidth, halfWidth]) {
    const line = new THREE.Line(geometry, material);
    line.position.x = x;
    scene.add(line);
  }
}

/** Guard rails along both sides of the road. */
function createGuardRails(scene) {
  const material = new THREE.MeshStandardMaterial({
    color: 0x9e9e9e,
    roughness: 0.5,
    metalness: 0.6,
  });
  const geometry = new THREE.BoxGeometry(0.2, 0.5, ROAD_LENGTH);
  for (const x of [-(ROAD_WIDTH / 2 + 0.4), ROAD_WIDTH / 2 + 0.4]) {
    const rail = new THREE.Mesh(geometry, material);
    rail.position.set(x, 0.45, ROAD_Z_CENTRE);
    rail.castShadow = true;
    scene.add(rail);
  }
}

/**
 * Trees and lamp posts beside the road. Every item is recycled when it passes
 * the camera, so scrolling costs nothing but a position update.
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

  const span = SCENERY_END_Z - SCENERY_START_Z;
  const spacing = 22;

  for (let z = SCENERY_START_Z; z < SCENERY_END_Z; z += spacing) {
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
        z + Math.abs((z * 3) % spacing),
      );
      group.add(tree);
    }
  }

  for (let z = SCENERY_START_Z; z < SCENERY_END_Z; z += spacing * 2) {
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
    windshield.position.set(0, 1.55, -0.85);
    windshield.rotation.x = -0.28;
    car.add(windshield);

    const rearWindow = new THREE.Mesh(geometries.glass, glassMaterial);
    rearWindow.position.set(0, 1.55, 1.25);
    rearWindow.rotation.x = 0.28;
    car.add(rearWindow);

    for (const side of [-1, 1]) {
      for (const axle of [-1.3, 1.3]) {
        const wheel = new THREE.Mesh(geometries.wheel, tyreMaterial);
        wheel.position.set(side * 0.98, 0.36, axle);
        wheel.castShadow = true;
        car.add(wheel);
      }
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
 * Create the scene, the cars and the state, and drive them from a
 * requestAnimationFrame loop.
 */
function createSimulation(canvas) {
  const renderer = createRenderer(canvas);
  const scene = createScene();
  const camera = createCamera();
  createLights(scene);
  createRoad(scene);
  const laneLines = createLaneLines(scene);
  createEdgeLines(scene);
  createGuardRails(scene);
  const scenery = createScenery(scene);
  const { buildCar, bodyMaterial } = createCarFactory();

  const state = {
    speed: 0, // m/s
    lateralX: 0,
    travelled: 0, // used to scroll the lane markings and the scenery
    controls: { steering: 0, throttle: 0, brake: 0, turn_signal: 0 },
  };

  const egoCar = buildCar(EGO_COLOUR);
  scene.add(egoCar);

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

  /** Place a traffic car at a random lane, far down the road, moving towards us. */
  function spawnTrafficCar() {
    const car = acquireTrafficCar();
    if (car === null) {
      return false;
    }
    car.userData.speed = pickTrafficSpeed();
    car.userData.body.material = bodyMaterial(pickTrafficColour());
    car.position.set(pickLane(), 0, pickSpawnZ());
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
      car.position.z += car.userData.speed * dt;
      if (car.position.z > DESPAWN_Z) {
        releaseTrafficCar(car);
      }
    }

    traffic.timer -= dt;
    if (traffic.timer <= 0) {
      spawnTrafficCar();
      traffic.timer = pickSpawnDelay();
    }
  }

  /** Scroll the markings and the scenery by the distance covered this frame. */
  function updateScrolling(distance) {
    const offset = dashScrollOffset(state.travelled);
    for (const line of laneLines) {
      line.position.z = offset;
    }
    const span = scenery.userData.span;
    for (const item of scenery.children) {
      item.position.z += distance;
      if (item.position.z > SCENERY_END_Z) {
        item.position.z -= span;
      }
    }
  }

  function update(dt) {
    const { steering, throttle, brake } = state.controls;

    state.speed = stepSpeed(state.speed, throttle, brake, dt);
    state.lateralX = stepLateralPosition(state.lateralX, steering, dt);
    state.travelled += state.speed * dt;

    egoCar.position.x = state.lateralX;
    // A slight roll and yaw make the steering readable on screen.
    egoCar.rotation.z = -steering * 0.06;
    egoCar.rotation.y = -steering * 0.08;

    updateTraffic(dt);
    updateScrolling(state.speed * dt);

    // The camera follows the car laterally instead of snapping to it.
    camera.position.x += (state.lateralX - camera.position.x) * clamp(dt * 4, 0, 1);
    camera.lookAt(state.lateralX * 0.6, 1, -20);

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
    spawnTrafficCar,
    scene,
    camera,
    renderer,
  };
}

/* ------------------------------------------------------------------------- *
 * 5. Browser bootstrap
 * ------------------------------------------------------------------------- */

/**
 * Placeholder for phase 5. The renderer receives validated commands here and
 * only logs them for now; the driving model is applied in phase 5.
 */
function applyControls(data) {
  console.log("[Controls]", data);
}

function createHud() {
  const status = document.getElementById("ws-status");
  const speed = document.getElementById("speed-value");
  const frameRate = document.getElementById("fps-value");

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
      if (socket?.readyState !== WebSocket.OPEN || socket.bufferedAmount > 1024) return false;
      try {
        socket.send(JSON.stringify(telemetryFrame(speedMs)));
        return true;
      } catch {
        return false;
      }
    },
  };
}

function boot() {
  const canvas = document.getElementById("scene-canvas");
  const hud = createHud();

  let simulation;
  try {
    simulation = createSimulation(canvas);
  } catch (error) {
    console.error("[Scene] could not create the WebGL renderer", error);
    hud.setStatus("WebGL در دسترس نیست", "disconnected");
    return;
  }

  // Keyboard commands drive the phase 2 scene directly. In phase 5 the
  // WebSocket commands reach the same state through applyControls.
  createKeyboardControls()((controls) => {
    simulation.state.controls = controls;
  });

  window.addEventListener("resize", simulation.resize);

  const clock = new THREE.Clock();
  let frameRate = 60;
  let hudTimer = 0;
  const renderLink = connectRenderSocket(hud);

  function loop() {
    const dt = Math.min(clock.getDelta(), 0.1); // ignore tab switch pauses
    simulation.update(dt);

    frameRate = smoothFrameRate(frameRate, dt);
    hudTimer += dt;
    if (hudTimer >= 0.25) {
      hud.setSpeed(simulation.state.speed);
      hud.setFrameRate(frameRate);
      renderLink.sendTelemetry(simulation.state.speed);
      hudTimer = 0;
    }
    requestAnimationFrame(loop);
  }

  loop();
}

// Expose the phase 5 entry point before any command can arrive. Guarded so the
// module stays importable outside a browser, for the Node tests.
if (typeof window !== "undefined") {
  window.applyControls = applyControls;

  if (typeof document !== "undefined" && document.getElementById("scene-canvas") !== null) {
    boot();
  }
}
