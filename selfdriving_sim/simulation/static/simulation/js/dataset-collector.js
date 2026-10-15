import { createDomainProfile, createSeededRandom } from "./domain-randomization.js";
import { createExpertDriver } from "./expert-driver.js";
import { createSimulation } from "./scene.js";

const FRAME_RATE = 20;
const RAW_WIDTH = 640;
const RAW_HEIGHT = 360;
const TRAFFIC_SETTINGS = Object.freeze({
  spawnDelayMinS: 0.8,
  spawnDelayMaxS: 1.5,
  spawnGapMinM: 55,
  spawnGapMaxM: 115,
  speedMin: 9,
  speedMax: 21,
});

const hud = {
  setTurnSignal() {},
  setSpeed() {},
  setHeading() {},
  setCollisionCount() {},
  setWallCount() {},
  setFrameRate() {},
  flash() {},
};

const query = new URLSearchParams(window.location.search);
const seed = Number(query.get("seed") ?? 1);
const random = createSeededRandom(seed);
const domain = createDomainProfile(random);
const canvas = document.getElementById("dataset-canvas");
const simulation = createSimulation(canvas, {
  hud,
  onTelemetry() {},
  onHazard() {},
}, {
  random,
  domain,
  preserveDrawingBuffer: true,
  trafficSettings: TRAFFIC_SETTINGS,
});
const driver = createExpertDriver({
  random,
  cruiseSpeed: 10 + random() * 5,
});
const captureCanvas = document.createElement("canvas");
captureCanvas.width = RAW_WIDTH;
captureCanvas.height = RAW_HEIGHT;
const captureContext = captureCanvas.getContext("2d", { willReadFrequently: true });

function captureFrame() {
  captureContext.drawImage(simulation.renderer.domElement, 0, 0, RAW_WIDTH, RAW_HEIGHT);
  return captureCanvas.toDataURL("image/jpeg", 0.96).slice("data:image/jpeg;base64,".length);
}

function snapshot(index, image, command, state, valid) {
  return {
    index,
    image,
    valid,
    command,
    state: {
      x: Number(state.x.toFixed(4)),
      z: Number(state.z.toFixed(4)),
      speed: Number(state.speed.toFixed(4)),
      heading: Number(state.heading.toFixed(5)),
      time: Number(state.time.toFixed(3)),
    },
  };
}

window.datasetCollectorReady = true;
window.collectSimulationFrames = async ({ count = 100 } = {}) => {
  if (!Number.isInteger(count) || count < 1 || count > 1000) {
    throw new RangeError("Episode frame count must be an integer from 1 to 1000.");
  }

  const frames = [];
  const dt = 1 / FRAME_RATE;
  simulation.update(0);
  for (let index = 0; index < count; index += 1) {
    const observed = {
      x: simulation.state.x,
      z: simulation.state.z,
      speed: simulation.state.speed,
      heading: simulation.state.heading,
      time: simulation.state.time,
      collisions: simulation.state.collisions,
      wallHits: simulation.state.wallHits,
      pendingReset: simulation.state.pendingReset,
      frozenUntil: simulation.state.frozenUntil,
    };
    const image = captureFrame();
    const command = driver.next(simulation.state, simulation.traffic);
    simulation.receiveCommand(command);
    simulation.update(dt);
    const valid = simulation.state.collisions === observed.collisions
      && simulation.state.wallHits === observed.wallHits
      && !observed.pendingReset
      && observed.time >= observed.frozenUntil;
    frames.push(snapshot(index, image, command, observed, valid));
    if (index % 20 === 19) await new Promise((resolve) => window.setTimeout(resolve, 0));
  }

  return {
    seed,
    frameRate: FRAME_RATE,
    rawSize: [RAW_WIDTH, RAW_HEIGHT],
    domain,
    laneOffset: driver.laneOffset,
    frames,
  };
};

window.disposeDatasetCollector = () => simulation.dispose();
