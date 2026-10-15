const SKY_COLOURS = [0xbcd4e6, 0xa9bdca, 0xc8c9c2, 0xb0aaa2, 0x9eafba];
const GROUND_COLOURS = [0x4a6b3a, 0x596849, 0x766f55, 0x657461, 0x59615a];
const ROAD_COLOURS = [0x26282a, 0x303538, 0x424242, 0x353834, 0x494641];
const MARKING_COLOURS = [0xffffff, 0xf5f0dd, 0xf0d67a, 0xe9e4d2];
const RAIL_COLOURS = [0x9e9e9e, 0x898989, 0xb1aaa0, 0x777d7d];
const POLE_COLOURS = [0x616161, 0x727272, 0x777d7d, 0x8a847b];
const FOLIAGE_COLOURS = [0x2e7d32, 0x3f6b31, 0x728343, 0x5d7450, 0x39634c];
const TRUNK_COLOURS = [0x5d4037, 0x70543c, 0x5b5247, 0x634d3b];
const SUN_COLOURS = [0xfff4e0, 0xffe0b2, 0xf5f5f5, 0xf7e7ce];

export const BASE_DOMAIN_PROFILE = Object.freeze({
  skyColor: 0xbcd4e6,
  groundColor: 0x4a6b3a,
  roadColor: 0x2a2a2a,
  markingColor: 0xffffff,
  railColor: 0x9e9e9e,
  poleColor: 0x616161,
  foliageColor: 0x2e7d32,
  trunkColor: 0x5d4037,
  sunColor: 0xfff4e0,
  fogNear: 90,
  fogFar: 300,
  ambientIntensity: 0.55,
  sunIntensity: 1.1,
  cameraFov: 60,
  cameraEyeHeight: 0.75,
  cameraOffsetX: 0,
  cameraPitchRadians: 0,
  cameraYawRadians: 0,
  cameraRollRadians: 0,
  dashSize: 3,
  dashGap: 3,
  textureSeed: 0,
  texturedSurfaces: false,
});

function pick(values, random) {
  return values[Math.min(values.length - 1, Math.floor(random() * values.length))];
}

function between(low, high, random) {
  return low + (high - low) * random();
}

/** A small deterministic generator for repeatable simulator episodes. */
export function createSeededRandom(seed) {
  if (!Number.isFinite(seed)) {
    throw new TypeError("Seed must be a finite number.");
  }

  let state = Number(seed) >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 0x1_0000_0000;
  };
}

/** Select one coherent appearance and camera profile for a simulator episode. */
export function createDomainProfile(random = Math.random) {
  const profile = {
    skyColor: pick(SKY_COLOURS, random),
    groundColor: pick(GROUND_COLOURS, random),
    roadColor: pick(ROAD_COLOURS, random),
    markingColor: pick(MARKING_COLOURS, random),
    railColor: pick(RAIL_COLOURS, random),
    poleColor: pick(POLE_COLOURS, random),
    foliageColor: pick(FOLIAGE_COLOURS, random),
    trunkColor: pick(TRUNK_COLOURS, random),
    sunColor: pick(SUN_COLOURS, random),
    fogNear: between(68, 108, random),
    fogFar: between(245, 345, random),
    ambientIntensity: between(0.36, 0.82, random),
    sunIntensity: between(0.68, 1.32, random),
    cameraFov: between(54, 70, random),
    cameraEyeHeight: between(0.63, 0.91, random),
    cameraOffsetX: between(-0.12, 0.12, random),
    cameraPitchRadians: between(-0.035, 0.035, random),
    cameraYawRadians: between(-0.022, 0.022, random),
    cameraRollRadians: between(-0.018, 0.018, random),
    dashSize: between(2.4, 4.0, random),
    dashGap: between(2.6, 5.0, random),
    textureSeed: Math.floor(random() * 0x1_0000_0000),
    texturedSurfaces: true,
  };

  if (profile.fogFar <= profile.fogNear + 100) {
    profile.fogFar = profile.fogNear + 100;
  }

  return Object.freeze(profile);
}

/** Return a deterministic integer seed for an independently randomized surface. */
export function nextTextureSeed(random) {
  return Math.floor(random() * 0x1_0000_0000) >>> 0;
}
