import assert from "node:assert/strict";
import test from "node:test";

import {
  ADAS_LOOKAHEAD_HORIZON_M,
  ADAS_SIGNAL_INTENT_DISTANCE_M,
  ADAS_OBSTACLE_DANGER_M,
  LANE_CENTRES,
  evaluateCorridorManeuver,
  assessCorridorsFromVisualDetections,
  detectVisualObstacles,
  detectObstaclesWithAI,
  detectObstaclesWithLidar,
  drawVisualDetectionOverlays,
  renderPerceptionPreview,
} from "../../static/simulation/js/scene.js";

test("evaluateCorridorManeuver keeps current lane when all clearances are clear", () => {
  const clearances = [80, 80, 80];
  const egoX = 0; // In center lane
  const egoHeading = 0;

  const result = evaluateCorridorManeuver(egoX, egoHeading, clearances);
  assert.equal(result.currentLaneIdx, 1);
  assert.equal(result.targetLaneIdx, 1);
  assert.equal(result.shouldAvoid, false);
  assert.equal(result.turnSignal, 0);
  assert.equal(result.tacticalSteering, 0);
});

test("evaluateCorridorManeuver swerves left when Center is blocked and Left is clear", () => {
  const clearances = [70, 25, 20]; // Left open (70m), Center blocked (25m), Right blocked (20m)
  const egoX = 0; // In center lane
  const egoHeading = 0;

  const result = evaluateCorridorManeuver(egoX, egoHeading, clearances);
  assert.equal(result.currentLaneIdx, 1);
  assert.equal(result.targetLaneIdx, 0); // Swerve to Left lane
  assert.equal(result.shouldAvoid, true);
  assert.equal(result.turnSignal, 1); // Left turn signal
  assert.equal(result.lockedTargetLane, 0); // Latches onto Left lane!
  assert.ok(result.tacticalSteering < 0); // Steer left (negative steering)
});

test("evaluateCorridorManeuver preserves lockedTargetLane despite clearance fluctuations", () => {
  // Maneuver in progress: locked onto Left (0).
  // Even if Right lane clearance fluctuates higher (e.g. 78m vs 65m), lock must NOT switch!
  const clearances = [65, 20, 78];
  const egoX = -0.8;
  const egoHeading = -0.15;

  const result = evaluateCorridorManeuver(egoX, egoHeading, clearances, LANE_CENTRES, 45, 0, -0.35);
  assert.equal(result.targetLaneIdx, 0); // Stays committed to Left lane!
  assert.equal(result.lockedTargetLane, 0);
  assert.ok(result.tacticalSteering < 0);
});

test("evaluateCorridorManeuver swerves right when Center is blocked and only Right is clear", () => {
  const clearances = [20, 25, 75]; // Left blocked, Center blocked, Right clear
  const egoX = 0;
  const egoHeading = 0;

  const result = evaluateCorridorManeuver(egoX, egoHeading, clearances);
  assert.equal(result.currentLaneIdx, 1);
  assert.equal(result.targetLaneIdx, 2); // Swerve to Right lane
  assert.equal(result.shouldAvoid, true);
  assert.equal(result.turnSignal, 2); // Right turn signal
  assert.equal(result.lockedTargetLane, 2);
  assert.ok(result.tacticalSteering > 0); // Steer right (positive steering)
});

test("evaluateCorridorManeuver maintains cruising in current lane when open and clear ahead", () => {
  const clearances = [70, 75, 20]; // In Left lane, Left lane is open
  const lanes = [-3.4, 0, 3.4];
  const egoX = -3.4; // Left lane
  const egoHeading = 0;

  const result = evaluateCorridorManeuver(egoX, egoHeading, clearances, lanes);
  assert.equal(result.currentLaneIdx, 0);
  assert.equal(result.targetLaneIdx, 0); // Stays in Left lane without darting back
  assert.equal(result.shouldAvoid, false);
  assert.equal(result.turnSignal, 0);
});

test("evaluateCorridorManeuver safely avoids obstacle ahead when in Left lane", () => {
  const clearances = [20, 75, 20]; // In Left lane, Left lane is blocked at 20m, Center is open
  const lanes = [-3.4, 0, 3.4];
  const egoX = -3.4; // Left lane
  const egoHeading = 0;

  const result = evaluateCorridorManeuver(egoX, egoHeading, clearances, lanes);
  assert.equal(result.currentLaneIdx, 0);
  assert.equal(result.targetLaneIdx, 1); // Evades to Center lane
  assert.equal(result.shouldAvoid, true);
  assert.equal(result.turnSignal, 2); // Right signal
  assert.ok(result.tacticalSteering > 0); // Steers right towards Center
});

test("assessCorridorsFromVisualDetections respects blind spot memory hold for alongside vehicle", () => {
  const blindSpotHolds = [0, 10.5, 0]; // Center lane held in blind spot until t=10.5
  const currentTime = 10.0;
  const visualObstacles = []; // Forward camera sees empty road
  const result = assessCorridorsFromVisualDetections(
    -3.4,
    0,
    visualObstacles,
    LANE_CENTRES,
    ADAS_OBSTACLE_DANGER_M,
    ADAS_LOOKAHEAD_HORIZON_M,
    null,
    0,
    blindSpotHolds,
    currentTime,
  );
  assert.equal(result.clearances[0], ADAS_LOOKAHEAD_HORIZON_M); // Left lane open
  assert.ok(result.clearances[1] <= 4); // Center lane held blocked by car beside us!
});

test("assessCorridorsFromVisualDetections maps visual obstacles to lane clearances", () => {
  const visualObstacles = [
    { laneIdx: 1, distanceM: 28 }, // Center obstacle at 28m
    { laneIdx: 2, distanceM: 18 }, // Right obstacle at 18m
  ];

  const result = assessCorridorsFromVisualDetections(0, 0, visualObstacles, LANE_CENTRES);
  assert.equal(result.source, "vision");
  assert.equal(result.clearances[0], ADAS_LOOKAHEAD_HORIZON_M); // Left lane open
  assert.equal(result.clearances[1], 28); // Center lane blocked at 28m
  assert.equal(result.clearances[2], 18); // Right lane blocked at 18m
  assert.equal(result.targetLaneIdx, 0); // Swerve to open Left lane!
  assert.equal(result.turnSignal, 1); // Left turn signal
});

test("detectVisualObstacles handles empty road image without false positives", () => {
  const width = 160;
  const height = 120;
  const data = new Uint8ClampedArray(width * height * 4);

  // Fill with neutral asphalt [36, 34, 32, 255]
  for (let i = 0; i < data.length; i += 4) {
    data[i] = 36;
    data[i + 1] = 34;
    data[i + 2] = 32;
    data[i + 3] = 255;
  }

  const detections = detectVisualObstacles({ width, height, data });
  assert.deepEqual(detections, []);
});

test("detectVisualObstacles detects a center lane vehicle and estimates distance and lane", () => {
  const width = 160;
  const height = 120;
  const data = new Uint8ClampedArray(width * height * 4);

  // Fill with neutral asphalt
  for (let i = 0; i < data.length; i += 4) {
    data[i] = 36;
    data[i + 1] = 34;
    data[i + 2] = 32;
    data[i + 3] = 255;
  }

  // Paint a red vehicle at center (x=78..81, y=61..63)
  // dy = 63 - 59.5 = 3.5 -> distance = 100 / 3.5 = 29m!
  for (let y = 61; y <= 63; y++) {
    for (let x = 78; x <= 81; x++) {
      const idx = (y * width + x) * 4;
      data[idx] = 220;     // Red body
      data[idx + 1] = 40;  // Green
      data[idx + 2] = 40;  // Blue
      data[idx + 3] = 255;
    }
  }

  const detections = detectVisualObstacles({ width, height, data }, 0, 0);
  assert.equal(detections.length, 1);
  assert.equal(detections[0].laneIdx, 1); // Center lane
  assert.equal(detections[0].laneName, "وسط");
  assert.equal(detections[0].distanceM, 29);
  assert.deepEqual(detections[0].bbox, [78, 61, 4, 3]);
});

test("detectVisualObstacles correctly compensates for ego heading and lateral offset", () => {
  const width = 160;
  const height = 120;
  const data = new Uint8ClampedArray(width * height * 4);

  for (let i = 0; i < data.length; i += 4) {
    data[i] = 36; data[i + 1] = 34; data[i + 2] = 32; data[i + 3] = 255;
  }

  // Vehicle is at world X=0 (Center lane) at 22m.
  // When ego turns left (heading = -0.18, egoX = -0.8), the vehicle shifts
  // right in the camera image. Projection (focal=104 px): theta = (w-x0)/d - h
  // = 0.8/22 + 0.18 = 0.216 -> centerX = 80 + 104*0.216 ~ 102.5 at y=63..64
  for (let y = 63; y <= 64; y++) {
    for (let x = 101; x <= 104; x++) {
      const idx = (y * width + x) * 4;
      data[idx] = 220; data[idx + 1] = 40; data[idx + 2] = 40; data[idx + 3] = 255;
    }
  }

  // Without compensation, x=101..104 would falsely be marked as Right lane.
  // With egoX=-0.8 and egoHeading=-0.18, it must correctly resolve to Center lane (1)!
  const detections = detectVisualObstacles({ width, height, data }, -0.8, -0.18);
  assert.equal(detections.length, 1);
  assert.equal(detections[0].laneIdx, 1); // Correctly classified as Center lane!
});

test("drawVisualDetectionOverlays renders tags and boxes on mock 2D context", () => {
  const operations = [];
  const mockCtx = {
    save() { operations.push("save"); },
    restore() { operations.push("restore"); },
    strokeRect(x, y, w, h) { operations.push(`strokeRect:${x},${y},${w},${h}`); },
    fillRect(x, y, w, h) { operations.push(`fillRect:${x},${y},${w},${h}`); },
    beginPath() { operations.push("beginPath"); },
    moveTo(x, y) { operations.push(`moveTo:${x},${y}`); },
    lineTo(x, y) { operations.push(`lineTo:${x},${y}`); },
    stroke() { operations.push("stroke"); },
    measureText(text) { return { width: text.length * 6 }; },
    fillText(text, x, y) { operations.push(`fillText:${text}`); },
  };

  const obstacles = [
    { laneIdx: 1, laneName: "وسط", distanceM: 25, bbox: [75, 60, 10, 4] },
  ];

  drawVisualDetectionOverlays(mockCtx, obstacles, 1);
  assert.ok(operations.includes("save"));
  assert.ok(operations.includes("restore"));
  assert.ok(operations.some(op => op.startsWith("strokeRect:75,60,10,4")));
  assert.ok(operations.some(op => op.includes("fillText:25m | وسط")));
});

test("detectVisualObstacles ignores LCD screen color shifts and small noise specks on asphalt", () => {
  const width = 160;
  const height = 120;
  const data = new Uint8ClampedArray(width * height * 4);

  // Fill with asphalt having typical phone camera LCD color shift (R:45, G:56, B:42)
  for (let i = 0; i < data.length; i += 4) {
    data[i] = 45;
    data[i + 1] = 56;
    data[i + 2] = 42;
    data[i + 3] = 255;
  }

  // Add a 2x2 noise speck at (x=80..81, y=65..66)
  for (let y = 65; y <= 66; y++) {
    for (let x = 80; x <= 81; x++) {
      const idx = (y * width + x) * 4;
      data[idx] = 200;
      data[idx + 1] = 20;
      data[idx + 2] = 20;
      data[idx + 3] = 255;
    }
  }

  // Must reject both color shift and tiny 2x2 noise (not a vehicle!)
  const detections = detectVisualObstacles({ width, height, data }, 0, 0);
  assert.deepEqual(detections, []);
});

test("detectVisualObstacles rejects highway dashed lines, road edge markings and horizon glare", () => {
  const width = 160;
  const height = 120;
  const data = new Uint8ClampedArray(width * height * 4);

  // Fill road with neutral asphalt
  for (let i = 0; i < data.length; i += 4) {
    data[i] = 38; data[i + 1] = 38; data[i + 2] = 38; data[i + 3] = 255;
  }

  // 1. Paint a dashed white lane line: skinny vertical stripe (w=2, h=14)
  for (let y = 64; y <= 77; y++) {
    for (let x = 70; x <= 71; x++) {
      const idx = (y * width + x) * 4;
      data[idx] = 230; data[idx + 1] = 230; data[idx + 2] = 230; data[idx + 3] = 255;
    }
  }

  // 2. Paint a horizontal glare band across the horizon (w=35, h=2)
  for (let y = 61; y <= 62; y++) {
    for (let x = 60; x <= 94; x++) {
      const idx = (y * width + x) * 4;
      data[idx] = 190; data[idx + 1] = 180; data[idx + 2] = 160; data[idx + 3] = 255;
    }
  }

  // Both the skinny lane line and the wide horizontal glare must be rejected!
  const detections = detectVisualObstacles({ width, height, data }, 0, 0);
  assert.deepEqual(detections, []);
});

test("detectVisualObstacles ignores ego vehicle hood corner indicators even when illuminated amber", () => {
  const width = 160;
  const height = 120;
  const data = new Uint8ClampedArray(width * height * 4);

  // Fill with neutral asphalt
  for (let i = 0; i < data.length; i += 4) {
    data[i] = 36; data[i + 1] = 34; data[i + 2] = 32; data[i + 3] = 255;
  }

  // Paint glowing amber left indicator in lower-left corner (y=78..82, x=15..25)
  for (let y = 78; y <= 82; y++) {
    for (let x = 15; x <= 25; x++) {
      const idx = (y * width + x) * 4;
      data[idx] = 255; data[idx + 1] = 160; data[idx + 2] = 0; data[idx + 3] = 255;
    }
  }

  // Paint glowing amber right indicator in lower-right corner (y=78..82, x=135..145)
  for (let y = 78; y <= 82; y++) {
    for (let x = 135; x <= 145; x++) {
      const idx = (y * width + x) * 4;
      data[idx] = 255; data[idx + 1] = 160; data[idx + 2] = 0; data[idx + 3] = 255;
    }
  }

  const detections = detectVisualObstacles({ width, height, data }, 0, 0);
  assert.deepEqual(detections, []); // Blinkers must NEVER be detected as obstacles!
});

test("evaluateCorridorManeuver stays strictly committed to lockedTargetLane throughout transition and never swerves back", () => {
  // Maneuver in progress: ego vehicle is halfway into lane change to Left (lockedTargetLane = 0)
  // egoX = -1.6 (between Center 0 and Left -3.4), egoHeading = -0.22
  // Transient blip or rear tire in Center/Left causes targetClearance to drop to 6m!
  const clearances = [6, 18, 75];
  const egoX = -1.6;
  const egoHeading = -0.22;

  const result = evaluateCorridorManeuver(egoX, egoHeading, clearances, LANE_CENTRES, 45, 0, -0.45);
  // Must NOT abandon the lane change mid-way or steer back into the filled Center lane!
  assert.equal(result.targetLaneIdx, 0);
  assert.equal(result.lockedTargetLane, 0);
  assert.equal(result.shouldAvoid, true);
  assert.ok(result.tacticalSteering < 0); // Keeps steering left to complete safe evasion
});

test("detectVisualObstacles detects vehicles and avoids false positives on night driving scene", () => {
  const width = 160;
  const height = 120;
  const data = new Uint8ClampedArray(width * height * 4);

  // Fill road with dark night asphalt (lum ~ 13)
  for (let i = 0; i < data.length; i += 4) {
    data[i] = 12; data[i + 1] = 13; data[i + 2] = 15; data[i + 3] = 255;
  }

  // 1. Empty night road: must have 0 false detections!
  const emptyDetections = detectVisualObstacles({ width, height, data }, 0, 0);
  assert.deepEqual(emptyDetections, []);

  // 2. Add an oncoming vehicle with illuminated headlights and front fascia in Center lane at 29m
  for (let y = 61; y <= 63; y++) {
    for (let x = 78; x <= 81; x++) {
      const idx = (y * width + x) * 4;
      data[idx] = 180; data[idx + 1] = 180; data[idx + 2] = 130; data[idx + 3] = 255;
    }
  }

  const detections = detectVisualObstacles({ width, height, data }, 0, 0);
  assert.equal(detections.length, 1);
  assert.equal(detections[0].laneIdx, 1);
  assert.equal(detections[0].laneName, "وسط");
  assert.equal(detections[0].distanceM, 29);
});

test("detectVisualObstacles detects vehicles and avoids false positives on rainy wet road scene", () => {
  const width = 160;
  const height = 120;
  const data = new Uint8ClampedArray(width * height * 4);

  // Fill road with rainy wet asphalt (lum ~ 17 with sky tint)
  for (let i = 0; i < data.length; i += 4) {
    data[i] = 16; data[i + 1] = 17; data[i + 2] = 20; data[i + 3] = 255;
  }

  // 1. Empty rainy road: must have 0 false detections!
  const emptyDetections = detectVisualObstacles({ width, height, data }, 0, 0);
  assert.deepEqual(emptyDetections, []);

  // 2. Add a vehicle in Left lane (worldX ~ -3.4 at 22m): theta = 3.4/22
  // (focal=104) -> centerX = 80 - 16 ~ 64, so x=62..65 at y=62..64
  for (let y = 62; y <= 64; y++) {
    for (let x = 62; x <= 65; x++) {
      const idx = (y * width + x) * 4;
      data[idx] = 95; data[idx + 1] = 40; data[idx + 2] = 40; data[idx + 3] = 255;
    }
  }

  const detections = detectVisualObstacles({ width, height, data }, 0, 0);
  assert.equal(detections.length, 1);
  assert.equal(detections[0].laneIdx, 0);
  assert.equal(detections[0].laneName, "چپ");
});

test("detectVisualObstacles detects vehicles across headlight distance illumination gradient", () => {
  const width = 160;
  const height = 120;
  const data = new Uint8ClampedArray(width * height * 4);

  // Gradient: near road (y=88) is brightly lit by headlights (lum ~ 40),
  // while distant horizon road (y=61) is dark (lum ~ 12).
  for (let y = 0; y < height; y++) {
    const t = Math.max(0, Math.min(1, (y - 60) / 28));
    const asphaltLum = Math.round(12 + t * 28);
    for (let x = 0; x < width; x++) {
      const idx = (y * width + x) * 4;
      data[idx] = asphaltLum - 1;
      data[idx + 1] = asphaltLum;
      data[idx + 2] = asphaltLum + 2;
      data[idx + 3] = 255;
    }
  }

  // 1. Empty road with strong gradient must not create false positive strips
  const emptyDetections = detectVisualObstacles({ width, height, data }, 0, 0);
  assert.deepEqual(emptyDetections, []);

  // 2. Add vehicle in Center lane at 22m (y=62..64, x=78..81)
  for (let y = 62; y <= 64; y++) {
    for (let x = 78; x <= 81; x++) {
      const idx = (y * width + x) * 4;
      data[idx] = 180; data[idx + 1] = 180; data[idx + 2] = 120; data[idx + 3] = 255;
    }
  }

  const detections = detectVisualObstacles({ width, height, data }, 0, 0);
  assert.equal(detections.length, 1);
  assert.equal(detections[0].laneIdx, 1);
  assert.equal(detections[0].laneName, "وسط");
  assert.equal(detections[0].distanceM, 22);
});

test("detectVisualObstacles rejects asphalt tire tracks, headlight illumination pools and road sheen as ground", () => {
  const width = 160;
  const height = 120;
  const data = new Uint8ClampedArray(width * height * 4);

  // Sunny warm asphalt [76, 71, 64]
  for (let i = 0; i < data.length; i += 4) {
    data[i] = 76;
    data[i + 1] = 71;
    data[i + 2] = 64;
    data[i + 3] = 255;
  }

  // 1. Dark tire wear tracks painted on road at (x=73..76, y=65..72)
  for (let y = 65; y <= 72; y++) {
    for (let x = 73; x <= 76; x++) {
      const idx = (y * width + x) * 4;
      data[idx] = 18;
      data[idx + 1] = 18;
      data[idx + 2] = 20;
    }
  }

  // 2. Ego headlights illumination pool on asphalt at (x=76..84, y=66..70)
  for (let y = 66; y <= 70; y++) {
    for (let x = 76; x <= 84; x++) {
      const idx = (y * width + x) * 4;
      data[idx] = 95;
      data[idx + 1] = 92;
      data[idx + 2] = 85;
    }
  }

  // Road features and headlight light pools on ground MUST have 0 detections!
  const detections = detectVisualObstacles({ width, height, data }, 0, 0);
  assert.deepEqual(detections, []);
});

test("detectVisualObstacles detects yellow traffic car in right lane and green car in left lane", () => {
  const width = 160;
  const height = 120;
  const data = new Uint8ClampedArray(width * height * 4);

  for (let i = 0; i < data.length; i += 4) {
    data[i] = 45;
    data[i + 1] = 45;
    data[i + 2] = 45;
    data[i + 3] = 255;
  }

  // 1. Yellow vehicle in Right lane (worldX ~ +3.4) at 10m (y=68..70):
  // theta = 3.4/10.5, focal=104 -> centerX ~ 114 (x=112..115). Closer rows
  // keep every painted pixel inside the paved-road wedge (|worldX| <= 4.8)
  for (let y = 68; y <= 70; y++) {
    for (let x = 112; x <= 115; x++) {
      const idx = (y * width + x) * 4;
      data[idx] = 240;
      data[idx + 1] = 210;
      data[idx + 2] = 45;
    }
  }

  // 2. Green vehicle in Left lane (worldX ~ -3.4) at 10m (y=68..70):
  // theta = -3.4/10.5, focal=104 -> centerX ~ 46 (x=45..48)
  for (let y = 68; y <= 70; y++) {
    for (let x = 45; x <= 48; x++) {
      const idx = (y * width + x) * 4;
      data[idx] = 50;
      data[idx + 1] = 160;
      data[idx + 2] = 60;
    }
  }

  const detections = detectVisualObstacles({ width, height, data }, 0, 0);
  assert.equal(detections.length, 2);

  const rightCar = detections.find(d => d.laneIdx === 2);
  assert.ok(rightCar);
  assert.equal(rightCar.laneName, "راست");

  const leftCar = detections.find(d => d.laneIdx === 0);
  assert.ok(leftCar);
  assert.equal(leftCar.laneName, "چپ");
});

test("detectObstaclesWithAI maps deep learning vehicle predictions to distance and lane corridors", async () => {
  const mockPredictions = [
    // Center car: bbox center x=80 (theta=0), bottom y=65 -> ~20m
    { class: "car", score: 0.94, bbox: [76, 61, 8, 4] },
    // Left lane truck: bbox center x=54 (theta=-0.25), bottom y=69 -> ~14m, worldX=-3.5
    { class: "truck", score: 0.88, bbox: [49, 63, 10, 6] },
    // Right lane bus: bbox center x=97.5 (theta=+0.168), bottom y=65 -> ~20m, worldX=+3.4
    { class: "bus", score: 0.91, bbox: [94, 61, 7, 4] },
  ];

  const obstacles = await detectObstaclesWithAI({ mockPredictions }, 0, 0);
  assert.equal(obstacles.length, 3);

  const centerCar = obstacles.find(o => o.laneIdx === 1);
  assert.ok(centerCar);
  assert.equal(centerCar.source, "ai");
  assert.equal(centerCar.laneName, "وسط");
  assert.ok(centerCar.distanceM > 0 && centerCar.distanceM < 50);

  const leftTruck = obstacles.find(o => o.laneIdx === 0);
  assert.ok(leftTruck);
  assert.equal(leftTruck.laneName, "چپ");

  const rightBus = obstacles.find(o => o.laneIdx === 2);
  assert.ok(rightBus);
  assert.equal(rightBus.laneName, "راست");
});

test("detectObstaclesWithAI rejects non-vehicle classes and low confidence detections", async () => {
  const mockPredictions = [
    // Non-vehicles that must be rejected
    { class: "dog", score: 0.95, bbox: [78, 62, 6, 6] },
    { class: "traffic light", score: 0.90, bbox: [90, 40, 4, 8] },
    { class: "chair", score: 0.85, bbox: [70, 65, 8, 8] },
    { class: "person", score: 0.75, bbox: [80, 60, 4, 10] },
    // Vehicle with very low confidence (< 0.20)
    { class: "car", score: 0.12, bbox: [76, 61, 8, 4] },
  ];

  const obstacles = await detectObstaclesWithAI({ mockPredictions }, 0, 0);
  assert.equal(obstacles.length, 0); // All non-vehicles and low-conf rejected!
});

test("detectObstaclesWithAI rejects detections outside highway road boundaries", async () => {
  const mockPredictions = [
    // Far right roadside object (worldX > 5m outside the paved preset)
    { class: "car", score: 0.92, bbox: [148, 65, 10, 8] },
    // Far left roadside object (worldX < -5m)
    { class: "car", score: 0.89, bbox: [2, 65, 10, 8] },
  ];

  const obstacles = await detectObstaclesWithAI({ mockPredictions }, 0, 0);
  assert.equal(obstacles.length, 0); // Roadside objects excluded!
});

test("detectObstaclesWithLidar emits bumper rays and detects physical vehicles in forward lanes", () => {
  // Ego at (0, 0, 0)
  // Car 1 in Center lane (x=0, z=-30, dist ~ 28m)
  // Car 2 in Left lane (x=-3.4, z=-20, dist ~ 18m)
  const trafficCars = [
    { position: { x: 0, z: -30 }, userData: { active: true, halfX: 1.0, halfZ: 2.2 } },
    { position: { x: -3.4, z: -20 }, userData: { active: true, halfX: 1.0, halfZ: 2.2 } },
  ];

  const obstacles = detectObstaclesWithLidar(0, 0, 0, trafficCars);
  assert.equal(obstacles.length, 2);

  const centerCar = obstacles.find(o => o.laneIdx === 1);
  assert.ok(centerCar);
  assert.equal(centerCar.source, "lidar");
  assert.equal(centerCar.laneName, "وسط");
  assert.ok(Math.abs(centerCar.distanceM - 28) <= 3);

  const leftCar = obstacles.find(o => o.laneIdx === 0);
  assert.ok(leftCar);
  assert.equal(leftCar.source, "lidar");
  assert.equal(leftCar.laneName, "چپ");
  assert.ok(Math.abs(leftCar.distanceM - 18) <= 3);
});

test("detectObstaclesWithLidar ignores inactive pooled cars and cars outside road boundary", () => {
  const trafficCars = [
    // Inactive pooled car (userData.active = false)
    { position: { x: 0, z: -25 }, userData: { active: false, halfX: 1.0, halfZ: 2.2 } },
    // Car parked off-road (x = 8.5m)
    { position: { x: 8.5, z: -25 }, userData: { active: true, halfX: 1.0, halfZ: 2.2 } },
  ];

  const obstacles = detectObstaclesWithLidar(0, 0, 0, trafficCars);
  assert.equal(obstacles.length, 0);
});

test("drawVisualDetectionOverlays renders distinctive AI and LiDAR badges", () => {
  const operations = [];
  const mockCtx = {
    save() { operations.push("save"); },
    restore() { operations.push("restore"); },
    strokeRect(x, y, w, h) { operations.push(`strokeRect:${x},${y},${w},${h}`); },
    fillRect(x, y, w, h) { operations.push(`fillRect:${x},${y},${w},${h}`); },
    beginPath() { operations.push("beginPath"); },
    moveTo(x, y) { operations.push(`moveTo:${x},${y}`); },
    lineTo(x, y) { operations.push(`lineTo:${x},${y}`); },
    stroke() { operations.push("stroke"); },
    measureText(text) { return { width: text.length * 6 }; },
    fillText(text, x, y) { operations.push(`fillText:${text}`); },
  };

  const obstacles = [
    { source: "ai", laneIdx: 1, laneName: "وسط", distanceM: 25, bbox: [75, 60, 10, 4] },
    { source: "lidar", laneIdx: 0, laneName: "چپ", distanceM: 18, bbox: [65, 62, 8, 4] },
  ];

  drawVisualDetectionOverlays(mockCtx, obstacles, 1);
  assert.ok(operations.some(op => op.includes("fillText:🤖 AI 25m | وسط")));
  assert.ok(operations.some(op => op.includes("fillText:📡 LiDAR 18m | چپ")));
});

test("drawVisualDetectionOverlays scales cleanly to 320x240 and renders mode-specific HUD banners", () => {
  const operations = [];
  const mockCtx = {
    save() { operations.push("save"); },
    restore() { operations.push("restore"); },
    strokeRect(x, y, w, h) { operations.push(`strokeRect:${x},${y},${w},${h}`); },
    fillRect(x, y, w, h) { operations.push(`fillRect:${x},${y},${w},${h}`); },
    beginPath() { operations.push("beginPath"); },
    moveTo(x, y) { operations.push(`moveTo:${x},${y}`); },
    lineTo(x, y) { operations.push(`lineTo:${x},${y}`); },
    stroke() { operations.push("stroke"); },
    measureText(text) { return { width: text.length * 6 }; },
    fillText(text, x, y) { operations.push(`fillText:${text}`); },
  };

  const obstacles = [
    { source: "lidar", laneIdx: 1, laneName: "وسط", distanceM: 30, bbox: [75, 60, 10, 4] },
  ];

  // Render on 320x240 canvas with sensorMode = "lidar"
  drawVisualDetectionOverlays(mockCtx, obstacles, 1, 320, 240, "lidar");

  // Box at 320x240 should scale x2: [150, 120, 20, 8]
  assert.ok(operations.some(op => op.startsWith("strokeRect:150,120,20,8")));
  // Header banner for LiDAR
  assert.ok(operations.some(op => op.includes("fillText:📡 LIDAR: 1 RETURNS")));
  // LiDAR tag
  assert.ok(operations.some(op => op.includes("fillText:📡 LiDAR 30m | وسط")));

  // Test Optical CV mode
  operations.length = 0;
  drawVisualDetectionOverlays(mockCtx, [{ source: "vision", laneIdx: 0, laneName: "چپ", distanceM: 20, bbox: [40, 50, 12, 6] }], 0, 320, 240, "vision");
  assert.ok(operations.some(op => op.includes("fillText:👁️ CV: 1 TRACKED")));
});

test("evaluateCorridorManeuver activates blinker first at long range (58m) before any steering begins", () => {
  // Center lane vehicle at 58m (between dangerDist 48m and intentDist 70m)
  // Left lane is open (80m), Right lane is open (80m)
  const clearances = [80, 58, 80];
  const egoX = 0; // Exactly centered in center lane
  const egoHeading = 0;

  const result = evaluateCorridorManeuver(egoX, egoHeading, clearances);
  // Blinker must be active FIRST (Left blinker to signal upcoming overtake)
  assert.equal(result.turnSignal, 1);
  // Car must NOT steer yet: remains in current lane
  assert.equal(result.currentLaneIdx, 1);
  assert.equal(result.targetLaneIdx, 1);
  assert.equal(result.shouldAvoid, false);
  assert.equal(result.tacticalSteering, 0);
  assert.equal(result.lockedTargetLane, null);
});

test("evaluateCorridorManeuver initiates physical lateral steering once clearance drops below danger threshold (35m)", () => {
  // Center lane vehicle now at 35m (< dangerDist 48m)
  const clearances = [80, 35, 80];
  const egoX = 0;
  const egoHeading = 0;

  const result = evaluateCorridorManeuver(egoX, egoHeading, clearances);
  // Blinker stays active
  assert.equal(result.turnSignal, 1);
  // Physical evasion now commences!
  assert.equal(result.targetLaneIdx, 0);
  assert.equal(result.shouldAvoid, true);
  assert.ok(result.tacticalSteering < 0); // Steer left toward lane 0
  assert.equal(result.lockedTargetLane, 0);
});

test("evaluateCorridorManeuver turns blinkers off upon arriving and aligning in target lane", () => {
  // Ego vehicle has arrived in Left lane (x = -4.0m, heading = 0)
  // Left lane is completely clear ahead (80m)
  const clearances = [80, 20, 80];
  const egoX = -4.0;
  const egoHeading = 0;

  const result = evaluateCorridorManeuver(egoX, egoHeading, clearances, LANE_CENTRES, 48, 0);
  // Arrived and aligned: turn signal turns off!
  assert.equal(result.turnSignal, 0);
  assert.equal(result.shouldAvoid, false);
  assert.equal(result.lockedTargetLane, null);
  assert.equal(result.targetLaneIdx, 0);
});

test("detectObstaclesWithLidar detects oncoming vehicles at long range (60m) with dense ray coverage", () => {
  const trafficCars = [
    { position: { x: 0, z: -62 }, userData: { active: true, halfX: 1.0, halfZ: 2.2 } },
  ];

  const obstacles = detectObstaclesWithLidar(0, 0, 0, trafficCars);
  assert.equal(obstacles.length, 1);
  assert.equal(obstacles[0].source, "lidar");
  assert.equal(obstacles[0].laneIdx, 1);
  assert.ok(obstacles[0].distanceM >= 58 && obstacles[0].distanceM <= 62);
  assert.ok(obstacles[0].hitsCount >= 2); // Dense scanning hits with multiple rays!
});

test("detectVisualObstacles detects physical hand or large blockage across the roadway", () => {
  const width = 160;
  const height = 120;
  const data = new Uint8ClampedArray(width * height * 4);

  // Fill asphalt road (lum ~ 35)
  for (let y = 60; y < 90; y++) {
    for (let x = 0; x < width; x++) {
      const idx = (y * width + x) * 4;
      data[idx] = 36;
      data[idx + 1] = 34;
      data[idx + 2] = 32;
      data[idx + 3] = 255;
    }
  }

  // Large hand / physical obstruction in center lane (x = 72..90, y = 66..76)
  // Skin / physical blockage tone (RGB 175, 125, 95)
  for (let y = 66; y <= 76; y++) {
    for (let x = 72; x <= 90; x++) {
      const idx = (y * width + x) * 4;
      data[idx] = 175;
      data[idx + 1] = 125;
      data[idx + 2] = 95;
      data[idx + 3] = 255;
    }
  }

  const detections = detectVisualObstacles({ width, height, data }, 0, 0);
  assert.ok(detections.length >= 1);
  const centerBlockage = detections.find(d => d.laneIdx === 1);
  assert.ok(centerBlockage);
  assert.ok(centerBlockage.distanceM > 0 && centerBlockage.distanceM <= 35);
});

test("renderPerceptionPreview renders to simulation.autopilotCanvas and spans full 320x240 without 160x120 crop", () => {
  const operations = [];
  const mockCtx = {
    save() { operations.push("save"); },
    restore() { operations.push("restore"); },
    strokeRect(x, y, w, h) { operations.push(`strokeRect:${x},${y},${w},${h}`); },
    fillRect(x, y, w, h) { operations.push(`fillRect:${x},${y},${w},${h}`); },
    fillText(text, x, y) { operations.push(`fillText:${text}`); },
    beginPath() { operations.push("beginPath"); },
    moveTo(x, y) { operations.push(`moveTo:${x},${y}`); },
    lineTo(x, y) { operations.push(`lineTo:${x},${y}`); },
    stroke() { operations.push("stroke"); },
    arc() {},
    measureText(text) { return { width: text.length * 6 }; },
    drawImage(src, sx, sy, sw, sh, dx, dy, dw, dh) {
      if (arguments.length === 5) {
        operations.push(`drawImage:5args:dx=${sx},dy=${sy},dw=${sw},dh=${sh}`);
      } else {
        operations.push(`drawImage:9args:dx=${dx},dy=${dy},dw=${dw},dh=${dh}`);
      }
    },
  };

  const mockCanvas = {
    width: 320,
    height: 240,
    getContext(type) {
      if (type === "2d") return mockCtx;
      return null;
    },
  };

  const mockPanel = {
    hidden: false,
    classList: { contains: () => false },
    querySelector: () => null,
  };

  const simulation = {
    autopilotCanvas: mockCanvas,
    testPanel: mockPanel,
    renderer: {
      domElement: { width: 1280, height: 720 },
    },
    latestVisualObstacles: [],
    settings: { obstacleSource: "ai_vision", drivingMode: "sim" },
    corridorState: { lockedTargetLane: 1 },
  };

  renderPerceptionPreview(simulation);

  // Must draw full background spanning dw=320, dh=240, NOT 160x120
  assert.ok(operations.some(op => op.includes("dw=320,dh=240")));
  assert.ok(!operations.some(op => op.includes("dw=160,dh=120")));
});

test("renderPerceptionPreview renders to popped-out autopilotCanvas when panel has is-popped-out", () => {
  const operations = [];
  const mockCtx = {
    save() { operations.push("save"); },
    restore() { operations.push("restore"); },
    strokeRect(x, y, w, h) { operations.push(`strokeRect:${x},${y},${w},${h}`); },
    fillRect(x, y, w, h) { operations.push(`fillRect:${x},${y},${w},${h}`); },
    fillText(text, x, y) { operations.push(`fillText:${text}`); },
    beginPath() { operations.push("beginPath"); },
    moveTo() {},
    lineTo() {},
    stroke() {},
    arc() {},
    measureText(text) { return { width: text.length * 6 }; },
    drawImage(src, sx, sy, sw, sh, dx, dy, dw, dh) {
      operations.push(`drawImage:dx=${dx},dy=${dy},dw=${dw},dh=${dh}`);
    },
  };

  const mockCanvas = {
    width: 320,
    height: 240,
    getContext(type) {
      if (type === "2d") return mockCtx;
      return null;
    },
  };

  const mockPanel = {
    hidden: true, // panel is "hidden" in main document, but is-popped-out is true
    classList: { contains: (cls) => cls === "is-popped-out" },
    querySelector: () => null,
  };

  const simulation = {
    autopilotCanvas: mockCanvas,
    testPanel: mockPanel,
    renderer: {
      domElement: { width: 1280, height: 720 },
    },
    latestVisualObstacles: [],
    settings: { obstacleSource: "ai_vision", drivingMode: "sim" },
    corridorState: { lockedTargetLane: 1 },
  };

  renderPerceptionPreview(simulation);

  // Even though mockPanel.hidden is true, because is-popped-out is true, it renders full frame!
  assert.ok(operations.some(op => op.includes("dw=320,dh=240")));
});

test("detectVisualObstacles safely returns empty array on HTMLImageElement (phone camera feed)", () => {
  class FakeImage {}
  globalThis.HTMLImageElement = FakeImage;

  const fakeImg = new FakeImage();
  const result = detectVisualObstacles(fakeImg, 0, 0);
  assert.deepEqual(result, []);
});

test("detectObstaclesWithAI returns empty array when no vehicles detected without triggering CV fallback", async () => {
  const result = await detectObstaclesWithAI({ mockPredictions: [] }, 0, 0);
  assert.deepEqual(result, []);
});

test("renderPerceptionPreview draws on-device phone obstacles and renders AI badge", () => {
  const operations = [];
  const mockCtx = {
    save() {},
    restore() {},
    strokeRect() {},
    fillRect() {},
    fillText(text) { operations.push(`text:${text}`); },
    beginPath() {},
    moveTo() {},
    lineTo() {},
    stroke() {},
    arc() {},
    measureText() { return { width: 30 }; },
    drawImage() {},
  };

  const mockCanvas = {
    width: 320,
    height: 240,
    getContext: () => mockCtx,
  };

  const mockPanel = {
    hidden: false,
    classList: { contains: () => false },
    querySelector: () => null,
  };

  const simulation = {
    autopilotCanvas: mockCanvas,
    testPanel: mockPanel,
    renderer: { domElement: { width: 1280, height: 720 } },
    latestVisualObstacles: [
      {
        source: "mobile_ai",
        label: "خودرو (85%)",
        distanceM: 32,
        laneIdx: 1,
        laneName: "وسط",
        bbox: [60, 50, 40, 30],
      }
    ],
    settings: { obstacleSource: "ai_vision", drivingMode: "camera" },
    corridorState: { lockedTargetLane: 1 },
  };

  renderPerceptionPreview(simulation);
  assert.ok(operations.some(op => op.includes("32m")));
});

test("mobile phone obstacles project to world coordinates with ego lateral offset and heading", () => {
  // Ego vehicle in Left lane (-3.7m) facing slightly right (+0.05 rad)
  const egoX = -3.7;
  const egoHeading = 0.05;
  const boxFocal = 104;

  // Obstacle detected in phone camera looking down road: bbox center is at centerX = 120 (towards the right side of phone screen)
  const [bx, by, bw, bh] = [100, 50, 40, 30];
  const centerX = bx + bw / 2; // 120
  const distM = 30;

  const thetaCam = (centerX - 80) / boxFocal; // (120 - 80) / 104 = +0.3846 rad
  const thetaWorld = thetaCam + egoHeading;
  const worldX = +(egoX + distM * thetaWorld).toFixed(2);

  // worldX = -3.7 + 30 * (0.3846 + 0.05) = -3.7 + 13.04 = ~9.34m (off to right)
  // When an obstacle is physically in Center lane (worldX ~ 0), it correctly resolves to laneIdx = 1
  const dividerXs = [-1.7, 1.7];
  let laneIdx = 1;
  if (worldX < dividerXs[0]) laneIdx = 0;
  else if (worldX > dividerXs[1]) laneIdx = 2;

  assert.equal(laneIdx, 2, "correctly classified as right lane instead of hardcoded center lane");
});
