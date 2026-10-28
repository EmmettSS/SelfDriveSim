import assert from "node:assert/strict";
import test from "node:test";

import {
  ADAS_LOOKAHEAD_HORIZON_M,
  ADAS_OBSTACLE_DANGER_M,
  LANE_CENTRES,
  evaluateCorridorManeuver,
  assessCorridorsFromVisualDetections,
  detectVisualObstacles,
  drawVisualDetectionOverlays,
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

  // Vehicle is at world X=0 (Center lane) at 25m.
  // When ego turns left (heading = -0.18, egoX = -0.8):
  // The vehicle shifts right in the camera image to x=89..92 at y=63..64
  for (let y = 63; y <= 64; y++) {
    for (let x = 89; x <= 92; x++) {
      const idx = (y * width + x) * 4;
      data[idx] = 220; data[idx + 1] = 40; data[idx + 2] = 40; data[idx + 3] = 255;
    }
  }

  // Without compensation, x=89..92 would falsely be marked as Right lane.
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
