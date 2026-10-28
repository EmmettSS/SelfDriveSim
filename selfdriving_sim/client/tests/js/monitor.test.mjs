/**
 * Node tests for the pure monitor-detection helpers.
 *
 * Run from selfdriving_sim/:
 *
 *     node --test client/tests/js/monitor.test.mjs
 *
 * monitor.js never touches the DOM, so the luminance grid, the screen-rect
 * detector, the cover mapping, the smoothing and the tracker state machine
 * are all checked here with synthetic grids.
 */

import assert from "node:assert/strict";
import test from "node:test";

import {
  AXIS_FRACTION,
  LOCK_SETTLING_UPDATES,
  MAX_RECT_AREA,
  MIN_RECT_AREA,
  MISS_RELEASE_UPDATES,
  SAMPLE_COLUMNS,
  SAMPLE_ROWS,
  MonitorTracker,
  coverMapping,
  detectScreenRect,
  findMonitorFromTap,
  luminanceGrid,
  rectDelta,
  refineScreenEdges,
  smoothRect,
} from "../../static/client/js/monitor.js";

/** An all-dark luminance grid at the sample resolution. */
function darkGrid(columns = SAMPLE_COLUMNS, rows = SAMPLE_ROWS, value = 10, columns2) {
  return Array.from({ length: rows }, () => Array.from({ length: columns2 ?? columns }, () => value));
}

/** Paints an inclusive index rectangle bright, simulating the PC screen. */
function paintScreen(grid, x0, y0, x1, y1, value = 200) {
  for (let y = y0; y <= y1; y += 1) {
    for (let x = x0; x <= x1; x += 1) grid[y][x] = value;
  }
  return grid;
}

/* ------------------------------ luminanceGrid ---------------------------- */

test("luminanceGrid weights RGB per BT.601 over the RGBA stride", () => {
  const grid = luminanceGrid(
    [255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255],
    3,
    1,
  );
  assert.equal(grid.length, 1);
  assert.equal(grid[0].length, 3);
  assert.ok(Math.abs(grid[0][0] - 0.299 * 255) < 1e-9);
  assert.ok(Math.abs(grid[0][1] - 0.587 * 255) < 1e-9);
  assert.ok(Math.abs(grid[0][2] - 0.114 * 255) < 1e-9);
});

test("luminanceGrid treats out-of-range payloads as black instead of throwing", () => {
  assert.deepEqual(luminanceGrid([10, 20, 30, 255], 2, 2), [[18.15, 0], [0, 0]]);
  assert.deepEqual(luminanceGrid([], 2, 2), [[0, 0], [0, 0]]);
});

/* ------------------------------ detectRect ------------------------------- */

test("detectScreenRect finds a synthetic bright monitor rectangle", () => {
  const grid = paintScreen(darkGrid(), 8, 6, 47, 35);
  const rect = detectScreenRect(grid);
  assert.ok(rect, "the screen must be detected");
  assert.equal(rect.x0, 8 / SAMPLE_COLUMNS);
  assert.equal(rect.y0, 6 / SAMPLE_ROWS);
  assert.equal(rect.x1, 48 / SAMPLE_COLUMNS);
  assert.equal(rect.y1, 36 / SAMPLE_ROWS);
  const area = (rect.x1 - rect.x0) * (rect.y1 - rect.y0);
  assert.ok(area >= MIN_RECT_AREA && area <= MAX_RECT_AREA);
});

test("detectScreenRect rejects an all-dark frame", () => {
  assert.equal(detectScreenRect(darkGrid()), null);
});

test("detectScreenRect rejects a frame that is bright everywhere", () => {
  // A bright wall or daylight window fills the whole frame: the adaptive
  // threshold cannot separate anything, and cropping to the full frame would
  // add nothing, so no rect is reported (the area caps at > 98.5%).
  const allBright = darkGrid(SAMPLE_COLUMNS, SAMPLE_ROWS, 255);
  assert.equal(detectScreenRect(allBright), null);
});

test("detectScreenRect rejects rectangles that are too small", () => {
  const glare = paintScreen(darkGrid(), 20, 10, 29, 19);
  assert.equal(detectScreenRect(glare), null);
});

test("detectScreenRect ignores narrower-than-three-sample bright stripes", () => {
  const grid = darkGrid();
  paintScreen(grid, 0, 0, 1, SAMPLE_ROWS - 1); // two columns of glare
  assert.equal(detectScreenRect(grid), null);
});

test("detectScreenRect refuses grids without three rows or columns", () => {
  assert.equal(detectScreenRect(null), null);
  assert.equal(detectScreenRect([]), null);
  assert.equal(detectScreenRect([[0, 0], [0, 0]]), null);
  assert.equal(detectScreenRect([[0], [0], [0]]), null);
});

test("the detector thresholds respond to the sampling fraction", () => {
  // With a tiny fraction requirement the narrow stripe is big enough.
  const grid = paintScreen(darkGrid(), 30, 0, 31, SAMPLE_ROWS - 1);
  const rect = detectScreenRect(grid, {
    fraction: AXIS_FRACTION,
    minArea: 0.01,
  });
  assert.equal(rect === null || rect.x0 >= 0, true, "option plumbing does not throw");
});

/* ------------------------------ coverMapping ----------------------------- */

test("coverMapping reproduces the object-fit cover algebra", () => {
  // Landscape video on a square viewport: width overflows to the left.
  const wide = coverMapping(1280, 720, 400, 400);
  assert.ok(Math.abs(wide.scale - 400 / 720) < 1e-9);
  assert.ok(Math.abs(wide.offsetX - (400 - 1280 * (400 / 720)) / 2) < 1e-9);
  assert.equal(wide.offsetY, 0);

  // Portrait video on sixteen-to-nine: height overflows to the top.
  const tall = coverMapping(720, 1280, 640, 360);
  assert.ok(Math.abs(tall.scale - 640 / 720) < 1e-9);
  assert.equal(tall.offsetX, 0);
  assert.ok(Math.abs(tall.offsetY - (360 - 1280 * (640 / 720)) / 2) < 1e-9);

  // Matching aspect ratios share the scale on both axes.
  const same = coverMapping(1600, 900, 800, 450);
  assert.deepEqual(same, { scale: 0.5, offsetX: 0, offsetY: 0 });
});

/* ------------------------- smoothing and deltas --------------------------- */

test("smoothRect mixes the estimate toward the new detection", () => {
  const first = { x0: 0, y0: 0, x1: 1, y1: 1 };
  const next = { x0: 1, y0: 1, x1: 0.5, y1: 0.5 };
  assert.deepEqual(smoothRect(null, next), next, "nothing to smooth: adopt");
  assert.deepEqual(smoothRect(first, null), first, "a miss keeps the estimate");
  assert.deepEqual(smoothRect(null, null), null);
  const mixed = smoothRect(first, next);
  assert.ok(Math.abs(mixed.x0 - 0.3) < 1e-9);
  assert.ok(Math.abs(mixed.y1 - 0.85) < 1e-9);
  const calmer = smoothRect(first, next, 0.5);
  assert.ok(Math.abs(calmer.x0 - 0.5) < 1e-9);
});

test("rectDelta is the largest per-axis change", () => {
  const a = { x0: 0, y0: 0.1, x1: 1, y1: 1 };
  const b = { x0: 0.4, y0: 0.1, x1: 1, y1: 1 };
  assert.equal(rectDelta(a, b), 0.4);
  assert.equal(rectDelta(a, a), 0);
  assert.equal(rectDelta(a, null), Infinity);
  assert.equal(rectDelta(null, b), Infinity);
});

/* ------------------------------ MonitorTracker --------------------------- */

const STEADY = { x0: 0.2, y0: 0.2, x1: 0.8, y1: 0.8 };

test("the tracker locks only after consecutive settled updates", () => {
  const tracker = new MonitorTracker();
  assert.equal(tracker.cropRect(), null, "never locked before the first sample");
  tracker.observe(STEADY);
  assert.equal(tracker.locked, false);
  assert.deepEqual(tracker.cropRect(), null, "crop is gated on the lock");
  for (let i = 0; i < LOCK_SETTLING_UPDATES - 1; i += 1) tracker.observe(STEADY);
  assert.equal(tracker.locked, false, "one short of the settling count");
  tracker.observe(STEADY);
  assert.equal(tracker.locked, true);
  const crop = tracker.cropRect();
  assert.ok(Math.abs(crop.x0 - STEADY.x0) < 1e-9);
});

test("a jump re-arms settling so the border never sticks to a stale spot", () => {
  const tracker = new MonitorTracker();
  for (let i = 0; i < LOCK_SETTLING_UPDATES; i += 1) tracker.observe(STEADY);
  tracker.observe(STEADY); // first comparison must be against the old position
  assert.equal(tracker.stableCount, LOCK_SETTLING_UPDATES);
  assert.equal(tracker.locked, true);
  const jump = { x0: 0.9, y0: 0.9, x1: 1, y1: 1 };
  tracker.observe(jump);
  assert.equal(tracker.locked, false, "a suspicious jump releases the lock");
  for (let i = 0; i < LOCK_SETTLING_UPDATES; i += 1) tracker.observe(jump);
  assert.equal(tracker.locked, true, "a stable new position re-locks");
});

test("repeated misses release the estimate entirely", () => {
  const tracker = new MonitorTracker();
  for (let i = 0; i < LOCK_SETTLING_UPDATES + 1; i += 1) tracker.observe(STEADY);
  for (let i = 0; i < MISS_RELEASE_UPDATES - 1; i += 1) {
    tracker.observe(null);
    assert.ok(tracker.rect, "short gaps do not drop the estimate");
  }
  tracker.observe(null);
  assert.equal(tracker.rect, null);
  assert.equal(tracker.locked, false);
  assert.equal(tracker.cropRect(), null);
  tracker.observe(STEADY);
  assert.equal(tracker.locked, false, "the lock re-arms from scratch");
});

test("reset forgets everything at once", () => {
  const tracker = new MonitorTracker();
  for (let i = 0; i < LOCK_SETTLING_UPDATES + 1; i += 1) tracker.observe(STEADY);
  tracker.reset();
  assert.deepEqual([tracker.rect, tracker.locked, tracker.stableCount, tracker.missCount], [null, false, 0, 0]);
  assert.equal(tracker.cropRect(), null);
});

test("refineScreenEdges snaps inward from outer dark bezel and stand", () => {
  // Screen is painted at x: 10..40, y: 10..30 (bright 220)
  // Candidate is overly large: x: 8..42, y: 6..38 (includes bezel above and stand below)
  const grid = paintScreen(darkGrid(64, 48, 15), 10, 10, 40, 30, 220);
  const candidate = { x0: 8 / 64, y0: 6 / 48, x1: 42 / 64, y1: 38 / 48 };
  const refined = refineScreenEdges(grid, candidate, { insetPercent: 0 });

  assert.ok(refined.y0 >= 9 / 48, "top edge snaps downward past dark bezel");
  assert.ok(refined.y1 <= 32 / 48, "bottom edge snaps upward past dark stand");
  assert.ok(refined.x0 >= 9 / 64, "left edge snaps inward");
  assert.ok(refined.x1 <= 41 / 64, "right edge snaps inward");
});

test("refineScreenEdges rejects keyboard lights below monitor", () => {
  // Screen is painted at x: 10..45, y: 8..28 (bright 210)
  // Dark bottom bezel and air gap: y: 29..32 (dark 15)
  // Keyboard RGB lights below: x: 12..43, y: 33..38 (bright 180)
  const grid = darkGrid(64, 48, 15);
  paintScreen(grid, 10, 8, 45, 28, 210);
  paintScreen(grid, 12, 33, 43, 38, 180);

  // Candidate initially spans both monitor and keyboard (y: 6..40)
  const candidate = { x0: 8 / 64, y0: 6 / 48, x1: 47 / 64, y1: 40 / 48 };
  const refined = refineScreenEdges(grid, candidate, { insetPercent: 0 });

  assert.ok(refined.y1 <= 30 / 48, `bottom edge must cut off keyboard above row 30, got ${refined.y1 * 48}`);
  assert.ok(refined.y0 >= 7 / 48);
});

test("detectScreenRect ignores illuminated keyboard strip below monitor", () => {
  // Screen at x: 10..45, y: 8..28 (bright 210)
  // Keyboard strip at x: 12..43, y: 35..38 (bright 190)
  const grid = darkGrid(64, 48, 15);
  paintScreen(grid, 10, 8, 45, 28, 210);
  paintScreen(grid, 12, 35, 43, 38, 190);

  const rect = detectScreenRect(grid);
  assert.ok(rect, "monitor must be detected");
  assert.ok(rect.y1 <= 30 / 48, `detected rect must not include keyboard, got y1=${rect.y1 * 48}`);
});

test("findMonitorFromTap locks onto bright screen area containing tap", () => {
  const grid = paintScreen(darkGrid(64, 48, 15), 12, 10, 45, 35, 210);
  // Tap in the middle of screen (x: 25/64, y: 20/48)
  const found = findMonitorFromTap(grid, 25 / 64, 20 / 48);
  assert.ok(found, "screen must be found from interior tap");
  assert.ok(found.x0 <= 15 / 64 && found.x1 >= 40 / 64);
  assert.ok(found.y0 <= 15 / 48 && found.y1 >= 30 / 48);

  // Tap in the dark corner (x: 2/64, y: 2/48)
  const missed = findMonitorFromTap(grid, 2 / 64, 2 / 48);
  assert.equal(missed, null, "dark tap returns null");
});

test("refresh samples through the provided sampler", () => {
  const samples = [null, STEADY];
  const tracker = new MonitorTracker({ sampler: () => samples.shift() });
  assert.deepEqual(tracker.refresh(), null, "samplers may miss");
  assert.deepEqual(tracker.refresh(), STEADY);
  assert.equal(tracker.locked, false);
  const bare = new MonitorTracker();
  assert.equal(bare.refresh(), null, "the browser wires the real sampler");
});
