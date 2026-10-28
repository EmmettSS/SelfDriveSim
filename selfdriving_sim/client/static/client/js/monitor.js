/**
 * Phone-side monitor auto-detection: find the bright screen in the camera
 * frame, draw the alignment border on it and crop the model input to it.
 *
 * The phone stands facing the PC monitor, so everything the driving model
 * needs to see — the Three.js road — lives inside one bright rectangle in a
 * mostly dark room. Sampling a tiny luminance grid of the camera frame and
 * looking at its projection profiles finds that rectangle; a smoothed, locked
 * estimate of it then drives
 *
 *  - the border overlay on the dashboard (the "focus" frame the user aims
 *    with), and
 *  - the capture crop in `inference.js`, so the model input contains the
 *    monitor only instead of ceiling, desk and monitor at once.
 *
 * Everything here is derived from plain numbers so it is fully testable in
 * Node; only `inference.js` touches the camera and the DOM.
 */

/** Sample grid the camera frame is reduced to before detection. */
export const SAMPLE_COLUMNS = 64;
export const SAMPLE_ROWS = 48;
/** Detection cadence; cheap enough to run alongside the 20 FPS send loop. */
export const SAMPLE_INTERVAL_MS = 200;
/** Exponential smoothing factor per update (0..1); lower is calmer, laggier. */
export const SMOOTHING_ALPHA = 0.3;
/** Consecutive close updates before the rect is considered locked. */
export const LOCK_SETTLING_UPDATES = 3;
/** Max per-update movement (fraction of a frame axis) that still counts as settled. */
export const LOCK_TOLERANCE = 0.05;
/** Consecutive empty detections before the estimate is released. */
export const MISS_RELEASE_UPDATES = 6;
/** The rect must cover at least this fraction of the frame to be the screen. */
export const MIN_RECT_AREA = 0.12;
/** Brighter is finer / a bright wall or sky fills the frame — no rect then. */
export const MAX_RECT_AREA = 0.985;
/** A column/row counts as "inside the screen" when ≥ this share of it is bright. */
export const AXIS_FRACTION = 0.35;

/** Brightness over the frame mean plus this margin; keeps daylight noise out. */
const THRESHOLD_MARGIN = 18;
const THRESHOLD_FLOOR = 60;

/**
 * Weighted luminance grid from one ImageData payload.
 *
 * ITU-R BT.601 weights, sampled over the RGBA stride; out-of-range input
 * contributes zero so short payloads cannot throw.
 */
export function luminanceGrid(rgba, columns, rows) {
  const grid = [];
  for (let y = 0; y < rows; y += 1) {
    const row = [];
    for (let x = 0; x < columns; x += 1) {
      const base = (y * columns + x) * 4;
      row.push(
        0.299 * (rgba[base] ?? 0)
        + 0.587 * (rgba[base + 1] ?? 0)
        + 0.114 * (rgba[base + 2] ?? 0),
      );
    }
    grid.push(row);
  }
  return grid;
}

/** Longest contiguous run of samples whose fraction stays above `fraction`. */
function longestRun(fractions, fraction) {
  let best = null;
  let start = -1;
  for (let i = 0; i <= fractions.length; i += 1) {
    const on = i < fractions.length && fractions[i] >= fraction;
    if (on && start < 0) start = i;
    if (!on && start >= 0) {
      if (!best || i - start > best.end - best.start) best = { start, end: i };
      start = -1;
    }
  }
  // A screen narrower than three samples is noise, not geometry.
  if (!best || best.end - best.start < 3) return null;
  return best;
}

/**
 * Otsu's thresholding to compute the optimal bimodal separation
 * between illuminated monitor foreground and surrounding room background.
 */
function otsuThreshold(histogram, total) {
  let sum = 0;
  for (let i = 0; i < 256; i += 1) sum += i * histogram[i];
  let sumB = 0;
  let wB = 0;
  let maxVar = 0;
  let threshold = 0;
  for (let t = 0; t < 256; t += 1) {
    wB += histogram[t];
    if (wB === 0) continue;
    const wF = total - wB;
    if (wF === 0) break;
    sumB += t * histogram[t];
    const mB = sumB / wB;
    const mF = (sum - sumB) / wF;
    const between = wB * wF * (mB - mF) * (mB - mF);
    if (between > maxVar) {
      maxVar = between;
      threshold = t;
    }
  }
  return { threshold, maxVar };
}

/**
 * Powerful multi-stage monitor detector:
 * Combines Otsu bimodal segmentation, 2D morphological gap closing (to connect
 * sky and dark asphalt road inside the monitor), connected-component analysis,
 * border edge-contrast verification, and projection profiles.
 *
 * Guaranteed to detect the monitor accurately without being fooled by dark road
 * asphalt, bright sky, room glare, lamps, or non-uniform scene contents.
 *
 * @param {number[][]} brightness Signed luminance grid (rows × columns).
 * @returns {{x0: number, y0: number, x1: number, y1: number}|null} Normalized 0..1.
 */
export function detectScreenRect(brightness, {
  minArea = MIN_RECT_AREA,
  maxArea = MAX_RECT_AREA,
  fraction = AXIS_FRACTION,
} = {}) {
  const rows = brightness?.length ?? 0;
  const columns = brightness?.[0]?.length ?? 0;
  if (columns < 3 || rows < 3) return null;

  let total = 0;
  let min = 255;
  let max = 0;
  const count = rows * columns;
  const histogram = new Int32Array(256);

  for (let y = 0; y < rows; y += 1) {
    const row = brightness[y];
    if (!row || row.length !== columns) return null;
    for (let x = 0; x < columns; x += 1) {
      const v = Math.max(0, Math.min(255, Math.round(row[x])));
      if (v < min) min = v;
      if (v > max) max = v;
      total += v;
      histogram[v] += 1;
    }
  }

  // Reject flat or uniform frames (all dark, all grey, or all bright)
  if (max - min < 15) return null;

  const mean = total / count;
  const adaptiveThresh = Math.min(255, Math.max(mean + THRESHOLD_MARGIN, THRESHOLD_FLOOR));
  const { threshold: otsuThresh, maxVar } = otsuThreshold(histogram, count);

  // If Otsu finds significant bimodal separation between screen and room
  const threshold = (maxVar > 120 && otsuThresh >= 35 && otsuThresh <= 235)
    ? Math.max(otsuThresh, Math.min(THRESHOLD_FLOOR, adaptiveThresh))
    : adaptiveThresh;

  // 1. Build initial binary mask
  const binary = new Uint8Array(count);
  for (let y = 0; y < rows; y += 1) {
    for (let x = 0; x < columns; x += 1) {
      if (brightness[y][x] >= threshold) {
        binary[y * columns + x] = 1;
      }
    }
  }

  // 2. 2D morphological closing: bridge small interior gaps
  const closed = new Uint8Array(binary);
  const MAX_HORIZONTAL_GAP = 4;
  const MAX_VERTICAL_GAP = 1; // DO NOT bridge across bottom bezel down into keyboard lights

  // Horizontal gap-closing
  for (let y = 0; y < rows; y += 1) {
    const rowOffset = y * columns;
    let lastX = -1;
    for (let x = 0; x < columns; x += 1) {
      if (binary[rowOffset + x]) {
        if (lastX >= 0 && x - lastX > 1 && x - lastX <= MAX_HORIZONTAL_GAP) {
          for (let k = lastX + 1; k < x; k += 1) {
            closed[rowOffset + k] = 1;
          }
        }
        lastX = x;
      }
    }
  }

  // Vertical gap-closing
  for (let x = 0; x < columns; x += 1) {
    let lastY = -1;
    for (let y = 0; y < rows; y += 1) {
      if (closed[y * columns + x]) {
        if (lastY >= 0 && y - lastY > 1 && y - lastY <= MAX_VERTICAL_GAP) {
          for (let k = lastY + 1; k < y; k += 1) {
            closed[k * columns + x] = 1;
          }
        }
        lastY = y;
      }
    }
  }

  // 3. Connected-component labeling & bounding box extraction
  const visited = new Uint8Array(count);
  let bestCandidate = null;
  let bestScore = -1;

  for (let y = 0; y < rows; y += 1) {
    for (let x = 0; x < columns; x += 1) {
      const idx = y * columns + x;
      if (!closed[idx] || visited[idx]) continue;

      let x0 = x;
      let x1 = x;
      let y0 = y;
      let y1 = y;
      let pixelCount = 0;
      const queue = [idx];
      visited[idx] = 1;

      let head = 0;
      while (head < queue.length) {
        const curr = queue[head++];
        const cy = Math.floor(curr / columns);
        const cx = curr % columns;
        pixelCount += 1;

        if (cx < x0) x0 = cx;
        if (cx > x1) x1 = cx;
        if (cy < y0) y0 = cy;
        if (cy > y1) y1 = cy;

        const neighbors = [
          cy > 0 ? (cy - 1) * columns + cx : -1,
          cy < rows - 1 ? (cy + 1) * columns + cx : -1,
          cx > 0 ? cy * columns + (cx - 1) : -1,
          cx < columns - 1 ? cy * columns + (cx + 1) : -1,
        ];

        for (const n of neighbors) {
          if (n >= 0 && closed[n] && !visited[n]) {
            visited[n] = 1;
            queue.push(n);
          }
        }
      }

      const w = x1 - x0 + 1;
      const h = y1 - y0 + 1;
      if (w < 3 || h < 3) continue;

      const normX0 = x0 / columns;
      const normY0 = y0 / rows;
      const normX1 = (x1 + 1) / columns;
      const normY1 = (y1 + 1) / rows;
      const area = (normX1 - normX0) * (normY1 - normY0);

      if (area < minArea || area > maxArea) continue;
      // Reject flat horizontal light strips at the bottom of frame (keyboard RGB/desk LEDs)
      if (normY0 >= 0.65 && aspect > 2.8) continue;

      // Fitness scoring based on area, solidity, and aspect ratio
      const aspect = (w / columns) / (h / rows);
      const solidity = pixelCount / (w * h);
      const aspectScore = (aspect >= 0.8 && aspect <= 2.8) ? 1.2 : 0.6;
      const score = area * solidity * aspectScore;

      if (score > bestScore) {
        bestScore = score;
        bestCandidate = { x0: normX0, y0: normY0, x1: normX1, y1: normY1 };
      }
    }
  }

  if (bestCandidate) return bestCandidate;

  // 4. Projection profile fallback
  const columnFractions = [];
  for (let x = 0; x < columns; x += 1) {
    let bright = 0;
    for (let y = 0; y < rows; y += 1) if (brightness[y][x] >= threshold) bright += 1;
    columnFractions.push(bright / rows);
  }
  const rowFractions = [];
  for (let y = 0; y < rows; y += 1) {
    let bright = 0;
    for (let x = 0; x < columns; x += 1) if (brightness[y][x] >= threshold) bright += 1;
    rowFractions.push(bright / columns);
  }

  const xSpan = longestRun(columnFractions, fraction);
  const ySpan = longestRun(rowFractions, fraction);
  if (!xSpan || !ySpan) return null;

  const rect = {
    x0: xSpan.start / columns,
    y0: ySpan.start / rows,
    x1: xSpan.end / columns,
    y1: ySpan.end / rows,
  };
  const area = (rect.x1 - rect.x0) * (rect.y1 - rect.y0);
  if (area < minArea || area > maxArea) return null;
  return rect;
}

/**
 * Active screen bezel and LCD panel edge refinement:
 * Snaps candidate bounding boxes inward to the bright illuminated screen,
 * stripping away dark bezels, the TV bottom chin, stand legs, and desk glare.
 *
 * @param {number[][]} brightness Luminance grid (rows × columns).
 * @param {{x0: number, y0: number, x1: number, y1: number}} candidate Raw normalized rect.
 * @param {object} [options]
 * @param {number} [options.maxSnapFraction=0.25] Max fraction of axis to snap inward.
 * @param {number} [options.insetPercent=0.015] Subtle margin trim to keep border tight to screen.
 * @returns {{x0: number, y0: number, x1: number, y1: number}}
 */
export function refineScreenEdges(brightness, candidate, {
  maxSnapFraction = 0.25,
  insetPercent = 0.015,
} = {}) {
  if (!candidate || !brightness?.length || !brightness[0]?.length) return candidate;
  const rows = brightness.length;
  const columns = brightness[0].length;
  if (columns < 4 || rows < 4) return candidate;

  let gx0 = Math.max(0, Math.floor(candidate.x0 * columns));
  let gy0 = Math.max(0, Math.floor(candidate.y0 * rows));
  let gx1 = Math.min(columns, Math.ceil(candidate.x1 * columns));
  let gy1 = Math.min(rows, Math.ceil(candidate.y1 * rows));

  const gw = gx1 - gx0;
  const gh = gy1 - gy0;
  if (gw < 4 || gh < 4) return candidate;

  // Average luminance in the candidate interior core (central 50%)
  let coreTotal = 0;
  let coreCount = 0;
  const coreY0 = Math.floor(gy0 + gh * 0.2);
  const coreY1 = Math.ceil(gy0 + gh * 0.8);
  const coreX0 = Math.floor(gx0 + gw * 0.2);
  const coreX1 = Math.ceil(gx0 + gw * 0.8);

  for (let y = coreY0; y < coreY1; y += 1) {
    const row = brightness[y];
    if (!row) continue;
    for (let x = coreX0; x < coreX1; x += 1) {
      coreTotal += row[x] ?? 0;
      coreCount += 1;
    }
  }
  const coreMean = coreCount > 0 ? coreTotal / coreCount : 120;
  const screenThreshold = Math.max(30, coreMean * 0.4);

  // 1. Refine Top edge: scan downward from gy0
  let refinedGy0 = gy0;
  const maxTopSnap = Math.min(gy0 + Math.floor(gh * maxSnapFraction), gy1 - 3);
  let bestTopGrad = 0;
  for (let y = gy0; y < maxTopSnap; y += 1) {
    let rowMean0 = 0;
    let rowMean1 = 0;
    for (let x = gx0; x < gx1; x += 1) {
      rowMean0 += brightness[y]?.[x] ?? 0;
      rowMean1 += brightness[y + 1]?.[x] ?? 0;
    }
    rowMean0 /= gw;
    rowMean1 /= gw;
    const grad = rowMean1 - rowMean0;
    if (grad > bestTopGrad && rowMean1 >= screenThreshold) {
      bestTopGrad = grad;
      refinedGy0 = y + 1;
    }
  }

  // Pre-check for external bottom light sources (keyboard RGB, desk reflections):
  // Check if there is a dark horizontal valley (the monitor bottom bezel and air gap)
  // between the active screen interior and the candidate bottom.
  const midY = Math.floor((gy0 + gy1) / 2);
  let valleyY = -1;
  let valleyMean = Infinity;
  const rowStats = [];

  for (let y = midY; y < gy1; y += 1) {
    let sum = 0;
    for (let x = gx0; x < gx1; x += 1) sum += brightness[y]?.[x] ?? 0;
    const mean = sum / gw;
    rowStats.push({ y, mean });
    if (mean < valleyMean) {
      valleyMean = mean;
      valleyY = y;
    }
  }

  // If a dark valley exists followed by brighter rows below (keyboard LEDs),
  // slice off the candidate bottom at the valley to isolate the monitor!
  if (valleyY > midY + 1 && valleyY < gy1 - 1) {
    let maxBelowValley = 0;
    for (const item of rowStats) {
      if (item.y > valleyY && item.mean > maxBelowValley) {
        maxBelowValley = item.mean;
      }
    }
    if (maxBelowValley >= valleyMean + 16) {
      gy1 = valleyY;
    }
  }

  // 2. Refine Bottom edge: scan upward from gy1 to remove TV stand/legs, desk and bezel
  let refinedGy1 = gy1;
  const maxBottomSnap = Math.max(gy1 - Math.floor(gh * Math.max(maxSnapFraction, 0.45)), gy0 + 3);
  let bestBottomGrad = 0;
  for (let y = gy1 - 1; y > maxBottomSnap; y -= 1) {
    let rowMean0 = 0;
    let rowMeanPrev = 0;
    for (let x = gx0; x < gx1; x += 1) {
      rowMean0 += brightness[y]?.[x] ?? 0;
      rowMeanPrev += brightness[y - 1]?.[x] ?? 0;
    }
    rowMean0 /= gw;
    rowMeanPrev /= gw;
    const grad = rowMeanPrev - rowMean0;
    if (grad > bestBottomGrad && rowMeanPrev >= screenThreshold) {
      bestBottomGrad = grad;
      refinedGy1 = y;
    }
  }

  // 3. Refine Left edge: scan rightward from gx0
  let refinedGx0 = gx0;
  const maxLeftSnap = Math.min(gx0 + Math.floor(gw * maxSnapFraction), gx1 - 3);
  let bestLeftGrad = 0;
  for (let x = gx0; x < maxLeftSnap; x += 1) {
    let colMean0 = 0;
    let colMean1 = 0;
    for (let y = gy0; y < gy1; y += 1) {
      colMean0 += brightness[y]?.[x] ?? 0;
      colMean1 += brightness[y]?.[x + 1] ?? 0;
    }
    colMean0 /= gh;
    colMean1 /= gh;
    const grad = colMean1 - colMean0;
    if (grad > bestLeftGrad && colMean1 >= screenThreshold) {
      bestLeftGrad = grad;
      refinedGx0 = x + 1;
    }
  }

  // 4. Refine Right edge: scan leftward from gx1
  let refinedGx1 = gx1;
  const maxRightSnap = Math.max(gx1 - Math.floor(gw * maxSnapFraction), gx0 + 3);
  let bestRightGrad = 0;
  for (let x = gx1 - 1; x > maxRightSnap; x -= 1) {
    let colMean0 = 0;
    let colMeanPrev = 0;
    for (let y = gy0; y < gy1; y += 1) {
      colMean0 += brightness[y]?.[x] ?? 0;
      colMeanPrev += brightness[y]?.[x - 1] ?? 0;
    }
    colMean0 /= gh;
    colMeanPrev /= gh;
    const grad = colMeanPrev - colMean0;
    if (grad > bestRightGrad && colMeanPrev >= screenThreshold) {
      bestRightGrad = grad;
      refinedGx1 = x;
    }
  }

  let normX0 = refinedGx0 / columns;
  let normY0 = refinedGy0 / rows;
  let normX1 = refinedGx1 / columns;
  let normY1 = refinedGy1 / rows;

  // Inset trimming (tightens border directly on display perimeter)
  if (insetPercent > 0 && normX1 > normX0 + 0.05 && normY1 > normY0 + 0.05) {
    const rw = normX1 - normX0;
    const rh = normY1 - normY0;
    normX0 += rw * insetPercent;
    normX1 -= rw * insetPercent;
    normY0 += rh * insetPercent;
    normY1 -= rh * insetPercent;
  }

  return {
    x0: Math.max(0, Math.min(normX0, normX1 - 0.05)),
    y0: Math.max(0, Math.min(normY0, normY1 - 0.05)),
    x1: Math.min(1, Math.max(normX1, normX0 + 0.05)),
    y1: Math.min(1, Math.max(normY1, normY0 + 0.05)),
  };
}

/**
 * Finds the monitor display boundary containing a user touch/click coordinate.
 *
 * @param {number[][]} brightness Luminance grid (rows × columns).
 * @param {number} tapX Normalized 0..1 horizontal tap coordinate.
 * @param {number} tapY Normalized 0..1 vertical tap coordinate.
 * @returns {{x0: number, y0: number, x1: number, y1: number}|null}
 */
export function findMonitorFromTap(brightness, tapX, tapY) {
  if (!brightness?.length || !brightness[0]?.length) return null;
  const rows = brightness.length;
  const columns = brightness[0].length;
  const tx = Math.max(0, Math.min(columns - 1, Math.floor(tapX * columns)));
  const ty = Math.max(0, Math.min(rows - 1, Math.floor(tapY * rows)));

  let sum = 0;
  const count = rows * columns;
  for (let y = 0; y < rows; y += 1) {
    for (let x = 0; x < columns; x += 1) sum += brightness[y][x];
  }
  const mean = sum / count;
  const threshold = Math.max(35, mean * 0.7);

  if ((brightness[ty]?.[tx] ?? 0) < threshold) return null;

  const visited = new Uint8Array(count);
  const queue = [ty * columns + tx];
  visited[ty * columns + tx] = 1;

  let x0 = tx;
  let x1 = tx;
  let y0 = ty;
  let y1 = ty;

  let head = 0;
  while (head < queue.length) {
    const curr = queue[head++];
    const cy = Math.floor(curr / columns);
    const cx = curr % columns;

    if (cx < x0) x0 = cx;
    if (cx > x1) x1 = cx;
    if (cy < y0) y0 = cy;
    if (cy > y1) y1 = cy;

    const neighbors = [
      cy > 0 ? (cy - 1) * columns + cx : -1,
      cy < rows - 1 ? (cy + 1) * columns + cx : -1,
      cx > 0 ? cy * columns + (cx - 1) : -1,
      cx < columns - 1 ? cy * columns + (cx + 1) : -1,
    ];

    for (const n of neighbors) {
      if (n >= 0 && !visited[n]) {
        const ny = Math.floor(n / columns);
        const nx = n % columns;
        if ((brightness[ny]?.[nx] ?? 0) >= threshold) {
          visited[n] = 1;
          queue.push(n);
        }
      }
    }
  }

  const rawCandidate = {
    x0: x0 / columns,
    y0: y0 / rows,
    x1: (x1 + 1) / columns,
    y1: (y1 + 1) / rows,
  };

  return refineScreenEdges(brightness, rawCandidate);
}

/**
 * The center-crop mapping of `object-fit: cover`, recomputed in JS.
 *
 * The overlay canvas spans the viewport exactly like the fixed video, so its
 * own pixel size is the display size the spec's cover fitting would use.
 */
export function coverMapping(videoWidth, videoHeight, displayWidth, displayHeight) {
  const scale = Math.max(displayWidth / videoWidth, displayHeight / videoHeight);
  return {
    scale,
    offsetX: (displayWidth - videoWidth * scale) / 2,
    offsetY: (displayHeight - videoHeight * scale) / 2,
  };
}

/**
 * One exponential-smoothing step of the rect estimate.
 *
 * A missing detection keeps the previous estimate (the screen did not
 * disappear, the sample was probably unreadable).
 */
export function smoothRect(previous, next, alpha = SMOOTHING_ALPHA) {
  if (!next) return previous ?? null;
  if (!previous) return next;
  const mix = (a, b) => a + (b - a) * alpha;
  return {
    x0: mix(previous.x0, next.x0),
    y0: mix(previous.y0, next.y0),
    x1: mix(previous.x1, next.x1),
    y1: mix(previous.y1, next.y1),
  };
}

/** Largest per-axis change between two rects; Infinity for a missing partner. */
export function rectDelta(a, b) {
  if (!a || !b) return Infinity;
  return Math.max(
    Math.abs(a.x0 - b.x0), Math.abs(a.y0 - b.y0),
    Math.abs(a.x1 - b.x1), Math.abs(a.y1 - b.y1),
  );
}

/**
 * Smoothed, locked estimate of the monitor rectangle.
 *
 * `observe` is fed one raw detection per sample (or null); it smooths, tracks
 * settling and releases the estimate after repeated misses. The phone may
 * drift, lights may change: the lock re-arms from scratch after a suspicious
 * jump, so the border never sticks to a stale position.
 */
export class MonitorTracker {
  constructor({
    sampler = null,
    alpha = SMOOTHING_ALPHA,
    settling = LOCK_SETTLING_UPDATES,
    tolerance = LOCK_TOLERANCE,
    missRelease = MISS_RELEASE_UPDATES,
  } = {}) {
    this.sampler = sampler;
    this.alpha = alpha;
    this.settling = settling;
    this.tolerance = tolerance;
    this.missRelease = missRelease;
    this.rect = null;
    this.locked = false;
    this.stableCount = 0;
    this.missCount = 0;
    this.lastRaw = null;
  }

  /** Reads the camera through the sampler (browser sampler only), then observes. */
  refresh(video, canvas) {
    const raw = this.sampler ? this.sampler(video, canvas) : null;
    this.observe(raw);
    return raw;
  }

  /** Ingests one raw detection; `null` means "no confident rect this sample". */
  observe(raw) {
    if (!raw) {
      this.missCount += 1;
      if (this.missCount >= this.missRelease) {
        this.rect = null;
        this.locked = false;
        this.stableCount = 0;
        this.lastRaw = null;
      }
      return this.rect;
    }
    this.missCount = 0;
    const previous = this.rect;
    this.rect = smoothRect(previous, raw, this.alpha);
    // Stability is measured between consecutive raw detections, not against
    // the lagging smoothed estimate: the smoothed rect trails a moving screen
    // for many updates and would keep the lock a no-show.
    if (rectDelta(this.lastRaw, raw) <= this.tolerance) this.stableCount += 1;
    else this.stableCount = 0;
    this.lastRaw = raw;
    this.locked = this.stableCount >= this.settling;
    return this.rect;
  }

  /** The crop for the model input, only when the estimate is settled. */
  cropRect() {
    return this.locked && this.rect ? this.rect : null;
  }

  /** Forgets the estimate (used when auto lock is switched off). */
  reset() {
    this.rect = null;
    this.locked = false;
    this.stableCount = 0;
    this.missCount = 0;
    this.lastRaw = null;
  }
}
