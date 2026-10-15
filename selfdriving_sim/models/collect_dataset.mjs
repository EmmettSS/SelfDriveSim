#!/usr/bin/env node

import { createServer } from "node:http";
import { createReadStream, existsSync } from "node:fs";
import { appendFile, mkdir, writeFile } from "node:fs/promises";
import { dirname, extname, relative, resolve, sep } from "node:path";
import { tmpdir } from "node:os";
import { fileURLToPath } from "node:url";

import { chromium } from "@playwright/test";
import chromiumBundle from "@sparticuz/chromium";

const PROJECT_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const DATA_ROOT = resolve(PROJECT_ROOT, "models/data");
const DEFAULT_OUTPUT = resolve(DATA_ROOT, "sim-dataset-pilot");
const DEFAULT_SEED = 20261014;
const MIME_TYPES = new Map([
  [".html", "text/html; charset=utf-8"],
  [".js", "text/javascript; charset=utf-8"],
  [".mjs", "text/javascript; charset=utf-8"],
]);

function parseArguments(argv) {
  const options = {
    frames: 256,
    episodeFrames: 500,
    seed: DEFAULT_SEED,
    output: DEFAULT_OUTPUT,
    width: 640,
    height: 360,
  };
  const names = new Map([
    ["--frames", "frames"],
    ["--episode-frames", "episodeFrames"],
    ["--seed", "seed"],
    ["--output", "output"],
    ["--width", "width"],
    ["--height", "height"],
  ]);

  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index];
    if (argument === "--help" || argument === "-h") {
      options.help = true;
      continue;
    }
    const name = names.get(argument);
    if (!name) throw new Error(`Unknown argument: ${argument}`);
    const value = argv[index + 1];
    if (value === undefined || value.startsWith("--")) throw new Error(`Missing value for ${argument}`);
    options[name] = name === "output" ? resolve(process.cwd(), value) : Number(value);
    index += 1;
  }

  if (options.help) return options;
  for (const key of ["frames", "episodeFrames", "width", "height"]) {
    if (!Number.isInteger(options[key]) || options[key] < 1) {
      throw new Error(`--${key.replace(/[A-Z]/g, (letter) => `-${letter.toLowerCase()}`)} must be a positive integer.`);
    }
  }
  if (options.frames > 100000) throw new Error("At most 100,000 frames can be collected per run.");
  if (options.episodeFrames > 1000) throw new Error("--episode-frames must not exceed 1,000.");
  if (!Number.isFinite(options.seed)) throw new Error("--seed must be a finite number.");
  return options;
}

function printHelp() {
  console.log(`Collect labeled frames from the local Three.js scene.

Usage: node models/collect_dataset.mjs [options]

  --frames N             Total frames to record (default: 256)
  --episode-frames N     Frames per randomized episode, 1..1000 (default: 500)
  --seed N               Reproducible base seed (default: ${DEFAULT_SEED})
  --output PATH          New output directory (default: models/data/sim-dataset-pilot)
  --width N              Browser render width (default: 640)
  --height N             Browser render height (default: 360)
`);
}

function isWithin(path, root) {
  const fromRoot = relative(root, path);
  return fromRoot === "" || (fromRoot !== ".." && !fromRoot.startsWith(`..${sep}`));
}

function isWithinProject(path) {
  return isWithin(path, PROJECT_ROOT);
}

function isWithinData(path) {
  return isWithin(path, DATA_ROOT);
}

async function prepareChromium() {
  const configuredPath = process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH
    ?? process.env.CHROMIUM_PATH
    ?? process.env.CHROME_PATH;
  if (configuredPath) {
    return { executablePath: configuredPath, args: ["--no-sandbox", "--enable-webgl"] };
  }

  const libraryDirectory = resolve(tmpdir(), "al2023", "lib");
  // Ask the bundle to extract its Amazon Linux libraries and SwiftShader on local Node runs too.
  const lambdaRuntime = process.env.AWS_LAMBDA_JS_RUNTIME;
  process.env.AWS_LAMBDA_JS_RUNTIME = "nodejs20.x";
  chromiumBundle.setGraphicsMode = true;
  let executablePath;
  try {
    executablePath = await chromiumBundle.executablePath();
  } finally {
    if (lambdaRuntime === undefined) delete process.env.AWS_LAMBDA_JS_RUNTIME;
    else process.env.AWS_LAMBDA_JS_RUNTIME = lambdaRuntime;
  }
  process.env.FONTCONFIG_PATH ??= resolve(tmpdir(), "fonts");
  process.env.LD_LIBRARY_PATH = [tmpdir(), libraryDirectory, process.env.LD_LIBRARY_PATH]
    .filter(Boolean)
    .join(":");
  return {
    executablePath,
    args: [
      ...chromiumBundle.args,
      "--no-sandbox",
      "--enable-webgl",
      "--ignore-gpu-blocklist",
      "--use-gl=angle",
      "--use-angle=swiftshader",
      "--enable-unsafe-swiftshader",
    ],
  };
}

async function launchBrowserWithRetry(options) {
  let lastError;
  for (let attempt = 1; attempt <= 3; attempt += 1) {
    try {
      return await chromium.launch({ headless: true, ...options });
    } catch (error) {
      lastError = error;
      if (attempt < 3) await new Promise((resolvePromise) => setTimeout(resolvePromise, attempt * 300));
    }
  }
  throw lastError;
}

function startStaticServer() {
  const server = createServer((request, response) => {
    let pathname;
    try {
      pathname = decodeURIComponent(new URL(request.url, "http://127.0.0.1").pathname);
    } catch {
      response.writeHead(400).end("Bad request");
      return;
    }
    if (pathname === "/") pathname = "/models/dataset-collector.html";
    const filePath = resolve(PROJECT_ROOT, `.${pathname}`);
    if (!isWithinProject(filePath)) {
      response.writeHead(403).end("Forbidden");
      return;
    }
    response.setHeader("Content-Type", MIME_TYPES.get(extname(filePath)) ?? "application/octet-stream");
    createReadStream(filePath)
      .on("error", () => {
        if (!response.headersSent) response.writeHead(404);
        response.end("Not found");
      })
      .pipe(response);
  });
  return new Promise((resolvePromise, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => {
      server.off("error", reject);
      resolvePromise({
        server,
        baseUrl: `http://127.0.0.1:${server.address().port}`,
      });
    });
  });
}

function updateStatistics(statistics, command, valid) {
  statistics.samples += 1;
  statistics.valid += Number(valid);
  statistics.invalid += Number(!valid);
  statistics.brakePositive += Number(command.brake > 0.1);
  statistics.signals[command.turn_signal] += 1;
  statistics.steeringMin = Math.min(statistics.steeringMin, command.steering);
  statistics.steeringMax = Math.max(statistics.steeringMax, command.steering);
  statistics.throttleMin = Math.min(statistics.throttleMin, command.throttle);
  statistics.throttleMax = Math.max(statistics.throttleMax, command.throttle);
}

function emptyStatistics() {
  return {
    samples: 0,
    valid: 0,
    invalid: 0,
    brakePositive: 0,
    signals: [0, 0, 0],
    steeringMin: Infinity,
    steeringMax: -Infinity,
    throttleMin: Infinity,
    throttleMax: -Infinity,
  };
}

function datasetMetadata(options, episodes, statistics, capturedAt, complete) {
  return {
    schema: "selfdrivesim/frame-labels/v1",
    source: "Three.js scene rendered with headless WebGL",
    captured_at: capturedAt,
    status: complete ? "complete" : "in_progress",
    requested_frames: options.frames,
    frame_rate_hz: 20,
    raw_image_size: [640, 360],
    image_format: "JPEG",
    labels: {
      steering: "float in [-1, 1], positive turns right",
      throttle: "float in [0, 1]",
      brake: "float in [0, 1]",
      turn_signal: "integer: 0 off, 1 left, 2 right",
    },
    scenario: "scripted lane following and oncoming-traffic avoidance",
    capture_alignment: "Each camera frame is captured before its labeled expert command is applied; both use the same simulation state.",
    invalid_frames_are_excluded_from_training: true,
    seed: options.seed,
    episodes,
    statistics: {
      ...statistics,
      steeringMin: Number.isFinite(statistics.steeringMin) ? statistics.steeringMin : null,
      steeringMax: Number.isFinite(statistics.steeringMax) ? statistics.steeringMax : null,
      throttleMin: Number.isFinite(statistics.throttleMin) ? statistics.throttleMin : null,
      throttleMax: Number.isFinite(statistics.throttleMax) ? statistics.throttleMax : null,
      signals: { off: statistics.signals[0], left: statistics.signals[1], right: statistics.signals[2] },
    },
  };
}

async function collect(options) {
  const output = options.output;
  if (!isWithinProject(output) || !isWithinData(output) || output === DATA_ROOT) {
    throw new Error(`Output must be a new directory inside ${DATA_ROOT}.`);
  }
  if (existsSync(output)) throw new Error(`${output} already exists; choose a new --output directory.`);
  await mkdir(dirname(output), { recursive: true });
  await mkdir(output, { recursive: false });
  const framesDirectory = resolve(output, "frames");
  await mkdir(framesDirectory);

  const manifestPath = resolve(output, "manifest.jsonl");
  const metadataPath = resolve(output, "metadata.json");
  const capturedAt = new Date().toISOString();
  const { server, baseUrl } = await startStaticServer();
  const episodes = [];
  const statistics = emptyStatistics();
  await writeFile(manifestPath, "");
  await writeFile(metadataPath, `${JSON.stringify(datasetMetadata(options, episodes, statistics, capturedAt, false), null, 2)}\n`);
  let browser;
  let context;

  try {
    const launchOptions = await prepareChromium();

    let remaining = options.frames;
    let episodeIndex = 0;
    while (remaining > 0) {
      const frameCount = Math.min(remaining, options.episodeFrames);
      const episodeSeed = (options.seed + episodeIndex) >>> 0;
      browser = await launchBrowserWithRetry(launchOptions);
      context = await browser.newContext({
        viewport: { width: options.width, height: options.height },
        deviceScaleFactor: 1,
      });
      const page = await context.newPage();
      page.setDefaultTimeout(120000);
      const errors = [];
      page.on("pageerror", (error) => errors.push(error.message));
      await page.goto(`${baseUrl}/models/dataset-collector.html?seed=${episodeSeed}`, { waitUntil: "load" });
      await page.waitForFunction(() => window.datasetCollectorReady === true);
      const result = await page.evaluate((count) => window.collectSimulationFrames({ count }), frameCount);
      if (errors.length > 0) throw new Error(`Episode ${episodeIndex} failed in the page: ${errors.join("; ")}`);
      if (result.frames.length !== frameCount) {
        throw new Error(`Episode ${episodeIndex} returned ${result.frames.length} frames, expected ${frameCount}.`);
      }

      const episodeName = `episode-${String(episodeIndex).padStart(4, "0")}`;
      const episodeDirectory = resolve(framesDirectory, episodeName);
      const episodeManifest = [];
      await mkdir(episodeDirectory);
      for (const frame of result.frames) {
        const filename = `frame-${String(frame.index).padStart(5, "0")}.jpg`;
        const framePath = resolve(episodeDirectory, filename);
        const image = Buffer.from(frame.image, "base64");
        await writeFile(framePath, image);
        const record = {
          image: relative(output, framePath).split(sep).join("/"),
          episode: episodeName,
          frame_index: frame.index,
          seed: episodeSeed,
          valid: frame.valid,
          steering: frame.command.steering,
          throttle: frame.command.throttle,
          brake: frame.command.brake,
          turn_signal: frame.command.turn_signal,
          state: frame.state,
        };
        episodeManifest.push(JSON.stringify(record));
        updateStatistics(statistics, frame.command, frame.valid);
      }
      episodes.push({
        name: episodeName,
        seed: result.seed,
        browser_version: browser.version(),
        frames: result.frames.length,
        raw_size: result.rawSize,
        domain: result.domain,
        lane_offset: result.laneOffset,
      });
      await appendFile(manifestPath, `${episodeManifest.join("\n")}\n`);
      await writeFile(metadataPath, `${JSON.stringify(
        datasetMetadata(options, episodes, statistics, capturedAt, remaining === frameCount),
        null,
        2,
      )}\n`);
      console.log(`${episodeName}: ${result.frames.length} frames, seed ${episodeSeed}`);
      await page.evaluate(() => window.disposeDatasetCollector());
      await context.close();
      context = null;
      await browser.close();
      browser = null;
      remaining -= frameCount;
      episodeIndex += 1;
    }

    console.log(`Saved ${statistics.samples} frames to ${relative(PROJECT_ROOT, output)}.`);
    console.log(`Valid ${statistics.valid}; ignored after collisions or wall contacts ${statistics.invalid}.`);
    console.log(`Brake-positive frames ${statistics.brakePositive}; turn signals off/left/right ${statistics.signals.join("/")}.`);
  } finally {
    await context?.close();
    await browser?.close();
    await new Promise((resolvePromise) => server.close(resolvePromise));
  }
}

async function main() {
  const options = parseArguments(process.argv.slice(2));
  if (options.help) {
    printHelp();
    return;
  }
  await collect(options);
}

main().catch((error) => {
  console.error(`Dataset collection failed: ${error.message}`);
  process.exitCode = 1;
});
