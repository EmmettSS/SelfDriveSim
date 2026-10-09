import { expect, test } from "@playwright/test";

async function ready(page) {
  await page.goto("/dashboard/");
  await expect(page.locator("#ws-status")).toHaveClass(/green/);
  await expect(page.locator("#btn-start")).toBeEnabled();
}

test("camera, local font/styles and socket initialise without sending early", async ({ page }) => {
  const failures = [];
  const pageErrors = [];
  const requested = [];
  page.on("response", (response) => { if (response.status() >= 400) failures.push(response.url()); });
  page.on("pageerror", (error) => pageErrors.push(error.message));
  page.on("request", (request) => requested.push(request.url()));
  await ready(page);
  await expect(page.locator("#camera-prompt")).toBeHidden();
  await expect(page.locator("#btn-stop")).toBeDisabled();
  await expect(page.locator("#packet-count")).toHaveText("0");
  await expect(page.locator("#speed-value")).toHaveText("—");
  expect(await page.locator("#camera-feed").evaluate((video) => ({
    live: video.srcObject.getVideoTracks()[0].readyState,
    ready: video.readyState >= 2,
    width: video.videoWidth,
    fit: getComputedStyle(video).objectFit,
  }))).toMatchObject({ live: "live", ready: true, width: 1280, fit: "cover" });
  await page.evaluate(() => document.fonts.ready);
  expect(await page.evaluate(() => document.fonts.check('12px "Vazirmatn"'))).toBe(true);
  const origin = new URL(page.url()).origin;
  expect(requested.every((url) => new URL(url).origin === origin)).toBe(true);
  expect(failures).toEqual([]);
  expect(pageErrors).toEqual([]);
});

test("20 Hz Mock controls traverse Django and Redis to the render socket, then stop", async ({ page }) => {
  await ready(page);
  await page.evaluate(() => new Promise((resolve, reject) => {
    const scheme = location.protocol === "https:" ? "wss:" : "ws:";
    window.renderFrames = [];
    window.renderSocket = new WebSocket(`${scheme}//${location.host}/ws/render/`);
    window.renderSocket.onopen = resolve;
    window.renderSocket.onerror = reject;
    window.renderSocket.onmessage = (event) => window.renderFrames.push(JSON.parse(event.data));
  }));
  await page.locator("#btn-start").click();
  await expect.poll(async () => Number(await page.locator("#fps-value").textContent())).toBeGreaterThanOrEqual(17);
  await expect(page.locator("#indicator-throttle")).toHaveClass(/active/);
  await expect(page.locator("#indicator-brake")).not.toHaveClass(/active/);
  await expect(page.locator("#indicator-left")).toHaveClass(/active/);
  await expect(page.locator("#indicator-left")).toHaveCSS("color", "rgb(245, 215, 100)");
  await expect(page.locator("#indicator-left")).toHaveCSS("opacity", "1");
  await expect(page.locator("#indicator-right")).not.toHaveClass(/active/);
  const frames = await page.evaluate(() => window.renderFrames);
  expect(frames.length).toBeGreaterThan(45);
  for (const frame of frames) {
    expect(Object.keys(frame).sort()).toEqual(["brake", "steering", "throttle", "turn_signal", "type"]);
    expect(frame.type).toBe("control_update");
    expect(frame.steering).toBeGreaterThanOrEqual(-1);
    expect(frame.steering).toBeLessThanOrEqual(1);
    expect(frame.throttle).toBeGreaterThanOrEqual(0);
    expect(frame.throttle).toBeLessThanOrEqual(1);
    expect(frame.brake).toBe(0);
    expect([0, 1, 2]).toContain(frame.turn_signal);
  }
  const capture = await page.locator("#process-canvas").evaluate((canvas) => ({
    width: canvas.width,
    height: canvas.height,
    hidden: canvas.hidden,
    pixels: canvas.getContext("2d").getImageData(0, 0, 200, 66).data.some((n, i) => i % 4 !== 3 && n > 0),
  }));
  expect(capture).toEqual({ width: 200, height: 66, hidden: true, pixels: true });
  await page.locator("#btn-stop").click();
  await expect(page.locator("#fps-value")).toHaveText("0");
  await expect(page.locator("#brake-value")).toHaveText("100%");
  await expect(page.locator("#btn-stop")).toBeDisabled();
  await expect.poll(() => page.evaluate(() => window.renderFrames.at(-1))).toEqual({
    type: "control_update", steering: 0, throttle: 0, brake: 1, turn_signal: 0,
  });
  const finalCount = await page.evaluate(() => window.renderFrames.length);
  await page.waitForTimeout(300);
  expect(await page.evaluate(() => window.renderFrames.length)).toBe(finalCount);
  await page.evaluate(() => window.renderSocket.close());
});

test("permission denial has recovery guidance and no control connection", async ({ page }) => {
  const sockets = [];
  page.on("websocket", (socket) => sockets.push(socket.url()));
  await page.addInitScript(() => {
    const original = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
    window.denyCamera = true;
    navigator.mediaDevices.getUserMedia = (constraints) => window.denyCamera
      ? Promise.reject(new DOMException("Denied", "NotAllowedError")) : original(constraints);
  });
  await page.goto("/dashboard/");
  await expect(page.locator("#camera-message")).toContainText("اجازهٔ دوربین");
  await expect(page.locator("#btn-camera")).toBeEnabled();
  await expect(page.locator("#btn-start")).toBeDisabled();
  expect(sockets).toEqual([]);
  await page.evaluate(() => { window.denyCamera = false; });
  await page.locator("#btn-camera").click();
  await expect(page.locator("#ws-status")).toHaveClass(/green/);
  await expect(page.locator("#camera-prompt")).toBeHidden();
  expect(sockets.length).toBe(1);
});

for (const [name, message] of [["NotFoundError", "دوربینی پیدا نشد"], ["NotReadableError", "دوربین در دسترس نیست"]]) {
  test(`camera ${name} keeps Start disabled and offers retry`, async ({ page }) => {
    await page.addInitScript((errorName) => {
      navigator.mediaDevices.getUserMedia = () => Promise.reject(new DOMException("Camera unavailable", errorName));
    }, name);
    await page.goto("/dashboard/");
    await expect(page.locator("#camera-message")).toContainText(message);
    await expect(page.locator("#btn-camera")).toBeEnabled();
    await expect(page.locator("#btn-start")).toBeDisabled();
    await expect(page.locator("#packet-count")).toHaveText("0");
  });
}

test("insecure origin explains HTTPS/ngrok before attempting media access", async ({ page }) => {
  await page.addInitScript(() => {
    Object.defineProperty(window, "isSecureContext", { value: false });
    window.mediaRequests = 0;
    navigator.mediaDevices.getUserMedia = () => { window.mediaRequests += 1; throw new Error("should not be called"); };
  });
  await page.goto("/dashboard/");
  await expect(page.locator("#camera-title")).toContainText("اتصال امن");
  await expect(page.locator("#camera-message")).toContainText("ngrok");
  await expect(page.locator("#btn-start")).toBeDisabled();
  expect(await page.evaluate(() => window.mediaRequests)).toBe(0);
});

test("a dropped control socket reconnects without restarting inference", async ({ page }) => {
  const connections = [];
  await page.routeWebSocket("**/ws/control/", (socket) => {
    connections.push(socket);
    socket.connectToServer();
  });
  await ready(page);
  await page.locator("#btn-start").click();
  await expect.poll(async () => Number(await page.locator("#packet-count").textContent())).toBeGreaterThan(2);
  await connections[0].close({ code: 1011, reason: "Test interruption" });
  await expect.poll(() => connections.length).toBe(2);
  await expect(page.locator("#ws-status")).toHaveClass(/green/);
  await expect(page.locator("#btn-start")).toBeEnabled();
  await expect(page.locator("#btn-stop")).toBeDisabled();
  await expect(page.locator("#fps-value")).toHaveText("0");
  const count = await page.locator("#packet-count").textContent();
  await page.waitForTimeout(200);
  await expect(page.locator("#packet-count")).toHaveText(count);
  await page.locator("#btn-start").click();
  await expect.poll(async () => Number(await page.locator("#packet-count").textContent())).toBeGreaterThan(Number(count));
});

test("server error replies stop further inference sends", async ({ page }) => {
  await page.routeWebSocket("**/ws/control/", (socket) => {
    socket.onMessage(() => socket.send(JSON.stringify({ type: "error", message: "Rejected in test" })));
  });
  await ready(page);
  await page.locator("#btn-start").click();
  await expect(page.locator("#status-message")).toContainText("سرور فرمان را نپذیرفت");
  await expect(page.locator("#btn-stop")).toBeDisabled();
  await expect(page.locator("#fps-value")).toHaveText("0");
});

test("background lifecycle releases the camera and pauses until a manual restart", async ({ page }) => {
  await ready(page);
  await page.locator("#btn-start").click();
  await page.evaluate(() => {
    window.previousTrack = document.querySelector("#camera-feed").srcObject.getVideoTracks()[0];
    Object.defineProperty(document, "hidden", { configurable: true, get: () => true });
    document.dispatchEvent(new Event("visibilitychange"));
  });
  await expect(page.locator("#btn-stop")).toBeDisabled();
  await expect(page.locator("#btn-start")).toBeDisabled();
  expect(await page.evaluate(() => window.previousTrack.readyState)).toBe("ended");
  expect(await page.locator("#camera-feed").evaluate((video) => video.srcObject)).toBeNull();
  await page.evaluate(() => {
    Object.defineProperty(document, "hidden", { configurable: true, get: () => false });
    document.dispatchEvent(new Event("visibilitychange"));
  });
  await expect(page.locator("#ws-status")).toHaveClass(/green/);
  await expect(page.locator("#btn-start")).toBeEnabled();
  await expect(page.locator("#btn-stop")).toBeDisabled();
  await expect(page.locator("#fps-value")).toHaveText("0");
});

for (const [label, width, height] of [
  ["portrait", 393, 851], ["small phone", 360, 640], ["landscape", 851, 393], ["desktop", 1440, 900],
]) {
  test(`${label} keeps steering, pedals and physical turn directions legible`, async ({ page }) => {
    await page.setViewportSize({ width, height });
    await ready(page);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    for (const id of ["steering-wheel", "indicator-left", "indicator-right", "throttle-value", "brake-value", "btn-start", "btn-stop"]) {
      await expect(page.locator(`#${id}`)).toBeVisible();
    }
    const left = await page.locator("#indicator-left").boundingBox();
    const right = await page.locator("#indicator-right").boundingBox();
    const wheel = await page.locator("#steering-wheel").boundingBox();
    expect(left.x).toBeLessThan(wheel.x);
    expect(right.x).toBeGreaterThan(wheel.x + wheel.width);
    expect(wheel.width).toBeGreaterThanOrEqual(110);
    if (height >= 800 || label === "landscape") {
      const stop = await page.locator("#btn-stop").boundingBox();
      expect(stop.y + stop.height).toBeLessThanOrEqual(height);
    }
  });
}

for (const [width, height] of [[393, 851], [851, 393], [1440, 900]]) {
  test(`camera recovery panel does not overlap the cockpit at ${width}×${height}`, async ({ page }) => {
    await page.setViewportSize({ width, height });
    await page.addInitScript(() => {
      navigator.mediaDevices.getUserMedia = () => Promise.reject(new DOMException("Denied", "NotAllowedError"));
    });
    await page.goto("/dashboard/");
    await expect(page.locator("#btn-camera")).toBeEnabled();
    const prompt = await page.locator("#camera-prompt").boundingBox();
    const cockpit = await page.locator(".cockpit").boundingBox();
    const intersects = prompt.x < cockpit.x + cockpit.width && prompt.x + prompt.width > cockpit.x
      && prompt.y < cockpit.y + cockpit.height && prompt.y + prompt.height > cockpit.y;
    expect(intersects).toBe(false);
    await page.locator("#btn-camera").click({ trial: true });
  });
}
