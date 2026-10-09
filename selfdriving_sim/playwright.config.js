import { defineConfig, devices } from "@playwright/test";

/** Redis must be running; this suite exercises the real Django/Channels path. */
export default defineConfig({
  testDir: "./client/tests/browser",
  fullyParallel: false,
  workers: 1,
  timeout: 30000,
  expect: { timeout: 8000 },
  reporter: "list",
  use: {
    ...devices["Pixel 7"],
    browserName: "chromium",
    baseURL: process.env.DASHBOARD_TEST_URL ?? "http://127.0.0.1:8000",
    permissions: ["camera"],
    locale: "fa-IR",
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
    launchOptions: {
      executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH || undefined,
      args: [
        "--use-fake-device-for-media-stream",
        "--use-fake-ui-for-media-stream",
        "--no-sandbox",
        "--disable-dev-shm-usage",
        ...JSON.parse(process.env.PLAYWRIGHT_CHROMIUM_ARGS || "[]"),
      ],
    },
  },
  webServer: process.env.DASHBOARD_TEST_URL ? undefined : {
    command: "python manage.py runserver 0.0.0.0:8000 --noreload",
    url: "http://127.0.0.1:8000/dashboard/",
    reuseExistingServer: !process.env.CI,
    timeout: 30000,
  },
});
