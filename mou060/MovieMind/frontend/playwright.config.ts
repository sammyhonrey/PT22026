import { defineConfig, devices } from "@playwright/test";

const baseURL = "http://127.0.0.1:5173";

/**
 * Critical-flow tests against a mocked FastAPI contract.
 *
 * The Vite dev server is started with VITE_API_BASE_URL pointing at itself so
 * the route mocks run same-origin, without needing a backend or CORS rules.
 */
export default defineConfig({
  testDir: "./e2e/mocked",
  fullyParallel: true,
  forbidOnly: Boolean(process.env.CI),
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? [["github"], ["html", { open: "never" }]] : [["list"], ["html", { open: "never" }]],
  use: {
    baseURL,
    screenshot: "only-on-failure",
    trace: "on-first-retry",
    video: "retain-on-failure",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: {
    command: "npm run dev -- --port 5173 --strictPort",
    env: { VITE_API_BASE_URL: baseURL },
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
    url: baseURL,
  },
});
