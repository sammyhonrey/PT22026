import { defineConfig, devices } from "@playwright/test";

const baseURL = process.env.E2E_BASE_URL ?? "http://127.0.0.1:5173";

/**
 * Live smoke tests. They run only when a safe, non-production backend,
 * frontend URL, and dedicated test account are provided:
 *
 *   E2E_BASE_URL=https://preview.example.com
 *   E2E_EMAIL=qa-user@example.com
 *   E2E_PASSWORD=...
 *
 * Without E2E_EMAIL/E2E_PASSWORD every test is skipped. Never point these at
 * production user data or a shared account.
 */
export default defineConfig({
  testDir: "./e2e/live",
  fullyParallel: false,
  retries: 0,
  reporter: [["list"], ["html", { open: "never" }]],
  use: {
    baseURL,
    screenshot: "only-on-failure",
    trace: "retain-on-failure",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
});
