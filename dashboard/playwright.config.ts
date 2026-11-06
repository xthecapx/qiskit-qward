import { defineConfig, devices } from "@playwright/test";

// Runs against the production build so the deployed relative paths are what is tested.
export default defineConfig({
  testDir: "tests/e2e",
  timeout: 30_000,
  fullyParallel: true,
  reporter: process.env.CI ? "github" : "list",
  use: { baseURL: "http://127.0.0.1:8775/", trace: "retain-on-failure" },
  webServer: {
    command: "npx vite preview --port 8775 --strictPort --host 127.0.0.1",
    url: "http://127.0.0.1:8775/",
    reuseExistingServer: !process.env.CI,
  },
  projects: [
    { name: "desktop", use: { ...devices["Desktop Chrome"] } },
    { name: "narrow", use: { ...devices["Pixel 7"] }, grep: /@narrow/ },
  ],
});
