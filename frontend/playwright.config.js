import { defineConfig } from '@playwright/test';

// End-to-end UI tests. These assume the stack is already running:
//   docker compose up         (backend :8000  +  frontend :5173)
//   — or — `npm run dev` here plus the backend on :8000.
//
// Run with:  npm run e2e        (headless)
//            npm run e2e -- --headed --project=chromium
//
// Locators are role/label/testid based with Playwright's auto-waiting
// web-first assertions — deliberately NOT `.first()` on generic selectors or
// regex-on-body, which is what made the earlier throwaway probes flaky.
export default defineConfig({
  testDir: './e2e',
  // These specs drive ONE shared, stateful backend and some flows are slow
  // (Cases note->verify ~18s). Running workers in parallel races on that single
  // backend and flakes, so this suite is serial by design.
  fullyParallel: false,
  workers: 1,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  reporter: 'list',
  timeout: 30_000,
  expect: { timeout: 10_000 },
  use: {
    baseURL: 'http://localhost:5173',
    headless: true,
    viewport: { width: 1440, height: 900 },
    trace: 'retain-on-failure',
  },
  projects: [{ name: 'chromium', use: { browserName: 'chromium' } }],
});
