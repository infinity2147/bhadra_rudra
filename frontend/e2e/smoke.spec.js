import { test, expect } from './fixtures.js';

// Every route must render its heading and raise no uncaught page error.
// This is the regression net for "a code change blanked a page" — the kind of
// break that a backend-only test suite can never see.
const ROUTES = [
  '/', '/incidents', '/cases', '/journey', '/graph', '/analytics', '/patterns',
  '/entities', '/model', '/live', '/copilot', '/sar', '/settings', '/aa',
  '/simulate', '/map', '/replay',
];

for (const route of ROUTES) {
  test(`route ${route} renders without errors`, async ({ page }) => {
    const pageErrors = [];
    page.on('pageerror', (e) => pageErrors.push(String(e)));

    await page.goto(route, { waitUntil: 'networkidle' });

    // Every page has a top-level heading; auto-waits up to expect timeout.
    await expect(page.locator('h1').first()).toBeVisible();
    expect(pageErrors, `uncaught errors on ${route}: ${pageErrors.join(' | ')}`).toHaveLength(0);
  });
}
