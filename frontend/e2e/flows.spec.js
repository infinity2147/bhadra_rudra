import { test, expect } from './fixtures.js';

const API = 'http://localhost:8000';

// Fetch a real alert id from the API so flows drive live data, not guesses.
async function firstAlertId(request) {
  const r = await request.get(`${API}/api/alerts`, { headers: { 'X-User-Role': 'ADMIN' } });
  const alerts = (await r.json()).alerts || [];
  expect(alerts.length, 'need at least one alert').toBeGreaterThan(0);
  return alerts[0].alert_id ?? alerts[0].id;
}

test('SAR: select an alert and generate a report', async ({ page }) => {
  await page.goto('/sar', { waitUntil: 'networkidle' });

  // The page has TWO <select>s (sidebar role switcher + alert picker). Target
  // the alert picker by its testid — this is exactly the ambiguity that made
  // the throwaway probe grab the wrong one.
  const select = page.getByTestId('sar-alert-select');
  await expect(select).toBeVisible();
  await expect.poll(async () => (await select.locator('option').count())).toBeGreaterThan(1);

  const value = await select.locator('option').nth(1).getAttribute('value');
  await select.selectOption(value);

  const generate = page.getByRole('button', { name: /generate/i });
  await expect(generate).toBeEnabled();
  await generate.click();

  await expect(page.getByText(/Report Summary|Report Text/i).first()).toBeVisible();
});

test('Cases: add a note and verify the audit chain', async ({ page, request }) => {
  const alertId = await firstAlertId(request);
  await page.goto(`/cases?focus=${alertId}`, { waitUntil: 'networkidle' });

  // Note field — target by placeholder, NOT the first textbox (the header
  // search box would otherwise win).
  const noteText = `e2e note ${Date.now()}`;
  const noteBox = page.getByPlaceholder(/investigator note/i);
  await expect(noteBox).toBeVisible();
  await noteBox.fill(noteText);

  await page.getByRole('button', { name: /add note only/i }).click();
  await expect(page.getByText(noteText)).toBeVisible();

  // Verify chain (ADMIN role is set by app default in this context).
  await page.getByRole('button', { name: /verify chain/i }).click();
  await expect(page.getByText(/Chain intact/i)).toBeVisible();
});

test('Settings: out-of-range number is clamped', async ({ page }) => {
  await page.goto('/settings', { waitUntil: 'networkidle' });
  // First numeric field is circular_amount_tolerance (max 0.5).
  const input = page.locator('input[type="number"]').first();
  await input.fill('0.9');
  await expect.poll(async () => Number(await input.inputValue())).toBeLessThanOrEqual(0.5);
});

test('Simulate: scoring a transaction yields a severity verdict', async ({ page }) => {
  await page.goto('/simulate', { waitUntil: 'networkidle' });
  await page.getByRole('button', { name: /score/i }).first().click();
  await expect(
    page.getByText(/CRITICAL|HIGH|MEDIUM|LOW/).first()
  ).toBeVisible();
});
