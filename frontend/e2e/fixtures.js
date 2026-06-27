import { test as base, expect } from '@playwright/test';

// All e2e specs run as ADMIN — the role gate is exercised at the API layer in
// the Python integration tests; here we want the full UI unlocked so the
// flows (Settings edit, Verify chain, etc.) are reachable. The role is read
// from localStorage, so inject it before any page script runs.
export const test = base.extend({
  context: async ({ context }, use) => {
    await context.addInitScript(() => localStorage.setItem('rudra_role', 'ADMIN'));
    await use(context);
  },
});

export { expect };
