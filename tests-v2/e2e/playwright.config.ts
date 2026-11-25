import { defineConfig } from '@playwright/test'

export default defineConfig({
  testDir: './specs',
  globalSetup: './server.ts',
  workers: 1,
  retries: 0,
  timeout: 12_000,
  forbidOnly: !!process.env.CI,
  use: { browserName: 'chromium', headless: true },
})
