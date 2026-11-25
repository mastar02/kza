import { test as base, expect, type Page } from '@playwright/test'
import { readFileSync } from 'node:fs'
import path from 'node:path'

export const test = base.extend({
  baseURL: async ({}, use) => {
    const origin = process.env.NEXA_E2E_ORIGIN
    if (!origin) throw new Error('Missing local test server')
    await use(origin)
  },
})
export { expect }

const seed = readFileSync(path.resolve(__dirname, '../../../crates/nexa-web/src/test_mode.rs'), 'utf8')
export type Role = 'HOME' | 'OPERATOR' | 'EVALUATOR'

export async function login(page: Page, role: Role, route: '/' | '/obs' = '/') {
  const token = new RegExp('pub const ' + role + '_TOKEN: &str = "([^"]+)";').exec(seed)?.[1]
  if (!token) throw new Error('Missing test-mode seed for ' + role)
  await page.goto(route)
  await page.getByLabel('Credencial de acceso').fill(token)
  const response = page.waitForResponse(r => r.url().endsWith('/api/admin/auth/login') && r.request().method() === 'POST')
  await page.getByRole('button', { name: 'Ingresar' }).click()
  expect((await response).ok()).toBe(true)
  await expect(page.getByRole('button', { name: 'Cerrar sesión' })).toBeVisible()
}

export async function expectStatus(page: Page, path: string, status: number) {
  const actual = await page.evaluate(async url => (await fetch(url)).status, path)
  expect(actual, path).toBe(status)
}
