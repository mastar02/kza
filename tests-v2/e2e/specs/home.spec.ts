import { test, expect, login, expectStatus } from './fixtures'

const assets = ['/api/home/model', '/api/home/plan']

test('El login público no solicita activos privados', async ({ page }) => {
  const requests: string[] = []
  page.on('request', request => requests.push(new URL(request.url()).pathname))
  await page.goto('/')
  await expect(page.getByLabel('Credencial de acceso')).toBeVisible()
  expect(requests).toContain('/api/admin/auth/whoami')
  expect(requests.filter(path => assets.includes(path))).toEqual([])
})

test('Los activos privados exigen sesión', async ({ page }) => {
  await page.goto('/')
  for (const path of assets) await expectStatus(page, path, 401)
})

test('Casa carga con cookie protegida y recursos privados sin caché', async ({ page, context }) => {
  const paths = ['/api/home/catalog', ...assets]
  const [responses] = await Promise.all([
    Promise.all(paths.map(path => page.waitForResponse(r => r.url().endsWith(path)))),
    login(page, 'HOME'),
  ])
  for (const response of responses) {
    expect(response.status(), response.url()).toBe(200)
    expect(response.headers()['cache-control'], response.url()).toMatch(/\bno-store\b/)
  }
  expect(await context.cookies()).toContainEqual(expect.objectContaining({
    name: 'kza_dashboard_token', httpOnly: true, sameSite: 'Strict',
  }))
  await expect(page.getByRole('heading', { name: 'Casa', exact: true })).toBeVisible()
  await expect(page.getByRole('heading', { name: 'Fixture light' })).toBeVisible()
})

for (const role of ['HOME', 'OPERATOR'] as const) {
  test('D1: ninguna sección NEXO enlaza a observabilidad para ' + role, async ({ page }) => {
    await login(page, role)
    const nav = page.getByRole('navigation', { name: 'Secciones del hogar' })
    const names = await nav.getByRole('button').allTextContents()
    expect(names).toContain('Casa')
    for (const name of names) {
      const tab = nav.getByRole('button', { name, exact: true })
      await tab.click()
      await expect(tab).toHaveAttribute('aria-current', 'page')
      const targets = await page.locator('[href], [formaction], [action]').evaluateAll(nodes =>
        nodes.map(node => node.getAttribute('href') || node.getAttribute('formaction') || node.getAttribute('action'))
          .filter(value => value && /^\/obs(?:\/|$)/.test(new URL(value, location.href).pathname)))
      expect(targets, name).toEqual([])
      for (const control of ['link', 'button'] as const) {
        await expect(page.getByRole(control, { name: /\bobs\b|observabilidad/i })).toHaveCount(0)
      }
    }
  })
}

test('Casa envía la acción tipada y muestra su recibo', async ({ page }) => {
  await login(page, 'HOME')
  const card = page.getByRole('article').filter({ has: page.getByRole('heading', { name: 'Fixture light' }) })
  const on = card.getByRole('button', { name: 'Encender' })
  await expect(on).toBeEnabled()
  const [receipt] = await Promise.all([
    page.waitForResponse(r => r.url().endsWith('/api/home/actions') && r.request().method() === 'POST'),
    on.click(),
  ])
  const action = receipt.request().postDataJSON()
  expect(action).toMatchObject({ entity_id: 'light.fixture', operation: 'turn_on' })
  expect(receipt.ok()).toBe(true)
  expect(await receipt.json()).toMatchObject({ request_id: action.request_id, entity_id: action.entity_id, status: 'confirmed' })
  await expect(card.getByRole('status')).toHaveText('Confirmado por Home Assistant')
  await expect(page.getByRole('alert')).toHaveCount(0)
})

test('HOME queda fuera de administración y evaluación', async ({ page }) => {
  await login(page, 'HOME')
  for (const path of ['/api/admin/users', '/api/evaluation/grants']) await expectStatus(page, path, 403)
})

test('Cerrar sesión purga Casa y retira la cookie de sesión', async ({ page }) => {
  await login(page, 'HOME')
  const device = page.getByRole('heading', { name: 'Fixture light' })
  await expect(device).toBeVisible()
  await page.getByRole('button', { name: 'Cerrar sesión' }).click()
  await expect(page.getByLabel('Credencial de acceso')).toBeVisible()
  await expect(device).toHaveCount(0)
  await expect(page.getByRole('navigation', { name: 'Secciones del hogar' })).toHaveCount(0)
  await expectStatus(page, '/api/admin/auth/whoami', 401)
  expect((await page.context().cookies()).map(cookie => cookie.name)).not.toContain('kza_dashboard_token')
})
