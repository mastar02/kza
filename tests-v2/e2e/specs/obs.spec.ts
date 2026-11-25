import { test, expect, login, expectStatus } from './fixtures'

test('EVALUATOR abre evaluación y queda fuera de Casa y administración', async ({ page }) => {
  await login(page, 'EVALUATOR', '/obs')
  await page.getByRole('button', { name: 'Evaluación' }).click()
  await expect(page.getByRole('heading', { name: 'Evaluación' })).toBeVisible()
  await expectStatus(page, '/api/evaluation/grants', 200)
  for (const path of ['/api/home/model', '/api/admin/users']) await expectStatus(page, path, 403)
})

test('OPERATOR negocia WS v3 y accede a diagnóstico y captura', async ({ page }) => {
  const frames: string[] = []
  page.on('websocket', socket => {
    if (new URL(socket.url()).pathname !== '/ws/live') return
    socket.on('framesent', ({ payload }) => {
      const frame = JSON.parse(String(payload))
      frames.push('send:' + frame.type + (frame.type === 'live_hello' ? ':' + frame.version : ''))
    })
    socket.on('framereceived', ({ payload }) => frames.push('receive:' + JSON.parse(String(payload)).type))
  })
  await login(page, 'OPERATOR', '/obs')
  await expect(page.getByRole('status')).toHaveText('Conectado')
  expect(frames.slice(0, 4)).toEqual(['send:live_hello:3', 'receive:live_reset', 'send:live_reset_ack', 'receive:rest_ready'])
  await expect(page.getByRole('heading', { name: 'En vivo' })).toBeVisible()
  await page.getByRole('button', { name: 'Diagnóstico' }).click()
  await expect(page.getByRole('heading', { name: 'Diagnóstico' })).toBeVisible()
  await page.getByRole('button', { name: 'Captura' }).click()
  await expect(page.getByRole('button', { name: 'Pausar captura global' })).toBeVisible()
})

test('Revocar la sesión activa cierra WS con 1008 y purga el panel', async ({ page }) => {
  // Observe the native close event without handling frames or replacing the transport.
  await page.addInitScript(() => {
    window.WebSocket = new Proxy(window.WebSocket, {
      construct(target, args) {
        const socket: WebSocket = Reflect.construct(target, args)
        if (new URL(socket.url).pathname === '/ws/live') {
          socket.addEventListener('close', event => {
            Reflect.set(window, 'liveCloseCode', event.code)
          })
        }
        return socket
      },
    })
  })
  await login(page, 'OPERATOR', '/obs')
  await expect(page.getByRole('status')).toHaveText('Conectado')
  const response = await page.evaluate(async () =>
    (await fetch('/api/admin/auth/logout', { method: 'POST' })).status)
  expect(response).toBeGreaterThanOrEqual(200)
  expect(response).toBeLessThan(300)
  await expect.poll(() => page.evaluate(() => Reflect.get(window, 'liveCloseCode')), { timeout: 3000 }).toBe(1008)
  await expect(page.getByLabel('Credencial de acceso')).toBeVisible({ timeout: 1000 })
  await expect(page.getByRole('heading', { name: 'Conversaciones' })).toHaveCount(0)
  await expectStatus(page, '/api/admin/auth/whoami', 401)
})

test('Las rutas retiradas y la documentación OpenAPI devuelven 404', async ({ page }) => {
  await login(page, 'OPERATOR', '/obs')
  for (const path of ['/openapi.json', '/docs', '/redoc', '/api/metrics',
    '/api/subsystems', '/api/failures', '/api/presence']) {
    await expectStatus(page, path, 404)
  }
})
