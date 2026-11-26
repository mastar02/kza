import { mount, unmount } from 'svelte'
import { afterEach, describe, expect, it, vi } from 'vitest'
import Routines from '../src/nexo/Routines.svelte'
import Diagnostics from '../src/obs/Diagnostics.svelte'
import { Session } from '../src/lib/session'
import { clickButton, identity, json, transport } from './helpers'

const disposers: (() => Promise<void>)[] = []

async function view(component: typeof Routines | typeof Diagnostics, fetcher: typeof fetch,
  props: Record<string, unknown> = {}) {
  const session = new Session(fetcher)
  await session.start()
  const target = document.createElement('div')
  document.body.append(target)
  const mounted = mount(component as typeof Routines, { target, props: { session, ...props } })
  disposers.push(async () => { await unmount(mounted); target.remove(); session.dispose() })
  return target
}

afterEach(async () => {
  vi.restoreAllMocks()
  for (const dispose of disposers.splice(0)) await dispose()
})

const routine = { routine_id: 'r-1', name: 'Buenas noches', enabled: true, revision: 1, schedule: null }

function routineFetcher(attempts: () => Response, perms = ['home:read']) {
  const who = identity(perms)
  return transport(path => {
    if (path === '/api/admin/auth/whoami') return who()
    if (path === '/api/routines') return json([routine])
    if (path === '/api/routines/r-1/attempts') return attempts()
    return json([])
  })
}

describe('historial de rutinas', () => {
  it('muestra ejecuciones con el contrato real de recibos bajo demanda', async () => {
    const fetcher = routineFetcher(() => json({
      routine_id: 'r-1',
      runs: [
        {
          run_id: 'run-99',
          routine_id: 'r-1',
          status: 'succeeded',
          source: 'manual',
          started_at_unix_ms: 1795680000000,
          finished_at_unix_ms: 1795680002000,
        },
        { status: 'late', at: 1795000000 },
      ],
    }))
    const target = await view(Routines, fetcher)
    await vi.waitFor(() => expect(target.textContent).toContain('Buenas noches'))
    expect(fetcher.mock.calls.some(([url]) => String(url).includes('attempts'))).toBe(false)
    clickButton(target, 'Historial')
    await vi.waitFor(() => expect(target.textContent).toContain('Manual'))
    expect(target.textContent).toContain('Aceptada')
    expect(target.textContent).toContain('Tardía')
    expect(target.textContent).not.toContain('Sin fecha')
    expect(target.textContent).not.toContain('r-1')
    expect(target.textContent).not.toContain('run-99')
  })

  it('acepta la clave attempts', async () => {
    const target = await view(Routines,
      routineFetcher(() => json({ attempts: [{ status: 'partial', created_at: '2026-11-20T22:00:00Z' }] })))
    await vi.waitFor(() => expect(target.textContent).toContain('Buenas noches'))
    clickButton(target, 'Historial')
    await vi.waitFor(() => expect(target.textContent).toContain('Parcial'))
  })

  it('muestra estado vacío limpio sin ejecuciones y con 503', async () => {
    for (const response of [() => json({ routine_id: 'r-1', runs: [] }), () => json({}, 503)]) {
      const target = await view(Routines, routineFetcher(response))
      await vi.waitFor(() => expect(target.textContent).toContain('Buenas noches'))
      clickButton(target, 'Historial')
      await vi.waitFor(() => expect(target.textContent).toContain('todavía no tiene ejecuciones'))
      expect(target.querySelector('.error')).toBeNull()
    }
  })

  it('informa un error de servidor distinto de 503', async () => {
    const target = await view(Routines, routineFetcher(() => json({}, 500)))
    await vi.waitFor(() => expect(target.textContent).toContain('Buenas noches'))
    clickButton(target, 'Historial')
    await vi.waitFor(() => expect(target.querySelector('[role=alert]')?.textContent).toContain('500'))
  })
})

function opsFetcher(permissions: string[], posts: string[]) {
  const who = identity(permissions, 'operator')
  let acked = false
  return transport((path, options) => {
    if (path === '/api/admin/auth/whoami') return who()
    if (options.method === 'POST') {
      posts.push(path)
      if (path.endsWith('/ack')) acked = true
      return json({ ok: true })
    }
    if (path === '/api/alerts') return json({ alerts: [{ id: 'a-1', title: 'GPU caliente', priority: 'high', body: 'Temperatura alta', acked }] })
    if (path === '/api/system/services') {
      return json([
        { name: 'nexa-brain', status: 'active', substate: 'running' },
        { name: 'nexa-web', status: 'active', substate: 'running' },
        { name: 'nexa-dashboard-proxy', status: 'active', substate: 'running' },
      ])
    }
    return json([])
  })
}

describe('alertas y servicios', () => {
  it('oculta Reconocer sin admin:manage', async () => {
    const target = await view(Diagnostics, opsFetcher(['home:read', 'diagnostics:read'], []))
    await vi.waitFor(() => expect(target.textContent).toContain('GPU caliente'))
    expect([...target.querySelectorAll('button')].some(b => b.textContent?.trim() === 'Reconocer')).toBe(false)
  })

  it('reconoce una alerta con admin:manage y refresca la lista', async () => {
    const posts: string[] = []
    const permissions = ['home:read', 'diagnostics:read', 'admin:manage']
    const target = await view(Diagnostics, opsFetcher(permissions, posts))
    await vi.waitFor(() => expect(target.textContent).toContain('Temperatura alta'))
    clickButton(target, 'Reconocer')
    await vi.waitFor(() => expect(posts).toEqual(['/api/admin/alerts/a-1/ack']))
    await vi.waitFor(() => expect(target.textContent).toContain('Reconocida'))
    expect([...target.querySelectorAll('button')].some(b => b.textContent?.trim() === 'Reconocer')).toBe(false)
  })

  it('reinicia servicios restartables solo con admin:manage y confirmación', async () => {
    const readonly = await view(Diagnostics, opsFetcher(['home:read', 'diagnostics:read'], []), { system: true })
    clickButton(readonly, 'Servicios')
    await vi.waitFor(() => expect(readonly.textContent).toContain('nexa-brain'))
    expect([...readonly.querySelectorAll('button')].some(b => b.textContent?.trim() === 'Reiniciar')).toBe(false)

    const posts: string[] = []
    const permissions = ['home:read', 'diagnostics:read', 'admin:manage']
    const target = await view(Diagnostics, opsFetcher(permissions, posts), { system: true })
    const confirm = vi.spyOn(window, 'confirm').mockReturnValueOnce(false).mockReturnValueOnce(true)
    const buttons = () => [...target.querySelectorAll('button')].filter(b => b.textContent?.trim() === 'Reiniciar')
    expect(buttons()).toHaveLength(0)
    clickButton(target, 'Servicios')
    await vi.waitFor(() => expect(target.textContent).toContain('nexa-dashboard-proxy'))
    await vi.waitFor(() => expect(buttons()).toHaveLength(1))
    clickButton(target, 'Reiniciar')
    expect(confirm).toHaveBeenCalledTimes(1)
    expect(posts).toEqual([])
    clickButton(target, 'Reiniciar')
    await vi.waitFor(() => expect(posts).toEqual(['/api/admin/services/nexa-brain/restart']))
  })
})
