import { mount, unmount } from 'svelte'
import { afterEach, describe, expect, it, vi } from 'vitest'
import App from '../src/App.svelte'
import { authPaths, Session } from '../src/lib/session'
import { clickButton, deferred, devices, identity, json, transport } from './helpers'

const disposers: (() => Promise<void>)[] = []

function app(fetcher: typeof fetch, pathname = '/') {
  const target = document.createElement('div')
  document.body.append(target)
  const session = new Session(fetcher)
  const component = mount(App, { target, props: { session, pathname } })
  disposers.push(async () => { await unmount(component); target.remove() })
  return { target, session }
}

afterEach(async () => {
  for (const dispose of disposers.splice(0)) await dispose()
})

describe('SPA', () => {
  it('no navega desde NEXO a observabilidad en ninguna sección (D1)', async () => {
    const operator = identity(['home:read', 'tasks:read', 'memory:read', 'memory:assign',
      'admin:manage', 'auth:manage'], 'operator')
    const fetcher = transport(path => {
      if (path === authPaths.whoami) return operator()
      if (path === '/api/home/catalog') return json(devices())
      if (path === '/api/memory') return json({ facts: [], proposals: [] })
      if (path === '/api/reminders/notices') return json({ source: 'reminder-store', notices: [] })
      return json([])
    })
    const { target } = app(fetcher)
    await vi.waitFor(() => expect(target.querySelector('h1')?.textContent).toBe('Casa'))
    for (const label of ['Casa', 'Rutinas', 'Tareas', 'Memoria', 'Voces y consentimientos', 'Acceso y sistema']) {
      clickButton(target, label)
      await vi.waitFor(() => expect(target.querySelector('h1')?.textContent).toBe(label))
      const navigation = [...target.querySelectorAll('[href], [action], [formaction]')]
      expect(navigation.some(element => ['href', 'action', 'formaction'].some(attribute => {
        const value = element.getAttribute(attribute)
        return value && /^\/obs(?:\/|$)/.test(new URL(value, document.baseURI).pathname)
      }))).toBe(false)
      expect(target.textContent).not.toMatch(/observabilidad|\/obs\b/i)
      expect(location.pathname).toBe('/')
    }
    const technical = app(fetcher, '/obs')
    await vi.waitFor(() => expect(technical.target.querySelector('h1')?.textContent).toBe('Observabilidad'))
  })

  it('espera la sesión antes de pedir modelo y plano y usa credenciales y no-store', async () => {
    const initial = deferred<Response>()
    let logged = false
    const authenticated = identity(['home:read'])
    const fetcher = transport((path, options) => {
      if (path === authPaths.whoami) return logged ? authenticated() : initial.promise
      if (path === authPaths.login) {
        expect(options.method).toBe('POST')
        expect(JSON.parse(String(options.body))).toEqual({ token: 'fixture-token' })
        logged = true
        return json({ ok: true })
      }
      if (path === '/api/home/catalog') return json(devices())
      if (path === '/api/home/model') return json({}, 404)
      if (path === '/api/home/plan') return json({}, 404)
      throw new Error('Unexpected request: ' + path)
    })
    const { target } = app(fetcher)
    await vi.waitFor(() => expect(fetcher.mock.calls.map(([url]) => url)).toEqual([authPaths.whoami]))
    initial.resolve(json({ ok: false }, 401))
    await vi.waitFor(() => expect(target.querySelector('input[name=token]')).not.toBeNull())
    expect(fetcher).toHaveBeenCalledTimes(1)
    target.querySelector<HTMLInputElement>('input[name=token]')!.value = 'fixture-token'
    target.querySelector('form')!.dispatchEvent(new SubmitEvent('submit', { cancelable: true, bubbles: true }))
    await vi.waitFor(() => expect(target.textContent).toContain('Habitación de prueba'))
    for (const endpoint of ['/api/home/model', '/api/home/plan']) {
      expect(fetcher).toHaveBeenCalledWith(endpoint, expect.objectContaining({
        credentials: 'same-origin', cache: 'no-store',
      }))
    }
    expect(target.textContent).not.toContain('Vista 3D')
    expect(target.textContent).not.toContain('Micrófonos')
    expect(target.textContent).not.toContain('Encender')
  })

  it('un 403 de los recursos 3D conserva habitaciones si whoami mantiene la autoridad', async () => {
    const home = identity(['home:read'])
    const fetcher = transport(path => {
      if (path === authPaths.whoami) return home()
      if (path === '/api/home/catalog') return json(devices())
      if (path === '/api/home/model') return json({}, 403)
      return json({ DEFAULT_YAW: 0 })
    })
    const { target, session } = app(fetcher)
    await vi.waitFor(() => expect(fetcher.mock.calls.filter(([url]) => url === authPaths.whoami)).toHaveLength(2))
    await vi.waitFor(() => expect(target.textContent).toContain('Habitación de prueba'))
    expect(target.textContent).not.toContain('Vista 3D')
    expect(session.state.status).toBe('ready')
  })

  it('gates assets and technical panels by permission', async () => {
    const fetcher = transport(() => identity(['tasks:read'])())
    const { target } = app(fetcher, '/obs')
    await vi.waitFor(() => expect(target.textContent).toContain('Esta sesión no tiene acceso a los paneles técnicos.'))
    expect(fetcher.mock.calls.map(([url]) => url)).toEqual([authPaths.whoami])
  })

  it.each([401, 403])('purga habitación y modelo ante %s de una operación protegida', async status => {
    const home = identity(['home:read'])
    let loseLease = false
    const fetcher = transport(path => {
      if (path === authPaths.whoami) return home()
      if (path === '/api/home/catalog') return loseLease ? json({}, status) : json(devices())
      if (path === '/api/home/model') return new Response(new Uint8Array([1, 2, 3]))
      return json({ DEFAULT_YAW: 0 })
    })
    const { target, session } = app(fetcher)
    await vi.waitFor(() => expect(target.textContent).toContain('Vista 3D'))
    expect(target.textContent).toContain('Habitación de prueba')
    const oldSignal = session.signal
    loseLease = true
    clickButton(target, 'Actualizar')
    await vi.waitFor(() => expect(target.querySelector('input[name=token]')).not.toBeNull())
    expect(oldSignal.aborted).toBe(true)
    expect(target.textContent).not.toMatch(/Habitación de prueba|Vista 3D/)
  })

  it('un 403 explícito de lease en el modelo retira la sesión sin esperar al plano', async () => {
    const home = identity(['home:read'])
    const model = deferred<Response>()
    const plan = deferred<Response>()
    const fetcher = transport(path => {
      if (path === authPaths.whoami) return home()
      if (path === '/api/home/catalog') return json(devices())
      return path === '/api/home/model' ? model.promise : plan.promise
    })
    const { target, session } = app(fetcher)
    await vi.waitFor(() => expect(target.textContent).toContain('Habitación de prueba'))
    model.resolve(json({ detail: 'dashboard_session_retired' }, 403))
    await vi.waitFor(() => expect(session.state.status).toBe('login'))
    expect(target.textContent).not.toContain('Habitación de prueba')
    expect(fetcher.mock.calls.filter(([url]) => url === authPaths.whoami)).toHaveLength(1)
    plan.resolve(json({ DEFAULT_YAW: 1 }))
  })

  it('retira los datos personales si cambia la cuenta detrás de la cookie', async () => {
    const expires_at = Date.now() / 1000 + 120
    let accountBinding = 'original'
    const fetcher = transport(path => {
      if (path === authPaths.whoami) return json({ ok: true, role: 'home', subject: 'home', account_binding: accountBinding,
        permissions: ['memory:own:read'], expires_at })
      expect(path).toBe('/api/memory/mine')
      accountBinding = 'replacement'
      return json({ facts: [{ id: 'fixture', category: 'Prueba', content: 'Recuerdo de otra cuenta', expires_at: null }] })
    })
    const { target, session } = app(fetcher)
    await vi.waitFor(() => expect(session.state.status).toBe('login'))
    expect(target.textContent).not.toContain('Recuerdo de otra cuenta')
    expect(fetcher.mock.calls.map(([url]) => url)).toEqual([
      authPaths.whoami, authPaths.whoami, '/api/memory/mine', authPaths.whoami,
    ])
  })

  it('purga antes de recibir la respuesta de logout y descarta la lectura tardía', async () => {
    const authenticated = identity(['home:read'])
    const logout = deferred<Response>()
    const plan = deferred<Response>()
    const fetcher = transport(path => {
      if (path === authPaths.whoami) return authenticated()
      if (path === authPaths.logout) return logout.promise
      if (path === '/api/home/catalog') return json(devices())
      if (path === '/api/home/plan') return plan.promise
      return new Response(new Uint8Array([1, 2, 3]))
    })
    const { target, session } = app(fetcher)
    await vi.waitFor(() => expect(target.textContent).toContain('Habitación de prueba'))
    clickButton(target, 'Cerrar sesión')
    expect(target.textContent).not.toContain('Habitación de prueba')
    expect(session.can('home:read')).toBe(false)
    plan.resolve(json({ DEFAULT_YAW: 1 }))
    logout.resolve(new Response(null, { status: 204 }))
    await vi.waitFor(() => expect(target.textContent).toContain('Sesión cerrada.'))
    expect(target.textContent).not.toContain('Vista 3D')
  })
})
