export const authPaths = {
  whoami: '/api/admin/auth/whoami',
  login: '/api/admin/auth/login',
  logout: '/api/admin/auth/logout',
} as const

export type Role = 'operator' | 'home' | 'evaluator'
type Identity = {
  role: Role
  permissions: ReadonlySet<string>
  expires: number
  subject: string | null
  accountBinding: string | null
  routineActor: string | null
}

export type SessionState =
  | { status: 'checking' | 'login' | 'unavailable'; message?: string }
  | { status: 'ready'; role: Role; generation: number }

export function record(value: unknown): Record<string, unknown> | undefined {
  if (value !== null && typeof value === 'object' && !Array.isArray(value)) {
    return value as Record<string, unknown>
  }
}

function identityFrom(value: unknown): Identity | null {
  const data = record(value)
  if (!data || data.ok !== true) return null
  const { role, permissions, expires_at } = data
  if ((role !== 'home' && role !== 'operator' && role !== 'evaluator') ||
      !Array.isArray(permissions) || !permissions.every(p => typeof p === 'string') ||
      typeof expires_at !== 'number' || !Number.isFinite(expires_at * 1000)) return null
  return {
    role, permissions: new Set(permissions as string[]), expires: expires_at * 1000,
    subject: typeof data.subject === 'string' ? data.subject : null,
    accountBinding: typeof data.account_binding === 'string' ? data.account_binding : null,
    routineActor: typeof data.routine_actor === 'string' ? data.routine_actor : null,
  }
}

function sameIdentity(first: Identity, second: Identity): boolean {
  return first.role === second.role && first.subject === second.subject &&
    first.accountBinding === second.accountBinding && first.routineActor === second.routineActor &&
    first.expires === second.expires && first.permissions.size === second.permissions.size &&
    [...first.permissions].every(permission => second.permissions.has(permission))
}

function discardBody(response: Response) {
  void response.body?.cancel().catch(() => {})
}

export class HttpError extends Error {
  constructor(readonly status: number) {
    super(status === 409 ? 'El estado cambió. Revisá los datos antes de repetir la acción.'
      : status === 404 ? 'El recurso ya no está disponible.'
      : 'El servidor no pudo completar la solicitud (' + status + ').')
  }
}

export function messageFor(error: unknown): string {
  return error instanceof HttpError ? error.message : 'No se pudo verificar la respuesta del servidor.'
}

export type RequestOptions = {
  method?: 'GET' | 'POST' | 'PUT' | 'PATCH' | 'DELETE'
  body?: unknown
  headers?: Record<string, string>
  signal?: AbortSignal
  verifyIdentity?: boolean
  permissions: readonly string[]
}

function makeRequest(options: Omit<RequestOptions, 'permissions'>, owner: AbortSignal): RequestInit {
  const signals = [owner, AbortSignal.timeout(12000)]
  if (options.signal) signals.push(options.signal)
  return {
    credentials: 'same-origin', cache: 'no-store', redirect: 'error',
    method: options.method ?? 'GET',
    headers: {
      ...(options.body === undefined ? {} : { 'Content-Type': 'application/json' }),
      ...options.headers,
    },
    body: options.body === undefined ? undefined : JSON.stringify(options.body),
    signal: AbortSignal.any(signals),
  }
}

export class Session {
  #identity: Identity | null = null
  #generation = 0
  #controller = new AbortController()
  #deadline = 0
  #timer?: ReturnType<typeof setTimeout>
  #state: SessionState = { status: 'checking' }
  #listeners = new Set<(state: SessionState) => void>()
  #purges = new Set<() => void>()
  #checking = false

  constructor(private readonly fetcher: typeof fetch = (input, init) => globalThis.fetch(input, init)) {}

  #fetch(input: RequestInfo | URL, init?: RequestInit): Promise<Response> {
    const fn = this.fetcher
    return fn(input, init)
  }

  get state(): SessionState { return this.#state }
  get signal(): AbortSignal { return this.#controller.signal }
  get generation(): number { return this.#generation }

  can(...permissions: string[]): boolean {
    const identity = this.#identity
    return identity !== null && this.#remaining() > 0 &&
      permissions.every(permission => identity.permissions.has(permission))
  }

  subscribe(listener: (state: SessionState) => void): () => void {
    this.#listeners.add(listener)
    return () => this.#listeners.delete(listener)
  }

  onPurge(purge: () => void): () => void {
    this.#purges.add(purge)
    return () => this.#purges.delete(purge)
  }

  #publish(state: SessionState) {
    this.#state = state
    for (const listener of this.#listeners) listener(state)
  }

  #remaining(): number {
    return this.#identity ? Math.min(this.#identity.expires - Date.now(), this.#deadline - performance.now()) : 0
  }

  #arm() {
    clearTimeout(this.#timer)
    const remaining = this.#remaining()
    if (remaining <= 0) this.retire()
    else this.#timer = setTimeout(() => this.#arm(), Math.max(1, Math.min(remaining, 2147483647)))
  }

  #clear(state: SessionState) {
    this.#generation++
    clearTimeout(this.#timer)
    this.#identity = null
    this.#deadline = 0
    this.#controller.abort()
    this.#controller = new AbortController()
    for (const purge of this.#purges) purge()
    this.#publish(state)
  }

  retire(message = 'La sesión venció o perdió acceso. Volvé a iniciar sesión.') {
    this.#clear({ status: 'login', message })
  }

  dispose() {
    this.#clear({ status: 'checking' })
    this.#listeners.clear()
    this.#purges.clear()
  }

  async start(token?: string): Promise<void> {
    this.#clear({ status: 'checking' })
    const owner = this.signal
    const signingIn = token !== undefined
    try {
      if (token !== undefined) {
        try {
          const response = await this.#fetch(authPaths.login, makeRequest({ method: 'POST', body: { token } }, owner))
          discardBody(response)
          if (!response.ok) throw new HttpError(response.status)
        } finally {
          token = undefined
        }
      }
      const response = await this.#fetch(authPaths.whoami, makeRequest({}, owner))
      owner.throwIfAborted()
      if (!response.ok) { discardBody(response); throw new HttpError(response.status) }
      const identity = identityFrom(await response.json())
      owner.throwIfAborted()
      if (!identity || identity.expires <= Date.now()) {
        this.#publish({ status: 'login' })
        return
      }
      this.#identity = identity
      this.#deadline = performance.now() + (identity.expires - Date.now())
      this.#publish({ status: 'ready', role: identity.role, generation: this.#generation })
      this.#arm()
    } catch (error) {
      if (owner.aborted) return
      if (error instanceof HttpError && [401, 403].includes(error.status)) {
        this.#publish({ status: 'login', message: signingIn
          ? 'La credencial no habilita el acceso.' : 'Ingresá con una credencial vigente.' })
      } else {
        this.#publish({ status: 'unavailable', message: messageFor(error) })
      }
    }
  }

  async revalidate(): Promise<boolean> {
    const identity = this.#identity
    if (!identity) return false
    if (this.#remaining() <= 0) { this.retire(); return false }
    const owner = this.signal
    try {
      const response = await this.#fetch(authPaths.whoami, makeRequest({}, owner))
      owner.throwIfAborted()
      if (!response.ok) { discardBody(response); throw new HttpError(response.status) }
      const current = identityFrom(await response.json())
      owner.throwIfAborted()
      if (current && sameIdentity(identity, current) && this.#remaining() > 0) return true
    } catch {
      if (owner.aborted) return false
    }
    this.retire('No se pudo verificar la sesión actual. Volvé a iniciar sesión.')
    return false
  }

  checkOnReturn(): void {
    if (!this.#identity || this.#checking) return
    this.#checking = true
    void this.revalidate().finally(() => { this.#checking = false })
  }

  async logout(): Promise<void> {
    this.#clear({ status: 'checking' })
    const owner = this.signal
    try {
      const response = await this.#fetch(authPaths.logout, makeRequest({ method: 'POST' }, owner))
      discardBody(response)
      owner.throwIfAborted()
      if (!response.ok && response.status !== 401) throw new HttpError(response.status)
      this.#publish({ status: 'login', message: 'Sesión cerrada.' })
    } catch {
      if (!owner.aborted) this.#publish({ status: 'login', message: 'Se limpiaron los datos de esta pestaña. El servidor no confirmó el cierre.' })
    }
  }

  async request(path: string, options: RequestOptions): Promise<unknown> {
    return this.#request(path, options, response => response.status === 204 ? null : response.json())
  }

  async homeAsset(kind: 'model', signal: AbortSignal): Promise<ArrayBuffer | null>
  async homeAsset(kind: 'plan', signal: AbortSignal): Promise<unknown>
  async homeAsset(kind: 'model' | 'plan', signal: AbortSignal): Promise<unknown> {
    try {
      return await this.#request<unknown>('/api/home/' + kind, { permissions: ['home:read'], signal },
        response => kind === 'model' ? response.arrayBuffer() : response.json(), true)
    } catch (error) {
      if (error instanceof HttpError && (error.status === 403 || error.status === 404)) return null
      throw error
    }
  }

  async #request<T>(path: string, options: RequestOptions, read: (response: Response) => T | Promise<T>,
    optionalAsset = false): Promise<T> {
    if (!/^\/api\/(?!\/)/.test(path)) throw new Error('Expected a local API path')
    if (!this.can(...options.permissions)) throw new HttpError(403)
    const owner = this.signal
    if (options.verifyIdentity && !await this.revalidate()) throw new HttpError(401)
    const init = makeRequest(options, owner)
    init.signal?.throwIfAborted()
    const response = await this.#fetch(path, init)
    if (owner.aborted) {
      discardBody(response)
      owner.throwIfAborted()
    }
    if (!response.ok) {
      if (response.status === 401 || response.status === 403 && !optionalAsset) {
        discardBody(response)
        this.retire()
      } else if (response.status === 403) {
        // Asset availability can be denied independently of the session lease.
        const failure: unknown = await response.json().catch(() => null)
        owner.throwIfAborted()
        const detail = record(failure)?.detail
        const code = typeof detail === 'string' ? detail : record(detail)?.code ?? record(failure)?.code
        if (code === 'dashboard_session_retired' || code === 'dashboard_authority_retired' ||
            code === 'dashboard_session_required') this.retire()
        else await this.revalidate()
      } else {
        discardBody(response)
      }
      throw new HttpError(response.status)
    }
    if (init.signal?.aborted) {
      discardBody(response)
      init.signal.throwIfAborted()
    }
    const data = await read(response)
    init.signal?.throwIfAborted()
    if (options.verifyIdentity && !await this.revalidate()) throw new HttpError(401)
    init.signal?.throwIfAborted()
    if (!this.can(...options.permissions)) {
      this.retire()
      throw new HttpError(401)
    }
    return data
  }
}
