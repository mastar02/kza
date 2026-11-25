import { record } from './session'
import type { RequestOptions, Session } from './session'

export type SocketPath = '/ws/live' | '/ws/voices' | '/ws/evaluation'
export type Channel = 'live' | 'conversations' | 'voices:profiles' | 'voices:captures' | 'voices:review' | 'evaluation:sample'
export type StreamStatus = 'connecting' | 'ready' | 'disconnected' | 'closed'
export type LiveRow = { type: string; ts: string; title: string; text: string; zone: string }
export type Projection<T> = (rows: readonly unknown[]) => T
type Receipt = { ticket: string; nonce: string }
type Entry = { id: string; receipt: Receipt; data: unknown; wall: number; mono: number }
type Phase = 'hello' | 'clearing' | 'reset-ack' | 'ready'
type Clock = { wall: number; mono: number }
type SocketFactory = (url: string) => WebSocket

const voiceChannels: Channel[] = ['voices:profiles', 'voices:captures', 'voices:review']
const channels: Record<SocketPath, readonly Channel[]> = {
  '/ws/live': ['live', 'conversations'],
  '/ws/voices': voiceChannels,
  '/ws/evaluation': ['evaluation:sample'],
}

const pathPermissions: Record<SocketPath, readonly string[]> = {
  '/ws/live': ['conversations:read'],
  '/ws/voices': ['voices:read', 'voices:review'],
  '/ws/evaluation': ['evaluation:read'],
}

const uuid = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/
const liveTypes = new Set(['turn', 'wake', 'tts', 'alert', 'cooldown'])
const retryDelays = [500, 1500, 3500, 7000]
let pageId: string | undefined

function isId(value: unknown): value is string {
  return typeof value === 'string' && uuid.test(value)
}

function receipt(value: unknown): Receipt | undefined {
  const row = record(value)
  if (isId(row?.ticket) && isId(row.nonce)) return { ticket: row.ticket, nonce: row.nonce }
}

function clock(): Clock { return { wall: Date.now(), mono: performance.now() } }

export function deadline(expires: unknown, observed?: unknown, started?: Clock): Clock | null {
  if (typeof expires !== 'number' || !Number.isFinite(expires * 1000)) return null
  const now = clock()
  const wallEnd = expires * 1000
  let budget = wallEnd - now.wall
  if (started) {
    if (typeof observed !== 'number' || !Number.isFinite(observed * 1000) || expires <= observed) return null
    const elapsed = Math.max(0, now.mono - started.mono, now.wall - started.wall)
    budget = Math.min(budget, (expires - observed) * 1000 - elapsed)
  }
  return { wall: Math.min(wallEnd, now.wall + budget), mono: now.mono + budget }
}

export function remaining(due: Clock): number {
  return Math.min(due.wall - Date.now(), due.mono - performance.now())
}

function bounded(value: unknown, max: number): string {
  return typeof value === 'string' && value.length <= max ? value : ''
}

type Hooks = {
  status: (status: StreamStatus) => void
  changed: (channel: Channel, data: unknown | null) => void
  cleared?: () => void
  retired?: (channel: Channel) => void
  committed: () => Promise<void>
}

/** Private data is retained only by this connection and acknowledged after its view releases it. */
export class ReceiptSocket {
  #entries = new Map<Channel, Entry>()
  #retired = new Set<string>()
  #reads = new Map<Channel, AbortController>()
  #inflight = new Set<Promise<void>>()
  #socket: WebSocket | null = null
  #phase: Phase = 'hello'
  #connection: string | null = null
  #epoch = 0
  #stopped = false
  #retry = 0
  #clientId = pageId ??= crypto.randomUUID()
  #retryTimer?: ReturnType<typeof setTimeout>
  #handshakeTimer?: ReturnType<typeof setTimeout>
  #expiryTimer?: ReturnType<typeof setTimeout>
  #commitBarrier = Promise.resolve()
  #unregister?: () => void

  constructor(private readonly session: Session, private readonly path: SocketPath, private readonly hooks: Hooks,
    private readonly connect: SocketFactory = url => new WebSocket(url)) {
    this.#unregister = session.onPurge(() => this.stop())
    this.#open()
  }

  get ready(): boolean { return this.#phase === 'ready' && this.#usable() }

  #usable(): boolean {
    return !this.#stopped && this.session.can(...pathPermissions[this.path]) &&
      (this.path !== '/ws/evaluation' || this.session.state.status === 'ready' && this.session.state.role === 'evaluator')
  }

  #open() {
    if (!this.#usable()) { this.stop(); return }
    this.hooks.status('connecting')
    this.#phase = 'hello'
    try {
      const url = new URL(this.path, location.href)
      url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:'
      const socket = this.connect(url.href)
      this.#socket = socket
      const current = () => this.#socket === socket && this.#usable()
      this.#handshakeTimer = setTimeout(() => this.#disconnect(1006), 6000)
      socket.addEventListener('open', () => {
        if (current()) this.#send({ type: 'live_hello', version: 3, client_id: this.#clientId })
      })
      socket.addEventListener('message', event => {
        if (!current()) return
        try {
          if (typeof event.data !== 'string' || event.data.length > 65536) throw new Error('Invalid frame')
          const frame = record(JSON.parse(event.data))
          if (!frame) throw new Error('Invalid frame')
          this.#message(frame, socket)
        } catch {
          this.#disconnect(1006)
        }
      })
      socket.addEventListener('close', event => { if (current()) this.#disconnect(event.code) })
    } catch {
      this.#disconnect(1006)
    }
  }

  #send(message: object) {
    const socket = this.#socket
    if (!socket || socket.readyState !== 1 || !this.#usable()) return
    try { socket.send(JSON.stringify(message)) } catch { this.#disconnect(1006) }
  }

  #afterCommit(): Promise<void> {
    const current = this.hooks.committed()
    this.#commitBarrier = Promise.all([this.#commitBarrier, current]).then(() => {})
    return this.#commitBarrier
  }

  #clear() {
    this.#epoch++
    this.#connection = null
    clearTimeout(this.#expiryTimer)
    for (const read of this.#reads.values()) read.abort()
    this.#reads.clear()
    for (const channel of this.#entries.keys()) this.hooks.changed(channel, null)
    this.#entries.clear()
    this.#retired.clear()
    this.hooks.cleared?.()
  }

  #disconnect(code: number) {
    const socket = this.#socket
    this.#socket = null
    this.#phase = 'hello'
    this.#clear()
    clearTimeout(this.#handshakeTimer)
    if (socket && socket.readyState < 2) socket.close()
    if (code === 1008) { this.session.retire(); return }
    if (!this.#usable()) { this.stop(); return }
    this.hooks.status('disconnected')
    const delay = retryDelays[this.#retry++]
    if (delay === undefined) { this.stop(); return }
    const retryEpoch = this.#epoch
    this.#retryTimer = setTimeout(async () => {
      try {
        await this.#afterCommit()
        if (this.#epoch === retryEpoch && this.#usable() && await this.session.revalidate() &&
            this.#epoch === retryEpoch) this.#open()
      } catch {
        if (this.#epoch === retryEpoch) this.stop()
      }
    }, delay)
  }

  stop() {
    if (this.#stopped) return
    this.#stopped = true
    clearTimeout(this.#retryTimer)
    clearTimeout(this.#handshakeTimer)
    const socket = this.#socket
    this.#socket = null
    this.#clear()
    this.#unregister?.()
    this.hooks.status('closed')
    if (socket && socket.readyState < 2) socket.close()
  }

  #ack(receipts: Receipt[]) {
    if (!receipts.length) return
    const epoch = this.#epoch
    void this.#afterCommit().then(() => {
      if (this.#epoch !== epoch || !this.ready) return
      const unique = [...new Map(receipts.map(item => [item.ticket, item])).values()]
      for (let offset = 0; offset < unique.length; offset += 16) {
        this.#send({ type: 'live_ack', receipts: unique.slice(offset, offset + 16) })
      }
    }).catch(() => { if (this.#epoch === epoch) this.#disconnect(1006) })
  }

  #message(frame: Record<string, unknown>, socket: WebSocket) {
    const payload = record(frame.payload)
    if (frame.type === 'live_reset') {
      if (!isId(payload?.id) || !isId(payload.nonce) ||
          this.#phase === 'clearing' || this.#phase === 'reset-ack') throw new Error('Invalid reset')
      const { id, nonce } = payload
      this.#phase = 'clearing'
      clearTimeout(this.#handshakeTimer)
      this.#handshakeTimer = setTimeout(() => this.#disconnect(1006), 6000)
      this.#clear()
      this.hooks.status('connecting')
      const epoch = this.#epoch
      void Promise.all([this.#afterCommit(), ...this.#inflight]).then(() => {
        if (this.#epoch !== epoch || this.#socket !== socket || !this.#usable()) return
        this.#phase = 'reset-ack'
        this.#send({ type: 'live_reset_ack', id, nonce })
      }).catch(() => { if (this.#epoch === epoch) this.#disconnect(1006) })
      return
    }
    if (frame.type === 'rest_ready') {
      const expected = channels[this.path]
      const advertised: unknown[] | null = Array.isArray(frame.channels) ? frame.channels : null
      if (this.#phase !== 'reset-ack' || !isId(frame.connection_id) ||
          (this.path !== '/ws/live' && (!advertised || advertised.length !== expected.length ||
            !expected.every(channel => advertised.includes(channel))))) throw new Error('Invalid connection')
      this.#connection = frame.connection_id
      this.#phase = 'ready'
      this.#retry = 0
      clearTimeout(this.#handshakeTimer)
      this.hooks.status('ready')
      return
    }
    if (!this.ready) return
    if (frame.type === 'retire' || frame.type === 'rest_retire' || frame.type === 'delete') {
      const channel = frame.type === 'retire' ? 'live' : frame.channel ?? (this.path === '/ws/live' ? 'conversations' : null)
      if (!channels[this.path].includes(channel as Channel)) throw new Error('Wrong retirement channel')
      this.#retire(channel as Channel, payload)
      return
    }
    if (this.path !== '/ws/live' || typeof frame.type !== 'string' || !liveTypes.has(frame.type)) return
    const capability = receipt(frame.receipt)
    const end = deadline(frame.expires_at)
    if (!payload || !capability || !end || !isId(frame.event_id)) throw new Error('Invalid live event')
    const row: LiveRow = {
      type: frame.type, ts: bounded(frame.ts, 32),
      title: bounded(payload.title ?? payload.intent ?? payload.endpoint_id, 512),
      text: bounded(payload.stt ?? payload.text ?? payload.body ?? payload.tts, 8192),
      zone: bounded(payload.zone, 512),
    }
    this.#replace('live', { ...end, id: frame.event_id, receipt: capability, data: row })
  }

  #retire(channel: Channel, payload?: Record<string, unknown>) {
    const ids: unknown = payload?.event_ids
    const capabilities: unknown = payload?.receipts ?? []
    if (!Array.isArray(ids) || !ids.length || ids.length > 32 || !ids.every(isId) ||
        !Array.isArray(capabilities) || capabilities.length > ids.length) throw new Error('Invalid retirement')
    const acknowledgements = capabilities.map(item => {
      const capability = receipt(item)
      const eventId = record(item)?.event_id
      if (!capability || !isId(eventId) || !ids.includes(eventId)) throw new Error('Invalid retirement receipt')
      return capability
    })
    for (const id of ids) this.#retired.add(id)
    this.#reads.get(channel)?.abort()
    this.#reads.delete(channel)
    const current = this.#entries.get(channel)
    if (current && ids.includes(current.id)) {
      this.#entries.delete(channel)
      this.hooks.changed(channel, null)
      acknowledgements.push(current.receipt)
    }
    this.hooks.retired?.(channel)
    this.#ack(acknowledgements)
    this.#armExpiry()
    // An overlong tombstone list must not permit an old reply to reappear.
    if (this.#retired.size > 256) this.#disconnect(1006)
  }

  discard(channel: Channel) {
    this.#reads.get(channel)?.abort()
    this.#reads.delete(channel)
    this.#replace(channel, null)
  }

  #replace(channel: Channel, next: Entry | null) {
    const old = this.#entries.get(channel)
    const acknowledgements: Receipt[] = []
    if (next && (remaining(next) <= 0 || this.#retired.has(next.id))) {
      acknowledgements.push(next.receipt)
      next = null
    }
    if (old && next?.id === old.id) {
      next.wall = Math.min(old.wall, next.wall)
      next.mono = Math.min(old.mono, next.mono)
    }
    if (old && old.receipt.ticket !== next?.receipt.ticket) acknowledgements.push(old.receipt)
    if (next) this.#entries.set(channel, next)
    else this.#entries.delete(channel)
    this.hooks.changed(channel, next?.data ?? null)
    this.#ack(acknowledgements)
    this.#armExpiry()
  }

  #armExpiry() {
    clearTimeout(this.#expiryTimer)
    if (!this.#entries.size) return
    const delay = Math.min(...[...this.#entries.values()].map(remaining))
    this.#expiryTimer = setTimeout(() => this.expire(), Math.max(1, Math.min(delay, 2147483647)))
  }

  expire() {
    for (const [channel, entry] of this.#entries) {
      if (remaining(entry) <= 0) this.#replace(channel, null)
    }
    this.#armExpiry()
  }

  read<T>(channel: Channel, path: string, options: RequestOptions, project: Projection<T>,
    latestExpiry = Infinity): Promise<void> {
    if (!this.ready || !channels[this.path].includes(channel) || channel === 'live' || this.#reads.has(channel)) {
      return Promise.resolve()
    }
    const owner = new AbortController()
    const connection = this.#connection
    const started = clock()
    this.#reads.set(channel, owner)
    const load = (async () => {
      try {
        const value = record(await this.session.request(path, {
          ...options, headers: { ...options.headers, 'X-KZA-Live-Connection': connection! },
          signal: options.signal ? AbortSignal.any([owner.signal, options.signal]) : owner.signal,
        }))
        if (owner.signal.aborted || this.#connection !== connection || !this.ready) return
        const matches = (given: unknown) => given === channel || channel === 'conversations' && given === undefined
        if (!value || value.connection_id !== connection || !matches(value.channel)) throw new Error('Wrong connection')
        if (value.snapshot === null) { this.#replace(channel, null); return }
        const batch = record(value.snapshot)
        const capability = receipt(batch?.receipt)
        if (!batch || !isId(batch.event_id) || !capability) throw new Error('Invalid snapshot')
        const end = deadline(batch.expires_at, batch.observed_at, started)
        try {
          if (!end || !matches(batch.channel) || !Array.isArray(batch.rows) || batch.rows.length > 50 ||
              typeof batch.expires_at !== 'number' || batch.expires_at > latestExpiry) throw new Error('Invalid snapshot')
          if (this.#retired.has(batch.event_id) || remaining(end) <= 0) {
            this.#ack([capability])
            return
          }
          const rowDeadline = Math.min(...batch.rows.flatMap(row => {
            const fields = record(row)
            return [fields?.expires_at, fields?.group_expires_at].filter((value): value is number => typeof value === 'number')
          }))
          const boundedEnd = deadline(Math.min(batch.expires_at, rowDeadline), batch.observed_at, started)
          if (!boundedEnd || remaining(boundedEnd) <= 0) { this.#ack([capability]); return }
          this.#replace(channel, { ...boundedEnd, id: batch.event_id, receipt: capability, data: project(batch.rows) })
        } catch (error) {
          this.#ack([capability])
          throw error
        }
      } catch (error) {
        if (owner.signal.aborted) return
        this.#replace(channel, null)
        throw error
      } finally {
        if (this.#reads.get(channel) === owner) this.#reads.delete(channel)
      }
    })()
    const settled = load.then(() => {}, () => {})
    this.#inflight.add(settled)
    void settled.then(() => this.#inflight.delete(settled))
    return load
  }
}
