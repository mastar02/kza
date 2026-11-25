import { afterEach, describe, expect, it, vi } from 'vitest'
import { Session } from '../src/lib/session'
import { ReceiptSocket } from '../src/lib/ws'
import type { Channel, SocketPath } from '../src/lib/ws'
import { deferred, identity, json, transport } from './helpers'

const ids = {
  reset: '00000000-0000-4000-8000-000000000001',
  nonce: '00000000-0000-4000-8000-000000000002',
  connection: '00000000-0000-4000-8000-000000000003',
  event: '00000000-0000-4000-8000-000000000004',
  ticket: '00000000-0000-4000-8000-000000000005',
}
const capability = { ticket: ids.ticket, nonce: ids.nonce }
const reset = { type: 'live_reset', ts: '', payload: { id: ids.reset, nonce: ids.nonce } }
const connected = { type: 'rest_ready', ts: '', connection_id: ids.connection }
const live = () => ({ type: 'turn', ts: '', event_id: ids.event, expires_at: Date.now() / 1000 + 120,
  receipt: capability, payload: { stt: 'transcripción de prueba' } })

class FakeSocket extends EventTarget {
  readyState = 0
  sent: Record<string, unknown>[] = []
  send(value: string) { this.sent.push(JSON.parse(value)) }
  open() { this.readyState = 1; this.dispatchEvent(new Event('open')) }
  frame(value: unknown) { this.dispatchEvent(new MessageEvent('message', { data: JSON.stringify(value) })) }
  close(code = 1000) { this.readyState = 3; this.dispatchEvent(new CloseEvent('close', { code })) }
}

const disposers: (() => void)[] = []
afterEach(() => {
  for (const dispose of disposers.splice(0)) dispose()
  vi.useRealTimers()
})

async function fixture(path: SocketPath, fetcher?: typeof fetch, committed: () => Promise<void> = async () => {}) {
  const isEvaluation = path === '/ws/evaluation'
  const permissions = path === '/ws/live' ? ['conversations:read'] :
    isEvaluation ? ['evaluation:read'] : ['voices:read', 'voices:review']
  const whoami = identity(permissions, isEvaluation ? 'evaluator' : 'operator')
  const session = new Session(fetcher ?? transport(endpoint => {
    expect(endpoint).toBe('/api/admin/auth/whoami')
    return whoami()
  }))
  await session.start()
  disposers.push(() => session.dispose())
  const sockets: FakeSocket[] = []
  const socketFactory = vi.fn((url: string) => {
    expect(new URL(url).pathname).toBe(path)
    const next = new FakeSocket()
    sockets.push(next)
    return next as unknown as WebSocket
  })
  const contents = new Map<Channel, unknown>()
  const client = new ReceiptSocket(session, path, {
    status() {},
    changed(channel, data) { if (data === null) contents.delete(channel); else contents.set(channel, data) },
    committed,
  }, socketFactory)
  const socket = sockets[0]!
  socket.open()
  return { session, client, socket, sockets, contents, socketFactory }
}

describe('WebSocket protocol 3', () => {
  it('sends hello and acknowledges reset and retire only after the view has cleared', async () => {
    let commit = deferred<void>()
    const { socket, contents, client } = await fixture('/ws/live', undefined, () => commit.promise)
    expect(socket.sent[0]).toMatchObject({ type: 'live_hello', version: 3, client_id: expect.any(String) })
    socket.frame(reset)
    expect(client.ready).toBe(false)
    expect(socket.sent).toHaveLength(1)
    commit.resolve()
    await vi.waitFor(() => expect(socket.sent[1]).toEqual({ type: 'live_reset_ack', id: ids.reset, nonce: ids.nonce }))
    socket.frame(connected)
    expect(client.ready).toBe(true)
    socket.frame(live())
    expect(contents.get('live')).toMatchObject({ text: 'transcripción de prueba' })
    expect(socket.sent).toHaveLength(2)
    commit = deferred<void>()
    socket.frame({ type: 'retire', payload: { event_ids: [ids.event],
      receipts: [{ event_id: ids.event, ...capability }] } })
    expect(contents.size).toBe(0)
    expect(socket.sent).toHaveLength(2)
    commit.resolve()
    await vi.waitFor(() => expect(socket.sent[2]).toEqual({ type: 'live_ack', receipts: [capability] }))
    socket.frame(live())
    expect(contents.size).toBe(0)
  })

  it('purga todo y no reabre una conexión cerrada con 1008', async () => {
    const { session, client, socket, contents, socketFactory } = await fixture('/ws/live')
    socket.frame(reset)
    await vi.waitFor(() => expect(socket.sent).toHaveLength(2))
    socket.frame(connected)
    socket.frame(live())
    expect(contents.size).toBe(1)
    const purge = vi.fn()
    session.onPurge(purge)
    vi.useFakeTimers()
    socket.close(1008)
    expect(contents.size).toBe(0)
    expect(client.ready).toBe(false)
    expect(session.state.status).toBe('login')
    expect(purge).toHaveBeenCalledOnce()
    await vi.advanceTimersByTimeAsync(60000)
    expect(socketFactory).toHaveBeenCalledOnce()
  })

  it('clears payloads at their lease deadline and reconnects a bounded number of times', async () => {
    const { socket, sockets, socketFactory, contents } = await fixture('/ws/live')
    socket.frame(reset)
    await vi.waitFor(() => expect(socket.sent).toHaveLength(2))
    socket.frame(connected)
    vi.useFakeTimers()
    socket.frame({ ...live(), expires_at: Date.now() / 1000 + .01 })
    expect(contents.size).toBe(1)
    await vi.advanceTimersByTimeAsync(20)
    expect(contents.size).toBe(0)
    expect(socket.sent.at(-1)).toEqual({ type: 'live_ack', receipts: [capability] })
    for (const wait of [500, 1500, 3500, 7000, 10000]) {
      const last = sockets.at(-1)!
      last.close(1006)
      await vi.advanceTimersByTimeAsync(wait)
      sockets.at(-1)!.open()
    }
    expect(socketFactory).toHaveBeenCalledTimes(5)
    expect(sockets[1]?.sent[0]?.client_id).toBe(sockets[0]?.sent[0]?.client_id)
  })

  it('restores the retry budget after the recovered socket completes the v3 handshake', async () => {
    vi.useFakeTimers()
    const { client, socket, sockets, socketFactory } = await fixture('/ws/live')
    socket.close(1006)
    await vi.advanceTimersByTimeAsync(500)
    const recovered = sockets[1]!
    recovered.open()
    recovered.frame(reset)
    await vi.advanceTimersByTimeAsync(0)
    expect(recovered.sent.at(-1)).toMatchObject({ type: 'live_reset_ack' })
    recovered.frame(connected)
    expect(client.ready).toBe(true)

    recovered.close(1006)
    await vi.advanceTimersByTimeAsync(499)
    expect(socketFactory).toHaveBeenCalledTimes(2)
    await vi.advanceTimersByTimeAsync(1)
    expect(socketFactory).toHaveBeenCalledTimes(3)
  })

  it.each([
    ['/ws/voices', 'voices:profiles', ['voices:captures', 'voices:profiles', 'voices:review'], ['voices:read', 'voices:review'], 'operator'],
    ['/ws/evaluation', 'evaluation:sample', ['evaluation:sample'], ['evaluation:read'], 'evaluator'],
  ] as const)('scopes REST snapshots to %s and discards a response retired while in flight', async (path, channel, channels, permissions, role) => {
    const response = deferred<Response>()
    const whoami = identity([...permissions], role)
    const fetcher = transport((endpoint, options) => {
      if (endpoint === '/api/admin/auth/whoami') return whoami()
      expect(options.headers).toMatchObject({ 'X-KZA-Live-Connection': ids.connection })
      expect(options.cache).toBe('no-store')
      return response.promise
    })
    const { client, socket, contents } = await fixture(path, fetcher)
    socket.frame(reset)
    await vi.waitFor(() => expect(socket.sent).toHaveLength(2))
    socket.frame({ ...connected, channels })
    const project = vi.fn(items => items)
    const read = client.read(channel, '/api/test', { permissions }, project)
    expect(fetcher.mock.calls).toHaveLength(2)
    socket.frame({ type: 'rest_retire', channel, ts: '', payload: { event_ids: [ids.event],
      receipts: [{ event_id: ids.event, ...capability }] } })
    response.resolve(json({ connection_id: ids.connection, channel, snapshot: {
      channel, event_id: ids.event, receipt: capability, rows: [{ text: 'private' }],
      observed_at: Date.now() / 1000 - 1, expires_at: Date.now() / 1000 + 60,
    } }))
    await read
    expect(project).not.toHaveBeenCalled()
    expect(contents.size).toBe(0)
    await vi.waitFor(() => expect(socket.sent.at(-1)).toEqual({ type: 'live_ack', receipts: [capability] }))
  })
})
