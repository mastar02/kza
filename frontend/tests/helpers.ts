import { vi } from 'vitest'
import type { Role } from '../src/lib/session'

export function json(value: unknown, status = 200): Response {
  return new Response(JSON.stringify(value), { status, headers: { 'Content-Type': 'application/json' } })
}

export function identity(permissions: string[], role: Role = 'home') {
  const expires_at = Date.now() / 1000 + 3600
  return () => json({ ok: true, role, permissions, expires_at, subject: 'fixture' })
}

export function transport(handler: (path: string, options: RequestInit) => Response | Promise<Response>) {
  return vi.fn(async (input: RequestInfo | URL, options: RequestInit = {}) => handler(String(input), options))
}

export function deferred<T>() {
  let resolve!: (value: T | PromiseLike<T>) => void
  const promise = new Promise<T>(done => { resolve = done })
  return { promise, resolve }
}

export function clickButton(target: HTMLElement, label: string) {
  const button = [...target.querySelectorAll('button')].find(candidate => candidate.textContent?.trim() === label)
  if (!button) throw new Error('Button missing: ' + label)
  button.click()
}

export function devices() {
  return {
    rooms: [{ room_id: 'laboratory', name: 'Habitación de prueba', entity_ids: ['light.fixture'] }],
    entities: [{ entity_id: 'light.fixture', domain: 'light', name: 'Luz de prueba', state: 'off', available: true,
      capabilities: ['turn_on', 'brightness', 'restart'], control_limits: { brightness_min: 20, brightness_max: 80 } }],
    observed_at: Date.now() / 1000 - 0.1, stale: false,
  }
}
