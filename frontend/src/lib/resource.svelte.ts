import type { Session } from './session'
import { messageFor, record } from './session'

/** A view-owned read with abort and purge on navigation or session retirement. */
export class Resource<T> {
  data = $state.raw<T | null>(null)
  error = $state('')
  busy = $state(false)
  #owner: AbortController | null = null
  #destroyed = false
  #unregister: () => void

  constructor(session: Session, private readonly read: (signal: AbortSignal) => Promise<T>) {
    this.#unregister = session.onPurge(() => this.clear())
  }

  async load(keepData = false) {
    if (this.#owner || this.#destroyed) return
    const owner = new AbortController()
    this.#owner = owner
    this.busy = true
    this.error = ''
    if (!keepData) this.data = null
    try {
      const data = await this.read(owner.signal)
      if (!owner.signal.aborted) this.data = data
    } catch (error) {
      if (!owner.signal.aborted) {
        this.data = null
        this.error = messageFor(error)
      }
    } finally {
      if (this.#owner === owner) {
        this.busy = false
        this.#owner = null
      }
    }
  }

  clear() {
    this.#owner?.abort()
    this.#owner = null
    this.data = null
    this.error = ''
    this.busy = false
  }

  destroy() {
    this.#destroyed = true
    this.clear()
    this.#unregister()
  }
}

export function rows(value: unknown, key?: string): Record<string, unknown>[] {
  const source = key ? record(value)?.[key] : value
  if (!Array.isArray(source)) return []
  return source.flatMap(item => {
    const row = record(item)
    return row ? [row] : []
  })
}


export function text(value: unknown, fallback = '—'): string {
  return typeof value === 'string' && value ? value : fallback
}
