import { messageFor } from './session'
import type { RequestOptions, Session } from './session'

/** Serializes writes within a panel and aborts their client side on retirement. */
export class Mutation {
  busy = $state(false)
  feedback = $state('')
  #owner: AbortController | null = null
  #unregister: () => void

  constructor(private readonly session: Session) {
    this.#unregister = session.onPurge(() => this.clear())
  }

  async run(path: string, options: RequestOptions, refresh: () => Promise<void>, describe?: (value: unknown) => string) {
    if (this.#owner) return false
    const owner = new AbortController()
    this.#owner = owner
    this.busy = true
    this.feedback = ''
    let completed = false
    try {
      if (!await this.session.revalidate() || owner.signal.aborted) return false
      const response = await this.session.request(path, { ...options, signal: owner.signal })
      if (!owner.signal.aborted) {
        this.feedback = describe?.(response) ?? 'El servidor recibió la solicitud. Consultá el estado actualizado.'
        completed = true
      }
    } catch (error) {
      if (!owner.signal.aborted) this.feedback = messageFor(error)
    } finally {
      if (!owner.signal.aborted) await refresh()
      if (this.#owner === owner) {
        this.#owner = null
        this.busy = false
      }
    }
    return completed
  }

  clear() {
    this.#owner?.abort()
    this.#owner = null
    this.feedback = ''
    this.busy = false
  }

  destroy() { this.clear(); this.#unregister() }
}
