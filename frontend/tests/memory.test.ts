import { mount, unmount } from 'svelte'
import { afterEach, describe, expect, it, vi } from 'vitest'
import Memory from '../src/nexo/Memory.svelte'
import { Session } from '../src/lib/session'
import { clickButton, identity, json, transport } from './helpers'

const disposers: (() => Promise<void>)[] = []
afterEach(async () => { for (const d of disposers.splice(0)) await d() })

describe('Memory view contract', () => {
  it('acepta fact_id y ausencia de expires_at en hechos y propuestas', async () => {
    const posts: string[] = []
    const who = identity(['memory:read', 'memory:review'], 'operator')
    const fetcher = transport((path, init) => {
      if (path === '/api/admin/auth/whoami') return who()
      if (path === '/api/memory') {
        return json({
          facts: [{ fact_id: 'fact_abc123', category: 'Hogar', content: 'El portón cierra suave.' }],
          proposals: [{ fact_id: 'prop_xyz789', category: 'Rutina', content: 'Cena a las 21h.' }],
          facts_next: null,
          proposals_next: null,
        })
      }
      if (init.method === 'POST') {
        posts.push(path)
        return json({ ok: true })
      }
      return json({})
    })
    const session = new Session(fetcher)
    await session.start()
    const target = document.createElement('div')
    document.body.append(target)
    const mounted = mount(Memory, { target, props: { session, personal: false } })
    disposers.push(async () => { await unmount(mounted); target.remove(); session.dispose() })

    await vi.waitFor(() => expect(target.textContent).toContain('El portón cierra suave.'))
    expect(target.textContent).toContain('Cena a las 21h.')
    clickButton(target, 'Conservar')
    await vi.waitFor(() => expect(posts).toEqual(['/api/memory/proposals/prop_xyz789/confirm']))
  })
})
