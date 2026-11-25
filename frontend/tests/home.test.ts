import { afterEach, describe, expect, it } from 'vitest'
import { canControl, catalogFrom, commandFor, readReceipt, submitAction } from '../src/lib/home'
import { Session } from '../src/lib/session'
import { devices, identity, json, transport } from './helpers'

const sessions: Session[] = []
afterEach(() => { for (const session of sessions.splice(0)) session.dispose() })

describe('typed home operations', () => {
  it('negates actions without a grant, capability, or current observation', async () => {
    const fetcher = transport(() => identity(['home:read'])())
    const session = new Session(fetcher)
    sessions.push(session)
    await session.start()
    const catalog = catalogFrom(devices())
    const entity = catalog.entities[0]!
    expect(canControl(session, catalog, entity, 'turn_on')).toBe(false)
    await expect(submitAction(session, catalog, entity, { operation: 'turn_on' }, 'test-id')).rejects.toMatchObject({ status: 403 })
    expect(fetcher.mock.calls.map(([url]) => url)).toEqual(['/api/admin/auth/whoami'])
  })

  it('admits only the declared parameters and confirms the original receipt', async () => {
    const home = identity(['home:read', 'home:control'])
    const receipt = { request_id: 'test-id', entity_id: 'light.fixture', status: 'confirmed' } as const
    const fetcher = transport((path, init) => {
      if (path === '/api/admin/auth/whoami') return home()
      if (init.method === 'POST') {
        expect(path).toBe('/api/home/actions')
        expect(JSON.parse(String(init.body))).toEqual({
          entity_id: 'light.fixture', operation: 'brightness', brightness_pct: 40, request_id: 'test-id',
        })
        return json({ ...receipt, request_id: 'wrong-id' })
      }
      expect(path).toBe('/api/home/actions?request_id=test-id')
      return json(receipt)
    })
    const session = new Session(fetcher)
    sessions.push(session)
    await session.start()
    const catalog = catalogFrom(devices())
    const entity = catalog.entities[0]!
    expect(entity.capabilities).not.toContain('restart')
    expect(canControl(session, catalog, entity, 'brightness', Date.now() + 7000)).toBe(false)
    expect(canControl(session, catalog, entity, 'activate_scene')).toBe(false)
    for (const value of ['', '5', '81', '40.5', 'Infinity']) expect(commandFor(entity, 'brightness', value)).toBeNull()
    const command = commandFor(entity, 'brightness', '40')
    expect(command).toEqual({ operation: 'brightness', brightness_pct: 40 })
    const withExtraFields = { ...command!, role: 'operator', request_id: 'other' }
    expect(await submitAction(session, catalog, entity, withExtraFields, 'test-id')).toEqual(receipt)
    expect(await readReceipt(session, receipt)).toEqual(receipt)
    expect(fetcher.mock.calls.filter(([, init]) => init?.method === 'POST')).toHaveLength(1)
  })
})
