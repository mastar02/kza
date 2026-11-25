import { HttpError, record } from './session'
import type { Session } from './session'

const operations = {
  turn_on: 'Encender', turn_off: 'Apagar', activate_scene: 'Activar escena',
  brightness: 'Brillo (%)', color_temp_kelvin: 'Temperatura de color (K)',
  set_temperature: 'Temperatura objetivo (°C)', set_hvac_mode: 'Modo del aire',
  open_cover: 'Abrir', close_cover: 'Cerrar', stop_cover: 'Detener', set_cover_position: 'Posición (% abierto)',
} as const

export type Operation = keyof typeof operations
type BareOperation = 'turn_on' | 'turn_off' | 'activate_scene' | 'open_cover' | 'close_cover' | 'stop_cover'

export type HomeCommand =
  | { operation: BareOperation }
  | { operation: 'brightness'; brightness_pct: number }
  | { operation: 'color_temp_kelvin'; color_temp_kelvin: number }
  | { operation: 'set_temperature'; temperature: number }
  | { operation: 'set_hvac_mode'; hvac_mode: string }
  | { operation: 'set_cover_position'; position: number }

export type Entity = {
  id: string; name: string; domain: string; state: string; available: boolean
  capabilities: Operation[]; limits: Record<string, unknown>; observation: unknown
}
type Room = { id: string; name: string; entityIds: string[]; temperature: number | null }
export type Catalog = { rooms: Room[]; entities: Entity[]; observation: unknown; observed: unknown; stale: boolean }

const byDomain: Record<string, readonly Operation[]> = {
  light: ['turn_on', 'turn_off', 'brightness', 'color_temp_kelvin'],
  switch: ['turn_on', 'turn_off'],
  climate: ['turn_on', 'turn_off', 'set_temperature', 'set_hvac_mode'],
  cover: ['open_cover', 'close_cover', 'stop_cover', 'set_cover_position'],
  scene: ['activate_scene'],
}

const knownStates: Record<string, readonly string[]> = {
  light: ['on', 'off'], switch: ['on', 'off'], cover: ['open', 'closed', 'opening', 'closing'],
  climate: ['off', 'cool', 'heat', 'dry', 'fan_only', 'auto', 'heat_cool'],
}

function entries(value: unknown): unknown[] { return Array.isArray(value) ? value : [] }
function strings(value: unknown): string[] {
  return entries(value).filter((item): item is string => typeof item === 'string')
}

export function catalogFrom(value: unknown): Catalog {
  const doc = record(value)
  if (!doc || !Array.isArray(doc.rooms) || !Array.isArray(doc.entities)) throw new Error('Invalid home catalog')
  const entityMap = new Map<string, Entity>()
  for (const raw of [...doc.entities, ...entries(doc.unassigned_entities)]) {
    const row = record(raw)
    if (typeof row?.entity_id !== 'string' || typeof row.domain !== 'string' ||
        !row.entity_id.startsWith(row.domain + '.')) continue
    const accepted = byDomain[row.domain] ?? []
    entityMap.set(row.entity_id, {
      id: row.entity_id, domain: row.domain,
      name: typeof row.name === 'string' && row.name ? row.name : row.entity_id,
      state: typeof row.state === 'string' ? row.state : '',
      available: row.available === true, observation: row.observation,
      capabilities: strings(row.capabilities).filter((op): op is Operation => accepted.includes(op as Operation)),
      limits: record(row.control_limits) ?? {},
    })
  }
  const rooms = doc.rooms.flatMap(raw => {
    const row = record(raw)
    if (typeof row?.room_id !== 'string' || typeof row.name !== 'string') return []
    return [{ id: row.room_id, name: row.name, entityIds: strings(row.entity_ids),
      temperature: typeof row.current_temperature === 'number' ? row.current_temperature : null }]
  })
  return { rooms, entities: [...entityMap.values()], stale: doc.stale !== false,
    observation: doc.observation, observed: doc.observed_at }
}

function millis(value: unknown): number {
  return typeof value === 'number' ? value * 1000 : typeof value === 'string' ? Date.parse(value) : NaN
}

function fresh(value: unknown, now: number): boolean {
  const observation = record(value)
  return (observation?.state === 'fresh' || observation?.state === 'partial') &&
    millis(observation.observed_at) <= now && now < millis(observation.expires_at)
}

export function canControl(session: Session, catalog: Catalog, entity: Entity, operation: Operation, now = Date.now()): boolean {
  const catalogFresh = catalog.observation != null ? fresh(catalog.observation, now)
    : millis(catalog.observed) <= now && now - millis(catalog.observed) < 6000
  const stateKnown = entity.domain === 'scene' ? entity.state !== 'unavailable'
    : knownStates[entity.domain]?.includes(entity.state) === true
  return session.can('home:read', 'home:control') && !catalog.stale && catalogFresh &&
    entity.available && stateKnown && (entity.observation == null || fresh(entity.observation, now)) &&
    byDomain[entity.domain]?.includes(operation) === true && entity.capabilities.includes(operation)
}

type Control = { operation: Operation; label: string } & (
  | { kind: 'button' }
  | { kind: 'number'; min: number; max: number; step: number }
  | { kind: 'select'; options: string[] }
)

export function controlFor(entity: Entity, operation: Operation): Control | null {
  const label = operations[operation]
  const limits = entity.limits
  const range = (min: unknown, max: unknown, step: unknown = 1): Control | null =>
    typeof min === 'number' && Number.isFinite(min) && typeof max === 'number' && Number.isFinite(max) && min <= max &&
    typeof step === 'number' && Number.isFinite(step) && step > 0
      ? { operation, label, kind: 'number', min, max, step } : null
  switch (operation) {
    case 'brightness': return range(limits.brightness_min ?? 0, limits.brightness_max ?? 100)
    case 'color_temp_kelvin': return range(limits.color_temp_min, limits.color_temp_max)
    case 'set_temperature': return range(limits.temperature_min, limits.temperature_max, limits.temperature_step)
    case 'set_cover_position': return range(limits.position_min, limits.position_max)
    case 'set_hvac_mode': {
      const options = strings(limits.hvac_modes)
      return options.length ? { operation, label, kind: 'select', options } : null
    }
    default: return { operation, label, kind: 'button' }
  }
}

export function commandFor(entity: Entity, operation: Operation, raw?: string): HomeCommand | null {
  const control = controlFor(entity, operation)
  if (!control) return null
  if (control.kind === 'button') return { operation: operation as BareOperation }
  if (control.kind === 'select') {
    return raw && control.options.includes(raw) ? { operation: 'set_hvac_mode', hvac_mode: raw } : null
  }
  if (!raw?.trim()) return null
  const value = Number(raw)
  const steps = (value - control.min) / control.step
  if (!Number.isFinite(value) || value < control.min || value > control.max ||
      Math.abs(steps - Math.round(steps)) > 1e-7) return null
  switch (operation) {
    case 'brightness': return Number.isInteger(value) && value >= 0 && value <= 100 ? { operation, brightness_pct: value } : null
    case 'color_temp_kelvin': return Number.isInteger(value) && value > 0 ? { operation, color_temp_kelvin: value } : null
    case 'set_temperature': return { operation, temperature: value }
    case 'set_cover_position': return Number.isInteger(value) && value >= 0 && value <= 100 ? { operation, position: value } : null
    default: return null
  }
}

export type ActionStatus = 'in_progress' | 'confirmed' | 'failed' | 'uncertain' | 'missing'
export type ActionReceipt = { request_id: string; entity_id: string; status: ActionStatus }
export const actionLabel: Record<ActionStatus, string> = {
  in_progress: 'En curso', confirmed: 'Confirmado por Home Assistant', failed: 'Falló',
  uncertain: 'Sin confirmar', missing: 'Recibo todavía no disponible',
}

export async function readReceipt(session: Session, expected: ActionReceipt): Promise<ActionReceipt> {
  try {
    return receiptFrom(await session.request('/api/home/actions?request_id=' + encodeURIComponent(expected.request_id),
      { permissions: ['home:read'] }), expected)
  } catch (error) {
    if (!session.can('home:read')) throw error
    return { ...expected, status: error instanceof HttpError && error.status === 404 ? 'missing' : 'uncertain' }
  }
}

function receiptFrom(value: unknown, expected: ActionReceipt): ActionReceipt {
  const row = record(value)
  const status = row?.status
  if (row?.request_id !== expected.request_id || row.entity_id !== expected.entity_id ||
      (status !== 'in_progress' && status !== 'confirmed' && status !== 'failed' && status !== 'uncertain')) {
    throw new Error('Invalid home action receipt')
  }
  return { ...expected, status }
}

function commandValue(command: HomeCommand): string | undefined {
  switch (command.operation) {
    case 'brightness': return String(command.brightness_pct)
    case 'color_temp_kelvin': return String(command.color_temp_kelvin)
    case 'set_temperature': return String(command.temperature)
    case 'set_hvac_mode': return command.hvac_mode
    case 'set_cover_position': return String(command.position)
    default: return undefined
  }
}

export async function submitAction(session: Session, catalog: Catalog, entity: Entity, command: HomeCommand,
  requestId: string = crypto.randomUUID()): Promise<ActionReceipt> {
  const { operation } = command
  if (!canControl(session, catalog, entity, operation)) throw new HttpError(403)
  const body = commandFor(entity, operation, commandValue(command))
  if (!body || !/^[A-Za-z0-9_.:-]{1,128}$/.test(requestId)) throw new Error('Invalid home action')
  const expected: ActionReceipt = { request_id: requestId, entity_id: entity.id, status: 'in_progress' }
  try {
    const value = await session.request('/api/home/actions', {
      permissions: ['home:read', 'home:control'], method: 'POST',
      body: { ...body, entity_id: entity.id, request_id: requestId },
    })
    return receiptFrom(value, expected)
  } catch (error) {
    if (!session.can('home:read', 'home:control')) throw error
    return readReceipt(session, expected)
  }
}

export async function loadHomeAssets(session: Session, signal: AbortSignal) {
  const [model, plan] = await Promise.allSettled([
    session.homeAsset('model', signal), session.homeAsset('plan', signal),
  ])
  signal.throwIfAborted()
  if (!session.can('home:read')) return { model: null, plan: null }
  return { model: model.status === 'fulfilled' ? model.value : null,
    plan: plan.status === 'fulfilled' ? record(plan.value) ?? null : null }
}
