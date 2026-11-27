import type { Entity, Room } from '../lib/home'

export type FloorKind = 'porcelanato' | 'laminado' | 'banio' | 'balcon'

export type ScanRoomSpec = {
  name: string
  floor: FloorKind
  aliases: string[]
}

export type BoxSpec = [number, number, number, number, number, number]

export type WallPatchSpec = {
  id: string
  note?: string
  box: BoxSpec
}

export type StairsSpec = {
  room: string
  treadBottom: number
  wallTop: number
  pit: { x0: number; x1: number; z0: number; z1: number }
  bottom: number
  steps: number
  rise: number
  landing: [number, number][]
  wall: number
  rail: { box: BoxSpec }
}

export type BalconySpec = {
  room: string
  slab: { x: number; width: number; thickness: number; z: number; depth: number }
  rails: { id: string; centre: [number, number, number]; size: [number, number, number] }[]
}

export type TintSpec = {
  prefix: string
  color: number
  roughness?: number
  metalness?: number
}

export type HousePlan = {
  SCAN_ROOMS: Record<string, ScanRoomSpec>
  FLOOR_ROOM: Record<string, string>
  MAIN_FLOOR: Record<string, string>
  DEFAULT_YAW: number
  CUT_HEIGHT: number
  WALL_PATCHES: WallPatchSpec[]
  SPLIT: { id: string; box: BoxSpec } | null
  STAIRS: StairsSpec | null
  BALCONY: BalconySpec | null
  HIDDEN_PREFIXES: string[]
  HIDDEN_NAMES: string[]
  DOOR_PREFIX: string
  WALL_GROUP: string
  GLASS: { prefixes: string[]; names: string[] }
  TINTS: TintSpec[]
  LIGHT_CABINETS: string[]
}

function isObject(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
}

export function parseHousePlan(raw: Record<string, unknown> | null): HousePlan | null {
  if (!isObject(raw) || !isObject(raw.SCAN_ROOMS) || !isObject(raw.FLOOR_ROOM)) return null
  if (!Object.keys(raw.SCAN_ROOMS).length || !Object.keys(raw.FLOOR_ROOM).length) return null
  return {
    SCAN_ROOMS: raw.SCAN_ROOMS as Record<string, ScanRoomSpec>,
    FLOOR_ROOM: raw.FLOOR_ROOM as Record<string, string>,
    MAIN_FLOOR: isObject(raw.MAIN_FLOOR) ? (raw.MAIN_FLOOR as Record<string, string>) : {},
    DEFAULT_YAW: typeof raw.DEFAULT_YAW === 'number' && Number.isFinite(raw.DEFAULT_YAW) ? raw.DEFAULT_YAW : 0.68,
    CUT_HEIGHT: typeof raw.CUT_HEIGHT === 'number' && Number.isFinite(raw.CUT_HEIGHT) ? raw.CUT_HEIGHT : 1.05,
    WALL_PATCHES: Array.isArray(raw.WALL_PATCHES) ? (raw.WALL_PATCHES as WallPatchSpec[]) : [],
    SPLIT: isObject(raw.SPLIT) ? (raw.SPLIT as { id: string; box: BoxSpec }) : null,
    STAIRS: isObject(raw.STAIRS) ? (raw.STAIRS as StairsSpec) : null,
    BALCONY: isObject(raw.BALCONY) ? (raw.BALCONY as BalconySpec) : null,
    HIDDEN_PREFIXES: Array.isArray(raw.HIDDEN_PREFIXES) ? (raw.HIDDEN_PREFIXES as string[]) : [],
    HIDDEN_NAMES: Array.isArray(raw.HIDDEN_NAMES) ? (raw.HIDDEN_NAMES as string[]) : [],
    DOOR_PREFIX: typeof raw.DOOR_PREFIX === 'string' ? raw.DOOR_PREFIX : 'Door',
    WALL_GROUP: typeof raw.WALL_GROUP === 'string' ? raw.WALL_GROUP : 'Walls',
    GLASS: isObject(raw.GLASS) ? (raw.GLASS as { prefixes: string[]; names: string[] }) : { prefixes: ['Window'], names: [] },
    TINTS: Array.isArray(raw.TINTS) ? (raw.TINTS as TintSpec[]) : [],
    LIGHT_CABINETS: Array.isArray(raw.LIGHT_CABINETS) ? (raw.LIGHT_CABINETS as string[]) : [],
  }
}

export function normalizeRoomText(value: unknown): string {
  return String(value ?? '')
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, ' ')
    .trim()
}

export function matchRooms(
  rooms: readonly Pick<Room, 'id' | 'name'>[],
  scanRooms: Record<string, ScanRoomSpec>,
): Map<string, string> {
  const aliases = Object.fromEntries(
    Object.entries(scanRooms).map(([id, room]) => [
      id,
      new Set([id, room.name, ...(room.aliases ?? [])].map(normalizeRoomText)),
    ]),
  )
  const bound = new Map<string, string>()
  const taken = new Set<string>()
  const claim = (roomId: string, scan: string | undefined) => {
    if (scan && !taken.has(scan)) {
      bound.set(roomId, scan)
      taken.add(scan)
    }
  }
  const byAlias = (text: string) =>
    Object.keys(aliases).find(scan => aliases[scan]?.has(text))
  for (const room of rooms) claim(room.id, byAlias(normalizeRoomText(room.id)))
  for (const room of rooms) {
    if (!bound.has(room.id)) claim(room.id, byAlias(normalizeRoomText(room.name)))
  }
  for (const room of rooms) {
    if (bound.has(room.id)) continue
    const words = normalizeRoomText(room.name).split(' ')
    claim(room.id, Object.keys(aliases).find(scan => words.some(word => aliases[scan]?.has(word))))
  }
  return bound
}

export type RoomReading = {
  fraction: number
  color: number
  actual: string | null
  active: boolean
  status: string
}

const THERMAL_ACTIONS: Record<string, [string, number]> = {
  cooling: ['Enfriando', 0x71bdff],
  heating: ['Calefaccionando', 0xffad77],
  fan: ['Ventilando', 0x62e8cd],
  drying: ['Deshumidificando', 0xb5a6ff],
  preheating: ['Precalentando', 0xffad77],
  defrosting: ['Descongelando', 0xffad77],
}

const THERMAL_MODES: Record<string, [string, number]> = {
  cool: ['Modo frío', 0x71bdff],
  heat: ['Modo calor', 0xffad77],
  fan_only: ['Ventilando', 0x62e8cd],
  dry: ['Modo seco', 0xb5a6ff],
  auto: ['Modo automático', 0x62e8cd],
  heat_cool: ['Modo frío y calor', 0x62e8cd],
}

const NEUTRAL_THERMAL = 0x8097a8

function degreesLabel(value: number | null | undefined): string | null {
  return typeof value === 'number' && Number.isFinite(value) ? `${value} °C` : null
}

export function lightReading(room: Room, entities: readonly Entity[]): RoomReading {
  const lights = entities.filter(entity => entity.domain === 'light' && entity.available && entity.state === 'on')
  const primary = lights.find(entity => entity.id === room.defaultLight) ?? lights[0]
  const kelvin = primary?.colorTempKelvin ?? null
  const pct = lights.length
    ? Math.max(
        ...lights.map(entity =>
          typeof entity.brightnessPct === 'number' && Number.isFinite(entity.brightnessPct)
            ? Math.max(0, Math.min(100, entity.brightnessPct))
            : 100,
        ),
      )
    : 0
  const color = kelvin !== null && kelvin > 0 && kelvin < 3300
    ? 0xffc885
    : kelvin !== null && kelvin >= 4800
      ? 0xadd5ff
      : 0xf1f1ff
  return { fraction: pct / 100, color, actual: degreesLabel(room.temperature), active: lights.length > 0, status: '' }
}

export function thermalReading(room: Room, entities: readonly Entity[]): RoomReading {
  const units = entities.filter(entity => entity.domain === 'climate')
  const unit = units.find(entity => entity.id === room.defaultClimate) ?? (units.length === 1 ? units[0] : null)
  const actual = degreesLabel(room.temperature ?? unit?.currentTemperature)
  if (!unit?.available || unit.state === 'unknown' || unit.state === 'unavailable') {
    return { fraction: 0, actual, active: false, status: unit ? 'Estado desconocido' : '', color: NEUTRAL_THERMAL }
  }
  const hvacAction = unit.hvacAction ?? ''
  const effect = (hvacAction && THERMAL_ACTIONS[hvacAction])
    || (!hvacAction ? THERMAL_MODES[unit.state] : undefined)
    || null
  const active = unit.state !== 'off' && effect !== null
  return {
    fraction: 0,
    actual,
    active,
    status: active && effect ? effect[0] : unit.state === 'off' || hvacAction === 'off' ? 'Apagado' : 'En pausa',
    color: active && effect ? effect[1] : NEUTRAL_THERMAL,
  }
}
