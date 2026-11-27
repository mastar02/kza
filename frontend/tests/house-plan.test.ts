import { describe, expect, it } from 'vitest'
import { lightReading, matchRooms, parseHousePlan, thermalReading } from '../src/nexo/house-plan'
import type { Entity, Room } from '../src/lib/home'

describe('house-plan & matching', () => {
  it('validates minimal plan structures cleanly', () => {
    expect(parseHousePlan(null)).toBeNull()
    expect(parseHousePlan({ DEFAULT_YAW: 0 })).toBeNull()
    const plan = parseHousePlan({
      SCAN_ROOMS: {
        living: { name: 'Living', floor: 'porcelanato', aliases: ['living', 'sala'] },
        dormitorio: { name: 'Dormitorio', floor: 'laminado', aliases: ['dormitorio', 'cuarto'] },
      },
      FLOOR_ROOM: { Floor_Living: 'living', Floor_Bedroom: 'dormitorio' },
    })
    expect(plan).not.toBeNull()
    const matched = matchRooms([
      { id: 'living', name: 'Living' },
      { id: 'cuarto', name: 'Dormitorio' },
    ], plan!.SCAN_ROOMS)
    expect(matched.get('living')).toBe('living')
    expect(matched.get('cuarto')).toBe('dormitorio')
  })

  it('computes light and thermal readings from room entities', () => {
    const room: Room = {
      id: 'living', name: 'Living', entityIds: ['light.living', 'climate.living'],
      temperature: 24.5, humidity: 50, defaultLight: 'light.living', defaultClimate: 'climate.living',
    }
    const entities: Entity[] = [
      {
        id: 'light.living', name: 'Luz Living', domain: 'light', state: 'on', available: true,
        capabilities: ['turn_on', 'brightness'], limits: {}, observation: null,
        brightnessPct: 75, colorTempKelvin: 2900, temperature: null, currentTemperature: null,
        currentHumidity: null, hvacAction: null, position: null,
      },
      {
        id: 'climate.living', name: 'Aire', domain: 'climate', state: 'cool', available: true,
        capabilities: ['set_temperature'], limits: {}, observation: null,
        brightnessPct: null, colorTempKelvin: null, temperature: 23, currentTemperature: 24.5,
        currentHumidity: 50, hvacAction: 'cooling', position: null,
      },
    ]
    const light = lightReading(room, entities)
    expect(light.fraction).toBeCloseTo(0.75)
    expect(light.color).toBe(0xffc885)

    const thermal = thermalReading(room, entities)
    expect(thermal.active).toBe(true)
    expect(thermal.actual).toBe('24.5 °C')
    expect(thermal.status).toBe('Enfriando')
  })
})
