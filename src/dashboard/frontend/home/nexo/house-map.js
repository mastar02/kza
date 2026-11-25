// Match configured plan rooms against the current house catalog.
import { plan } from './house-plan.js';

export const { SCAN_ROOMS, FLOOR_ROOM, MAIN_FLOOR } = plan;

/** Lowercase, accent-free, single-spaced text for comparing ids and names. */
export function normalize(value) {
  return String(value ?? '').normalize('NFD').replace(/[\u0300-\u036f]/g, '').toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim();
}

/**
 * Match catalog rooms to scanned rooms by id, then exact name, then whole words.
 * Each pass only claims unbound rooms. An id match takes precedence over a name match.
 *
 * @param {Array<{room_id: string, name?: string}>} rooms Catalog rooms in catalog order.
 * @param {object} [scanRooms] Plan room definitions with aliases.
 * @returns {Map<string, string>} catalog room_id -> scanned room id.
 */
export function matchRooms(rooms, scanRooms = SCAN_ROOMS) {
  const aliases = Object.fromEntries(Object.entries(scanRooms).map(([id, room]) =>
    [id, new Set(room.aliases.map(normalize))]));
  const bound = new Map(), taken = new Set();
  const claim = (room, scan) => { if (scan && !taken.has(scan)) { bound.set(room.room_id, scan); taken.add(scan); } };
  const byAlias = text => Object.keys(aliases).find(scan => aliases[scan].has(text));
  for (const room of rooms) claim(room, byAlias(normalize(room.room_id)));
  for (const room of rooms) {
    if (!bound.has(room.room_id)) claim(room, byAlias(normalize(room.name)));
  }
  for (const room of rooms) {
    if (bound.has(room.room_id)) continue;
    const words = normalize(room.name).split(' ');
    claim(room, Object.keys(aliases).find(scan => words.some(word => aliases[scan].has(word))));
  }
  return bound;
}
