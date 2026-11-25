const assert = require('node:assert/strict');
const { existsSync } = require('node:fs');
const { test } = require('node:test');
const { pathToFileURL } = require('node:url');

const nexo = new URL('../../src/dashboard/frontend/home/nexo/', pathToFileURL(__filename));
const load = name => import(new URL(name, nexo));
const room = (room_id, name) => ({ room_id, name });

async function syntheticMatcher() {
  const { matchRooms } = await load('house-map.js');
  const { SCAN_ROOMS } = await load('house-plan.example.js');
  return rooms => Object.fromEntries(matchRooms(rooms, SCAN_ROOMS));
}

test('the room id takes precedence over the display name', async () => {
  const bind = await syntheticMatcher();
  assert.deepEqual(bind([room('reading', 'Una sala'), room('workshop', 'Taller'), room('music', 'Música')]),
    { reading: 'lectura', workshop: 'taller', music: 'musica' });
});

test('a display name only claims a room no id took, regardless of the input order', async () => {
  const bind = await syntheticMatcher();
  const primary = room('workshop', 'Taller');
  const annex = room('annex', 'Taller');
  assert.deepEqual(bind([primary, annex]), { workshop: 'taller' });
  assert.deepEqual(bind([annex, primary]), { workshop: 'taller' });
});

test('names are matched without accents, case or punctuation, and by whole words', async () => {
  const bind = await syntheticMatcher();
  const { normalize } = await load('house-map.js');
  assert.deepEqual(bind([room('x1', 'MÚSICA principal'), room('x2', 'Sala de lectura')]), { x1: 'musica', x2: 'lectura' });
  assert.equal(normalize('  Sala-de  Música '), 'sala de musica');
  assert.deepEqual(bind([room('x3', 'Talleres del fondo')]), {});
});

test('an exact room name beats an earlier whole-word match', async () => {
  const bind = await syntheticMatcher();
  assert.deepEqual(bind([room('r1', 'Taller de pintura'), room('r2', 'Taller')]), { r2: 'taller' });
  assert.deepEqual(bind([room('r1', 'Lectura infantil'), room('r2', 'Sala de lectura')]), { r2: 'lectura' });
});

test('unrecognized rooms are left unbound', async () => {
  const bind = await syntheticMatcher();
  assert.deepEqual(bind([room('garage', 'Garage'), room('reading', 'Lectura')]), { reading: 'lectura' });
  assert.deepEqual(bind([]), {});
  assert.deepEqual(bind([{ room_id: 'reading' }, { room_id: 'x', name: null }]), { reading: 'lectura' });
});

test('map and renderer use the local plan when installed and the example in a public checkout', async () => {
  const privateURL = new URL('private/house-plan.js', nexo);
  const privateExists = existsSync(privateURL);
  const expected = await (privateExists ? import(privateURL) : load('house-plan.example.js'));
  const { hasPrivatePlan } = await load('house-plan.js');
  const { SCAN_ROOMS, FLOOR_ROOM, MAIN_FLOOR, matchRooms } = await load('house-map.js');
  const { BALCONY, WALL_PATCHES } = await load('house-patches.js');
  assert.equal(hasPrivatePlan, privateExists);
  assert.equal(SCAN_ROOMS, expected.SCAN_ROOMS);
  assert.equal(FLOOR_ROOM, expected.FLOOR_ROOM);
  assert.equal(MAIN_FLOOR, expected.MAIN_FLOOR);
  assert.equal(WALL_PATCHES, expected.WALL_PATCHES);
  for (const scan of Object.values(FLOOR_ROOM)) assert.ok(SCAN_ROOMS[scan], scan);
  for (const [scan, { aliases }] of Object.entries(SCAN_ROOMS)) {
    assert.equal(matchRooms([room(aliases[0], '')]).get(aliases[0]), scan);
    if (scan !== BALCONY?.room) assert.ok(MAIN_FLOOR[scan], scan);
  }
});
