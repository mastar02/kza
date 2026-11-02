const assert = require('node:assert/strict');
const { test } = require('node:test');
const nexo = require('../../src/dashboard/frontend/home/nexo/state.js');
const home = require('../../src/dashboard/frontend/home/home.js');

const identity = (permissions, role = 'home', expires_at = 10) => ({ ok: true, role, permissions, expires_at });
const deferred = () => { let resolve, reject; const promise = new Promise((yes, no) => { resolve = yes; reject = no; }); return { promise, resolve, reject }; };

test('pending bootstrap and HOME cannot query private or diagnostic sources', async () => {
  const session = nexo.createSession({ wall: () => 0, monotonic: () => 0 });
  const pending = deferred(), requests = [];
  const api = nexo.createScopedApi({ session, transport: (path) => { requests.push(path); return path.endsWith('whoami') ? pending.promise : Promise.resolve({}); } });
  const bootstrap = api('/api/admin/auth/whoami');
  await assert.rejects(api('/api/home/catalog'), error => error.sent === false);
  assert.deepEqual(requests, ['/api/admin/auth/whoami']);
  pending.resolve(identity(['home:read', 'home:control']));
  assert.equal(session.accept(await bootstrap), true);
  await api('/api/home/catalog');
  for (const path of ['/api/home/microphones', '/api/users', '/api/voices', '/api/memory/facts', '/api/conversations']) {
    await assert.rejects(api(path), error => error.status === 403 && error.sent === false);
  }
  assert.deepEqual(requests, ['/api/admin/auth/whoami', '/api/home/catalog']);
  assert.deepEqual(session.views(), ['home', 'actions']);
});

test('EVALUATOR and a read-only identity cannot submit household effects', async () => {
  const session = nexo.createSession({ wall: () => 0, monotonic: () => 0 });
  let requests = 0;
  const api = nexo.createScopedApi({ session, transport: async () => { requests++; } });
  session.accept(identity(['evaluation:read'], 'evaluator'));
  assert.deepEqual(session.views(), []);
  await assert.rejects(api('/api/home/catalog'), error => error.sent === false);
  session.accept(identity(['home:read']));
  await assert.rejects(api('/api/home/actions', { method: 'POST' }), error => error.sent === false);
  await assert.rejects(api('https://other.example/api/home/catalog'), error => error.sent === false);
  assert.equal(requests, 0);
});

test('retired catalog and whoami results cannot restore a later login', async () => {
  for (const path of ['/api/home/catalog', '/api/admin/auth/whoami']) {
    const session = nexo.createSession({ wall: () => 0, monotonic: () => 0 });
    session.accept(identity(['home:read'], 'operator'));
    const pending = deferred(); let withdrawals = 0;
    const api = nexo.createScopedApi({ session, transport: () => pending.promise, onRetired: () => withdrawals++ });
    const result = api(path);
    session.clear(); session.accept(identity(['home:read']));
    pending.resolve({ private: 'old session material' });
    await assert.rejects(result, error => error.detail === 'dashboard_session_retired');
    assert.equal(session.role(), 'home'); assert.equal(withdrawals, 0);
  }
});

test('ordinary operation denial and retired session are distinct; old 401 cannot expire new login', async () => {
  const session = nexo.createSession({ wall: () => 0, monotonic: () => 0 });
  session.accept(identity(['home:read'])); let withdrawals = 0;
  for (const detail of ['home_policy_denied', 'dashboard_session_retired', 'dashboard_authority_retired']) {
    const api = nexo.createScopedApi({ session, transport: async () => { throw { status: 403, detail }; }, onRetired: () => withdrawals++ });
    await assert.rejects(api('/api/home/catalog'));
  }
  assert.equal(withdrawals, 2);
  const pending = deferred();
  const api = nexo.createScopedApi({ session, transport: () => pending.promise, onRetired: () => withdrawals++ });
  const request = api('/api/home/catalog');
  session.clear(); session.accept(identity(['home:read'])); pending.reject({ status: 401 });
  await assert.rejects(request); assert.equal(withdrawals, 2); assert.equal(session.active(), true);
});

test('wall advance during suspension and monotonic advance during wall rollback both expire session', async () => {
  for (const suspension of [true, false]) {
    let wall = 0, mono = 0, sends = 0;
    const session = nexo.createSession({ wall: () => wall, monotonic: () => mono });
    session.accept(identity(['home:read'], 'home', 1));
    const api = nexo.createScopedApi({ session, transport: async () => sends++ });
    if (suspension) wall = 1001; else { wall = -5000; mono = 1001; }
    assert.equal(session.active(), false);
    await assert.rejects(api('/api/home/catalog'), error => error.sent === false);
    assert.equal(sends, 0);
  }
});

test('a frozen performance clock cannot extend a catalog or entity observation after suspension', () => {
  const received = Date.parse('2026-11-02T00:00:00Z'); let wall = received;
  const observation = { state: 'fresh', observed_at: new Date(received).toISOString(), expires_at: new Date(received + 1000).toISOString() };
  const entity = { entity_id: 'light.test', domain: 'light', state: 'on', available: true, capabilities: ['turn_off'], observation };
  const clock = home.createSnapshotClock({ observation, entities: [entity] }, { wallClock: () => wall, monotonic: () => 0 });
  assert.equal(home.entityCanControl(entity, 'turn_off', clock.read().stale, clock.now()), true);
  wall += 1001;
  assert.equal(clock.read().stale, true);
  assert.equal(home.entityCanControl(entity, 'turn_off', clock.read().stale, clock.now()), false);
});

test('room associations use backend IDs despite homonymous room and entity labels', () => {
  const first = { entity_id: 'light.a', name: 'Ambiente', domain: 'light' };
  const second = { entity_id: 'climate.b', name: 'Ambiente', domain: 'climate', state: 'cool', available: true };
  const catalog = { rooms: [{ room_id: 'r1', name: 'Ambiente', entity_ids: ['light.a'] }, { room_id: 'r2', name: 'Ambiente', entity_ids: ['climate.b'] }], entities: [first, second] };
  assert.deepEqual(nexo.roomEntities(catalog, 'r1', 'all'), [first]);
  assert.deepEqual(nexo.roomEntities(catalog, 'r1', 'climate'), []);
  assert.deepEqual(nexo.roomEntities(catalog, 'r2', 'climate'), [second]);
  assert.equal(nexo.entityState(second), 'Modo Frío');
});
