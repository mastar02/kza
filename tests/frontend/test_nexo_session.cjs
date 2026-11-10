const assert = require('node:assert/strict');
const { test } = require('node:test');
const nexo = require('../../src/dashboard/frontend/home/nexo/state.js');
const home = require('../../src/dashboard/frontend/home/home.js');
const people = require('../../src/dashboard/frontend/home/nexo/people.js');

const identity = (permissions, role = 'home', expires_at = 10) => ({ ok: true, role, permissions, expires_at });
const deferred = () => { let resolve, reject; const promise = new Promise((yes, no) => { resolve = yes; reject = no; }); return { promise, resolve, reject }; };

test('Access and Evaluation views require both their permission and their dashboard role', () => {
  const session = nexo.createSession({ wall: () => 0, monotonic: () => 0 });
  for (const [role, expected] of [['operator', ['access']], ['home', []], ['evaluator', ['evaluation']]]) {
    session.accept(identity(['auth:manage', 'evaluation:read'], role));
    assert.deepEqual(session.views(), expected);
  }
  session.accept(identity([], 'operator'));
  assert.deepEqual(session.views(), []);
});

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
  // Rutinas is readable with home:read; writes stay gated by their own permissions.
  assert.deepEqual(session.views(), ['home', 'actions', 'routines']);
});

test('EVALUATOR and a read-only identity cannot submit household effects', async () => {
  const session = nexo.createSession({ wall: () => 0, monotonic: () => 0 });
  let requests = 0;
  const api = nexo.createScopedApi({ session, transport: async () => { requests++; } });
  session.accept(identity(['evaluation:read'], 'evaluator'));
  assert.deepEqual(session.views(), ['evaluation']);
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

test('account paths have closed methods and deletion needs both account and grant authority', async () => {
  const session = nexo.createSession({ wall: () => 0, monotonic: () => 0 });
  const sends = [];
  const api = nexo.createScopedApi({ session, transport: async (path, options) => { sends.push([path, options.method || 'GET']); } });
  session.accept(identity(['admin:manage'], 'operator'));
  assert.deepEqual(session.views(), ['people']);
  await api('/api/admin/users');
  await api('/api/admin/users', { method: 'POST' });
  await api('/api/admin/users/synthetic/permissions', { method: 'PUT' });
  for (const [path, method] of [['/api/admin/users/synthetic', 'DELETE'], ['/api/admin/users', 'DELETE'],
      ['/api/admin/users/synthetic', 'PUT'], ['/api/admin/users/synthetic/permissions', 'GET'], ['/api/admin/users/synthetic/enroll', 'POST']]) {
    await assert.rejects(api(path, { method }), error => error.status === 403 && error.sent === false);
  }
  assert.equal(sends.length, 3);
  session.accept(identity(['admin:manage', 'auth:manage'], 'operator'));
  await api('/api/admin/users/synthetic', { method: 'DELETE' });
  assert.equal(sends.length, 4);
  session.accept(identity(['auth:manage'], 'operator'));
  await assert.rejects(api('/api/admin/users/synthetic', { method: 'DELETE' }), error => error.sent === false);
  session.accept(identity(['voices:read'], 'operator'));
  assert.deepEqual(session.views(), []);
});

test('a late account response cannot appear in another original login', async () => {
  const session = nexo.createSession({ wall: () => 0, monotonic: () => 0 });
  session.accept(identity(['admin:manage'], 'operator'));
  const response = deferred(); let withdrawn = 0;
  const api = nexo.createScopedApi({ session, transport: () => response.promise, onRetired: () => withdrawn++ });
  const pending = api('/api/admin/users');
  session.clear(); session.accept(identity(['home:read']));
  response.resolve([{ id: 'old-account', name: 'Private', permission_level: 'admin', is_active: true }]);
  await assert.rejects(pending, error => error.detail === 'dashboard_session_retired');
  assert.equal(withdrawn, 0);
  assert.equal(nexo.sessionRetired({ status: 403, detail: { code: 'dashboard_authority_retired', credentials_retired: null } }), true);
});

test('account metadata is closed and account role or voice material cannot select session permissions', () => {
  const account = { id: 'synthetic', name: '<script>synthetic</script>', permission_level: 'admin', is_active: false };
  const result = people.parseAccounts([account]);
  assert.deepEqual(result, [account]); assert.notEqual(result[0], account);
  for (const value of [null, {}, [account, account], [{ ...account, is_active: 'false' }],
      [{ ...account, permission_level: ['adult'] }], [{ ...account, permission_level: 'operator' }],
      [{ ...account, voice_embedding: [1, 2, 3] }], [{ ...account, preferences: {} }]]) {
    assert.throws(() => people.parseAccounts(value));
  }
  assert.match(people.mutationError({ detail: { credentials_retired: true } }), /Se confirmó el retiro/);
  assert.match(people.mutationError({ detail: { credentials_retired: null } }), /No se pudo confirmar/);
  assert.match(people.mutationError(new Error('private raw detail')), /Puede haberse aplicado/);
});

test('routine routes mirror the effective server requirement and refuse unmapped paths locally', async () => {
  const session = nexo.createSession({ wall: () => 0, monotonic: () => 0 });
  const sent = [];
  const api = nexo.createScopedApi({ session, transport: async (path, options = {}) => { sent.push(`${options.method || 'GET'} ${path}`); return {}; } });
  const routes = [
    ['GET', '/api/routines'], ['GET', '/api/routines/r1'], ['POST', '/api/routines'], ['POST', '/api/routines/preview'],
    ['PUT', '/api/routines/r1'], ['DELETE', '/api/routines/r1?expected_revision=1'], ['POST', '/api/routines/r1/enable'],
    ['POST', '/api/routines/r1/disable'], ['POST', '/api/routines/r1/execute'], ['GET', '/api/routines/r1/runs/run1'],
    ['GET', '/api/routines/r1/attempts?request_id=x'],
  ];
  const attempt = async () => { const allowed = []; for (const [method, path] of routes) { try { await api(path, { method }); allowed.push(`${method} ${path}`); } catch (error) { assert.equal(error.sent, false); } } return allowed; };
  session.accept(identity(['home:read', 'home:control', 'routines:execute'], 'home'));
  assert.deepEqual(await attempt(), ['GET /api/routines', 'GET /api/routines/r1']);
  // A role label never substitutes a permission: an 'operator' without routines:manage gets the same refusals.
  session.accept(identity(['home:read', 'home:control'], 'operator'));
  assert.deepEqual(await attempt(), ['GET /api/routines', 'GET /api/routines/r1']);
  session.accept(identity(['home:read', 'routines:manage'], 'operator'));
  assert.ok(!(await attempt()).includes('POST /api/routines/r1/execute'), 'execute also needs home:control');
  session.accept(identity(['home:read', 'routines:manage', 'home:control'], 'operator'));
  assert.equal((await attempt()).length, routes.length);
  sent.length = 0;
  for (const [method, path] of [['GET', '/api/routines/preview/x'], ['PATCH', '/api/routines/r1'], ['POST', '/api/routines/r1/unknown'], ['GET', '/api/routines/r1/runs']]) {
    await assert.rejects(api(path, { method }), error => error.sent === false);
  }
  assert.deepEqual(sent, []);
  session.accept(identity(['evaluation:read'], 'evaluator'));
  assert.ok(!session.views().includes('routines'));
});

test('extra identity fields such as voice or person links never change permissions or views', () => {
  const plain = nexo.createSession({ wall: () => 0, monotonic: () => 0 }), linked = nexo.createSession({ wall: () => 0, monotonic: () => 0 });
  plain.accept(identity(['home:read', 'home:control', 'routines:execute']));
  linked.accept({ ...identity(['home:read', 'home:control', 'routines:execute']), user_id: 'person-1', person_link: { user_id: 'person-1', role: 'admin' },
    voice_profile: 'vp_admin', speaker_label: 'Administración', permissions_hint: ['routines:manage'] });
  for (const permission of ['routines:manage', 'routines:execute', 'home:control', 'admin:manage']) assert.equal(linked.can(permission), plain.can(permission));
  assert.deepEqual(linked.views(), plain.views());
});

test('a response discarded after the server answered is marked as answered, not as a local refusal', async () => {
  const session = nexo.createSession({ wall: () => 0, monotonic: () => 0 });
  session.accept(identity(['home:read', 'routines:manage', 'home:control'], 'operator'));
  const pending = deferred(); let retired = 0;
  const api = nexo.createScopedApi({ session, transport: () => pending.promise, onRetired: () => retired++ });
  const result = api('/api/routines/r1/execute', { method: 'POST', body: { request_id: 'x' } });
  session.clear(); session.accept(identity(['home:read', 'routines:manage', 'home:control'], 'operator'));
  pending.resolve({ run_id: 'run1' });
  await assert.rejects(result, error => error.status === 403 && error.detail === 'dashboard_session_retired' && error.sent === true && error.answered === true);
  assert.equal(retired, 0);
});
