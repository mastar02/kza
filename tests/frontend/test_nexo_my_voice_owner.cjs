const assert = require('node:assert/strict');
const { test } = require('node:test');
const owner = require('../../src/dashboard/frontend/home/nexo/my-voice-owner.js');

const { createMyVoiceOwner, parseStatus, parseRooms, classifyWrite, resolveMarker, allowed, PREFIX, REFUSALS, OPT_IN_DISCLOSURE } = owner;
const tick = () => new Promise(resolve => setImmediate(resolve));
async function settle() { for (let index = 0; index < 12; index++) await tick(); }
const deferred = () => { let resolve, reject; const promise = new Promise((yes, no) => { resolve = yes; reject = no; }); return { promise, resolve, reject }; };

const off = (state = null, revision = 0) => ({ state, revision, enabled: false, window: null });
const on = (state = null, revision = 0, extra = {}) => ({ state, revision, enabled: true, window: null, failures_left: state === 'pending_verification' ? 3 : null,
  message: null, disclosure: OPT_IN_DISCLOSURE, ...extra });
const WINDOW = { room_id: 'living', phrase: 'mesa verde lluvia río', expires_at: 100060 };
const CATALOG = { rooms: [{ room_id: 'living', name: 'Living', entity_ids: [] }, { room_id: 'kitchen', name: 'Cocina', entity_ids: ['light.k'] }] };
const PERSON = ['home:read', 'memory:own:read', 'memory:own:write', 'memory:own:voice'];
function storageMock() {
  const data = new Map();
  return { data, get length() { return data.size; }, key: index => [...data.keys()][index] ?? null,
    getItem: key => data.has(key) ? data.get(key) : null, setItem: (key, value) => { data.set(key, String(value)); }, removeItem: key => { data.delete(key); } };
}
function sessionMock(subject = 'synthetic-home-a', permissions = PERSON, role = 'home') {
  return { subject: () => subject, role: () => role, can: permission => permissions.includes(permission) };
}
const whoami = (subject = 'synthetic-home-a', role = 'home', permissions = PERSON) => ({ ok: true, subject, role, permissions });
const fail = (status, detail, extra = {}) => () => { throw Object.assign(Error('http'), { status, detail, ...extra }); };
const network = () => { throw new TypeError('Failed to fetch'); };

function fixture({ storage = storageMock(), session = sessionMock(), identity = () => whoami(), onAccountChanged = () => {} } = {}) {
  const calls = [], reads = [], writes = [], bodies = [], timers = [];
  let time = 100000 * 1000;
  const api = (path, options = {}) => {
    const method = options.method || 'GET';
    calls.push(method + ' ' + path);
    if (path === '/api/admin/auth/whoami') return Promise.resolve().then(identity);
    if (method === 'GET' && path === '/api/home/catalog') return Promise.resolve(CATALOG);
    if (method === 'GET') {
      const handler = reads.shift();
      if (!handler) return Promise.reject(Object.assign(Error('no read scripted'), { status: 599 }));
      return new Promise((resolve, reject) => {
        options.signal?.addEventListener('abort', () => reject(Object.assign(Error('aborted'), { name: 'AbortError' })));
        Promise.resolve().then(() => typeof handler === 'function' ? handler() : handler).then(resolve, reject);
      });
    }
    assert.equal(options.signal, undefined, 'writes are never abortable');
    bodies.push(options.body);
    const handler = writes.shift();
    if (!handler) return Promise.reject(Object.assign(Error('no write scripted'), { status: 599 }));
    return Promise.resolve().then(() => typeof handler === 'function' ? handler(path, options) : handler);
  };
  const created = createMyVoiceOwner({ api, session, storage, onAccountChanged, wall: () => time,
    schedule: (callback, delay) => { timers.push({ callback, delay }); return timers.length; }, cancel: () => {} });
  return { owner: created, calls, bodies, timers, storage, read: (...handlers) => reads.push(...handlers), write: (...handlers) => writes.push(...handlers),
    writes: () => calls.filter(call => !call.startsWith('GET')), advance: ms => { time += ms; } };
}

test('parseStatus accepts exactly the flag-off and the full shapes', () => {
  assert.deepEqual(parseStatus(off()), { state: null, revision: 0, enabled: false, configured: false, window: null, failuresLeft: null });
  assert.equal(parseStatus(on('pending_verification', 1, { window: WINDOW })).window.phrase, WINDOW.phrase);
  assert.equal(parseStatus(on('active', 2, { enabled: false })).enabled, false);
  const rows = [
    v => null, v => [], v => ({ ...v, extra: 1 }), v => ({ ...v, state: 'unknown' }), v => ({ ...v, revision: -1 }), v => ({ ...v, revision: 1.5 }),
    v => ({ ...v, revision: '1' }), v => ({ ...v, enabled: 'false' }), v => ({ ...v, enabled: true }), v => ({ ...v, window: WINDOW }),
    v => { const { window, ...rest } = v; return rest; }, v => ({ ...v, failures_left: 3 }), v => ({ ...v, person_id: 'user_x' }),
  ];
  const full = [
    v => ({ ...v, failures_left: null }), v => ({ ...v, failures_left: -1 }), v => ({ ...v, failures_left: 1.5 }), v => ({ ...v, message: 3 }),
    v => ({ ...v, disclosure: null }), v => ({ ...v, window: { ...WINDOW, extra: 1 } }), v => ({ ...v, window: { ...WINDOW, room_id: 'Living Room' } }),
    v => ({ ...v, window: { ...WINDOW, phrase: 7 } }), v => ({ ...v, window: { ...WINDOW, phrase: '' } }), v => ({ ...v, window: { ...WINDOW, expires_at: null } }),
    v => ({ ...v, window: { ...WINDOW, expires_at: Infinity } }), v => ({ ...v, profile_id: 'x' }), v => ({ ...v, scores: [] }), v => ({ ...v, room: 'living' }),
    v => { const { disclosure, ...rest } = v; return rest; }, v => ({ ...v, enabled: 1 }), v => ({ ...v, state: 'active', failures_left: 3 }),
  ];
  let count = 0;
  for (const mutate of rows) { assert.throws(() => parseStatus(mutate(off())), /invalid_my_voice/); count++; }
  for (const mutate of full) { assert.throws(() => parseStatus(mutate(on('pending_verification', 1))), /invalid_my_voice/); count++; }
  assert.throws(() => parseStatus(on('active', 2, { window: WINDOW })), /invalid_my_voice/, 'a window only while pending');
  assert.ok(count >= 30, count);
});

test('parseRooms keeps only id and name and fails closed on duplicates and bad ids', () => {
  assert.deepEqual(parseRooms(CATALOG), [{ room_id: 'living', name: 'Living' }, { room_id: 'kitchen', name: 'Cocina' }]);
  for (const value of [null, {}, { rooms: {} }, { rooms: [{ room_id: 'living' }] }, { rooms: [{ room_id: 'Living', name: 'x' }] },
    { rooms: [{ room_id: 'a', name: 'x' }, { room_id: 'a', name: 'y' }] }, { rooms: [null] }]) assert.throws(() => parseRooms(value));
});

test('allowed: the flag or the switch off leaves only the revocation', () => {
  const rows = [
    [parseStatus(off(null, 0)), { request: false, revoke: false, verify: false }],
    [parseStatus(off('pending_verification', 1)), { request: false, revoke: true, verify: false }],
    [parseStatus(off('active', 2)), { request: false, revoke: true, verify: false }],
    [parseStatus(off('verification_locked', 3)), { request: false, revoke: true, verify: false }],
    [parseStatus(off('revoked', 3)), { request: false, revoke: false, verify: false }],
    [parseStatus(off('invalidated', 3)), { request: false, revoke: true, verify: false }],
    [parseStatus(on(null, 0)), { request: true, revoke: false, verify: false }],
    [parseStatus(on('revoked', 2)), { request: true, revoke: false, verify: false }],
    [parseStatus(on('invalidated', 2)), { request: true, revoke: true, verify: false }],
    [parseStatus(on('pending_verification', 1)), { request: false, revoke: true, verify: true }],
    [parseStatus(on('active', 2)), { request: false, revoke: true, verify: false }],
    [parseStatus(on('verification_locked', 2)), { request: false, revoke: true, verify: false }],
    [parseStatus(on('pending_verification', 1, { enabled: false })), { request: false, revoke: true, verify: false }],
    [null, { request: false, revoke: false, verify: false }],
  ];
  for (const [status, expected] of rows) assert.deepEqual(allowed(status), expected, JSON.stringify(status));
});

test('classifyWrite: 401/403 are uncertain, 409/422/429 are refusals with Spanish copy, malformed bodies are uncertain', () => {
  const rows = [
    ['request', { value: { state: 'pending_verification', revision: 1 } }, 'applied'],
    ['revoke', { value: { state: 'revoked', revision: 3 } }, 'applied'],
    ['revoke', { value: { state: 'verification_locked', revision: 4 } }, 'applied'],
    ['verify', { value: { window: WINDOW } }, 'applied'],
    ['verify', { value: { window: null } }, 'uncertain'], ['verify', { value: { window: WINDOW, extra: 1 } }, 'uncertain'],
    ['request', { value: { state: 'x', revision: 1 } }, 'uncertain'], ['request', { value: null }, 'uncertain'], ['request', { value: { state: 'active', revision: 1, x: 1 } }, 'uncertain'],
    ['request', { error: { sent: false } }, 'not_applied'], ['request', { error: { answered: true, status: 403 } }, 'uncertain'],
    ['request', { error: new TypeError('offline') }, 'uncertain'], ['request', { error: { status: 401 } }, 'uncertain'], ['revoke', { error: { status: 403, detail: 'dashboard_permission_denied' } }, 'uncertain'],
    ['request', { error: { status: 500 } }, 'uncertain'], ['request', { error: { status: 503, detail: 'memory_unavailable' } }, 'uncertain'], ['request', { error: { status: 418 } }, 'uncertain'],
  ];
  for (const code of Object.keys(REFUSALS)) rows.push(['verify', { error: { status: code === 'memory_voice_rate_limited' ? 429 : code === 'memory_voice_invalid_request' ? 422 : 409, detail: code } }, 'refused']);
  rows.push(['request', { error: { status: 409, detail: { code: 'memory_voice_profile_ambiguous', notice: 'x' } } }, 'refused']);
  for (const [op, outcome, kind] of rows) assert.equal(classifyWrite(op, outcome).kind, kind, JSON.stringify([op, outcome]));
  assert.equal(classifyWrite('request', { error: { status: 409, detail: { code: 'memory_voice_profile_ambiguous', notice: 'x' } } }).text, owner.REENROLMENT_NOTICE);
  assert.match(classifyWrite('verify', { error: { status: 409, detail: 'memory_voice_room_unavailable' } }).text, /Cuenta como uno de los intentos/);
  assert.throws(() => classifyWrite('confirm', { value: {} }));
  assert.ok(rows.length >= 30, rows.length);
});

test('resolveMarker: only an observed revision step of the right kind is Comprobado; restored markers never are', () => {
  const status = (state, revision, window = null) => ({ state, revision, window });
  assert.equal(resolveMarker({ op: 'request', expected_revision: 0 }, status('pending_verification', 1)).result, 'applied');
  assert.equal(resolveMarker({ op: 'request', expected_revision: 0, restored: true }, status('pending_verification', 1)).result, 'final');
  assert.equal(resolveMarker({ op: 'request', expected_revision: 0 }, status(null, 0)).text, 'Tu estado todavía no cambió; puede que el pedido no se haya aplicado. Podés volver a intentarlo.');
  assert.match(resolveMarker({ op: 'request', expected_revision: 0 }, status('active', 2)).text, /no podemos confirmar/);
  assert.equal(resolveMarker({ op: 'revoke', expected_revision: 2 }, status('revoked', 3)).result, 'applied');
  assert.equal(resolveMarker({ op: 'revoke', expected_revision: 2 }, status('verification_locked', 3)).result, 'applied');
  assert.equal(resolveMarker({ op: 'revoke', expected_revision: 2 }, status('pending_verification', 3)).result, 'final');
  assert.equal(resolveMarker({ op: 'verify', expected_revision: 1 }, status('pending_verification', 1, WINDOW)).result, 'applied');
  assert.equal(resolveMarker({ op: 'verify', expected_revision: 1, restored: true }, status('pending_verification', 1, WINDOW)).result, 'final');
  assert.match(resolveMarker({ op: 'verify', expected_revision: 1 }, status('pending_verification', 1)).text, /No figura una ventana abierta/);
  assert.match(resolveMarker({ op: 'verify', expected_revision: 1 }, status('active', 2)).text, /no podemos confirmar/);
});

test('open: whoami, then the status; flag off reads no rooms and offers only the revocation', async () => {
  const f = fixture();
  f.read(off('active', 2));
  await f.owner.open(); await settle();
  assert.deepEqual(f.calls, ['GET /api/admin/auth/whoami', 'GET /api/memory/mine/voice']);
  const state = f.owner.getState();
  assert.deepEqual(state.allowed, { request: false, revoke: true, verify: false });
  assert.equal(state.rooms, null); assert.equal(state.message, null);
});

test('a pending person with the flag on gets the rooms (id and name only) and can open a window', async () => {
  const f = fixture();
  f.read(on('pending_verification', 1), on('pending_verification', 1, { window: WINDOW }));
  await f.owner.open(); await settle();
  assert.deepEqual(f.calls, ['GET /api/admin/auth/whoami', 'GET /api/memory/mine/voice', 'GET /api/home/catalog']);
  assert.deepEqual(f.owner.getState().rooms, [{ room_id: 'living', name: 'Living' }, { room_id: 'kitchen', name: 'Cocina' }]);
  assert.deepEqual(await f.owner.openWindow('living'), { sent: false }, 'an explicit confirmation is required');
  assert.deepEqual(await f.owner.openWindow('garage', { confirmed: true }), { sent: false }, 'only a listed room');
  f.write((path, options) => {
    assert.equal(path, '/api/memory/mine/voice/verification'); assert.equal(options.method, 'POST');
    assert.deepEqual(options.body, { room_id: 'living', expected_revision: 1 });
    const stored = JSON.parse(f.storage.getItem(PREFIX + 'consent'));
    assert.deepEqual(Object.keys(stored).sort(), ['expected_revision', 'op', 'sent_at', 'subject'], 'no room, no phrase');
    return { window: WINDOW };
  });
  const before = f.calls.length;
  assert.equal((await f.owner.openWindow('living', { confirmed: true })).applied, true); await settle();
  assert.equal(f.calls[before], 'GET /api/admin/auth/whoami', 'whoami precedes the write');
  assert.equal(f.owner.getState().status.window.phrase, WINDOW.phrase);
  assert.ok(![...f.storage.data.values()].some(value => value.includes('mesa') || value.includes('living')), 'the phrase and the room never reach storage');
  assert.ok(f.timers.some(timer => timer.delay > 0), 'a re-read is scheduled at the deadline');
  f.owner.leave();
  assert.equal(f.owner.getState().status, null, 'leaving drops the phrase');
});

test('outcome notices belong to the visit: the window notice clears once a read shows it closed, and leaving clears the rest', async () => {
  const f = fixture();
  f.read(on('pending_verification', 1), on('pending_verification', 1, { window: WINDOW }));
  await f.owner.open(); await settle();
  f.write({ window: WINDOW });
  await f.owner.openWindow('living', { confirmed: true }); await settle();
  assert.equal(f.owner.getState().notice, 'Ventana de verificación abierta.');
  f.read(on('active', 2));
  await f.owner.refresh(); await settle();
  assert.equal(f.owner.getState().notice, null, 'the window closed: its notice goes with it');
  f.write(fail(409, 'memory_voice_conflict'));
  await f.owner.revoke({ confirmed: true }); await settle();
  assert.ok(f.owner.getState().notice, 'a refusal is shown during the visit');
  f.owner.leave();
  assert.equal(f.owner.getState().notice, null, 'leaving drops the outcome notice');
});

test('request and revoke send the CAS revision the person saw, after a whoami', async () => {
  const f = fixture();
  f.read(on(null, 0), on('pending_verification', 1), on('pending_verification', 1), on('revoked', 2));
  await f.owner.open(); await settle();
  f.write({ state: 'pending_verification', revision: 1 });
  assert.equal((await f.owner.requestConsent({ confirmed: true })).applied, true); await settle();
  assert.deepEqual(f.bodies.at(-1), { expected_revision: 0 });
  assert.equal(f.owner.getState().notice, 'Inscripción pedida. Falta verificar tu voz.');
  f.write((path, options) => { assert.equal(options.method, 'DELETE'); assert.equal(path, '/api/memory/mine/voice/consent'); return { state: 'revoked', revision: 2 }; });
  assert.equal((await f.owner.revoke({ confirmed: true })).applied, true); await settle();
  assert.deepEqual(f.bodies.at(-1), { expected_revision: 1 });
  assert.equal(f.storage.getItem(PREFIX + 'consent'), null, 'an applied write leaves no marker');
});

test('a refusal (409/429) changes nothing, shows the Spanish reason and is never retried', async () => {
  for (const [code, status] of [['memory_voice_room_unavailable', 409], ['memory_voice_rate_limited', 429], ['memory_voice_profile_suspended', 409], ['memory_voice_conflict', 409]]) {
    const f = fixture();
    f.read(on('pending_verification', 1), on('pending_verification', 1));
    await f.owner.open(); await settle();
    f.write(fail(status, code));
    assert.equal((await f.owner.openWindow('kitchen', { confirmed: true })).refused, true); await settle();
    assert.equal(f.owner.getState().notice, REFUSALS[code]);
    assert.equal(f.owner.getState().marker, null);
    assert.equal(f.writes().length, 1);
  }
});

test('401/403/network on a write are uncertain: a content-free marker, resolved only by a later read', async () => {
  for (const failure of [fail(401), fail(403, 'dashboard_permission_denied'), network, fail(500)]) {
    const f = fixture();
    const later = deferred();
    f.read(on(null, 0), () => later.promise);
    await f.owner.open(); await settle();
    f.write(failure);
    assert.deepEqual(await f.owner.requestConsent({ confirmed: true }), { uncertain: true });
    assert.match(f.owner.getState().notice, /^Sin confirmar\./);
    assert.deepEqual(JSON.parse(f.storage.getItem(PREFIX + 'consent')).op, 'request');
    later.resolve(on('pending_verification', 1)); await settle();
    assert.equal(f.owner.getState().notice, 'Comprobado: tu inscripción quedó pedida.');
    assert.equal(f.storage.getItem(PREFIX + 'consent'), null);
  }
});

test('a read that started before the loss decides nothing', async () => {
  const f = fixture();
  const early = deferred();
  f.read(on(null, 0), () => early.promise, on(null, 0));
  await f.owner.open(); await settle();
  void f.owner.refresh(); await settle();
  f.write(network);
  await f.owner.requestConsent({ confirmed: true }); await settle();
  early.resolve(on('pending_verification', 1)); await settle();
  // The early read cannot settle the marker; the follow-up read (unchanged) says it may not have applied.
  assert.equal(f.owner.getState().notice, 'Tu estado todavía no cambió; puede que el pedido no se haya aplicado. Podés volver a intentarlo.');
});

test('accounts: another subject, role or a missing grant is a change; an unknown whoami sends nothing', async () => {
  for (const identity of [() => whoami('synthetic-home-b'), () => whoami('synthetic-home-a', 'operator'), () => whoami('synthetic-home-a', 'home', ['home:read', 'memory:own:read'])]) {
    let changed = 0;
    const f = fixture({ identity, onAccountChanged: () => changed++ });
    await f.owner.open(); await settle();
    assert.equal(changed, 1); assert.deepEqual(f.calls, ['GET /api/admin/auth/whoami']);
  }
  const g = fixture({ identity: () => { throw new TypeError('offline'); } });
  await g.owner.open(); await settle();
  assert.deepEqual(g.calls, ['GET /api/admin/auth/whoami']);
  // Account change between the click and the write: nothing is sent.
  let calls = 0;
  const h = fixture({ identity: () => (++calls === 1 ? whoami() : whoami('synthetic-home-b')) });
  h.read(on(null, 0));
  await h.owner.open(); await settle();
  assert.deepEqual(await h.owner.requestConsent({ confirmed: true }), { sent: false, accountChanged: true });
  assert.deepEqual(h.writes(), []);
});

test('accounts without the voice grant (operator, evaluator, household without a person) never read or write', async () => {
  for (const session of [sessionMock('synthetic-operator', ['memory:read', 'memory:assign', 'admin:manage'], 'operator'),
    sessionMock('synthetic-evaluator', ['evaluation:read'], 'evaluator'), sessionMock('synthetic-home', ['home:read'])]) {
    const f = fixture({ session });
    await f.owner.open(); await settle();
    assert.deepEqual(f.calls, []);
    assert.deepEqual(await f.owner.requestConsent({ confirmed: true }), { sent: false });
    assert.deepEqual(await f.owner.revoke({ confirmed: true }), { sent: false });
  }
});

test('a marker of another account is wiped without resending; a restored marker is never Comprobado', async () => {
  const storage = storageMock();
  storage.setItem(PREFIX + 'consent', JSON.stringify({ op: 'request', subject: 'synthetic-home-b', sent_at: 1, expected_revision: 0 }));
  const f = fixture({ storage });
  f.read(on(null, 0));
  await f.owner.open(); await settle();
  assert.equal(storage.getItem(PREFIX + 'consent'), null);
  assert.deepEqual(f.writes(), []);
  const own = storageMock();
  own.setItem(PREFIX + 'consent', JSON.stringify({ op: 'request', subject: 'synthetic-home-a', sent_at: 1, expected_revision: 0 }));
  const g = fixture({ storage: own });
  g.read(on('pending_verification', 1));
  await g.owner.open(); await settle();
  assert.match(g.owner.getState().notice, /no podemos confirmar/);
  assert.deepEqual(g.writes(), []);
});

const BIND_A = 'a'.repeat(32), BIND_B = 'b'.repeat(32);
const bound = binding => () => ({ ...whoami(), account_binding: binding });
const storedConsent = extra => { const storage = storageMock();
  storage.setItem(PREFIX + 'consent', JSON.stringify({ op: 'request', subject: 'synthetic-home-a', sent_at: 1, expected_revision: 0, ...extra })); return storage; };

test('R3 binding: a restored consent marker with the same account_binding resolves like this page\'s own (Comprobado)', async () => {
  const storage = storedConsent({ binding: BIND_A });
  const f = fixture({ storage, identity: bound(BIND_A) });
  f.read(on('pending_verification', 1));
  await f.owner.open(); await settle();
  assert.equal(f.owner.getState().notice, 'Comprobado: tu inscripción quedó pedida.');
  assert.equal(f.owner.getState().marker, null); assert.equal(storage.getItem(PREFIX + 'consent'), null);
  assert.deepEqual(f.writes(), []);
  // Verify: the same binding lets an open window count as Comprobado.
  const g = fixture({ storage: storedConsent({ op: 'verify', expected_revision: 1, binding: BIND_A }), identity: bound(BIND_A) });
  g.read(on('pending_verification', 1, { window: WINDOW }));
  await g.owner.open(); await settle();
  assert.equal(g.owner.getState().notice, 'Comprobado: tenés una ventana de verificación abierta.');
});

test('R3 binding: another grant\'s restored marker is dropped unsent; no binding (stored or named) keeps the old rule', async () => {
  const f = fixture({ storage: storedConsent({ binding: BIND_B }), identity: bound(BIND_A) });
  f.read(on('pending_verification', 1));
  await f.owner.open(); await settle();
  assert.equal(f.owner.getState().notice, 'Una operación de otra cuenta quedó sin confirmar; no se reintentó.');
  assert.equal(f.owner.getState().marker, null); assert.equal(f.storage.getItem(PREFIX + 'consent'), null);
  assert.deepEqual(f.writes(), []);
  for (const [name, storage, identity] of [
    ['no stored binding', storedConsent({}), bound(BIND_A)],
    ['server names none', storedConsent({ binding: BIND_A }), () => whoami()],
    ['malformed stored binding', storedConsent({ binding: 'A'.repeat(32) }), bound('a'.repeat(32))],
  ]) {
    const g = fixture({ storage, identity });
    g.read(on('pending_verification', 1));
    await g.owner.open(); await settle();
    assert.match(g.owner.getState().notice, /no podemos confirmar/, name);
    assert.ok(!/Comprobado/.test(g.owner.getState().notice), name);
    assert.deepEqual(g.writes(), [], name);
  }
});

test('R3 binding: the stored marker carries the binding (no room, phrase or person); a later different or malformed binding is a change', async () => {
  for (const [name, later] of [['another grant', BIND_B], ['none', null], ['malformed', 'x'.repeat(32)], ['not a string', 7]]) {
    let calls = 0, changed = 0;
    const f = fixture({ identity: () => (++calls === 1 ? bound(BIND_A)() : { ...whoami(), account_binding: later }), onAccountChanged: () => changed++ });
    f.read(on(null, 0));
    await f.owner.open(); await settle();
    assert.equal(changed, 0, name);
    assert.deepEqual(await f.owner.requestConsent({ confirmed: true }), { sent: false, accountChanged: true }, name);
    assert.equal(changed, 1, name); assert.deepEqual(f.writes(), [], name);
  }
  const g = fixture({ identity: bound(BIND_A) });
  g.read(on(null, 0), on(null, 0));
  await g.owner.open(); await settle();
  let stored = null;
  g.write(() => { stored = JSON.parse(g.storage.getItem(PREFIX + 'consent')); return { state: 'pending_verification', revision: 1 }; });
  assert.equal((await g.owner.requestConsent({ confirmed: true })).applied, true);
  assert.deepEqual(Object.keys(stored).sort(), ['binding', 'expected_revision', 'op', 'sent_at', 'subject']);
  assert.equal(stored.binding, BIND_A);
  let changed = 0;
  const h = fixture({ identity: bound('Z'.repeat(32)), onAccountChanged: () => changed++ });
  await h.owner.open(); await settle();
  assert.equal(changed, 1); assert.deepEqual(h.calls, ['GET /api/admin/auth/whoami']);
});

test('R3 binding: the end of a session forgets the binding, so the next login of the same subject starts fresh', async () => {
  let current = BIND_A, changed = 0;
  const f = fixture({ identity: () => bound(current)(), onAccountChanged: () => changed++ });
  f.read(on(null, 0), on(null, 0));
  await f.owner.open(); await settle();
  await f.owner.clear(); current = BIND_B;
  await f.owner.open(); await settle();
  assert.equal(changed, 0, 'a new session is not a change of the previous one');
  f.write({ state: 'pending_verification', revision: 1 });
  assert.equal((await f.owner.requestConsent({ confirmed: true })).applied, true);
});

test('a write during an in-flight write, without confirmation or with a stale revision is not sent', async () => {
  const f = fixture();
  const hold = deferred();
  f.read(on(null, 0), on('pending_verification', 1));
  await f.owner.open(); await settle();
  f.write(() => hold.promise);
  const first = f.owner.requestConsent({ confirmed: true });
  await settle();
  assert.deepEqual(await f.owner.requestConsent({ confirmed: true }), { sent: false }, 'double tap');
  assert.deepEqual(await f.owner.requestConsent(), { sent: false });
  hold.resolve({ state: 'pending_verification', revision: 1 });
  await first; await settle();
  assert.equal(f.calls.filter(call => call.startsWith('POST')).length, 1);
  // Stale: a refresh changes the revision while the whoami of the write runs.
  const gate = deferred();
  let asked = 0;
  const g = fixture({ identity: () => (++asked === 2 ? gate.promise.then(() => whoami()) : whoami()) });
  g.read(on(null, 0), on('pending_verification', 1));
  await g.owner.open(); await settle();
  const pending = g.owner.requestConsent({ confirmed: true });
  await settle();
  await g.owner.refresh(); await settle();
  gate.resolve(); assert.deepEqual(await pending, { sent: false });
  assert.equal(g.owner.getState().notice, REFUSALS.memory_voice_conflict);
  assert.deepEqual(g.writes(), []);
});

test('the copy names the 48 h ambient transcript, the evaluators and that the voice grants nothing (N6)', () => {
  assert.match(OPT_IN_DISCLOSURE, /48 horas en la transcripción ambiental/);
  assert.match(OPT_IN_DISCLOSURE, /un evaluador autorizado/);
  assert.match(OPT_IN_DISCLOSURE, /Tu voz no te da permisos ni acceso/);
  for (const text of [...Object.values(REFUSALS), ...Object.values(owner.STATE_LABELS), owner.OFF, owner.NOT_ENROLLED]) assert.match(text, /[áéíóúñ]|[a-z]/i);
});

test('a revision that moves while whoami is in flight stops the write with the conflict notice', async () => {
  const gate = deferred();
  let calls = 0;
  const f = fixture({ identity: () => (++calls === 2 ? gate.promise : whoami()) });
  // Another tab requested and revoked: a request is still allowed, but not against revision 0.
  f.read(on(null, 0), on('revoked', 2));
  await f.owner.open(); await settle();
  const sending = f.owner.requestConsent({ confirmed: true }); await settle();
  await f.owner.refresh(); await settle();
  assert.equal(f.owner.getState().status.revision, 2, 'another tab moved the consent');
  gate.resolve(whoami());
  assert.deepEqual(await sending, { sent: false });
  assert.deepEqual(f.writes(), [], 'nothing is sent against a revision the person did not see');
  assert.equal(f.owner.getState().notice, REFUSALS.memory_voice_conflict);
});

test('S: pagehide (wipe false) keeps an uncertain marker; the next load restores it and resolves it without «Comprobado»', async () => {
  const storage = storageMock();
  const f = fixture({ storage });
  f.read(on(null, 0));
  await f.owner.open(); await settle();
  f.write(network);
  assert.deepEqual(await f.owner.requestConsent({ confirmed: true }), { uncertain: true });
  await f.owner.clear({ wipe: false });
  assert.ok(storage.getItem(PREFIX + 'consent'), 'the marker survives a pagehide');
  assert.equal(storage.getItem(owner.ENDED_KEY), null, 'no end-of-session reminder on pagehide');
  const g = fixture({ storage });
  g.read(on('pending_verification', 1));
  await g.owner.open(); await settle();
  assert.ok(!/Comprobado/.test(g.owner.getState().notice || ''), 'a restored marker is never Comprobado');
  assert.match(g.owner.getState().notice, /no podemos confirmar/);
  assert.equal(storage.getItem(PREFIX + 'consent'), null, 'resolved and dropped');
  assert.deepEqual(g.writes(), [], 'never resent');
});

test('S: the end of the session (wipe) removes the marker and leaves a content-free reminder for the next open', async () => {
  const storage = storageMock();
  const f = fixture({ storage });
  f.read(on(null, 0));
  await f.owner.open(); await settle();
  f.write(fail(401));
  await f.owner.requestConsent({ confirmed: true });
  await f.owner.clear();
  assert.equal(storage.getItem(PREFIX + 'consent'), null);
  assert.equal(storage.getItem(owner.ENDED_KEY), '1');
  const g = fixture({ storage });
  g.read(on('pending_verification', 1));
  await g.owner.open(); await settle();
  assert.match(g.owner.getState().notice, /quedó sin confirmar al terminar la sesión anterior/);
  assert.equal(storage.getItem(owner.ENDED_KEY), null, 'shown once');
  const clean = storageMock(), h = fixture({ storage: clean });
  h.read(on(null, 0));
  await h.owner.open(); await settle();
  await h.owner.clear();
  assert.equal(clean.getItem(owner.ENDED_KEY), null, 'no reminder without a marker');
});
