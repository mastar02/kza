const test = require('node:test');
const assert = require('node:assert/strict');
const { createApi } = require('../../src/dashboard/frontend/home/home.js');
const { createSession, createScopedApi, NOTICE_LIST, NOTICE_DISMISS } = require('../../src/dashboard/frontend/home/nexo/state.js');
const { createNoticesOwner } = require('../../src/dashboard/frontend/home/nexo/notices-owner.js');

const WHOAMI = '/api/admin/auth/whoami';
const EPOCH = Date.parse('2026-11-20T12:00:00Z');
const trigger_at = EPOCH / 1000 - 60;
const due = { reminder_id: 'synthetic-shared', trigger_at, text: 'Regar las plantas', status: 'pending',
  reason: null, label: 'no entregado', pending: true, attempts: 1, failed_attempts: 1, claimed_at: null };
const uncertain = { ...due, reminder_id: 'synthetic-cut', text: 'Sacar la basura', status: 'uncertain',
  reason: 'delivery_cancelled', label: 'cortado', pending: false, failed_attempts: 0 };
const sending = { ...due, reminder_id: 'synthetic-sending', status: 'sending', label: 'anunciando' };
const confirmed = { ...due, status: 'confirmed', reason: 'dismissed', label: 'descartado', pending: false };
const writeGrant = ['home:read', 'tasks:read', 'tasks:write'];
const grant = (permissions = writeGrant, role = 'home', subject = 'synthetic-account', account_binding = null) =>
  ({ ok: true, permissions, role, subject, account_binding, expires_at: EPOCH / 1000 + 60 });
const response = (data, status = 200, headers) => new Response(JSON.stringify(data), { status, headers });
const success = status => response({ status, outcome: { status: 'completed', persistence: 'committed',
  effect: 'observed', observation_scope: 'software_state' } });
const flush = () => new Promise(resolve => setImmediate(resolve));
function deferred() {
  let resolve;
  const promise = new Promise(done => { resolve = done; });
  return { promise, resolve };
}

function setup({ identity = grant(), entries = [] } = {}) {
  let wall = EPOCH, mono = 0, serial = 0;
  const timers = new Map(), requests = [], accountChanges = [];
  const session = createSession({ wall: () => wall, monotonic: () => mono });
  if (identity) assert.equal(session.accept(identity), true);
  const fixture = {
    identity, entries, requests, accountChanges, session,
    advance(ms) { wall += ms; mono += ms; },
    sample(rows = fixture.entries, options = {}) {
      const observed_at = new Date(wall - 500).toISOString();
      const expires_at = new Date(wall + 5500).toISOString();
      return response({ source: 'reminder-store', observed_at, counts: {}, notices: rows, ...options.body }, 200, {
        'X-KZA-Source-Id': 'reminder-store', 'X-KZA-Observation-State': 'fresh',
        'X-KZA-Observed-At': observed_at, 'X-KZA-Expires-At': expires_at, ...options.headers
      });
    },
    posts() { return requests.filter(({ path }) => path === NOTICE_DISMISS); },
    reads() { return requests.filter(({ path }) => path === NOTICE_LIST); }
  };
  const transport = createApi({ fetch(path, options) {
    requests.push({ path, ...options });
    if (path === WHOAMI) return fixture.whoami ? fixture.whoami() : response(fixture.identity);
    if (path === NOTICE_LIST) return fixture.list ? fixture.list() : fixture.sample();
    if (path === NOTICE_DISMISS) return fixture.post();
    assert.fail(`Unexpected route ${path}`);
  } });
  const api = createScopedApi({ session, transport, onRetired: () => { session.clear(); owner.clear(); } });
  const owner = createNoticesOwner({ session, api, onAccountChanged: generation => accountChanges.push(generation),
    clockOptions: { wallClock: () => wall, monotonic: () => mono }, now: () => mono,
    schedule(callback, delay) { timers.set(++serial, { callback, delay }); return serial; },
    cancel(id) { timers.delete(id); }
  });
  return { ...fixture, fixture, api, owner, async open() { assert.equal(owner.open(), true); await flush(); },
    key(id) { return owner.getState().notices.find(entry => entry.reminder_id === id).key; } };
}

test('only a fresh, well-formed shared snapshot establishes an empty panel', async () => {
  const { owner, open, reads } = setup({ entries: [confirmed] });
  assert.deepEqual(owner.getState(), { visible: false });
  await open();
  assert.equal(reads().length, 1);
  assert.equal(reads()[0].cache, 'no-store');
  assert.equal(reads()[0].credentials, 'same-origin');
  assert.equal(Object.hasOwn(reads()[0], 'decode'), false);
  assert.equal(owner.getState().status, 'empty');
  assert.deepEqual(owner.getState().notices, []);
});

test('pending, sending and uncertain rows keep distinct labels, and only display fields are retained', async () => {
  const { owner, open, key, posts } = setup({ entries: [due, uncertain, sending,
    { ...uncertain, reminder_id: 'synthetic-null', text: null, label: 'incompleto', room_id: 'unexpected', owner: 'unexpected' }] });
  await open();
  const state = owner.getState();
  assert.equal(state.status, 'ready');
  assert.deepEqual(state.notices.map(({ status, label, text }) => [status, label, text]), [
    ['pending', 'No entregado', 'Regar las plantas'], ['uncertain', 'Entrega incierta · Cortado', 'Sacar la basura'],
    ['sending', 'Anunciando', 'Regar las plantas'], ['uncertain', 'Entrega incierta · Incompleto', null]
  ]);
  assert.deepEqual(Object.keys(state.notices[3]), ['key', 'reminder_id', 'trigger_at', 'text', 'status', 'label']);
  assert.equal(await owner.dismiss(key(sending.reminder_id)), false);
  assert.equal(posts().length, 0);
});

test('the server expiry makes even an otherwise valid snapshot stale and blocks dismiss until refresh', async () => {
  const { owner, open, key, advance, posts, fixture } = setup({ entries: [due] });
  await open();
  advance(5499);
  assert.equal(owner.getState().status, 'ready');
  advance(1);
  assert.equal(owner.getState().status, 'stale');
  assert.equal(owner.getState().locked, true);
  assert.equal(await owner.dismiss(key(due.reminder_id)), false);
  assert.equal(posts().length, 0);
  fixture.entries = [];
  await owner.refresh();
  assert.equal(owner.getState().status, 'empty');
  advance(5500);
  assert.equal(owner.getState().status, 'stale');
});

test('a failed refresh preserves a marked last observation and an invalid first read never looks empty', async () => {
  const { owner, open, fixture } = setup({ entries: [due] });
  fixture.list = () => fixture.sample([], { headers: { 'X-KZA-Source-Id': 'demo' } });
  await open();
  assert.equal(owner.getState().status, 'error');
  assert.match(owner.getState().error, /verificar/);
  delete fixture.list;
  await owner.refresh();
  fixture.list = () => { throw new TypeError('synthetic network fault'); };
  await owner.refresh();
  assert.equal(owner.getState().status, 'stale');
  assert.equal(owner.getState().notices[0].text, due.text);
  assert.match(owner.getState().error, /contactar/);
  fixture.list = () => response({ detail: 'private internal text' }, 403);
  await owner.refresh();
  assert.equal(owner.getState().status, 'error');
  assert.deepEqual(owner.getState().notices, []);
  assert.equal(owner.getState().canWrite, false);
  assert.match(owner.getState().error, /permiso/);
  assert.doesNotMatch(owner.getState().error, /private/);
});

test('a read-only grant sees notices while evaluator, missing read authority and logged out never request the channel', async () => {
  const reader = setup({ identity: grant(['home:read', 'tasks:read']), entries: [due] });
  await reader.open();
  assert.equal(reader.owner.getState().status, 'ready');
  assert.equal(reader.owner.getState().canWrite, false);
  assert.equal(await reader.owner.dismiss(reader.key(due.reminder_id)), false);
  await assert.rejects(reader.api(NOTICE_DISMISS, { method: 'POST' }), { sent: false, status: 403 });
  assert.equal(reader.posts().length, 0);
  for (const identity of [null, grant(['home:read', 'tasks:write']), grant(['tasks:read', 'tasks:write']), grant(writeGrant, 'evaluator')]) {
    const { owner, requests, api } = setup({ identity });
    assert.equal(owner.open(), false);
    assert.equal(await owner.dismiss('invented'), false);
    await assert.rejects(api(NOTICE_LIST), { sent: false, status: 403 });
    await assert.rejects(api(NOTICE_DISMISS, { method: 'POST' }), { sent: false, status: 403 });
    assert.deepEqual(owner.getState(), { visible: false });
    assert.deepEqual(requests, []);
  }
  const operator = setup({ identity: grant(writeGrant, 'operator') });
  await operator.open();
  assert.equal(operator.owner.getState().canWrite, true);
  for (const [path, method] of [[NOTICE_LIST, 'POST'], [NOTICE_DISMISS, 'GET'], ['/api/reminders/status', 'GET']]) {
    await assert.rejects(operator.api(path, { method }), { sent: false, status: 403 });
  }
});

test('a changed live write grant keeps the read panel and stops the write before transport', async () => {
  const { fixture, owner, open, key, posts } = setup({ entries: [due] });
  await open();
  fixture.identity = grant(['home:read', 'tasks:read']);
  assert.equal(await owner.dismiss(key(due.reminder_id)), false);
  assert.equal(owner.getState().canWrite, false);
  assert.match(owner.getState().feedback.text, /permiso/);
  assert.equal(owner.getState().notices.length, 1);
  assert.equal(posts().length, 0);
});

test('losing live read permission during preflight clears all notice data immediately', async () => {
  const { fixture, owner, open, key, posts } = setup({ entries: [due] });
  await open();
  fixture.identity = grant(['home:read', 'tasks:write']);
  assert.equal(await owner.dismiss(key(due.reminder_id)), false);
  assert.equal(owner.getState().status, 'error');
  assert.deepEqual(owner.getState().notices, []);
  assert.equal(owner.getState().canWrite, false);
  assert.match(owner.getState().error, /permiso/);
  assert.equal(posts().length, 0);
});

test('double dismiss sends once and an older list response cannot restore the committed occurrence', async () => {
  const successor = { ...due, trigger_at: due.trigger_at + 30 };
  const { owner, open, fixture, posts } = setup({ entries: [due, successor] });
  await open();
  const [original, next] = owner.getState().notices;
  const write = deferred(), oldRead = deferred();
  fixture.post = () => write.promise;
  const first = owner.dismiss(original.key);
  assert.equal(await owner.dismiss(original.key), false);
  await flush();
  assert.deepEqual(posts().map(request => [request.method, JSON.parse(request.body)]), [
    ['POST', { reminder_id: due.reminder_id, trigger_at: due.trigger_at }]
  ]);
  const held = fixture.sample([due, successor]);
  fixture.list = () => oldRead.promise;
  const reading = owner.refresh();
  await flush();
  delete fixture.list;
  fixture.entries = [confirmed, successor];
  write.resolve(success('dismissed'));
  assert.equal(await first, true);
  oldRead.resolve(held);
  await reading;
  assert.equal(await owner.dismiss(original.key), false);
  assert.equal(posts().length, 1);
  assert.equal(owner.getState().feedback.kind, 'success');
  assert.deepEqual(owner.getState().notices.map(entry => entry.key), [next.key]);
});

test('an uncertain delivery can be cleared with a committed purge receipt', async () => {
  const { owner, open, key, fixture, posts } = setup({ entries: [uncertain] });
  await open();
  fixture.post = () => { fixture.entries = []; return success('purged'); };
  assert.equal(await owner.dismiss(key(uncertain.reminder_id)), true);
  await flush();
  assert.equal(owner.getState().status, 'empty');
  assert.equal(owner.getState().feedback.kind, 'success');
  assert.equal(posts().length, 1);
});

test('conflict, absence and service-unavailable errors have distinct feedback and no automatic POST', async () => {
  for (const [status, detail, expected] of [
    [409, 'reminder_notice_busy', /anunciando/],
    [404, 'reminder_notice_not_found', /ya no está disponible/],
    [503, 'reminder_notices_unavailable', /servicio de avisos no está disponible/]
  ]) {
    const { owner, open, key, fixture, posts } = setup({ entries: [due] });
    await open();
    fixture.post = () => response({ detail }, status);
    assert.equal(await owner.dismiss(key(due.reminder_id)), false);
    await flush();
    assert.equal(owner.getState().notices.length, 1);
    assert.equal(owner.getState().feedback.kind, 'error');
    assert.match(owner.getState().feedback.text, expected);
    assert.equal(posts().length, 1);
  }
});

test('lost or unvalidated receipts stay uncertain even if a refreshed list is empty', async () => {
  for (const result of [() => { throw new TypeError('receipt lost'); }, () => response({ status: 'dismissed' })]) {
    const { owner, open, key, fixture, posts } = setup({ entries: [due] });
    await open();
    fixture.post = () => { fixture.entries = []; return result(); };
    assert.equal(await owner.dismiss(key(due.reminder_id)), false);
    await flush();
    assert.equal(owner.getState().status, 'empty');
    assert.equal(owner.getState().feedback.kind, 'uncertain');
    assert.match(owner.getState().feedback.text, /No se pudo confirmar/);
    assert.equal(posts().length, 1);
  }
});

test('account switch during a list read clears and discards the response', async () => {
  const { owner, open, fixture, accountChanges, session } = setup({ entries: [due] });
  await open();
  const pending = deferred(), saved = fixture.sample([uncertain]);
  fixture.list = () => pending.promise;
  const read = owner.refresh();
  await flush();
  fixture.identity = grant(writeGrant, 'home', 'synthetic-other');
  pending.resolve(saved);
  await read;
  assert.deepEqual(accountChanges, [session.generation()]);
  assert.deepEqual(owner.getState(), { visible: false });
});

test('switching personal grants under the same subject stops dismiss and clears old data', async () => {
  const { owner, open, key, fixture, posts, accountChanges } = setup({ identity: grant(writeGrant, 'home', 'same', 'binding-a'), entries: [due] });
  await open();
  fixture.identity = grant(writeGrant, 'home', 'same', 'binding-b');
  assert.equal(await owner.dismiss(key(due.reminder_id)), false);
  assert.equal(posts().length, 0);
  assert.equal(accountChanges.length, 1);
  assert.deepEqual(owner.getState(), { visible: false });
});

test('leaving aborts the read and a late response from a former generation cannot repopulate Home', async () => {
  const { owner, open, fixture, session, reads } = setup({ entries: [due] });
  await open();
  const pending = deferred(), saved = fixture.sample([due]);
  fixture.list = () => pending.promise;
  const read = owner.refresh();
  await flush();
  owner.leave();
  assert.equal(reads().at(-1).signal.aborted, true);
  assert.deepEqual(owner.getState(), { visible: false });
  fixture.identity = grant(['home:read', 'tasks:read'], 'home', 'synthetic-other');
  assert.equal(session.accept(fixture.identity), true);
  fixture.entries = [uncertain];
  delete fixture.list;
  await open();
  pending.resolve(saved);
  await read;
  assert.equal(owner.getState().canWrite, false);
  assert.deepEqual(owner.getState().notices.map(entry => entry.text), [uncertain.text]);
});

test('a sent dismissal survives navigation but its late receipt cannot affect another session', async () => {
  const { owner, open, key, fixture, session, posts } = setup({ entries: [due] });
  await open();
  const receipt = deferred();
  fixture.post = () => receipt.promise;
  const writing = owner.dismiss(key(due.reminder_id));
  await flush();
  owner.leave();
  await open();
  assert.equal(owner.getState().locked, true);
  assert.equal(posts().length, 1);
  owner.clear();
  fixture.identity = grant(writeGrant, 'operator', 'synthetic-other');
  assert.equal(session.accept(fixture.identity), true);
  await open();
  receipt.resolve(success('dismissed'));
  assert.equal(await writing, false);
  assert.equal(owner.getState().feedback, null);
  assert.equal(owner.getState().locked, false);
  assert.equal(owner.getState().notices.length, 1);
});
