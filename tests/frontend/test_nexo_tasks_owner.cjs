const test = require('node:test');
const assert = require('node:assert/strict');
const { createApi } = require('../../src/dashboard/frontend/home/home.js');
const { createSession, createScopedApi } = require('../../src/dashboard/frontend/home/nexo/state.js');
const { createTasksOwner } = require('../../src/dashboard/frontend/home/nexo/tasks-owner.js');

const WHOAMI = '/api/admin/auth/whoami';
const grant = ['tasks:read', 'tasks:write'];
const identity = (permissions = grant, role = 'home') => ({
  ok: true, role, subject: 'synthetic', expires_at: Date.now() / 1000 + 60,
  account_binding: permissions.includes('tasks:personal') ? 'synthetic-binding' : null,
  permissions
});
const response = (data, status = 200) => new Response(JSON.stringify(data), { status });
const wait = async predicate => {
  for (let i = 0; i < 20; i++) {
    if (predicate()) return;
    await new Promise(resolve => setImmediate(resolve));
  }
  assert.fail('Owner did not settle');
};
const sharedReminder = { id: 'shared-reminder', scope: 'shared', text: 'Cerrar el patio',
  trigger_at: 1800000000, ha_actions: [{ domain: 'light', service: 'turn_off', entity_id: 'light.patio', room: 'Patio' }] };
const privateReminder = { id: 'private-reminder', scope: 'private', text: 'Agenda de prueba' };

function setup(initial = identity()) {
  const session = createSession();
  assert.equal(session.accept(initial), true);
  const calls = [], changes = [], retired = [];
  const fixture = { identity: initial, shared: [sharedReminder], overrides: new Map() };
  const transport = createApi({ fetch(path, options) {
    const method = options.method || 'GET';
    calls.push({ path, method, body: options.body ? JSON.parse(options.body) : undefined, signal: options.signal });
    if (fixture.overrides.has(path)) return fixture.overrides.get(path)(options);
    if (path === WHOAMI) return response(fixture.identity);
    if (method !== 'GET') {
      if (method === 'DELETE' && path === '/api/reminders/shared-reminder') fixture.shared = [];
      return response({ outcome: { status: 'completed', persistence: 'committed' } });
    }
    if (path === '/api/lists?scope=shared') return response([{ id: 'house', name: 'Despensa', owner_type: 'shared' }]);
    if (path === '/api/lists/house/items') return response([{ id: 'item', text: 'Arroz', completed: false }]);
    if (path === '/api/reminders?scope=shared') return response(fixture.shared);
    if (path === '/api/lists?scope=personal') return response([]);
    if (path === '/api/reminders?scope=personal') return response([privateReminder]);
    assert.fail('Unexpected request ' + path);
  } });
  const api = createScopedApi({ session, transport, onRetired: error => retired.push(error.status) });
  const owner = createTasksOwner({ api, session, onAccountChanged: g => changes.push(g) });
  return { session, owner, calls, fixture, changes, retired };
}

test('scoped API maps task methods and refuses invalid or unauthorized scopes locally', async () => {
  const session = createSession(), requests = [];
  session.accept(identity(grant));
  const api = createScopedApi({ session, transport: (path, options) => requests.push([path, options.method || 'GET']) });
  const deny = (path, options) => assert.rejects(api(path, options), { status: 403, sent: false });
  assert.ok(session.views().includes('tasks'));
  for (const base of ['/api/lists', '/api/reminders']) {
    await api(base + '?scope=shared');
    await api(base, { method: 'POST', body: base.endsWith('lists') ? { shared: true } : { scope: 'shared' } });
    for (const query of ['scope=personal', 'scope=wrong', 'scope=shared&scope=personal', '%75ser_id=synthetic']) {
      await deny(base + '?' + query);
    }
    await deny(base, { method: 'POST', body: { user_id: 'synthetic' } });
  }
  await api('/api/lists/list-id/items');
  await api('/api/lists/list-id/items', { method: 'POST' });
  await api('/api/lists/list-id/items/item-id', { method: 'DELETE' });
  await api('/api/reminders/reminder-id', { method: 'DELETE' });
  await deny('/api/lists/list-id/items?scope=shared');
  await deny('/api/reminders/reminder-id?scope=shared', { method: 'DELETE' });
  await deny('/api/reminders/status', { method: 'DELETE' });
  await deny('/api/reminders/notices', { method: 'DELETE' });
  assert.equal(requests.length, 8);
  session.accept(identity(['tasks:read']));
  await deny('/api/lists', { method: 'POST' });
  await deny('/api/reminders/reminder-id', { method: 'DELETE' });
  session.accept(identity(['tasks:write']));
  await deny('/api/reminders');
  assert.ok(!session.views().includes('tasks'));
  session.accept(identity([...grant, 'tasks:personal']));
  await api('/api/reminders?scope=personal');
  await api('/api/reminders', { method: 'POST', body: { scope: 'private' } });
  assert.equal(requests.length, 10);
});

test('visitor and unlinked operator read shared tasks only; deletion verifies, writes once and rereads', async t => {
  for (const role of ['home', 'operator']) await t.test(role, async () => {
    const { owner, calls } = setup(identity(grant, role));
    assert.equal(owner.open(), true);
    await wait(() => owner.getState().data);
    assert.equal(calls.at(0).path, WHOAMI);
    assert.equal(calls.at(-1).path, WHOAMI);
    assert.equal(owner.getState().data.personal, null);
    assert.deepEqual(owner.getState().data.shared.reminders, [sharedReminder]);
    assert.equal(owner.getState().data.shared.lists[0].items[0].text, 'Arroz');
    assert.equal(await owner.createList('personal', 'No permitida'), false);
    const deleting = owner.removeReminder('shared', sharedReminder.id);
    assert.equal(await owner.removeReminder('shared', sharedReminder.id), false);
    assert.equal(await deleting, true);
    const deletion = calls.findIndex(call => call.method === 'DELETE');
    assert.equal(calls[deletion - 1].path, WHOAMI);
    assert.equal(calls[deletion].path, '/api/reminders/shared-reminder');
    assert.ok(calls.slice(deletion + 1).some(call => call.path === '/api/reminders?scope=shared'));
    assert.deepEqual(owner.getState().data.shared.reminders, []);
    assert.equal(owner.getState().feedback, 'Recordatorio eliminado.');
    assert.ok(calls.every(call => !/[?&](scope=personal|user_id=)/.test(call.path)));
  });
});

test('linked account gets scoped data and explicitly sends shared or private creation', async () => {
  const { owner, calls } = setup(identity([...grant, 'tasks:personal']));
  owner.open();
  await wait(() => owner.getState().data);
  assert.deepEqual(owner.getState().data.personal.reminders, [privateReminder]);
  assert.equal(await owner.createList('personal', 'Lecturas'), true);
  assert.equal(await owner.createReminder('shared', 'Ventilar', 1800000000), true);
  assert.equal(await owner.createReminder('personal', 'Correo', 1800000000), true);
  assert.deepEqual(calls.filter(call => call.method === 'POST').map(call => [call.path, call.body]), [
    ['/api/lists', { name: 'Lecturas', shared: false }],
    ['/api/reminders', { text: 'Ventilar', trigger_at: 1800000000, scope: 'shared' }],
    ['/api/reminders', { text: 'Correo', trigger_at: 1800000000, scope: 'private' }]
  ]);
});

test('a read-only snapshot can be refreshed and offers no mutation', async () => {
  const { owner, calls } = setup(identity(['tasks:read']));
  owner.open();
  await wait(() => owner.getState().data);
  assert.equal(owner.getState().canWrite, false);
  assert.equal(await owner.removeReminder('shared', sharedReminder.id), false);
  assert.equal(await owner.refresh(), true);
  assert.ok(calls.every(call => call.method === 'GET'));
});

test('a different account before or after loading retires the owner without publishing personal data', async t => {
  for (const phase of ['before', 'after']) await t.test(phase, async () => {
    const personal = identity([...grant, 'tasks:personal']);
    const { owner, fixture, calls, changes, session } = setup(personal);
    const changed = { ...personal, account_binding: 'another-binding' };
    if (phase === 'before') fixture.identity = changed;
    else fixture.overrides.set('/api/reminders?scope=personal', () => {
      fixture.identity = changed;
      return response([privateReminder]);
    });
    const seen = [];
    owner.subscribe(() => seen.push(owner.getState().data));
    owner.open();
    await wait(() => changes.length);
    assert.deepEqual(changes, [session.generation()]);
    assert.deepEqual(owner.getState(), { visible: false });
    assert.ok(seen.every(data => !data));
    assert.equal(calls.some(call => call.path.includes('scope=personal')), phase === 'after');
  });
});

test('leave aborts reads and clear prevents a late write response from restoring the old snapshot', async () => {
  const linked = identity([...grant, 'tasks:personal']);
  const { owner, calls, fixture, session } = setup(linked);
  let releaseRead, releaseWrite;
  fixture.overrides.set('/api/reminders?scope=personal', () => new Promise(resolve => { releaseRead = resolve; }));
  owner.open();
  await wait(() => releaseRead);
  await owner.leave();
  assert.ok(calls.find(call => call.path === '/api/reminders?scope=personal').signal.aborted);
  fixture.overrides.delete('/api/reminders?scope=personal');
  owner.open();
  releaseRead(response([]));
  await wait(() => owner.getState().data?.personal);
  fixture.overrides.set('/api/reminders/shared-reminder', () => new Promise(resolve => { releaseWrite = resolve; }));
  const pending = owner.removeReminder('shared', sharedReminder.id);
  await wait(() => releaseWrite);
  const write = calls.find(call => call.method === 'DELETE');
  await owner.clear({ wipe: true });
  assert.equal(write.signal.aborted, false);
  session.accept(identity()); fixture.identity = identity();
  owner.open();
  await wait(() => owner.getState().data);
  const seen = owner.getState().data;
  releaseWrite(response({ outcome: { status: 'completed', persistence: 'committed' } }));
  await pending;
  assert.equal(owner.getState().data, seen);
  assert.equal(owner.getState().feedback, '');
});

test('401, 403, 503 and network failures drop the snapshot and allow a new read', async t => {
  for (const [status, copy] of [[401, /sesión venció/], [403, /no tiene permiso/], [503, /no está disponible/], [0, /conexión/]]) {
    await t.test(String(status), async () => {
      const { owner, fixture, retired } = setup();
      fixture.overrides.set('/api/reminders?scope=shared', () => {
        if (status) return response({ detail: 'synthetic' }, status);
        throw new TypeError('offline');
      });
      owner.open();
      await wait(() => owner.getState().error);
      assert.match(owner.getState().error, copy);
      assert.equal(owner.getState().data, null);
      assert.deepEqual(retired, status === 401 ? [401] : []);
      fixture.overrides.clear();
      assert.equal(await owner.refresh(), true);
    });
  }
});

test('conflicting and unconfirmed writes refresh without claiming success or replaying DELETE', async t => {
  for (const [status, copy] of [[409, /cambió/], [0, /confirmar el cambio/]]) await t.test(String(status), async () => {
    const { owner, fixture, calls } = setup();
    owner.open();
    await wait(() => owner.getState().data);
    fixture.overrides.set('/api/reminders/shared-reminder', () => {
      fixture.shared = [];
      if (status) return response({ detail: 'reminder_not_active' }, status);
      throw new TypeError('lost reply');
    });
    assert.equal(await owner.removeReminder('shared', sharedReminder.id), false);
    assert.match(owner.getState().feedback, copy);
    assert.deepEqual(owner.getState().data.shared.reminders, []);
    assert.equal(calls.filter(call => call.method === 'DELETE').length, 1);
  });
});
