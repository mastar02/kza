const assert = require('node:assert/strict');
const { test } = require('node:test');
const home = require('../../src/dashboard/frontend/home/home.js');
const { createSession } = require('../../src/dashboard/frontend/home/nexo/state.js');
const { activityPermissions, createActivityOwner } = require('../../src/dashboard/frontend/home/nexo/activity-owner.js');
const { createNoticesOwner } = require('../../src/dashboard/frontend/home/nexo/notices-owner.js');

const READ = 'home:read', CONTROL = 'home:control', NOTICES = 'tasks:read';
const EXECUTE = 'routines:execute', MANAGE = 'routines:manage';
const WHOAMI = '/api/admin/auth/whoami', ROUTINES = '/api/routines', NOTICE_LIST = '/api/reminders/notices';
const NOW = 1_800_000_000_000;
const identity = (role = 'home', permissions = [READ, CONTROL, NOTICES, EXECUTE], extra = {}) => ({
  ok: true, role, permissions, expires_at: NOW / 1000 + 100,
  subject: 'private-subject', account_binding: 'private-binding', routine_actor: 'private-actor', ...extra
});
const routine = (name, run = {}) => ({
  routine_id: 'private-' + name, revision: 1, name, description: '', enabled: true, timezone: 'America/Montevideo',
  schedule: null, conditions: [], actions: [], created_by: 'private-creator',
  latest_run: {
    routine_id: 'private-' + name, run_id: 'private-run-' + name, status: 'succeeded', source: 'manual',
    scheduled_for: null, reason: 'private-reason', actor: 'private-actor', occurrence_key: 'private-occurrence',
    actions: [{ index: 0, entity_id: 'light.private_device', status: 'confirmed' },
      { index: 1, entity_id: 'light.private_other', status: 'failed' }], ...run
  }
});

function fixture({ login = identity(), respond, noticeState = { status: 'empty', notices: [] }, reconcile } = {}) {
  const session = createSession({ wall: () => NOW, monotonic: () => 0 });
  assert.equal(session.accept(login), true);
  let now = NOW;
  const events = [], changes = [];
  const api = async (path, options) => {
    events.push(path);
    return respond ? respond(path, options) : path === WHOAMI ? login : [];
  };
  const actions = { async reconcile() { events.push('actions:reconcile'); await reconcile?.(); } };
  const notices = {
    open() { events.push('notices:open'); },
    leave() { events.push('notices:leave'); },
    refresh() { events.push('notices:refresh'); },
    getState() { return noticeState; },
    subscribe() { return () => {}; }
  };
  const owner = createActivityOwner({ api, session, actions, notices, clock: () => now,
    deviceLabel: id => id === 'light.private_device' ? 'Luz de techo · Sala' : null,
    onAccountChanged: generation => changes.push(generation) });
  return { session, owner, events, changes, setTime(value) { now = value; } };
}

test('view and source gates are an intersection and exclude evaluator even with extra grants', async () => {
  const all = [READ, CONTROL, NOTICES, MANAGE];
  for (const [login, expected] of [
    [identity('operator', all), ['action', 'notice', 'routine']],
    [identity(), ['action', 'notice', 'routine']],
    [identity('home', all, { account_binding: null }), ['action', 'notice', 'routine']],
    [identity('home', all, { subject: null }), ['action', 'routine']],
    [identity('home', [READ, CONTROL]), ['action']],
    [identity('home', [READ, NOTICES]), ['notice']],
    [identity('home', [READ, EXECUTE]), ['routine']],
    [identity('home', [READ]), []],
    [identity('operator', all.slice(1)), null],
    [identity('evaluator', all), null]
  ]) {
    const { session, owner, events } = fixture({ login });
    const { visible, ...grants } = activityPermissions(session);
    assert.equal(visible, expected !== null);
    assert.equal(session.views().includes('activity'), visible);
    assert.deepEqual(Object.keys(grants).filter(key => grants[key]), expected ?? []);
    assert.equal(await owner.open(), visible);
    await owner.refresh();
    assert.deepEqual(Object.keys(owner.getState().blocks ?? {}), expected ?? []);
    assert.equal(events.includes('notices:open'), grants.notice);
    assert.equal(events.includes('notices:refresh'), grants.notice);
    assert.equal(events.includes('actions:reconcile'), grants.action);
    assert.equal(events.includes(ROUTINES), grants.routine);
    owner.leave();
  }
});

test('actual action controller keeps two receipts for the same operation and timestamps at send', async () => {
  const login = identity('home', [READ, CONTROL]);
  const session = createSession({ wall: () => NOW, monotonic: () => 0 });
  session.accept(login);
  let now = NOW, sequence = 0;
  const actions = home.createActionController({
    api: async (_path, { body }) => {
      now += 100;
      return { status: body.request_id.endsWith('1') ? 'confirmed' : 'failed',
        effects: [{ entity_id: 'light.private_device', status: 'confirmed', reason: 'private-reason' }] };
    },
    makeId: () => 'private-request-' + ++sequence,
    onChange: (_key, record) => owner.recordAction(record)
  });
  const owner = createActivityOwner({ api: async () => login, session, actions, clock: () => now,
    deviceLabel: () => 'Luz de techo · Sala' });
  await actions.execute('light.private_device', 'turn_on');
  await actions.execute('light.private_device', 'turn_on');
  await owner.open();
  const rows = owner.getState().blocks.action.rows;
  assert.deepEqual(rows.map(({ at, status }) => [at, status]), [[NOW + 100, 'failed'], [NOW, 'confirmed']]);
  assert.deepEqual(rows[0], { block: 'action', at: NOW + 100, atKind: 'sent_here', what: 'Encender',
    where: 'Luz de techo · Sala', origin: 'Desde esta pestaña', status: 'failed', label: 'Falló', detail: '1 confirmado' });
  assert.doesNotMatch(JSON.stringify(rows), /private[-_]/);
});

test('cap keeps the latest 50 sends and an evicted or cleared receipt cannot reappear on completion', async () => {
  const { owner, setTime } = fixture({ login: identity('home', [READ, CONTROL]) });
  const makeRecord = index => ({
    entity_id: 'scene.private_scene', request_id: 'private-request-' + index, operation: 'activate_scene'
  });
  for (let index = 0; index < 51; index++) {
    setTime(NOW + index);
    owner.recordAction({ ...makeRecord(index), sending: true, status: 'in_progress' });
  }
  owner.recordAction({ ...makeRecord(0), sending: false, status: 'confirmed' });
  await owner.open();
  const rows = owner.getState().blocks.action.rows;
  assert.equal(rows.length, 50);
  assert.equal(rows[0].where, 'Una escena');
  assert.equal(rows.at(-1).at, NOW + 1);
  assert.ok(rows.every(row => row.status === 'running'));
  owner.leave();
  await owner.open();
  assert.equal(owner.getState().blocks.action.rows.length, 50);
  owner.clear();
  assert.equal(owner.getState().visible, false);
  owner.recordAction({ ...makeRecord(50), sending: false, status: 'confirmed' });
  await owner.open();
  assert.equal(owner.getState().blocks.action.status, 'empty');
});

test('real notices owner supplies shared occurrences and filters confirmed deliveries', async () => {
  const login = identity('home', [READ, NOTICES]);
  const session = createSession({ wall: () => NOW, monotonic: () => 0 });
  session.accept(login);
  const observed = new Date().toISOString();
  const trigger = NOW / 1000;
  const events = [];
  const api = async (path, { decode } = {}) => {
    events.push(path);
    if (path === WHOAMI) return login;
    assert.equal(path, NOTICE_LIST);
    const headers = { 'X-KZA-Observation-State': 'fresh', 'X-KZA-Observed-At': observed,
      'X-KZA-Expires-At': new Date(Date.parse(observed) + 6000).toISOString(), 'X-KZA-Source-Id': 'reminder-store' };
    const entry = { reminder_id: 'private-reminder', trigger_at: trigger, text: 'Aviso de prueba', status: 'pending', label: 'pendiente' };
    return decode({ headers: { get: key => headers[key] }, json: async () => ({ source: 'reminder-store',
      observed_at: observed, notices: [entry, { ...entry, trigger_at: trigger + 10, text: null },
        { status: 'confirmed', text: 'private-confirmed' }] }) });
  };
  const notices = createNoticesOwner({ api, session });
  const owner = createActivityOwner({ api, session, notices });
  const ready = new Promise(resolve => {
    owner.subscribe(() => { if (owner.getState().blocks?.notice.status === 'ready') resolve(); });
  });
  try {
    await owner.open();
    await ready;
    const { rows } = owner.getState().blocks.notice;
    assert.deepEqual(events, [WHOAMI, NOTICE_LIST, WHOAMI]);
    assert.deepEqual(rows.map(({ what, at, atKind }) => [what, at, atKind]), [
      ['Aviso sin texto', NOW + 10000, 'scheduled'], ['Aviso de prueba', NOW, 'scheduled']
    ]);
    assert.doesNotMatch(JSON.stringify(rows), /private-/);
  } finally {
    owner.leave();
  }
});

test('latest routine runs keep only scheduled times and use safe receipt text', async () => {
  const login = identity('operator', [READ, MANAGE]);
  const scheduled = new Date(NOW + 1000).toISOString();
  const { owner, events } = fixture({ login, respond: path => path === WHOAMI ? login : [
    routine('Zeta', { source: 'household_voice', scheduled_for: scheduled }),
    routine('Alfa', { status: 'running' }),
    routine('Primera', { source: 'scheduled', scheduled_for: scheduled }),
    { ...routine('Sin ejecución'), latest_run: null }
  ] });
  await owner.open();
  const { rows, updatedAt } = owner.getState().blocks.routine;
  assert.deepEqual(events, [WHOAMI, ROUTINES, WHOAMI]);
  assert.equal(updatedAt, NOW);
  assert.deepEqual(rows.map(({ what, at, origin, label }) => [what, at, origin, label]), [
    ['Primera', NOW + 1000, 'Programada', 'Confirmada'], ['Alfa', null, 'Manual', 'En curso'],
    ['Zeta', null, 'Por voz', 'Confirmada']
  ]);
  assert.equal(rows[0].detail, '1 de 2 acciones confirmadas');
  assert.match(rows[1].detail, /Actualizá para ver el resultado/);
  assert.doesNotMatch(JSON.stringify(rows), /private[-_]|request_id|routine_id|actor|reason/);
});

test('a failed or stale source leaves healthy blocks available and can recover on refresh', async () => {
  const login = identity();
  const noticeState = { status: 'stale', notices: [] };
  let available = false;
  const { owner, events } = fixture({ login, noticeState, respond(path) {
    if (path === WHOAMI) return login;
    if (!available) throw Object.assign(Error('private-service-detail'), { status: 503 });
    return [routine('Otra rutina')];
  } });
  await owner.open();
  assert.equal(owner.getState().status, 'partial');
  assert.deepEqual(Object.values(owner.getState().blocks).map(({ status }) => status), ['empty', 'stale', 'error']);
  available = true;
  noticeState.status = 'empty';
  await owner.refresh();
  assert.equal(owner.getState().status, 'ready');
  assert.deepEqual(Object.values(owner.getState().blocks).map(({ status }) => status), ['empty', 'empty', 'ready']);
  assert.equal(events.filter(event => event === 'actions:reconcile').length, 2);
});

test('routine response is discarded on account switch during the read and old action receipts are cleared', async () => {
  const login = identity('home', [READ, CONTROL, EXECUTE]);
  let changeAccount = false;
  const { session, owner, changes } = fixture({ login, respond(path) {
    if (path === WHOAMI) return changeAccount ? { ...login, routine_actor: 'other-actor' } : login;
    changeAccount = true;
    return [routine('Private result')];
  } });
  owner.recordAction({ entity_id: 'light.private_device', request_id: 'private-receipt',
    operation: 'turn_on', sending: true, status: 'in_progress' });
  assert.equal(await owner.open(), false);
  assert.deepEqual(changes, [session.generation()]);
  assert.deepEqual(owner.getState(), { visible: false });
  session.accept(identity('home', [READ, CONTROL]));
  changeAccount = false;
  assert.equal(await owner.open(), true);
  assert.equal(owner.getState().blocks.action.status, 'empty');
});

test('leave and session replacement prevent a late read from restoring rows', async () => {
  const login = identity('home', [READ, EXECUTE]);
  let started, release;
  const waiting = new Promise(resolve => { started = resolve; });
  const reply = new Promise(resolve => { release = resolve; });
  const { owner, session, events } = fixture({ login, respond(path, { signal }) {
    if (path === WHOAMI) return login;
    started(signal);
    return reply;
  } });
  const opening = owner.open();
  const signal = await waiting;
  owner.leave();
  assert.equal(signal.aborted, true);
  session.accept(identity('home', [READ]));
  await owner.open();
  release([routine('Late result')]);
  await opening;
  assert.deepEqual(owner.getState().blocks, {});
  assert.deepEqual(events, [WHOAMI, ROUTINES]);
});

test('routine preflight suppresses a request when the actual account lost the grant or gained management', async () => {
  for (const permissions of [[READ], [READ, MANAGE]]) {
    const login = identity('home', [READ, EXECUTE]);
    const { owner, events } = fixture({ login, respond: () => ({ ...login, permissions }) });
    await owner.open();
    assert.deepEqual(events, [WHOAMI]);
    assert.equal(owner.getState().blocks.routine.status, 'error');
  }
});
