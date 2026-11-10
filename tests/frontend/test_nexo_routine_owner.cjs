const assert = require('node:assert/strict');
const { test } = require('node:test');
const home = require('../../src/dashboard/frontend/home/home.js');
const routines = require('../../src/dashboard/frontend/home/nexo/routine-owner.js');

const TZ = 'America/Montevideo';
const lamp = { entity_id: 'light.test_lamp', name: 'Luz principal', domain: 'light', capabilities: ['turn_on', 'turn_off', 'color_temp_kelvin', 'brightness'],
  control_limits: { brightness_min: 0, brightness_max: 100, color_temp_min: 2000, color_temp_max: 6500 } };
const ac = { entity_id: 'climate.test_ac', domain: 'climate', available: true, state: 'cool', capabilities: ['set_temperature', 'set_hvac_mode'],
  control_limits: { temperature_min: 18.5, temperature_max: 25, temperature_step: 0.5, hvac_modes: ['off', 'cool', 'dry'] } };
const spec = (name = 'Noche') => ({ name, description: '', enabled: true, timezone: TZ, schedule: null, conditions: [], actions: [{ entity_id: lamp.entity_id, operation: 'turn_on' }] });
const failure = (status, detail, extra = {}) => Object.assign(new Error('synthetic'), { status, detail, ...extra });
const MANAGER = ['home:read', 'home:control', 'routines:manage'];

function memoryStorage() {
  const map = new Map();
  return { map, get length() { return map.size; }, key: index => [...map.keys()][index] ?? null, getItem: key => map.has(key) ? map.get(key) : null,
    setItem: (key, value) => { map.set(key, String(value)); }, removeItem: key => { map.delete(key); } };
}
function session(permissions = MANAGER, subject = 'synthetic-operator') { const set = new Set(permissions); return { can: permission => set.has(permission), subject: () => subject, set }; }

/** Minimal server with the store semantics the owner relies on (CAS, actor-scoped idempotent claims). */
function fakeServer() {
  // `subject` is the account the shared cookie names now (what whoami answers).
  const server = { routines: new Map(), runs: new Map(), calls: [], hooks: [], serial: 0, subject: 'synthetic-operator' };
  server.add = (definition, extra = {}) => {
    const id = 'r' + (++server.serial);
    server.routines.set(id, { routine_id: id, revision: 1, created_by: 'synthetic-operator', created_at: '2026-11-10T00:00:00Z', ...JSON.parse(JSON.stringify(definition)), latest_run: null, ...extra });
    return server.routines.get(id);
  };
  server.run = (routineId, status, actions) => ({ run_id: 'run' + (++server.serial), routine_id: routineId, occurrence_key: '[]', actor: 'synthetic-operator', source: 'manual', status,
    snapshot: {}, scheduled_for: null, reason: null, actions: actions || [{ index: 0, entity_id: lamp.entity_id, status: status === 'running' ? 'pending' : 'confirmed', reason: null }] });
  server.handle = (path, options = {}) => {
    const method = options.method || 'GET', url = new URL(path, 'http://x'), parts = url.pathname.split('/').filter(Boolean);
    if (url.pathname === '/api/admin/auth/whoami') { if (!server.subject) throw failure(401, 'not_authenticated'); return { ok: true, role: 'operator', permissions: MANAGER, subject: server.subject }; }
    if (url.pathname === '/api/home/catalog') return { rooms: [], entities: [lamp, ac], unassigned_entities: [] };
    if (url.pathname === '/api/routines' && method === 'GET') return [...server.routines.values()].map(item => JSON.parse(JSON.stringify(item)));
    if (url.pathname === '/api/routines' && method === 'POST') { const created = server.add(options.body); return JSON.parse(JSON.stringify(created)); }
    if (url.pathname === '/api/routines/preview') return JSON.parse(JSON.stringify(options.body));
    const routine = server.routines.get(parts[2]);
    if (!routine) throw failure(404, 'routine_not_found');
    const tail = parts.slice(3).join('/');
    if (method === 'PUT') {
      if (options.body.expected_revision !== routine.revision) throw failure(409, 'revision_conflict');
      const { expected_revision: _, ...body } = options.body; Object.assign(routine, body, { revision: routine.revision + 1 }); return JSON.parse(JSON.stringify(routine));
    }
    if (method === 'DELETE') {
      if (Number(url.searchParams.get('expected_revision')) !== routine.revision) throw failure(409, 'revision_conflict');
      server.routines.delete(routine.routine_id); return { success: true };
    }
    if (['enable', 'disable'].includes(tail)) {
      if (options.body.expected_revision !== routine.revision) throw failure(409, 'revision_conflict');
      routine.enabled = tail === 'enable'; routine.revision++; return JSON.parse(JSON.stringify(routine));
    }
    if (tail === 'execute') {
      const key = routine.routine_id + ':' + options.body.request_id;
      if (!server.runs.has(key)) {
        if (!routine.enabled) throw failure(409, 'routine_disabled');
        const run = server.run(routine.routine_id, 'succeeded'); server.runs.set(key, run); routine.latest_run = run;
      }
      return JSON.parse(JSON.stringify(server.runs.get(key)));
    }
    if (tail === 'attempts') {
      const run = server.runs.get(routine.routine_id + ':' + url.searchParams.get('request_id'));
      if (!run) throw failure(404, 'routine_not_found'); return JSON.parse(JSON.stringify(run));
    }
    if (parts[3] === 'runs') { const run = [...server.runs.values(), routine.latest_run].find(item => item?.run_id === parts[4]); if (!run) throw failure(404, 'routine_not_found'); return JSON.parse(JSON.stringify(run)); }
    throw failure(404, 'not_found');
  };
  server.api = async (path, options = {}) => {
    const method = options.method || 'GET';
    server.calls.push({ method, path, body: options.body, signal: options.signal });
    const hook = server.hooks.find(item => item.method === method && path.startsWith(item.path));
    if (hook) {
      server.hooks.splice(server.hooks.indexOf(hook), 1);
      if (hook.mode === 'reject') throw hook.error;
      if (hook.mode === 'hold') await hook.until;
      if (hook.mode === 'drop' || hook.mode === 'hold-drop') { if (hook.mode === 'hold-drop') await hook.until; server.handle(path, options); throw new TypeError('Synthetic lost response'); }
      if (hook.mode === 'commit-then') { server.handle(path, options); throw hook.error; }
    }
    await Promise.resolve();
    return server.handle(path, options);
  };
  return server;
}
function owner({ server = fakeServer(), permissions = MANAGER, storage = memoryStorage(), ids, now, subject = 'synthetic-operator' } = {}) {
  let serial = 0;
  const instance = routines.createRoutineOwner({ api: server.api, session: session(permissions, subject), storage, makeId: ids || (() => 'req-' + (++serial)),
    schedule: () => 1, cancel: () => {}, ...(now ? { now } : {}) });
  return { instance, server, storage };
}
async function opened(options) { const value = owner(options); await value.instance.open(); await value.instance.refresh(); return value; }
const executes = server => server.calls.filter(call => call.path.endsWith('/execute'));

test('legacy contract: typed validation and raw brightness serialization (carried over from the retired legacy controller)', async () => {
  const draft = { ...spec('test'), actions: [{ entity_id: ac.entity_id, operation: 'set_temperature', temperature: 23.5 }] };
  assert.equal(routines.validateRoutineDraft(draft, { entities: [ac] }).valid, true);
  for (const temperature of [undefined, null, 30, 22.2]) assert.equal(routines.validateRoutineDraft({ ...draft, actions: [{ ...draft.actions[0], temperature }] }, { entities: [ac] }).valid, false);
  assert.equal(routines.validateRoutineDraft({ ...draft, actions: [{ entity_id: ac.entity_id, operation: 'set_hvac_mode', hvac_mode: 'heat' }] }, { entities: [ac] }).valid, false);
  const raw = { ...spec('test'), actions: [{ entity_id: lamp.entity_id, operation: 'color_temp_kelvin', color_temp_kelvin: 2700, brightness: 179 }] };
  const { instance, server } = await opened();
  instance.newDraft(); instance.setDraft(raw);
  assert.equal((await instance.save()).ok, true);
  const post = server.calls.find(call => call.method === 'POST' && call.path === '/api/routines');
  assert.deepEqual(post.body.actions, raw.actions);
  assert.deepEqual(Object.keys(post.body).sort(), ['actions', 'conditions', 'description', 'enabled', 'name', 'schedule', 'timezone'], 'only editable fields leave the browser');
  assert.equal(home.controlFields(lamp, 'color_temp_kelvin', raw.actions[0])[1].name, 'brightness');
});

test('LD7: a new schedule, condition or draft has no invented values', () => {
  assert.deepEqual(routines.emptyDraft(), { name: '', description: '', enabled: true, timezone: TZ, schedule: null, conditions: [], actions: [] });
  const errors = routines.validateRoutineDraft({ ...spec(), schedule: { at: '', weekdays: [] }, conditions: [{ entity_id: '', state: '' }] }, { entities: [lamp] }).errors;
  assert.ok(errors.includes('Elegí una hora válida.') && errors.includes('Elegí al menos un día, sin repetir.'));
  assert.ok(errors.some(text => text.includes('condición 1')));
});

test('execute persists the request identity before sending; without storage nothing is sent', async () => {
  const storage = memoryStorage(), { instance, server } = await opened({ storage });
  server.add(spec()); await instance.refresh();
  let storedAtSend = null;
  const original = server.api;
  server.api = async (path, options) => { if (path.endsWith('/execute')) storedAtSend = storage.getItem(routines.PREFIX + 'r1'); return original(path, options); };
  const { instance: second } = owner({ server, storage });
  await second.open(); await second.refresh();
  await second.execute('r1');
  assert.equal(JSON.parse(storedAtSend).request_id, 'req-1');
  const failing = { ...memoryStorage(), setItem() { throw new Error('quota'); } };
  const third = owner({ server, storage: failing });
  await third.instance.open(); await third.instance.refresh();
  const before = executes(server).length;
  assert.deepEqual(await third.instance.execute('r1'), { sent: false });
  assert.equal(executes(server).length, before);
  assert.match(third.instance.getState().notices.r1, /no se envió/);
  instance.leave();
});

test('lost execute response stays unresolved, retries reuse the identity and a new one appears only after a terminal receipt', async () => {
  const { instance, server } = await opened();
  server.add(spec()); await instance.refresh();
  server.hooks.push({ method: 'POST', path: '/api/routines/r1/execute', mode: 'drop' });
  const lost = await instance.execute('r1');
  assert.equal(lost.uncertain, true);
  assert.equal(instance.getState().attempts.r1.state, 'unresolved');
  assert.deepEqual(await instance.execute('r1'), { sent: false }, 'no fresh execute while unresolved');
  await instance.retryAttempt('r1');
  assert.deepEqual(executes(server).map(call => call.body.request_id), ['req-1', 'req-1']);
  assert.equal(server.runs.size, 1, 'the server holds exactly one run');
  assert.equal(instance.getState().attempts.r1.state, 'terminal');
  await instance.execute('r1');
  assert.equal(executes(server).at(-1).body.request_id, 'req-2');
});

test('F1: pending identities survive a reload (new owner, same tab storage) and are wiped only at session end', async () => {
  const storage = memoryStorage(), server = fakeServer();
  server.add(spec());
  const first = owner({ server, storage }); await first.instance.open(); await first.instance.refresh();
  server.hooks.push({ method: 'POST', path: '/api/routines/r1/execute', mode: 'drop' });
  await first.instance.execute('r1');
  server.hooks.push({ method: 'PUT', path: '/api/routines/r1', mode: 'reject', error: new TypeError('offline') });
  // pagehide: the page clears its state but keeps this tab's identities.
  await first.instance.clear({ wipe: false });
  assert.deepEqual(JSON.parse(storage.getItem(routines.PREFIX + 'r1')), { request_id: 'req-1', run_id: null, state: 'unresolved', subject: 'synthetic-operator' });
  const reloaded = owner({ server, storage, ids: () => 'fresh' }); await reloaded.instance.open(); await reloaded.instance.refresh();
  assert.equal(reloaded.instance.getState().attempts.r1.state, 'unresolved');
  await reloaded.instance.consult('r1');
  const state = reloaded.instance.getState();
  assert.equal(state.attempts.r1.run.run_id, [...server.runs.values()][0].run_id, 'consult finds the same run');
  assert.equal(server.runs.size, 1);
  storage.setItem(routines.WRITE_PREFIX + 'r1', JSON.stringify({ kind: 'disable', routine_id: 'r1', expected_revision: 9, subject: 'synthetic-operator' }));
  storage.setItem('kza:other', 'kept');
  await reloaded.instance.clear();
  assert.deepEqual([...storage.map.keys()].sort(), ['kza:other', routines.ENDED_KEY].sort(), 'logout or session end wipes attempts and write markers, leaving only a content-free reminder');
});

test('a terminal flag in storage is not trusted and malformed records are discarded', async () => {
  const storage = memoryStorage(), server = fakeServer(); server.add(spec());
  storage.setItem(routines.PREFIX + 'r1', JSON.stringify({ request_id: 'own-id', run_id: 'run-x', state: 'terminal', subject: 'synthetic-operator' }));
  storage.setItem(routines.PREFIX + 'r2', '{not json');
  const { instance } = owner({ server, storage }); await instance.open(); await instance.refresh();
  assert.equal(instance.getState().attempts.r1.state, 'unresolved');
  assert.equal(storage.getItem(routines.PREFIX + 'r2'), null);
});

test('F3: only answers that prove no claim are definitive; retries and post-response refusals stay uncertain', () => {
  const cases = [
    [{ sent: false, status: 403 }, true, 'none'], [{ sent: false, status: 403 }, false, 'none'],
    [failure(401), true, 'none'], [failure(403, 'dashboard_permission_denied'), true, 'none'], [failure(403, 'routine_write_denied'), true, 'none'],
    [failure(401), false, 'unknown'], [failure(403, 'permission denied'), false, 'unknown'],
    [failure(404, 'routine_not_found'), false, 'none'], [failure(422, [{ loc: ['body'], msg: 'x' }]), false, 'none'],
    [failure(409, 'routine_disabled'), false, 'none'], [failure(409, 'routine_busy'), false, 'none'], [failure(409, 'occurrence_conflict'), true, 'unknown'],
    [failure(503, 'routines unavailable'), true, 'unknown'], [failure(500), true, 'unknown'], [new TypeError('Failed to fetch'), true, 'unknown'],
    [failure(403, 'dashboard_session_retired', { sent: true, answered: true }), true, 'unknown'],
  ];
  for (const [error, first, expected] of cases) assert.equal(routines.executeOutcome(error, first), expected, JSON.stringify({ status: error.status, detail: error.detail, first }));
});

test('F3/LD1/LD11: a definitive refusal drops the attempt and restores Ejecutar; a refusal on retry keeps it', async () => {
  const { instance, server } = await opened();
  server.add(spec()); await instance.refresh();
  server.hooks.push({ method: 'POST', path: '/api/routines/r1/execute', mode: 'reject', error: failure(403, 'routine_write_denied') });
  assert.equal((await instance.execute('r1')).executed, false);
  assert.equal(instance.getState().attempts.r1, undefined);
  assert.match(instance.getState().notices.r1, /^No se ejecutó/);
  server.hooks.push({ method: 'POST', path: '/api/routines/r1/execute', mode: 'drop' });
  await instance.execute('r1');
  server.hooks.push({ method: 'POST', path: '/api/routines/r1/execute', mode: 'reject', error: failure(403, 'permission denied') });
  await instance.retryAttempt('r1');
  assert.equal(instance.getState().attempts.r1.state, 'unresolved', '403 on a retry says nothing about the first send');
  server.hooks.push({ method: 'POST', path: '/api/routines/r1/execute', mode: 'reject', error: failure(409, 'routine_busy') });
  await instance.retryAttempt('r1');
  assert.equal(instance.getState().attempts.r1, undefined, 'busy is evaluated after the claim lookup: no claim for this id');
  server.routines.get('r1').enabled = false;
  await instance.execute('r1');
  assert.equal(instance.getState().attempts.r1, undefined);
  assert.match(instance.getState().notices.r1, /desactivada/);
});

test('F4: a new execute after an uncertain, interrupted or partially failed run needs explicit confirmation', async () => {
  const { instance, server } = await opened();
  const uncertain = server.add(spec('Incierta'));
  uncertain.latest_run = { ...server.run(uncertain.routine_id, 'uncertain'), source: 'schedule', actions: [{ index: 0, entity_id: lamp.entity_id, status: 'uncertain', reason: null }] };
  const partial = server.add(spec('Parcial'));
  partial.latest_run = server.run(partial.routine_id, 'failed', [{ index: 0, entity_id: lamp.entity_id, status: 'confirmed', reason: null }, { index: 1, entity_id: ac.entity_id, status: 'failed', reason: 'x' }]);
  const fine = server.add(spec('Bien'));
  fine.latest_run = server.run(fine.routine_id, 'succeeded');
  const running = server.add(spec('En curso'));
  running.latest_run = server.run(running.routine_id, 'running');
  await instance.refresh();
  const asked = await instance.execute(uncertain.routine_id);
  assert.deepEqual(asked.needsConfirmation, [{ index: 0, entity_id: lamp.entity_id, status: 'uncertain' }]);
  assert.equal(executes(server).length, 0);
  assert.deepEqual((await instance.execute(partial.routine_id)).needsConfirmation.map(item => item.entity_id), [ac.entity_id]);
  assert.equal((await instance.execute(fine.routine_id)).sent, true);
  assert.equal((await instance.execute(running.routine_id)).sent, false);
  assert.equal((await instance.execute(uncertain.routine_id, { confirmed: true })).sent, true);
  // Own attempt uncertain: the next run asks again even if the list has not refreshed.
  server.hooks.push({ method: 'POST', path: `/api/routines/${fine.routine_id}/execute`, mode: 'drop' });
  await instance.execute(fine.routine_id);
  const own = [...server.runs.values()].at(-1); own.status = 'interrupted';
  await instance.consult(fine.routine_id);
  assert.ok((await instance.execute(fine.routine_id)).needsConfirmation);
});

test('F2: leaving the view never aborts a write and its late receipt is applied', async () => {
  const { instance, server } = await opened();
  server.add(spec()); await instance.refresh();
  let release; const until = new Promise(resolve => { release = resolve; });
  server.hooks.push({ method: 'POST', path: '/api/routines/r1/execute', mode: 'hold', until });
  const pending = instance.execute('r1');
  await Promise.resolve();
  const call = executes(server)[0];
  assert.equal(call.signal, undefined, 'writes carry no abort signal');
  assert.equal(await instance.leave(), true);
  release(); await pending;
  assert.equal(instance.getState().attempts.r1.state, 'terminal');
  assert.equal(server.runs.size, 1);
  const reads = server.calls.filter(item => item.method === 'GET');
  assert.ok(reads.every(item => item.signal instanceof AbortSignal), 'reads remain abortable');
});

test('F5: lost write responses are resolved by comparison before any retry is offered', async () => {
  for (const [label, act, mutate, expected] of [
    ['update applied', o => { o.edit('r1'); const d = o.getState().draft.value; d.name = 'Cambiada'; o.setDraft(d); return o.save(); }, null, 'El cambio se aplicó.'],
    ['update not applied', o => { o.edit('r1'); const d = o.getState().draft.value; d.name = 'Cambiada'; o.setDraft(d); return o.save(); }, 'reject', 'El cambio no se aplicó. Podés volver a intentarlo.'],
    ['update changed elsewhere', o => { o.edit('r1'); const d = o.getState().draft.value; d.name = 'Cambiada'; o.setDraft(d); return o.save(); }, 'other', 'La rutina cambió en otra sesión. Revisá antes de volver a intentarlo.'],
    ['disable applied', o => o.setEnabled('r1', false), null, 'La rutina quedó desactivada.'],
    ['delete applied', o => o.remove('r1'), null, 'La rutina se borró.'],
    ['delete not applied', o => o.remove('r1'), 'reject', 'El cambio no se aplicó. Podés volver a intentarlo.'],
  ]) {
    const { instance, server, storage } = await opened();
    server.add(spec()); await instance.refresh();
    const method = label.startsWith('update') ? 'PUT' : label.startsWith('disable') ? 'POST' : 'DELETE';
    server.hooks.push({ method, path: '/api/routines/r1', mode: mutate === 'reject' ? 'reject' : 'drop', error: new TypeError('offline') });
    if (mutate === 'other') { server.hooks[0] = { method, path: '/api/routines/r1', mode: 'reject', error: failure(502) }; server.routines.get('r1').revision = 3; }
    const resultPromise = act(instance);
    assert.ok(storage.getItem(routines.WRITE_PREFIX + 'r1'), `${label}: marker stored before the response`);
    const result = await resultPromise;
    assert.equal(result.unknown, true, label);
    await instance.refresh();
    const state = instance.getState();
    assert.equal(state.notices.r1, expected, label);
    assert.equal(state.writes.r1, undefined, label);
    assert.equal(storage.getItem(routines.WRITE_PREFIX + 'r1'), null, label);
    if (label === 'update changed elsewhere') assert.equal(state.draft.error.conflict, true);
    if (label === 'update applied') assert.equal(state.draft, null, 'an applied save closes the editor');
    if (label === 'update not applied') assert.equal(state.draft.unknown, false, 'retry becomes possible');
  }
});

test('F5: a lost create is found by its definition or reported as not saved; markers survive reload', async () => {
  const storage = memoryStorage(), server = fakeServer();
  const first = owner({ server, storage }); await first.instance.open(); await first.instance.refresh();
  first.instance.newDraft(); first.instance.setDraft(spec('Nueva'));
  server.hooks.push({ method: 'POST', path: '/api/routines', mode: 'drop' });
  const pending = first.instance.save();
  await first.instance.clear({ wipe: false });
  await pending;
  const second = owner({ server, storage }); await second.instance.open(); await second.instance.refresh();
  assert.equal(second.instance.getState().notices.new, 'La rutina se guardó.');
  assert.equal(server.routines.size, 1);
  second.instance.newDraft(); second.instance.setDraft(spec('Otra'));
  server.hooks.push({ method: 'POST', path: '/api/routines', mode: 'reject', error: new TypeError('offline') });
  await second.instance.save();
  await second.instance.refresh();
  assert.equal(second.instance.getState().notices.new, 'El cambio no se aplicó. Podés volver a intentarlo.');
  assert.equal(second.instance.getState().draft.unknown, false);
});

test('a definitive write answer removes the marker and keeps the draft; 409 revision_conflict offers reload', async () => {
  const { instance, server, storage } = await opened();
  server.add(spec()); await instance.refresh();
  instance.edit('r1'); const draft = instance.getState().draft.value; draft.name = 'Mía'; instance.setDraft(draft);
  server.routines.get('r1').revision = 2;
  const result = await instance.save();
  assert.equal(result.ok, false);
  assert.equal(storage.getItem(routines.WRITE_PREFIX + 'r1'), null);
  assert.equal(instance.getState().draft.error.conflict, true);
  assert.equal(instance.getState().draft.value.name, 'Mía');
  await instance.discardAndReload();
  assert.equal(instance.getState().draft.editingRevision, 2);
});

test('F8: error copy covers every 403 detail, unknown 409 codes and list-shaped 422 without rendering raw detail', () => {
  assert.equal(routines.describeError(failure(403, 'dashboard_permission_denied')), 'La sesión no tiene permiso para esta operación.');
  assert.equal(routines.describeError(failure(403, 'permission denied')), 'La sesión no tiene permiso para esta operación.');
  assert.equal(routines.describeError(failure(403, 'routine_write_denied')), 'La sesión no tiene permiso para esta operación.');
  assert.equal(routines.describeError(failure(403, 'origin denied')), 'Este navegador no está autorizado para operar KZA desde este origen.');
  assert.equal(routines.describeError(failure(403, 'dashboard_authority_retired')), 'La sesión dejó de estar vigente.');
  assert.equal(routines.describeError(failure(409, 'occurrence_conflict')), 'La rutina no aceptó el cambio; actualizá.');
  const list = routines.describeError(failure(422, [{ loc: ['body', 'name'], msg: 'Field required', type: 'missing' }]));
  assert.equal(list, 'El servidor rechazó la definición. Revisá sus campos; conservamos tu borrador.');
  assert.equal(routines.describeError(failure(422, 'invalid_routine')), list);
  assert.equal(routines.describeError(failure(503, 'routines unavailable')), 'Rutinas no disponibles en este servidor.');
  assert.equal(routines.describeError(new TypeError('Failed to fetch')), 'No se pudo contactar al servidor.');
});

test('LD4/LD12: list and catalog keep separate status; the catalog is refreshed and a stale one blocks saving', async () => {
  let clock = 1000;
  const server = fakeServer(); server.add(spec());
  const original = server.handle; let catalogFails = true;
  server.handle = (path, options) => { if (path === '/api/home/catalog' && catalogFails) throw failure(503); return original(path, options); };
  const { instance } = await opened({ server, now: () => clock });
  let state = instance.getState();
  assert.equal(state.list.error, null); assert.equal(state.routines.length, 1);
  assert.ok(state.catalogStatus.error); assert.equal(state.catalogUsable, false);
  instance.edit('r1');
  assert.equal((await instance.save()).invalid, true);
  assert.ok(instance.getState().draft.validation.includes('Actualizá el catálogo de dispositivos antes de guardar.'));
  catalogFails = false; await instance.refresh();
  assert.equal(instance.getState().catalogUsable, true);
  clock += 300001;
  assert.equal(instance.getState().catalogUsable, false, 'catalog older than 5 minutes is stale');
  const catalogReads = server.calls.filter(call => call.path === '/api/home/catalog').length;
  await instance.refresh();
  assert.equal(server.calls.filter(call => call.path === '/api/home/catalog').length, catalogReads + 1);
});

test('LD1/LD6/F14: household and role-labelled sessions are gated by permission; household polling reads only the list', async () => {
  const server = fakeServer(); const routine = server.add(spec()); routine.latest_run = server.run(routine.routine_id, 'running');
  let scheduled = null;
  const household = routines.createRoutineOwner({ api: server.api, session: session(['home:read', 'home:control', 'routines:execute']), storage: memoryStorage(),
    schedule: callback => { scheduled = callback; return 1; }, cancel: () => {} });
  await household.open(); await household.refresh();
  assert.deepEqual(household.getState().capabilities, { read: true, manage: false, execute: false });
  assert.deepEqual(await household.execute('r1'), { sent: false });
  assert.equal(household.newDraft(), false); assert.equal(household.edit('r1'), false);
  assert.deepEqual(await household.setEnabled('r1', false), { ok: false }); assert.deepEqual(await household.remove('r1'), { ok: false });
  assert.ok(scheduled, 'a running receipt is polled');
  const before = server.calls.length; await scheduled();
  assert.ok(server.calls.slice(before).every(call => call.method === 'GET' && ['/api/routines', '/api/home/catalog'].includes(call.path)), 'no runs/attempts reads without routines:manage');
  const labelled = routines.createRoutineOwner({ api: server.api, session: session(['home:read', 'home:control']), storage: memoryStorage(), schedule: () => 1, cancel: () => {} });
  assert.deepEqual(labelled.capabilities(), { read: true, manage: false, execute: false });
  assert.equal(server.calls.filter(call => call.method !== 'GET').length, 0);
});

test('responses that arrive after a session boundary are ignored and never re-persist identities', async () => {
  const storage = memoryStorage(), { instance, server } = await opened({ storage });
  server.add(spec()); await instance.refresh();
  let release; const until = new Promise(resolve => { release = resolve; });
  server.hooks.push({ method: 'POST', path: '/api/routines/r1/execute', mode: 'hold', until });
  const pending = instance.execute('r1');
  await Promise.resolve();
  await instance.clear();
  release(); await pending;
  assert.deepEqual([...storage.map.keys()], [routines.ENDED_KEY], 'no identity re-persisted; only the content-free reminder of an unconfirmed run');
  assert.deepEqual(instance.getState().attempts, {});
});

test('receipts with another routine or run identity are not accepted', async () => {
  const { instance, server } = await opened();
  server.add(spec()); await instance.refresh();
  const original = server.handle;
  server.handle = (path, options) => { const value = original(path, options); return path.endsWith('/execute') ? { ...value, routine_id: 'other' } : value; };
  await instance.execute('r1');
  assert.equal(instance.getState().attempts.r1.state, 'unresolved');
  assert.match(instance.getState().notices.r1, /Resultado pendiente de consulta/);
});

test('announcements change only on transitions, not on polls or refreshes', async () => {
  const { instance, server } = await opened();
  server.add(spec()); await instance.refresh();
  const before = instance.getState().announcement;
  await instance.refresh(); await instance.refresh();
  assert.deepEqual(instance.getState().announcement, before);
  await instance.execute('r1');
  assert.notDeepEqual(instance.getState().announcement, before);
});

test('polls are quiet and announce the end of an active run once', async () => {
  const { instance, server } = await opened();
  server.add(spec()); await instance.refresh();
  const handle = server.handle;
  server.handle = (path, options) => {
    const value = handle(path, options);
    if (path.endsWith('/execute')) { const stored = [...server.runs.values()].at(-1); stored.status = 'running'; return { ...value, status: 'running' }; }
    return value;
  };
  await instance.execute('r1');
  assert.equal(instance.getState().attempts.r1.state, 'active');
  [...server.runs.values()].at(-1).status = 'succeeded';
  let release; const until = new Promise(resolve => { release = resolve; });
  server.hooks.push({ method: 'GET', path: '/api/routines/r1/runs/', mode: 'hold', until });
  const polling = instance.consult('r1', true);
  await Promise.resolve(); await Promise.resolve();
  assert.deepEqual(instance.getState().busy, [], 'a poll never disables controls');
  release(); await polling;
  assert.equal(instance.getState().attempts.r1.state, 'terminal');
  assert.equal(instance.getState().announcement.text, 'La ejecución terminó.');
});

// R1: the reviewer's probe. A list read in flight before a lost write must not decide its outcome.
async function heldListOwner() {
  const server = fakeServer();
  let releaseList = null;
  const handle = server.api;
  server.api = async (path, options = {}) => {
    if (path === '/api/routines' && (options.method || 'GET') === 'GET' && server.holdNextList) {
      server.holdNextList = false;
      const snapshot = server.handle(path, options); // evaluated before the write commits
      server.calls.push({ method: 'GET', path });
      await new Promise(resolve => { releaseList = resolve; });
      return snapshot;
    }
    return handle(path, options);
  };
  const { instance } = await opened({ server });
  return { server, instance, release: () => releaseList() };
}
test('R1: a lost create resolved only by a read issued after the loss; the retry offered cannot duplicate', async () => {
  const { server, instance, release } = await heldListOwner();
  instance.newDraft(); instance.setDraft(spec('Nueva'));
  server.holdNextList = true; const inflight = instance.refresh();
  await new Promise(resolve => setImmediate(resolve));
  server.hooks.push({ method: 'POST', path: '/api/routines', mode: 'drop' });
  assert.equal((await instance.save()).unknown, true);
  release(); await inflight; await instance.refresh();
  const state = instance.getState();
  assert.equal(state.notices.new, 'La rutina se guardó.', 'the committed create is found');
  assert.equal(state.draft, null);
  assert.equal(server.routines.size, 1);
  assert.deepEqual(await instance.save(), { ok: false }, 'nothing left to resubmit');
  assert.equal(server.routines.size, 1);
});
for (const [label, act, method, expected] of [
  ['update', o => { o.edit('r1'); const d = o.getState().draft.value; d.name = 'Cambiada'; o.setDraft(d); return o.save(); }, 'PUT', 'El cambio se aplicó.'],
  ['enable/disable', o => o.setEnabled('r1', false), 'POST', 'La rutina quedó desactivada.'],
  ['delete', o => o.remove('r1'), 'DELETE', 'La rutina se borró.'],
]) {
  test(`R1: a lost ${label} is not judged by a read that was already in flight`, async () => {
    const { server, instance, release } = await heldListOwner();
    server.add(spec()); await instance.refresh();
    server.holdNextList = true; const inflight = instance.refresh();
    await new Promise(resolve => setImmediate(resolve));
    server.hooks.push({ method, path: '/api/routines/r1', mode: 'drop' });
    assert.equal((await act(instance)).unknown, true);
    release(); await inflight; await instance.refresh();
    assert.equal(instance.getState().notices.r1, expected);
  });
}

test('R2: records of another account are dropped on restore and never consulted or retried', async () => {
  const storage = memoryStorage(), server = fakeServer();
  server.add(spec()); server.subject = 'operator-a';
  const first = owner({ server, storage, subject: 'operator-a' }); await first.instance.open(); await first.instance.refresh();
  server.hooks.push({ method: 'POST', path: '/api/routines/r1/execute', mode: 'drop' });
  await first.instance.execute('r1');
  server.hooks.push({ method: 'POST', path: '/api/routines/r1/disable', mode: 'reject', error: new TypeError('offline') });
  await first.instance.setEnabled('r1', false);
  storage.setItem(routines.PREFIX + 'r9', JSON.stringify({ request_id: 'legacy-unbound', run_id: null, state: 'unresolved' }));
  await first.instance.clear({ wipe: false }); // pagehide
  server.subject = 'operator-b';
  const before = server.calls.length;
  const second = owner({ server, storage, subject: 'operator-b' }); await second.instance.open(); await second.instance.refresh();
  const state = second.instance.getState();
  assert.deepEqual(state.attempts, {}); assert.deepEqual(state.writes, {});
  assert.deepEqual([...storage.map.keys()], [], 'foreign and unbound records are removed');
  assert.match(state.notices.session, /otra sesión quedó sin confirmar/);
  assert.deepEqual(await second.instance.retryAttempt('r1'), { sent: false });
  assert.ok(server.calls.slice(before).every(call => call.method === 'GET' && !call.path.includes('attempts')), 'no consult or resend of the foreign identity');
});

test('R2: a retry refuses an in-memory attempt whose account differs from the session', async () => {
  let current = 'operator-a';
  const server = fakeServer(); server.add(spec()); server.subject = 'operator-a';
  const instance = routines.createRoutineOwner({ api: server.api, session: { can: () => true, subject: () => current }, storage: memoryStorage(), makeId: () => 'id-a', schedule: () => 1, cancel: () => {} });
  await instance.open(); await instance.refresh();
  server.hooks.push({ method: 'POST', path: '/api/routines/r1/execute', mode: 'drop' });
  await instance.execute('r1');
  current = 'operator-b';
  assert.deepEqual(await instance.retryAttempt('r1'), { sent: false });
  assert.equal(executes(server).length, 1);
  assert.equal(instance.getState().attempts.r1, undefined);
});

test('R2: without a session subject nothing is sent', async () => {
  const { instance, server } = await opened({ subject: null });
  server.add(spec()); await instance.refresh();
  assert.deepEqual(await instance.execute('r1'), { sent: false });
  assert.match(instance.getState().notices.r1, /identificar la cuenta/);
  assert.equal(executes(server).length, 0);
});

test('R5: a session end that wipes an unconfirmed run leaves only a reminder shown to the next session', async () => {
  const storage = memoryStorage(), server = fakeServer(); server.add(spec()); server.subject = 'operator-a';
  const first = owner({ server, storage, subject: 'operator-a' }); await first.instance.open(); await first.instance.refresh();
  server.hooks.push({ method: 'POST', path: '/api/routines/r1/execute', mode: 'drop' });
  await first.instance.execute('r1');
  await first.instance.clear(); server.subject = 'operator-b';
  assert.equal(storage.getItem(routines.ENDED_KEY), '1');
  assert.ok(![...storage.map.keys()].some(key => key.startsWith(routines.PREFIX)));
  const second = owner({ server, storage, subject: 'operator-b' }); await second.instance.open();
  assert.match(second.instance.getState().notices.session, /sin confirmar/);
  assert.equal(storage.getItem(routines.ENDED_KEY), null);
});

test('R6: closing the editor during an uncertain save keeps the draft for recovery if it was not applied', async () => {
  const { instance, server } = await opened();
  instance.newDraft(); instance.setDraft(spec('Conservada'));
  server.hooks.push({ method: 'POST', path: '/api/routines', mode: 'reject', error: new TypeError('offline') });
  assert.equal((await instance.save()).unknown, true);
  instance.closeEditor();
  assert.equal(instance.getState().draft.open, false);
  await instance.refresh();
  let draft = instance.getState().draft;
  assert.equal(draft.unknown, false); assert.equal(draft.recoverable, true); assert.equal(draft.value.name, 'Conservada');
  instance.reopenDraft();
  draft = instance.getState().draft;
  assert.equal(draft.open, true); assert.equal(draft.value.name, 'Conservada');
  instance.closeEditor();
  assert.equal(instance.getState().draft, null, 'an explicit close after recovery discards');
});

/** Probe 2B: the cookie switches to another account while this tab keeps its cached session. */
async function takenOver() {
  const server = fakeServer(); server.add(spec()); server.subject = 'operator-a';
  // Server claims are actor-scoped: the same request_id under another cookie is a new run.
  const handle = server.handle;
  server.handle = (path, options = {}) => (path.endsWith('/execute') ? handle(path, { ...options, body: { request_id: server.subject + ':' + options.body.request_id } }) : handle(path, options));
  const storage = memoryStorage(); let changed = 0;
  const instance = routines.createRoutineOwner({ api: server.api, session: session(MANAGER, 'operator-a'), storage, makeId: () => 'id-a', schedule: () => 1, cancel: () => {},
    onAccountChanged: () => { changed++; } });
  await instance.open(); await instance.refresh();
  server.hooks.push({ method: 'POST', path: '/api/routines/r1/execute', mode: 'drop' });
  await instance.execute('r1');
  server.subject = 'operator-b';
  return { instance, server, storage, changed: () => changed };
}
const whoamis = server => server.calls.filter(call => call.path === '/api/admin/auth/whoami');

test('R2b: returning to a hidden tab after another tab switched accounts sends nothing and drops the records', async () => {
  const { instance, server, storage, changed } = await takenOver();
  await instance.leave(); await instance.open(); await instance.refresh();
  assert.ok(whoamis(server).length >= 1, 'the account is confirmed on return');
  assert.equal(changed(), 1, 'the page is asked to re-read its session');
  const state = instance.getState();
  assert.deepEqual(state.attempts, {});
  assert.match(state.notices.session, /Revisá el último resultado/);
  assert.ok(![...storage.map.keys()].some(key => key.startsWith(routines.PREFIX)), 'no identity survives');
  assert.equal(storage.getItem(routines.ENDED_KEY), '1', 'the next session is reminded');
  assert.deepEqual(await instance.retryAttempt('r1'), { sent: false });
  assert.equal(executes(server).length, 1);
  assert.equal(server.runs.size, 1, 'no run under the second account');
});

test('R2b: a retry or a consult confirms the account first even without leaving the view', async () => {
  for (const act of [instance => instance.retryAttempt('r1'), instance => instance.consult('r1')]) {
    const { instance, server, changed } = await takenOver();
    const before = server.calls.length;
    await act(instance);
    const after = server.calls.slice(before);
    assert.equal(after[0]?.path, '/api/admin/auth/whoami');
    assert.ok(!after.some(call => call.path.includes('/execute') || call.path.includes('/attempts')), 'nothing about the attempt is sent under the other account');
    assert.equal(server.runs.size, 1);
    assert.equal(changed(), 1);
    assert.deepEqual(instance.getState().attempts, {});
  }
});

test('R2b: when the account cannot be confirmed nothing is sent and the identity is kept', async () => {
  const { instance, server, storage, changed } = await takenOver();
  server.subject = 'operator-a';
  server.hooks.push({ method: 'GET', path: '/api/admin/auth/whoami', mode: 'reject', error: new TypeError('offline') });
  assert.deepEqual(await instance.retryAttempt('r1'), { sent: false });
  assert.match(instance.getState().notices.r1, /No se pudo confirmar la cuenta/);
  assert.equal(executes(server).length, 1);
  assert.ok(storage.getItem(routines.PREFIX + 'r1'));
  assert.equal(changed(), 0);
  assert.equal((await instance.retryAttempt('r1')).sent, true, 'the confirmed retry reuses the identity');
  assert.equal(server.runs.size, 1, 'and does not duplicate');
});

test('R2b: a whoami answer without a subject counts as another account', async () => {
  const { instance, server, changed } = await takenOver();
  server.subject = 'operator-a';
  const handle = server.handle;
  server.handle = (path, options) => (path === '/api/admin/auth/whoami' ? { ok: true, role: 'operator', permissions: MANAGER } : handle(path, options));
  assert.equal((await instance.retryAttempt('r1')).sent, false);
  assert.equal(changed(), 1);
  assert.equal(executes(server).length, 1);
});

test('R2b: the not-registered message no longer promises a duplicate-free retry without the account check', async () => {
  const { instance, server } = await takenOver();
  server.subject = 'operator-a';
  server.routines.get('r1').latest_run = null; server.runs.clear();
  await instance.consult('r1');
  const text = instance.getState().notices.r1;
  assert.match(text, /no registró este intento para esta cuenta/);
  assert.doesNotMatch(text, /no se duplica/);
});

test('R6b: a kept draft is not replaced by Crear rutina or Editar unless confirmed', async () => {
  const { instance, server } = await opened();
  server.add(spec('Existente')); await instance.refresh();
  instance.newDraft(); instance.setDraft(spec('Conservada'));
  server.hooks.push({ method: 'POST', path: '/api/routines', mode: 'reject', error: new TypeError('offline') });
  await instance.save(); instance.closeEditor(); await instance.refresh();
  const kept = instance.getState().draft;
  assert.equal(kept.recoverable, true);
  assert.equal(instance.hiddenDraft(), true);
  assert.equal(instance.newDraft(), false);
  assert.equal(instance.edit('r1'), false);
  assert.equal(instance.getState().draft.value.name, 'Conservada');
  assert.equal(instance.edit('r1', { replace: true }), true);
  assert.equal(instance.getState().draft.value.name, 'Existente');
});
