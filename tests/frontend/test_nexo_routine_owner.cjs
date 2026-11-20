const assert = require('node:assert/strict');
const { test } = require('node:test');
const home = require('../../src/dashboard/frontend/home/home.js');
const routines = require('../../src/dashboard/frontend/home/nexo/routine-owner.js');
const { mountRoutines } = require('../../src/dashboard/frontend/home/nexo/routines.js');
const { createScopedApi, createSession, ROUTINE_PROPOSALS } = require('../../src/dashboard/frontend/home/nexo/state.js');
const { createDocument, icon, visibleText } = require('../mocks/dom_shim.cjs');

const TZ = 'America/Montevideo';
const lamp = { entity_id: 'light.test_lamp', name: 'Luz principal', domain: 'light', capabilities: ['turn_on', 'turn_off', 'color_temp_kelvin', 'brightness'],
  control_limits: { brightness_min: 0, brightness_max: 100, color_temp_min: 2000, color_temp_max: 6500 } };
const ac = { entity_id: 'climate.test_ac', domain: 'climate', available: true, state: 'cool', capabilities: ['set_temperature', 'set_hvac_mode'],
  control_limits: { temperature_min: 18.5, temperature_max: 25, temperature_step: 0.5, hvac_modes: ['off', 'cool', 'dry'] } };
const spec = (name = 'Noche') => ({ name, description: '', enabled: true, timezone: TZ, schedule: null, conditions: [], actions: [{ entity_id: lamp.entity_id, operation: 'turn_on' }] });
const failure = (status, detail, extra = {}) => Object.assign(new Error('synthetic'), { status, detail, ...extra });
const MANAGER = ['home:read', 'home:control', 'routines:manage'];
const HOUSEHOLD = ['home:read', 'home:control', 'routines:execute'];
const actorFor = subject => subject == null ? null : 'fixture-actor:' + subject;

function memoryStorage() {
  const map = new Map();
  return { map, get length() { return map.size; }, key: index => [...map.keys()][index] ?? null, getItem: key => map.has(key) ? map.get(key) : null,
    setItem: (key, value) => { map.set(key, String(value)); }, removeItem: key => { map.delete(key); } };
}
function session(permissions = MANAGER, subject = 'synthetic-operator', actor = actorFor(subject)) {
  const set = new Set(permissions);
  return { can: permission => set.has(permission), subject: () => subject, routineActor: () => actor, set };
}

/** Minimal server with the store semantics the owner relies on (CAS, actor-scoped idempotent claims). */
function fakeServer() {
  const server = { routines: new Map(), runs: new Map(), proposals: new Map(), calls: [], hooks: [], serial: 0, subject: 'synthetic-operator', permissions: MANAGER };
  const actor = () => server.routineActor ?? actorFor(server.subject);
  const attemptKey = (routineId, requestId) => JSON.stringify([routineId, actor(), requestId]);
  const canReadRun = run => run && (server.permissions.includes('routines:manage') || run.source === 'manual' && run.actor === actor());
  server.add = (definition, extra = {}) => {
    const id = 'r' + (++server.serial);
    server.routines.set(id, { routine_id: id, revision: 1, created_by: 'synthetic-operator', created_at: '2026-11-10T00:00:00Z', ...JSON.parse(JSON.stringify(definition)), latest_run: null, ...extra });
    return server.routines.get(id);
  };
  server.run = (routineId, status, actions) => ({ run_id: 'run' + (++server.serial), routine_id: routineId, occurrence_key: '[]', actor: actor(), source: 'manual', status,
    snapshot: {}, scheduled_for: null, reason: null, actions: actions || [{ index: 0, entity_id: lamp.entity_id, status: status === 'running' ? 'pending' : 'confirmed', reason: null }] });
  server.handle = (path, options = {}) => {
    const method = options.method || 'GET', url = new URL(path, 'http://x'), parts = url.pathname.split('/').filter(Boolean);
    if (url.pathname === '/api/admin/auth/whoami') { if (!server.subject) throw failure(401, 'not_authenticated'); return { ok: true, role: server.role || (server.permissions.includes('routines:manage') ? 'operator' : 'home'), permissions: server.permissions, subject: server.subject, routine_actor: actor() }; }
    if (url.pathname === '/api/home/catalog') return { rooms: [], entities: [lamp, ac], unassigned_entities: [] };
    if (url.pathname === '/api/routines' && method === 'GET') return [...server.routines.values()].map(item => JSON.parse(JSON.stringify({ ...item, latest_run: canReadRun(item.latest_run) ? item.latest_run : null })));
    if (url.pathname === '/api/routines' && method === 'POST') { const created = server.add(options.body); return JSON.parse(JSON.stringify(created)); }
    if (url.pathname === '/api/routines/preview') return JSON.parse(JSON.stringify(options.body));
    if (url.pathname === ROUTINE_PROPOSALS || url.pathname.startsWith(ROUTINE_PROPOSALS + '/')) {
      if (!server.permissions.includes('routines:manage') || server.role === 'evaluator') throw failure(403, 'dashboard_permission_denied');
      if (url.pathname === ROUTINE_PROPOSALS) return {
        proposals: [...server.proposals.values()].filter(item => !item.receipt).map(item => item.summary), cleared_on_restart: true
      };
      const proposal = server.proposals.get(parts[3]);
      if (!proposal || method === 'GET' && proposal.receipt) throw failure(404, 'routine_proposal_not_found');
      if (method === 'GET') return { ...proposal.summary, stale: proposal.stale, diff: proposal.diff };
      const status = parts[4] === 'confirm' ? 'confirmed' : 'rejected';
      if (proposal.receipt) {
        if (proposal.receipt.status !== status) throw failure(409, 'proposal_' + proposal.receipt.status);
        return proposal.receipt;
      }
      if (status === 'confirmed' && proposal.stale) throw failure(409, 'revision_conflict');
      const routine = status === 'confirmed' ? server.add(proposal.diff.after) : null;
      proposal.receipt = { proposal_id: parts[3], status, routine_id: routine?.routine_id ?? null, revision: routine?.revision ?? null };
      return proposal.receipt;
    }
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
      const key = attemptKey(routine.routine_id, options.body.request_id);
      if (!server.runs.has(key)) {
        if (!routine.enabled) throw failure(409, 'routine_disabled');
        const run = server.run(routine.routine_id, 'succeeded'); server.runs.set(key, run); routine.latest_run = run;
      }
      return JSON.parse(JSON.stringify(server.runs.get(key)));
    }
    if (tail === 'attempts') {
      const run = server.runs.get(attemptKey(routine.routine_id, url.searchParams.get('request_id')));
      if (!run) throw failure(404, 'routine_not_found'); return JSON.parse(JSON.stringify(run));
    }
    if (parts[3] === 'runs') { const run = [...server.runs.values(), routine.latest_run].find(item => item?.run_id === parts[4] && item.routine_id === routine.routine_id); if (!canReadRun(run)) throw failure(404, 'routine_not_found'); return JSON.parse(JSON.stringify(run)); }
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
  assert.deepEqual(JSON.parse(storage.getItem(routines.PREFIX + 'r1')), { request_id: 'req-1', run_id: null, state: 'unresolved', actor: actorFor('synthetic-operator') });
  const reloaded = owner({ server, storage, ids: () => 'fresh' }); await reloaded.instance.open(); await reloaded.instance.refresh();
  assert.equal(reloaded.instance.getState().attempts.r1.state, 'unresolved');
  await reloaded.instance.consult('r1');
  const state = reloaded.instance.getState();
  assert.equal(state.attempts.r1.run.run_id, [...server.runs.values()][0].run_id, 'consult finds the same run');
  assert.equal(server.runs.size, 1);
  storage.setItem(routines.WRITE_PREFIX + 'r1', JSON.stringify({ kind: 'disable', routine_id: 'r1', expected_revision: 9, actor: actorFor('synthetic-operator') }));
  storage.setItem('kza:other', 'kept');
  await reloaded.instance.clear();
  assert.deepEqual([...storage.map.keys()].sort(), ['kza:other', routines.ENDED_KEY].sort(), 'logout or session end wipes attempts and write markers, leaving only a content-free reminder');
});

test('a terminal flag in storage is not trusted and malformed records are discarded', async () => {
  const storage = memoryStorage(), server = fakeServer(); server.add(spec());
  storage.setItem(routines.PREFIX + 'r1', JSON.stringify({ request_id: 'own-id', run_id: 'run-x', state: 'terminal', actor: actorFor('synthetic-operator') }));
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

test('T033-U6: household capabilities allow execution and polling of own receipts while management stays closed', async () => {
  const server = fakeServer(); const routine = server.add(spec());
  routine.latest_run = { ...server.run(routine.routine_id, 'running'), actor: 'other-account' };
  server.permissions = HOUSEHOLD; server.subject = 'household-a';
  const identity = createSession({ wall: () => 0, monotonic: () => 0 });
  identity.accept({ ok: true, role: 'home', permissions: HOUSEHOLD, subject: server.subject, routine_actor: actorFor(server.subject), expires_at: 60 });
  const api = createScopedApi({ session: identity, transport: server.api });
  const handle = server.handle;
  server.handle = (path, options) => {
    const value = handle(path, options);
    if (path.endsWith('/execute')) { routine.latest_run.status = 'running'; return { ...value, status: 'running' }; }
    return value;
  };
  let scheduled = null;
  const household = routines.createRoutineOwner({ api, session: identity, storage: memoryStorage(),
    makeId: () => 'home-attempt', schedule: callback => { scheduled = callback; return 1; }, cancel: () => {} });
  await household.open(); await household.refresh();
  assert.deepEqual(household.getState().capabilities, { read: true, manage: false, consult: true, execute: true });
  assert.equal(household.getState().routines[0].latest_run, null);
  assert.equal(scheduled, null, 'a foreign running receipt is not polled');
  assert.equal(household.newDraft(), false); assert.equal(household.edit('r1'), false);
  assert.deepEqual(await household.setEnabled('r1', false), { ok: false }); assert.deepEqual(await household.remove('r1'), { ok: false });
  const { sent, run } = await household.execute('r1');
  assert.equal(sent, true); assert.equal(run.status, 'running');
  assert.ok(scheduled, 'a running receipt is polled');
  routine.latest_run.status = 'succeeded';
  const before = server.calls.length; await scheduled();
  assert.ok(server.calls.slice(before).some(call => call.path === '/api/routines/r1/runs/' + run.run_id));
  assert.equal(household.getState().attempts.r1.state, 'terminal');
  const labelled = routines.createRoutineOwner({ api: server.api, session: session(['home:read', 'home:control']), storage: memoryStorage(), schedule: () => 1, cancel: () => {} });
  assert.deepEqual(labelled.capabilities(), { read: true, manage: false, consult: false, execute: false });
  assert.deepEqual(server.calls.filter(call => call.method !== 'GET').map(call => call.path), ['/api/routines/r1/execute']);
});

test('T033-U6: household consults a lost response by attempt and retries with the same account and ID', async () => {
  for (const recover of [instance => instance.consult('r1'), instance => instance.retryAttempt('r1')]) {
    const server = fakeServer(); server.add(spec()); server.permissions = HOUSEHOLD; server.subject = 'household-a';
    const { instance, storage } = await opened({ server, permissions: HOUSEHOLD, subject: server.subject });
    server.hooks.push({ method: 'POST', path: '/api/routines/r1/execute', mode: 'drop' });
    assert.equal((await instance.execute('r1')).uncertain, true);
    const pending = JSON.parse(storage.getItem(routines.PREFIX + 'r1'));
    assert.equal(pending.actor, actorFor('household-a'));
    const before = server.calls.length;
    await recover(instance);
    const [whoami, request] = server.calls.slice(before);
    assert.equal(whoami.path, '/api/admin/auth/whoami');
    if (request.method === 'POST') assert.equal(request.body.request_id, pending.request_id);
    else assert.equal(request.path, '/api/routines/r1/attempts?request_id=' + pending.request_id);
    assert.equal(server.runs.size, 1);
    assert.equal(instance.getState().attempts.r1.state, 'terminal');
  }
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

test('a voice brake refreshes the saved state and keeps the execution receipt', async () => {
  const server = fakeServer(), routine = server.add(spec());
  routine.latest_run = { ...server.run(routine.routine_id, 'running'), source: 'household_voice' };
  let poll;
  const instance = routines.createRoutineOwner({ api: server.api, session: session(), storage: memoryStorage(),
    schedule: callback => { poll = callback; return 1; }, cancel: () => {} });
  await instance.open(); await instance.refresh();
  assert.equal(instance.getState().routines[0].latest_voice_brake, null);
  routine.enabled = false; routine.revision++;
  routine.latest_run.status = 'interrupted';
  routine.latest_voice_brake = { receipt_id: 'brake1', source: 'household_voice', operation: 'disable', status: 'disabled' };
  await poll();
  const view = instance.getState().routines[0];
  assert.equal(view.enabled, false);
  assert.equal(view.latest_run.run_id, routine.latest_run.run_id);
  assert.equal(view.latest_run.status, 'interrupted');
  assert.deepEqual(view.latest_voice_brake, { receipt_id: 'brake1', operation: 'disable', status: 'disabled' });
  instance.leave();
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
  storage.setItem(routines.PREFIX + 'r9', JSON.stringify({ request_id: 'legacy-unbound', run_id: null, state: 'unresolved', subject: 'operator-b' }));
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
  const instance = routines.createRoutineOwner({ api: server.api, session: { can: () => true, routineActor: () => actorFor(current) }, storage: memoryStorage(), makeId: () => 'id-a', schedule: () => 1, cancel: () => {} });
  await instance.open(); await instance.refresh();
  server.hooks.push({ method: 'POST', path: '/api/routines/r1/execute', mode: 'drop' });
  await instance.execute('r1');
  current = 'operator-b';
  assert.deepEqual(await instance.retryAttempt('r1'), { sent: false });
  assert.equal(executes(server).length, 1);
  assert.equal(instance.getState().attempts.r1, undefined);
});

test('R2: without a session actor nothing is sent', async () => {
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
async function takenOver(permissions = MANAGER) {
  const server = fakeServer(); server.add(spec()); server.subject = 'operator-a';
  server.permissions = permissions;
  const storage = memoryStorage(); let changed = 0;
  const instance = routines.createRoutineOwner({ api: server.api, session: session(permissions, 'operator-a'), storage, makeId: () => 'id-a', schedule: () => 1, cancel: () => {},
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
  for (const permissions of [MANAGER, HOUSEHOLD]) {
    for (const act of [instance => instance.retryAttempt('r1'), instance => instance.consult('r1')]) {
      const { instance, server, changed } = await takenOver(permissions);
      const before = server.calls.length;
      await act(instance);
      const after = server.calls.slice(before);
      assert.deepEqual(after.map(call => call.path), ['/api/admin/auth/whoami']);
      assert.equal(server.runs.size, 1);
      assert.equal(changed(), 1);
      assert.deepEqual(instance.getState().attempts, {});
    }
  }
});

test('T033-U6: household credentials with the same name cannot recover each other’s attempts', async () => {
  for (const recover of [instance => instance.retryAttempt('r1'), instance => instance.consult('r1')]) {
    const server = fakeServer(); server.add(spec()); server.permissions = HOUSEHOLD;
    server.subject = 'same-home-name'; server.routineActor = 'fixture-credential-a';
    const storage = memoryStorage();
    const identity = createSession({ wall: () => 0, monotonic: () => 0 });
    const accept = () => identity.accept({ ...server.handle('/api/admin/auth/whoami'), expires_at: 60 });
    assert.equal(accept(), true);
    const api = createScopedApi({ session: identity, transport: server.api });
    const options = { api, session: identity, storage, makeId: () => 'shared-id', schedule: () => 1, cancel: () => {} };
    const instance = routines.createRoutineOwner(options);
    await instance.open(); await instance.refresh();
    server.hooks.push({ method: 'POST', path: '/api/routines/r1/execute', mode: 'drop' });
    assert.equal((await instance.execute('r1')).uncertain, true);
    const stored = storage.getItem(routines.PREFIX + 'r1');
    assert.equal(JSON.parse(stored).actor, 'fixture-credential-a');

    server.routineActor = 'fixture-credential-b';
    assert.equal(identity.subject(), server.subject);
    const before = server.calls.length;
    await recover(instance);
    assert.deepEqual(server.calls.slice(before).map(call => call.path), ['/api/admin/auth/whoami']);
    assert.equal(server.runs.size, 1);
    assert.deepEqual(instance.getState().attempts, {});

    storage.setItem(routines.PREFIX + 'r1', stored);
    assert.equal(accept(), true);
    const reloaded = routines.createRoutineOwner(options);
    await reloaded.open(); await reloaded.refresh();
    assert.deepEqual(reloaded.getState().attempts, {});
    assert.equal(reloaded.getState().routines[0].latest_run, null);
    assert.equal(storage.getItem(routines.PREFIX + 'r1'), null);
    assert.equal((await reloaded.execute('r1')).sent, true);
    assert.equal(server.runs.size, 2, 'a new attempt with the same request_id belongs to the second credential');
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

test('R2b: a whoami answer with only the same display name cannot confirm the account', async () => {
  const { instance, server, changed } = await takenOver();
  server.subject = 'operator-a';
  const handle = server.handle;
  server.handle = (path, options) => (path === '/api/admin/auth/whoami' ? { ok: true, role: 'operator', permissions: MANAGER, subject: server.subject } : handle(path, options));
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

const proposalId = 'a'.repeat(32);
const proposalPath = ROUTINE_PROPOSALS + '/' + proposalId;
const proposalCalls = server => server.calls.filter(call => call.path.startsWith(ROUTINE_PROPOSALS));
const turn = () => new Promise(resolve => setImmediate(resolve));
function proposalFixture({ role = 'operator', permissions = MANAGER, storage = memoryStorage(), server = fakeServer() } = {}) {
  server.permissions = permissions; server.role = role;
  const identity = createSession({ wall: () => 0, monotonic: () => 0 });
  assert.equal(identity.accept({ ...server.handle('/api/admin/auth/whoami'), expires_at: 60 }), true);
  const api = createScopedApi({ session: identity, transport: server.api });
  let changes = 0;
  const instance = routines.createRoutineOwner({ api, session: identity, storage, schedule: () => 1, cancel: () => {}, onAccountChanged: () => { changes++; } });
  function add(before = null, after = spec('Rutina propuesta')) {
    const proposal = { summary: { proposal_id: proposalId, operation: before ? 'update' : 'create', source: 'household_voice', expires_at: '2026-11-22T00:00:00Z' },
      diff: { before, after }, stale: false };
    server.proposals.set(proposalId, proposal);
    return proposal;
  }
  function mount() {
    const doc = createDocument(), container = doc.createElement('div');
    const dispose = mountRoutines({ container, dialog: doc.createElement('dialog'), confirmDialog: doc.createElement('dialog'), owner: instance, icon: icon(doc) });
    const control = suffix => container.querySelector('[data-key="' + proposalId + ':' + suffix + '"]');
    return { doc, container, control, dispose };
  }
  return { instance, api, server, storage, add, mount, changes: () => changes };
}

test('T033-U3b: the proposal UI and all four routes require routines:manage and exclude Evaluación', async () => {
  for (const [role, permissions, allowed] of [
    ['operator', MANAGER, true], ['home', MANAGER, true], ['home', HOUSEHOLD, false],
    ['operator', ['home:read'], false], ['evaluator', MANAGER, false]
  ]) {
    const f = proposalFixture({ role, permissions });
    f.add();
    const view = f.mount();
    await f.instance.open(); await turn();
    assert.equal(Boolean(view.container.querySelector('#routine-proposals')), allowed);
    if (allowed) {
      assert.deepEqual(proposalCalls(f.server).map(call => call.path), [ROUTINE_PROPOSALS]);
      assert.equal(f.instance.getState().proposals.items.length, 1);
      for (const [path, method] of [[ROUTINE_PROPOSALS, 'POST'], [proposalPath, 'DELETE'], [proposalPath + '/unknown', 'POST'], [ROUTINE_PROPOSALS + '/invalid', 'GET']]) {
        await assert.rejects(f.api(path, { method }), { status: 403, sent: false });
      }
    } else {
      await f.instance.refreshProposals(); await f.instance.reviewProposal(proposalId); await f.instance.decideProposal(proposalId, 'confirm');
      for (const [path, method] of [[ROUTINE_PROPOSALS, 'GET'], [proposalPath, 'GET'], [proposalPath + '/confirm', 'POST'], [proposalPath + '/reject', 'POST']]) {
        await assert.rejects(f.api(path, { method }), { status: 403, sent: false });
      }
      assert.equal(f.instance.getState().proposals, null);
      assert.deepEqual(proposalCalls(f.server), []);
    }
    view.dispose();
  }
});

test('T033-U3b: preview renders both definitions without writing and retains focus through conflict, 503 and expiry', async () => {
  const f = proposalFixture();
  const proposal = f.add(spec('Rutina anterior'), { ...spec('Rutina nueva'), enabled: false,
    actions: [{ entity_id: lamp.entity_id, operation: 'color_temp_kelvin', color_temp_kelvin: 2700, brightness_pct: 25 }] });
  const view = f.mount();
  await f.instance.open(); await turn();
  view.control('review').focus(); view.control('review').click(); await turn();
  assert.equal(view.doc.activeElement, view.control('review'));
  const panel = view.container.querySelector('#routine-proposals');
  assert.equal(panel.getAttribute('aria-labelledby'), 'routine-proposals-title');
  assert.match(visibleText(panel.querySelector('[data-proposal-diff="before"]')), /Antes.*Rutina anterior/);
  assert.match(visibleText(panel.querySelector('[data-proposal-diff="after"]')), /Después.*Rutina nueva.*Desactivada.*2700.*25/);
  assert.ok(proposalCalls(f.server).every(call => call.method === 'GET'));
  assert.equal(f.server.routines.size, 0);

  proposal.stale = true;
  view.control('confirm').focus(); view.control('confirm').click(); await turn();
  assert.match(visibleText(panel), /La rutina cambió/);
  assert.equal(view.control('confirm'), null);
  assert.equal(view.doc.activeElement, view.control('review'));
  await f.instance.reviewProposal(proposalId);
  assert.match(visibleText(panel), /Conflicto/);
  assert.ok(view.control('reject'));

  f.server.hooks.push({ method: 'GET', path: proposalPath, mode: 'reject', error: failure(503, 'routine_proposals_unavailable') });
  await f.instance.reviewProposal(proposalId);
  assert.match(visibleText(panel), /no están disponibles/);
  assert.equal(view.control('reject'), null);
  f.server.proposals.delete(proposalId);
  await f.instance.reviewProposal(proposalId);
  assert.match(visibleText(panel), /expiró o ya no está disponible/);
  assert.equal(view.doc.activeElement.dataset.key, 'proposals-refresh');
  view.dispose();
});

test('T033-U3b: confirm and reject replay the same decision after a lost response, including reload', async () => {
  for (const [action, expected] of [['confirm', 'confirmada'], ['reject', 'rechazada']]) {
    const f = proposalFixture(), proposal = f.add();
    const view = f.mount();
    await f.instance.open(); await turn(); await f.instance.reviewProposal(proposalId);
    const path = proposalPath + '/' + action;
    f.server.hooks.push({ method: 'POST', path, mode: 'drop' });
    view.control(action).focus(); view.control(action).click();
    assert.equal(view.control(action).getAttribute('aria-disabled'), 'true');
    await turn();
    assert.equal(view.doc.activeElement, view.control(action));
    assert.match(visibleText(view.container.querySelector('#routine-proposals')), /Resultado incierto/);
    assert.match(view.control(action).textContent, /Reintentar/);
    assert.equal(view.control(action === 'confirm' ? 'reject' : 'confirm'), null);
    await f.instance.decideProposal(proposalId, action === 'confirm' ? 'reject' : 'confirm');
    assert.equal(proposalCalls(f.server).filter(call => call.method === 'POST').length, 1);
    const [stored] = [...f.storage.map.values()].map(JSON.parse);
    assert.deepEqual(stored, { action, actor: actorFor(f.server.subject) });
    view.dispose(); await f.instance.clear({ wipe: false });

    const after = proposalFixture({ server: f.server, storage: f.storage });
    const second = after.mount();
    await after.instance.open(); await turn(); await after.instance.reviewProposal(proposalId);
    assert.equal(proposalCalls(f.server).filter(call => call.method === 'POST').length, 1, 'reload does not replay automatically');
    second.control(action).focus(); second.control(action).click(); await turn();
    assert.deepEqual(proposalCalls(f.server).filter(call => call.method === 'POST').map(call => [call.path, call.body, call.signal]),
      [[path, undefined, undefined], [path, undefined, undefined]]);
    assert.equal(f.server.routines.size, action === 'confirm' ? 1 : 0);
    assert.ok(proposal.receipt);
    assert.match(visibleText(second.container.querySelector('#routine-proposals')), new RegExp('Propuesta ' + expected));
    assert.equal(second.doc.activeElement.dataset.key, 'proposals-refresh');
    assert.equal(f.storage.length, 0);
    second.dispose();
  }
});

test('T033-U3b: an expired replay remains uncertain and a 503 permits only the original decision', async () => {
  const f = proposalFixture(); f.add();
  await f.instance.open(); await turn(); await f.instance.reviewProposal(proposalId);
  const path = proposalPath + '/confirm';
  f.server.hooks.push({ method: 'POST', path, mode: 'drop' });
  await f.instance.decideProposal(proposalId, 'confirm');
  f.server.hooks.push({ method: 'POST', path, mode: 'reject', error: failure(503, 'routine_proposals_unavailable') });
  await f.instance.decideProposal(proposalId, 'confirm');
  assert.match(f.instance.getState().proposals.decisions[proposalId].message, /incierto.*no están disponibles/i);
  f.server.proposals.delete(proposalId);
  await f.instance.decideProposal(proposalId, 'confirm');
  const state = f.instance.getState().proposals;
  assert.equal(state.selected.status, 'uncertain');
  assert.match(state.selected.message, /incierto.*expiró/i);
  assert.deepEqual(state.decisions, {});
  assert.equal(f.storage.length, 0);
});

test('T033-U3b: a refreshed list that omits the selected proposal removes its diff and decisions', async () => {
  for (const empty of [true, false]) {
    const f = proposalFixture(), proposal = f.add();
    const view = f.mount();
    await f.instance.open(); await turn(); await f.instance.reviewProposal(proposalId);
    assert.ok(view.container.querySelector('[data-proposal-diff="after"]'));
    view.control('confirm').focus();
    f.server.proposals.delete(proposalId);
    if (!empty) {
      const id = 'b'.repeat(32);
      f.server.proposals.set(id, { ...proposal, summary: { ...proposal.summary, proposal_id: id } });
    }
    await f.instance.refreshProposals();

    const state = f.instance.getState().proposals;
    assert.equal(state.loaded, true);
    assert.equal(state.items.length, empty ? 0 : 1);
    assert.equal(state.selected, null);
    assert.equal(view.container.querySelector('[data-proposal-id="' + proposalId + '"]'), null);
    assert.equal(view.container.querySelector('[data-proposal-diff="after"]'), null);
    assert.equal(view.control('confirm'), null);
    assert.equal(view.control('reject'), null);
    assert.equal(view.doc.activeElement.dataset.key, 'proposals-refresh');
    await f.instance.decideProposal(proposalId, 'confirm');
    await f.instance.decideProposal(proposalId, 'reject');
    assert.ok(proposalCalls(f.server).every(call => call.method === 'GET'));
    view.dispose();
  }
});

test('T033-U3b: an omitted proposal keeps only its original uncertain decision for replay', async () => {
  const f = proposalFixture(); f.add();
  const view = f.mount();
  await f.instance.open(); await turn(); await f.instance.reviewProposal(proposalId);
  const path = proposalPath + '/confirm';
  f.server.hooks.push({ method: 'POST', path, mode: 'drop' });
  await f.instance.decideProposal(proposalId, 'confirm');
  assert.ok(view.container.querySelector('[data-proposal-diff="after"]'));
  await f.instance.refreshProposals();

  const state = f.instance.getState().proposals;
  assert.deepEqual(state.items, []);
  assert.equal(state.selected, null);
  assert.equal(state.decisions[proposalId].action, 'confirm');
  assert.deepEqual([...f.storage.map.values()].map(value => JSON.parse(value)),
    [{ action: 'confirm', actor: actorFor(f.server.subject) }]);
  assert.equal(view.container.querySelector('[data-proposal-diff="after"]'), null);
  assert.equal(view.control('confirm'), null);
  assert.equal(view.control('reject'), null);
  await f.instance.reviewProposal(proposalId);
  assert.equal(view.control('confirm').textContent, 'Reintentar confirmación');
  assert.equal(view.control('reject'), null);
  await f.instance.decideProposal(proposalId, 'reject');
  await f.instance.decideProposal(proposalId, 'confirm');
  assert.deepEqual(proposalCalls(f.server).filter(call => call.method === 'POST').map(call => call.path), [path, path]);
  assert.equal(f.server.routines.size, 1);
  view.dispose();
});

test('T033-U3b: a successful detail response cannot republish a proposal omitted by a newer list', async () => {
  const server = fakeServer(), transport = server.api;
  let release;
  const delayed = new Promise(resolve => { release = resolve; });
  server.api = async (path, options) => {
    const result = await transport(path, options);
    if (path === proposalPath) await delayed;
    return result;
  };
  const f = proposalFixture({ server }); f.add();
  const view = f.mount();
  await f.instance.open(); await turn();
  const preview = f.instance.reviewProposal(proposalId);
  await turn();
  assert.equal(f.instance.getState().proposals.selected.loading, true);
  assert.equal(proposalCalls(server).at(-1).path, proposalPath);
  server.proposals.delete(proposalId);
  await f.instance.refreshProposals();
  release(); await preview;
  assert.equal(f.instance.getState().proposals.selected, null);
  assert.equal(view.container.querySelector('[data-proposal-diff="after"]'), null);
  assert.equal(view.control('confirm'), null);
  assert.equal(view.control('reject'), null);
  view.dispose();
});

test('T033-U3b: account change after the POST asks the user to check current state', async () => {
  const f = proposalFixture(), proposal = f.add();
  const view = f.mount();
  await f.instance.open(); await turn(); await f.instance.reviewProposal(proposalId);
  const handle = f.server.handle, path = proposalPath + '/confirm';
  f.server.handle = (requestPath, options) => {
    const response = handle(requestPath, options);
    if (requestPath === path) f.server.routineActor = 'fixture-other-credential';
    return response;
  };
  const before = f.server.calls.length;
  await f.instance.decideProposal(proposalId, 'confirm');
  assert.deepEqual(f.server.calls.slice(before).map(call => [call.method, call.path]),
    [['GET', '/api/admin/auth/whoami'], ['POST', path], ['GET', '/api/admin/auth/whoami']]);
  assert.equal(proposal.receipt.status, 'confirmed');
  assert.equal(f.server.routines.size, 1);
  assert.equal(f.changes(), 1);
  assert.equal(f.instance.getState().proposals.selected, null);
  assert.deepEqual(f.instance.getState().proposals.decisions, {});
  assert.deepEqual([...f.storage.map.entries()], [[routines.ENDED_KEY, '1']]);
  const text = visibleText(view.container);
  assert.match(text, /cambió.*Revisá.*estado actual/);
  assert.doesNotMatch(text, /no se envió nada|propuesta confirmada|rutina propuesta/i);
  view.dispose();
});

test('T033-U3b: a new account cannot replay a decision or receive a late proposal preview', async () => {
  for (const inFlight of ['preview', 'retry']) {
    const f = proposalFixture(); f.server.routineActor = actorFor(f.server.subject); f.add();
    const view = f.mount();
    await f.instance.open(); await turn();
    let release;
    const held = new Promise(resolve => { release = resolve; });
    let delayed;
    if (inFlight === 'preview') {
      f.server.hooks.push({ method: 'GET', path: proposalPath, mode: 'hold', until: held });
      delayed = f.instance.reviewProposal(proposalId);
      await turn();
    } else {
      await f.instance.reviewProposal(proposalId);
      f.server.hooks.push({ method: 'POST', path: proposalPath + '/reject', mode: 'drop' });
      await f.instance.decideProposal(proposalId, 'reject');
    }
    f.server.routineActor = 'fixture-other-credential';
    const before = proposalCalls(f.server).length;
    if (delayed) { release(); await delayed; }
    else await f.instance.decideProposal(proposalId, 'reject');
    assert.equal(f.changes(), 1);
    assert.equal(proposalCalls(f.server).length, before);
    assert.equal(f.instance.getState().proposals.selected, null);
    assert.deepEqual(f.instance.getState().proposals.decisions, {});
    assert.ok(![...f.storage.map.keys()].some(key => key.includes(proposalId)));
    assert.doesNotMatch(visibleText(view.container), /Rutina propuesta|Resultado incierto/);
    view.dispose();
  }
});

test('T033-U3b: leaving hides the pending decision until the account is verified again', async () => {
  const f = proposalFixture(); f.add();
  await f.instance.open(); await turn(); await f.instance.reviewProposal(proposalId);
  f.server.hooks.push({ method: 'POST', path: proposalPath + '/reject', mode: 'drop' });
  await f.instance.decideProposal(proposalId, 'reject');
  assert.ok(f.instance.getState().proposals.decisions[proposalId]);
  await f.instance.leave();
  const view = f.mount();
  assert.equal(view.control('review'), null);
  const before = proposalCalls(f.server).length;
  f.server.routineActor = 'fixture-second-account';
  await f.instance.open(); await turn();
  assert.equal(f.changes(), 1);
  assert.equal(proposalCalls(f.server).length, before);
  assert.equal(view.control('review'), null);
  assert.ok(![...f.storage.map.keys()].some(key => key.includes(proposalId)));
  view.dispose();
});
