'use strict';

/* T045-U3: Deterministic C06 error matrix (C02 simulation) across home createApi,
 * every NEXO owner (loaded read-only), and obs-data / obs session (loaded read-only). */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { test } = require('node:test');

const home = require('../../src/dashboard/frontend/home/home.js');
const state = require('../../src/dashboard/frontend/home/nexo/state.js');
const { createNoticesOwner } = require('../../src/dashboard/frontend/home/nexo/notices-owner.js');
const { createTasksOwner } = require('../../src/dashboard/frontend/home/nexo/tasks-owner.js');
const people = require('../../src/dashboard/frontend/home/nexo/people.js');
const routines = require('../../src/dashboard/frontend/home/nexo/routine-owner.js');
const memory = require('../../src/dashboard/frontend/home/nexo/memory-owner.js');
const myMemory = require('../../src/dashboard/frontend/home/nexo/my-memory-owner.js');
const myVoice = require('../../src/dashboard/frontend/home/nexo/my-voice-owner.js');
const voiceConsents = require('../../src/dashboard/frontend/home/nexo/voice-consents-owner.js');
const { createAccessOwner } = require('../../src/dashboard/frontend/home/nexo/access-owner.js');
const { createVoiceOwner } = require('../../src/dashboard/frontend/home/nexo/voice-owner.js');
const { createEvaluationOwner } = require('../../src/dashboard/frontend/home/nexo/evaluation-owner.js');
const obsData = require('../../src/dashboard/frontend/obs/src/obs-data.js');
const obsSession = require('../../src/dashboard/frontend/obs/src/session.js');

const RAW_ENGLISH = /Failed to fetch|AbortError|TimeoutError|NetworkError|TypeError|ECONNREFUSED|The operation was aborted|origin denied/i;
const MATRIX_PATH = path.resolve(__dirname, '../../docs/research/2026-11-20-autonomous-t045-error-matrix.json');
const C06_CLASSES = Object.freeze([
  '401', '403', '409', '503', 'timeout', 'disconnect-reconnect', 'expired', 'partial-uncertain',
]);

const EPOCH = Date.parse('2026-11-20T12:00:00Z');
const FACT_ID = 'fact_' + 'a'.repeat(64);
const UUID_1 = '10000000-0000-4000-8000-000000000001';
const UUID_2 = '20000000-0000-4000-8000-000000000002';
const PROFILE_ID = '30000000-0000-4000-8000-000000000003';
const FP = 'vf_' + 'a'.repeat(64);
const WHOAMI = '/api/admin/auth/whoami';

const flush = async (turns = 10) => {
  for (let i = 0; i < turns; i++) await new Promise(resolve => setImmediate(resolve));
};
const deferred = () => {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
};
const makeHeaders = (map = {}) => {
  const lower = Object.fromEntries(Object.entries(map).map(([k, v]) => [k.toLowerCase(), v]));
  return { get: name => lower[name.toLowerCase()] ?? null };
};
const jsonReply = (status, body, extraHeaders = {}) => ({
  ok: status >= 200 && status < 300, status,
  headers: makeHeaders(extraHeaders),
  body: { cancel: async () => {} },
  json: async () => body,
  arrayBuffer: async () => new ArrayBuffer(0),
});
const realObsHeaders = (extra = {}) => ({
  'X-KZA-Source': 'real',
  'X-KZA-Source-Id': 'alert_manager',
  'X-KZA-Observed-At': '2026-11-20T12:00:00+00:00',
  'X-KZA-Expires-At': '2026-11-20T12:00:06+00:00',
  ...extra,
});
function assertCleanSpanish(message, context) {
  assert.equal(typeof message, 'string', `${context}: message must be a string`);
  assert.ok(message.trim().length > 0, `${context}: message must be non-empty`);
  assert.doesNotMatch(message, RAW_ENGLISH, `${context}: raw English leaked in "${message}"`);
}
function fakeClock(start = EPOCH) {
  let current = start;
  const timers = [];
  return {
    now: () => new Date(current),
    nowMs: () => current,
    setTimer(callback, ms) { const timer = { at: current + ms, callback }; timers.push(timer); return timer; },
    clearTimer(timer) { const index = timers.indexOf(timer); if (index >= 0) timers.splice(index, 1); },
    advance(ms) {
      current += ms;
      for (;;) {
        const due = timers.filter(timer => timer.at <= current).sort((a, b) => a.at - b.at)[0];
        if (!due) break;
        timers.splice(timers.indexOf(due), 1);
        due.callback();
      }
    },
    pending: () => timers.length,
  };
}
class FakeSocket extends EventTarget {
  constructor() { super(); this.readyState = 0; this.sent = []; }
  event(type, data) { const ev = new Event(type); if (data !== undefined) ev.data = JSON.stringify(data); this.dispatchEvent(ev); }
  open() { this.readyState = 1; this.event('open'); }
  send(data) { this.sent.push(JSON.parse(data)); }
  close() { this.readyState = 3; this.event('close'); }
}
function memoryStorage() {
  const map = new Map();
  return {
    get length() { return map.size; },
    key: i => [...map.keys()][i] ?? null,
    getItem: k => map.has(k) ? map.get(k) : null,
    setItem: (k, v) => { map.set(k, String(v)); },
    removeItem: k => { map.delete(k); },
  };
}

// ---------------------------------------------------------------------------
// 1. home-createApi (all 8 C06 classes)
// ---------------------------------------------------------------------------
test('C06 [home-createApi]: 401, 403, 409, 503, timeout/abort, disconnect-reconnect, expired, partial-uncertain', async t => {
  // 401, 403, 409, 503
  for (const [status, kind, detail, rx] of [
    [401, 'unauthorized', 'session_expired', /autorizada/],
    [403, 'forbidden', 'origin denied', /permiso/],
    [409, 'conflict', 'revision_conflict', /Conflicto/],
    [503, 'unavailable', 'service_down', /no está disponible/],
  ]) {
    let unauth = 0;
    const api = home.createApi({ fetch: async () => jsonReply(status, { detail }), onUnauthorized: () => { unauth++; } });
    await assert.rejects(api('/api/home/catalog'), err => {
      assert.equal(err.status, status);
      assert.equal(err.kind, kind);
      assert.equal(err.detail, detail);
      assert.match(err.message, rx);
      assertCleanSpanish(err.message, `home-createApi/${status}`);
      return true;
    });
    assert.equal(unauth, status === 401 ? 1 : 0);
  }

  // timeout vs caller abort
  {
    const caller = new AbortController();
    const apiAbort = home.createApi({
      fetch: (_p, { signal }) => new Promise((_, reject) => {
        signal.addEventListener('abort', () => reject(new DOMException('The operation was aborted.', 'AbortError')), { once: true });
      }),
      timeoutMs: 60000,
    });
    const req = apiAbort('/api/home/catalog', { signal: caller.signal });
    caller.abort();
    await assert.rejects(req, err => {
      assert.equal(err.kind, 'aborted');
      assert.equal(err.status, null);
      assertCleanSpanish(err.message, 'home-createApi/aborted');
      return true;
    });

    t.mock.timers.enable({ apis: ['setTimeout'] });
    const apiTimeout = home.createApi({
      fetch: (_p, { signal }) => new Promise((_, reject) => {
        signal.addEventListener('abort', () => reject(new DOMException('The operation was aborted.', 'AbortError')), { once: true });
      }),
      timeoutMs: 200,
    });
    const timed = apiTimeout('/api/home/catalog');
    t.mock.timers.tick(200);
    await assert.rejects(timed, err => {
      assert.equal(err.kind, 'timeout');
      assert.equal(err.status, null);
      assertCleanSpanish(err.message, 'home-createApi/timeout');
      return true;
    });
  }

  // disconnect-reconnect via createApi + createPoller
  {
    let offline = true;
    const errors = [], dataSeen = [];
    const api = home.createApi({
      fetch: async () => {
        if (offline) throw new TypeError('Failed to fetch');
        return jsonReply(200, { rooms: [], entities: [], unassigned_entities: [] });
      },
    });
    const clock = fakeClock();
    const poller = home.createPoller({
      load: signal => api('/api/home/catalog', { signal }),
      onData: data => dataSeen.push(data),
      onError: err => errors.push(err),
      now: clock.nowMs,
      schedule: clock.setTimer,
      cancel: clock.clearTimer,
    });
    poller.start();
    await flush();
    assert.equal(errors.length, 1);
    assert.equal(errors[0].kind, 'network');
    assertCleanSpanish(errors[0].message, 'home-createApi/disconnect');
    offline = false;
    await poller.refresh();
    assert.equal(dataSeen.length, 1);
    poller.stop();
  }

  // expired snapshot disables controls
  {
    let wall = EPOCH;
    const observation = { state: 'fresh', observed_at: new Date(EPOCH).toISOString(), expires_at: new Date(EPOCH + 1000).toISOString() };
    const entity = { entity_id: 'light.l1', domain: 'light', state: 'on', available: true, capabilities: ['turn_off'], observation };
    const snap = home.createSnapshotClock({ observation, entities: [entity] }, { wallClock: () => wall, monotonic: () => 0 });
    assert.equal(home.entityCanControl(entity, 'turn_off', snap.read().stale, snap.now()), true);
    wall += 1001;
    assert.equal(snap.read().stale, true);
    assert.equal(home.entityCanControl(entity, 'turn_off', snap.read().stale, snap.now()), false);
  }

  // partial-uncertain action receipt reconciles once without reposting
  {
    const calls = [];
    let receiptCalls = 0;
    const api = home.createApi({
      fetch: async (url, init = {}) => {
        calls.push([init.method || 'GET', url]);
        if (init.method === 'POST') throw new TypeError('Failed to fetch');
        receiptCalls++;
        return jsonReply(200, {
          status: 'uncertain',
          effects: [{ status: 'confirmed' }, { status: 'uncertain' }, { status: 'not_executed' }],
        });
      },
    });
    const actions = home.createActionController({ api, makeId: () => 'req-partial-1' });
    await actions.execute('scene.evening', 'activate_scene', {});
    const rec = actions.get('scene.evening', 'activate_scene');
    assert.equal(rec.status, 'uncertain');
    assert.equal(home.effectSummary(rec.effects), '1 confirmado · 1 sin confirmar · 1 sin ejecutar');
    assert.equal(calls.filter(([m]) => m === 'POST').length, 1);
    assert.equal(receiptCalls, 1);
  }
});

// ---------------------------------------------------------------------------
// 2. nexo-notices-owner (all 8 C06 classes)
// ---------------------------------------------------------------------------
test('C06 [nexo-notices-owner]: 401, 403, 409, 503, timeout, disconnect-reconnect, expired, partial-uncertain', async () => {
  const triggerAt = EPOCH / 1000 - 30;
  const due = { reminder_id: 'r-notice', trigger_at: triggerAt, text: 'Aviso C06', status: 'pending',
    reason: null, label: 'pendiente', pending: true, attempts: 1, failed_attempts: 0, claimed_at: null };
  const grant = { ok: true, role: 'home', subject: 'sub-notice', account_binding: null,
    permissions: ['home:read', 'tasks:read', 'tasks:write'], expires_at: EPOCH / 1000 + 120 };

  function makeNoticesFixture() {
    let wall = EPOCH, mono = 0;
    const clock = fakeClock(EPOCH);
    const session = state.createSession({ wall: () => wall, monotonic: () => mono });
    session.accept(grant);
    const handlers = { list: null, post: null };
    const sample = () => {
      const obs = new Date(wall - 200).toISOString();
      const exp = new Date(wall + 4000).toISOString();
      return jsonReply(200, { source: 'reminder-store', observed_at: obs, notices: [due] }, {
        'X-KZA-Source-Id': 'reminder-store', 'X-KZA-Observation-State': 'fresh',
        'X-KZA-Observed-At': obs, 'X-KZA-Expires-At': exp,
      });
    };
    const transport = home.createApi({
      fetch: async (url, init) => {
        if (url === WHOAMI) return jsonReply(200, grant);
        if (url === state.NOTICE_LIST) return handlers.list ? handlers.list(init) : sample();
        if (url === state.NOTICE_DISMISS) return handlers.post ? handlers.post(init) : jsonReply(200, {});
        assert.fail(`unexpected ${url}`);
      },
    });
    let retired = 0;
    const api = state.createScopedApi({ session, transport, onRetired: () => { retired++; } });
    const owner = createNoticesOwner({
      api, session,
      clockOptions: { wallClock: () => wall, monotonic: () => mono },
      now: () => mono, schedule: clock.setTimer, cancel: clock.clearTimer,
    });
    return {
      owner, handlers, session,
      retired: () => retired,
      advance(ms) { wall += ms; mono += ms; clock.advance(ms); },
    };
  }

  // 401, 403, 503, timeout on list read
  for (const [label, fail, rx] of [
    ['401', () => jsonReply(401, {}), /Iniciá sesión/],
    ['403', () => jsonReply(403, {}), /permiso/],
    ['503', () => jsonReply(503, { detail: 'reminder_notices_unavailable' }), /no está disponible/],
    ['timeout', () => { throw new DOMException('timed out', 'TimeoutError'); }, /verificar/],
  ]) {
    const f = makeNoticesFixture();
    f.handlers.list = fail;
    f.owner.open();
    await flush();
    assert.equal(f.owner.getState().status, 'error', label);
    assert.match(f.owner.getState().error, rx, label);
    assertCleanSpanish(f.owner.getState().error, `nexo-notices-owner/${label}`);
    if (label === '401') assert.equal(f.retired(), 1);
    f.owner.clear();
  }

  // disconnect-reconnect + expired + 409 + partial-uncertain
  {
    const f = makeNoticesFixture();
    f.owner.open();
    await flush();
    assert.equal(f.owner.getState().status, 'ready');
    // disconnect
    f.handlers.list = () => { throw new TypeError('Failed to fetch'); };
    await f.owner.refresh();
    assert.equal(f.owner.getState().status, 'stale');
    assert.match(f.owner.getState().error, /contactar/);
    assertCleanSpanish(f.owner.getState().error, 'nexo-notices-owner/disconnect');
    // reconnect
    f.handlers.list = null;
    await f.owner.refresh();
    assert.equal(f.owner.getState().status, 'ready');
    // 409 on dismiss
    const key = f.owner.getState().notices[0].key;
    f.handlers.post = () => jsonReply(409, { detail: 'reminder_notice_busy' });
    assert.equal(await f.owner.dismiss(key), false);
    assert.equal(f.owner.getState().feedback.kind, 'error');
    assert.match(f.owner.getState().feedback.text, /anunciando/);
    assertCleanSpanish(f.owner.getState().feedback.text, 'nexo-notices-owner/409');
    await flush();
    // partial-uncertain on dismiss (lost receipt)
    f.handlers.post = () => { throw new TypeError('Failed to fetch'); };
    assert.equal(await f.owner.dismiss(key), false);
    assert.equal(f.owner.getState().feedback.kind, 'uncertain');
    assert.match(f.owner.getState().feedback.text, /No se pudo confirmar/);
    assertCleanSpanish(f.owner.getState().feedback.text, 'nexo-notices-owner/partial-uncertain');
    await flush();
    // expired blocks dismiss
    f.advance(4500);
    assert.equal(f.owner.getState().status, 'stale');
    assert.equal(f.owner.getState().locked, true);
    f.owner.clear();
  }
});

// ---------------------------------------------------------------------------
// 3. nexo-tasks-owner (7 C06 classes)
// ---------------------------------------------------------------------------
test('C06 [nexo-tasks-owner]: 401, 403, 409, 503, timeout, disconnect-reconnect, partial-uncertain', async () => {
  const grant = { ok: true, role: 'home', subject: 'sub-tasks', account_binding: null,
    permissions: ['tasks:read', 'tasks:write'], expires_at: EPOCH / 1000 + 120 };
  const reminder = { id: 'rem-1', scope: 'shared', text: 'Revisar filtro', trigger_at: EPOCH / 1000 + 60 };

  function makeTasksFixture() {
    const session = state.createSession({ wall: () => EPOCH, monotonic: () => 0 });
    session.accept(grant);
    const overrides = new Map();
    const transport = home.createApi({
      fetch: async (url, init = {}) => {
        if (overrides.has(url)) return overrides.get(url)(init);
        if (url === WHOAMI) return jsonReply(200, grant);
        if (url === '/api/lists?scope=shared') return jsonReply(200, []);
        if (url === '/api/reminders?scope=shared') return jsonReply(200, [reminder]);
        return jsonReply(200, { outcome: { status: 'completed', persistence: 'committed' } });
      },
    });
    const api = state.createScopedApi({ session, transport });
    const owner = createTasksOwner({ api, session });
    return { owner, overrides };
  }

  for (const [label, fn, rx] of [
    ['401', () => jsonReply(401, {}), /sesión venció/],
    ['403', () => jsonReply(403, {}), /permiso/],
    ['503', () => jsonReply(503, {}), /no está disponible/],
    ['timeout', () => { throw new DOMException('timed out', 'TimeoutError'); }, /conexión/],
  ]) {
    const f = makeTasksFixture();
    f.overrides.set('/api/reminders?scope=shared', fn);
    f.owner.open();
    await flush();
    assert.equal(f.owner.getState().data, null, label);
    assert.match(f.owner.getState().error, rx, label);
    assertCleanSpanish(f.owner.getState().error, `nexo-tasks-owner/${label}`);
    f.owner.clear();
  }

  // disconnect-reconnect + 409 + partial-uncertain
  {
    const f = makeTasksFixture();
    f.overrides.set('/api/reminders?scope=shared', () => { throw new TypeError('Failed to fetch'); });
    f.owner.open();
    await flush();
    assert.match(f.owner.getState().error, /conexión/);
    assertCleanSpanish(f.owner.getState().error, 'nexo-tasks-owner/disconnect');
    f.overrides.clear();
    assert.equal(await f.owner.refresh(), true);
    assert.equal(f.owner.getState().data.shared.reminders.length, 1);

    f.overrides.set('/api/reminders/rem-1', () => jsonReply(409, { detail: 'reminder_conflict' }));
    assert.equal(await f.owner.removeReminder('shared', 'rem-1'), false);
    assert.match(f.owner.getState().feedback, /cambió/);
    assertCleanSpanish(f.owner.getState().feedback, 'nexo-tasks-owner/409');

    f.overrides.set('/api/reminders/rem-1', () => { throw new TypeError('Failed to fetch'); });
    assert.equal(await f.owner.removeReminder('shared', 'rem-1'), false);
    assert.match(f.owner.getState().feedback, /confirmar/);
    assertCleanSpanish(f.owner.getState().feedback, 'nexo-tasks-owner/partial-uncertain');
    f.owner.clear();
  }
});

// ---------------------------------------------------------------------------
// 4. nexo-people (6 C06 classes)
// ---------------------------------------------------------------------------
test('C06 [nexo-people]: 401, 403, 409, 503, timeout, partial-uncertain', async () => {
  const api = home.createApi({
    fetch: async url => {
      if (url === '/401') return jsonReply(401, {});
      if (url === '/403') return jsonReply(403, { detail: 'denied' });
      if (url === '/409-last') return jsonReply(409, { detail: 'No puedes eliminar al único administrador' });
      if (url === '/409') return jsonReply(409, { detail: 'conflict' });
      if (url === '/503') return jsonReply(503, { detail: 'user_catalog_unavailable' });
      throw new DOMException('timed out', 'TimeoutError');
    },
  });
  const capture = async url => { try { await api(url); assert.fail(); } catch (err) { return err; } };
  for (const [label, url, rx] of [
    ['401', '/401', /No se confirmó/],
    ['403', '/403', /no permite este cambio/],
    ['409', '/409-last', /último administrador/],
    ['409-generic', '/409', /rechazó el cambio/],
    ['503', '/503', /No se confirmó/],
    ['timeout', '/timeout', /No se confirmó/],
  ]) {
    const msg = people.mutationError(await capture(url));
    assert.match(msg, rx, label);
    assertCleanSpanish(msg, `nexo-people/${label}`);
  }
  const partialTrue = people.mutationError({ detail: { credentials_retired: true } });
  const partialNull = people.mutationError({ detail: { credentials_retired: null } });
  assert.match(partialTrue, /Se confirmó el retiro/);
  assert.match(partialNull, /No se pudo confirmar/);
  assertCleanSpanish(partialTrue, 'nexo-people/partial-true');
  assertCleanSpanish(partialNull, 'nexo-people/partial-null');
});

test('C06 [nexo-routine-owner]: HTTP 200 with non-JSON body on setEnabled is unknown and reconciles via GET /api/routines', async () => {
  const grant = { ok: true, role: 'operator', subject: 'sub-r', routine_actor: 'actor-r',
    permissions: ['home:read', 'home:control', 'routines:manage'], expires_at: EPOCH / 1000 + 120 };
  const session = state.createSession({ wall: () => EPOCH, monotonic: () => 0 });
  session.accept(grant);
  const routine = { routine_id: 'r1', revision: 1, name: 'Noche', description: '', enabled: true, timezone: 'America/Montevideo',
    schedule: null, conditions: [], actions: [{ entity_id: 'light.test_lamp', operation: 'turn_on' }], created_by: 'sub-r',
    created_at: '2026-11-10T00:00:00Z', latest_run: null };
  const calls = [];
  let applied = false;
  const lamp = { entity_id: 'light.test_lamp', name: 'Luz', domain: 'light', capabilities: ['turn_on', 'turn_off'], control_limits: {} };
  const scoped = state.createScopedApi({
    session,
    transport: home.createApi({
      fetch: async (url, opts = {}) => {
        const method = opts.method || 'GET';
        calls.push(method + ' ' + url);
        if (url === WHOAMI) return jsonReply(200, grant);
        if (url === '/api/home/catalog') return jsonReply(200, { rooms: [], entities: [lamp], unassigned_entities: [] });
        if (url === '/api/routines' && method === 'GET') return jsonReply(200, [applied ? { ...routine, enabled: false, revision: 2 } : routine]);
        if (method === 'POST' && url === '/api/routines/r1/disable') {
          applied = true;
          return { ...jsonReply(200, null), json: async () => { throw new SyntaxError('Unexpected token < in JSON'); } };
        }
        return jsonReply(200, []);
      },
    }),
  });
  const instance = routines.createRoutineOwner({ api: scoped, session, storage: memoryStorage(), now: () => EPOCH });
  await instance.open();
  await instance.refresh();
  await flush();
  assert.equal(instance.getState().routines.length, 1);
  calls.length = 0;
  const result = await instance.setEnabled('r1', false);
  await flush();
  assert.equal(result.unknown, true);
  assert.equal(instance.getState().routines[0].enabled, false, 'reconciliation reads the applied state');
  assert.ok(calls.includes('GET /api/routines'), 'reconciliation GET /api/routines must fire: ' + calls.join(', '));
  const err = await (async () => { try { await home.createApi({ fetch: async () => ({ ...jsonReply(200, null), json: async () => { throw new SyntaxError('x'); } }) })('/w', { method: 'POST' }); } catch (e) { return e; } })();
  assert.equal(routines.writeOutcome(err), 'unknown');
});

// ---------------------------------------------------------------------------
// 5. nexo-routine-owner (7 C06 classes)
// ---------------------------------------------------------------------------
test('C06 [nexo-routine-owner]: 401, 403, 409, 503, timeout, disconnect-reconnect, partial-uncertain', async () => {
  const api = home.createApi({
    fetch: async url => {
      if (url === '/401') return jsonReply(401, {});
      if (url === '/403') return jsonReply(403, { detail: 'origin denied' });
      if (url === '/409') return jsonReply(409, { detail: 'revision_conflict' });
      if (url === '/503') return jsonReply(503, { detail: 'routines_unavailable' });
      if (url === '/timeout') throw new DOMException('timed out', 'TimeoutError');
      throw new TypeError('Failed to fetch');
    },
  });
  const capture = async url => { try { await api(url); assert.fail(); } catch (err) { return err; } };
  for (const [label, url, rx] of [
    ['401', '/401', /Iniciá sesión/],
    ['403', '/403', /origen/],
    ['409', '/409', /cambió en otra sesión/],
    ['503', '/503', /no disponibles/],
    ['timeout', '/timeout', /contactar al servidor/],
    ['network', '/net', /contactar al servidor/],
  ]) {
    const err = await capture(url);
    const msg = routines.describeError(err);
    assert.match(msg, rx, label);
    assertCleanSpanish(msg, `nexo-routine-owner/${label}`);
  }
  assert.equal(routines.executeOutcome(await capture('/503'), true), 'unknown');
  assert.equal(routines.executeOutcome(await capture('/net'), true), 'unknown');
  assert.equal(routines.writeOutcome(await capture('/409')), 'rejected');
  assert.equal(routines.writeOutcome(await capture('/timeout')), 'unknown');
  assert.equal(routines.riskyRun({ status: 'uncertain', actions: [] }), true);

  // disconnect-reconnect via createRoutineOwner
  const grant = { ok: true, role: 'home', subject: 'sub-r', routine_actor: 'actor-r',
    permissions: ['home:read'], expires_at: EPOCH / 1000 + 120 };
  const session = state.createSession({ wall: () => EPOCH, monotonic: () => 0 });
  session.accept(grant);
  let offline = true;
  const scoped = state.createScopedApi({
    session,
    transport: home.createApi({
      fetch: async url => {
        if (url === WHOAMI) return jsonReply(200, grant);
        if (offline) throw new TypeError('Failed to fetch');
        if (url === '/api/home/catalog') return jsonReply(200, { rooms: [], entities: [], unassigned_entities: [] });
        return jsonReply(200, []);
      },
    }),
  });
  const owner = routines.createRoutineOwner({ api: scoped, session, storage: memoryStorage(), now: () => EPOCH });
  await owner.open();
  await flush();
  assertCleanSpanish(owner.getState().list.error.message, 'nexo-routine-owner/disconnect');
  offline = false;
  await owner.refresh();
  assert.equal(owner.getState().list.error, null);
  assert.equal(owner.getState().list.loaded, true);
  await owner.clear({ wipe: true });
});

// ---------------------------------------------------------------------------
// 6. nexo-memory-owner (all 8 C06 classes)
// ---------------------------------------------------------------------------
test('C06 [nexo-memory-owner]: 401, 403, 409, 503, timeout, disconnect-reconnect, expired, partial-uncertain', async () => {
  for (const [label, err, rx] of [
    ['401', { status: 401, kind: 'unauthorized' }, /Iniciá sesión/],
    ['403', { status: 403, kind: 'forbidden' }, /no permite/],
    ['503', { status: 503, kind: 'unavailable', detail: 'memory_unavailable' }, /no está disponible/],
    ['timeout', Object.assign(new Error('El servidor no respondió a tiempo.'), { name: 'TimeoutError', status: null, kind: 'timeout' }), /consultar/],
  ]) {
    const msg = memory.describeReadError(err);
    assert.match(msg, rx, label);
    assertCleanSpanish(msg, `nexo-memory-owner/read/${label}`);
  }
  const c409 = memory.classifyWrite('assign', FACT_ID, { error: { status: 409, kind: 'conflict', detail: 'memory_owner_conflict' } });
  assert.equal(c409.kind, 'provisional');
  assertCleanSpanish(c409.text, 'nexo-memory-owner/409');
  const cUncertain = memory.classifyWrite('delete', FACT_ID, { error: Object.assign(new TypeError('No se pudo conectar con el servidor.'), { status: null, kind: 'network' }) });
  assert.equal(cUncertain.kind, 'uncertain');
  assertCleanSpanish(cUncertain.text, 'nexo-memory-owner/partial-uncertain');
  const cInconclusive = memory.resolveMarker('delete', { where: null, state: null, item: null, factsComplete: false, proposalsComplete: true });
  assert.equal(cInconclusive.result, 'inconclusive');
  assertCleanSpanish(cInconclusive.text, 'nexo-memory-owner/inconclusive');

  // disconnect-reconnect + expired fact filtering via createMemoryOwner
  let wall = EPOCH;
  const grant = { ok: true, role: 'operator', subject: 'sub-mem', permissions: ['memory:read'], expires_at: EPOCH / 1000 + 300 };
  const session = state.createSession({ wall: () => wall, monotonic: () => wall - EPOCH });
  session.accept(grant);
  const clock = fakeClock(EPOCH);
  let offline = true;
  const pagePayload = () => ({
    observed_at: EPOCH / 1000,
    facts: [{
      id: FACT_ID, content: 'Recuerdo temporal', category: 'general', state: 'automatic',
      expires_at: EPOCH / 1000 + 2, revision: 1, confirmed_at: null, confirmed_by: null,
      sources: [{ origin: 'voice', captured_at: EPOCH / 1000 }],
      owner: null, owner_revision: 0, owner_change: null,
    }],
    proposals: [], facts_next: null, proposals_next: null,
  });
  const scoped = state.createScopedApi({
    session,
    transport: home.createApi({
      fetch: async url => {
        if (url === WHOAMI) return jsonReply(200, grant);
        if (offline) throw new TypeError('Failed to fetch');
        return jsonReply(200, pagePayload());
      },
    }),
  });
  const owner = memory.createMemoryOwner({
    api: scoped, session, storage: memoryStorage(),
    wall: () => wall, monotonic: () => wall - EPOCH,
    schedule: clock.setTimer, cancel: clock.clearTimer,
  });
  await owner.open();
  await flush();
  assertCleanSpanish(owner.getState().read.error, 'nexo-memory-owner/disconnect');
  offline = false;
  await owner.refresh();
  assert.equal(owner.getState().read.error, null);
  assert.equal(owner.getState().facts.length, 1);
  wall += 3000;
  assert.equal(owner.getState().facts.length, 0, 'expired automatic fact drops from visible list');
  await owner.clear({ wipe: true });
});

// ---------------------------------------------------------------------------
// 7. nexo-my-memory-owner (all 8 C06 classes)
// ---------------------------------------------------------------------------
test('C06 [nexo-my-memory-owner]: 401, 403, 409, 503, timeout, disconnect-reconnect, expired, partial-uncertain', async () => {
  for (const [label, err, rx] of [
    ['401', { status: 401, kind: 'unauthorized' }, /Iniciá sesión/],
    ['403', { status: 403, kind: 'forbidden' }, /no permite/],
    ['503', { status: 503, kind: 'unavailable', detail: 'memory_unavailable' }, /no están disponibles/],
    ['timeout', Object.assign(new Error('El servidor no respondió a tiempo.'), { name: 'TimeoutError', status: null, kind: 'timeout' }), /consultar/],
  ]) {
    const msg = myMemory.describeReadError(err);
    assert.match(msg, rx, label);
    assertCleanSpanish(msg, `nexo-my-memory-owner/read/${label}`);
  }
  const c409 = myMemory.classifyWrite('return', { error: { status: 409, kind: 'conflict', detail: 'conflict' } });
  assert.equal(c409.kind, 'uncertain');
  assertCleanSpanish(c409.text, 'nexo-my-memory-owner/409');
  const cUncertain = myMemory.classifyWrite('delete', { error: Object.assign(new TypeError('No se pudo conectar con el servidor.'), { status: null, kind: 'network' }) });
  assert.equal(cUncertain.kind, 'uncertain');
  assertCleanSpanish(cUncertain.text, 'nexo-my-memory-owner/partial-uncertain');
  const cInconclusive = myMemory.resolveMarker('delete', { present: false, complete: false });
  assert.equal(cInconclusive.result, 'inconclusive');
  assertCleanSpanish(cInconclusive.text, 'nexo-my-memory-owner/inconclusive');

  let wall = EPOCH;
  const grant = { ok: true, role: 'home', subject: 'sub-mm', account_binding: '0'.repeat(32),
    permissions: ['memory:own:read', 'memory:own:write'], expires_at: EPOCH / 1000 + 300 };
  const session = state.createSession({ wall: () => wall, monotonic: () => wall - EPOCH });
  session.accept(grant);
  const clock = fakeClock(EPOCH);
  let offline = true;
  const scoped = state.createScopedApi({
    session,
    transport: home.createApi({
      fetch: async url => {
        if (url === WHOAMI) return jsonReply(200, grant);
        if (offline) throw new TypeError('Failed to fetch');
        return jsonReply(200, {
          limit: 100, next: null,
          facts: [{
            id: FACT_ID, content: 'Recuerdo propio', category: 'personal', state: 'automatic',
            created_at: EPOCH / 1000, expires_at: EPOCH / 1000 + 2, owner_method: 'person_session',
            owner_assigned_at: EPOCH / 1000, owned_dependents: [], has_hidden_dependents: false,
          }],
        });
      },
    }),
  });
  const owner = myMemory.createMyMemoryOwner({
    api: scoped, session, storage: memoryStorage(),
    wall: () => wall, monotonic: () => wall - EPOCH,
    schedule: clock.setTimer, cancel: clock.clearTimer,
  });
  await owner.open();
  await flush();
  assertCleanSpanish(owner.getState().read.error, 'nexo-my-memory-owner/disconnect');
  offline = false;
  await owner.refresh();
  assert.equal(owner.getState().read.error, null);
  assert.equal(owner.getState().facts.length, 1);
  wall += 3000;
  assert.equal(owner.getState().facts.length, 0, 'expired own fact drops from visible list');
  await owner.clear({ wipe: true });
});

// ---------------------------------------------------------------------------
// 8. nexo-my-voice-owner (all 8 C06 classes)
// ---------------------------------------------------------------------------
test('C06 [nexo-my-voice-owner]: 401, 403, 409, 503, timeout, disconnect-reconnect, expired, partial-uncertain', async () => {
  for (const [label, err, rx] of [
    ['401', { status: 401, kind: 'unauthorized' }, /Iniciá sesión/],
    ['403', { status: 403, kind: 'forbidden' }, /no permite/],
    ['503', { status: 503, kind: 'unavailable' }, /no está disponible/],
    ['timeout', Object.assign(new Error('El servidor no respondió a tiempo.'), { name: 'TimeoutError', status: null, kind: 'timeout' }), /consultar/],
  ]) {
    const msg = myVoice.describeReadError(err);
    assert.match(msg, rx, label);
    assertCleanSpanish(msg, `nexo-my-voice-owner/read/${label}`);
  }
  const c409 = myVoice.classifyWrite('request', { error: { status: 409, kind: 'conflict', detail: 'memory_voice_conflict' } });
  assert.equal(c409.kind, 'refused');
  assertCleanSpanish(c409.text, 'nexo-my-voice-owner/409');
  const cUncertain = myVoice.classifyWrite('verify', { error: Object.assign(new TypeError('No se pudo conectar con el servidor.'), { status: null, kind: 'network' }) });
  assert.equal(cUncertain.kind, 'uncertain');
  assertCleanSpanish(cUncertain.text, 'nexo-my-voice-owner/partial-uncertain');

  // disconnect-reconnect + verification window expiry triggers re-read
  let wall = EPOCH;
  const clock = fakeClock(EPOCH);
  const grant = { ok: true, role: 'home', subject: 'sub-mv', account_binding: '1'.repeat(32),
    permissions: ['memory:own:voice'], expires_at: EPOCH / 1000 + 300 };
  const session = state.createSession({ wall: () => wall, monotonic: () => 0 });
  session.accept(grant);
  let offline = true, reads = 0;
  const scoped = state.createScopedApi({
    session,
    transport: home.createApi({
      fetch: async url => {
        if (url === WHOAMI) return jsonReply(200, grant);
        if (offline) throw new TypeError('Failed to fetch');
        reads++;
        const activeWindow = reads === 1 ? { room_id: 'living', phrase: 'frase de prueba', expires_at: EPOCH / 1000 + 2 } : null;
        return jsonReply(200, {
          enabled: true, state: 'pending_verification', revision: 1,
          disclosure: myVoice.OPT_IN_DISCLOSURE, failures_left: 3, message: null, window: activeWindow,
        });
      },
    }),
  });
  const owner = myVoice.createMyVoiceOwner({
    api: scoped, session, storage: memoryStorage(),
    wall: () => wall, schedule: clock.setTimer, cancel: clock.clearTimer,
  });
  await owner.open();
  await flush();
  assertCleanSpanish(owner.getState().read.error, 'nexo-my-voice-owner/disconnect');
  offline = false;
  await owner.refresh();
  assert.equal(owner.getState().read.error, null);
  assert.ok(owner.getState().status.window !== null);
  wall += 2500;
  clock.advance(2500);
  await flush();
  assert.equal(owner.getState().status.window, null, 'window expiry triggers re-read and clears phrase');
  await owner.clear({ wipe: true });
});

// ---------------------------------------------------------------------------
// 9. nexo-voice-consents-owner (7 C06 classes)
// ---------------------------------------------------------------------------
test('C06 [nexo-voice-consents-owner]: 401, 403, 409, 503, timeout, disconnect-reconnect, partial-uncertain', async () => {
  for (const [label, err, rx] of [
    ['401', { status: 401, kind: 'unauthorized' }, /Iniciá sesión/],
    ['403', { status: 403, kind: 'forbidden' }, /no permite/],
    ['503', { status: 503, kind: 'unavailable' }, /no está disponible/],
    ['timeout', Object.assign(new Error('El servidor no respondió a tiempo.'), { name: 'TimeoutError', status: null, kind: 'timeout' }), /consultar/],
  ]) {
    const msg = voiceConsents.describeReadError(err);
    assert.match(msg, rx, label);
    assertCleanSpanish(msg, `nexo-voice-consents-owner/read/${label}`);
  }
  const c409 = voiceConsents.classifyWrite('reopen', { error: { status: 409, kind: 'conflict', detail: 'memory_voice_conflict' } });
  assert.equal(c409.kind, 'refused');
  assertCleanSpanish(c409.text, 'nexo-voice-consents-owner/409');
  const cUncertain = voiceConsents.classifyWrite('resume', { error: Object.assign(new TypeError('No se pudo conectar con el servidor.'), { status: null, kind: 'network' }) });
  assert.equal(cUncertain.kind, 'uncertain');
  assertCleanSpanish(cUncertain.text, 'nexo-voice-consents-owner/partial-uncertain');

  const grant = { ok: true, role: 'operator', subject: 'sub-vc',
    permissions: ['memory:read', 'memory:assign'], expires_at: EPOCH / 1000 + 300 };
  const session = state.createSession({ wall: () => EPOCH, monotonic: () => 0 });
  session.accept(grant);
  let offline = true;
  const scoped = state.createScopedApi({
    session,
    transport: home.createApi({
      fetch: async url => {
        if (url === WHOAMI) return jsonReply(200, grant);
        if (offline) throw new TypeError('Failed to fetch');
        if (url === '/api/memory/voice/switch') return jsonReply(200, { configured: true, enabled: false, effective: false, revision: 1 });
        return jsonReply(409, { detail: 'memory_voice_disabled' });
      },
    }),
  });
  const owner = voiceConsents.createVoiceConsentsOwner({ api: scoped, session, storage: memoryStorage(), wall: () => EPOCH });
  await owner.open();
  await flush();
  assertCleanSpanish(owner.getState().read.error, 'nexo-voice-consents-owner/disconnect');
  offline = false;
  await owner.refresh();
  assert.equal(owner.getState().read.error, null);
  assert.equal(owner.getState().listReason, 'disabled');
  await owner.clear({ wipe: true });
});

// ---------------------------------------------------------------------------
// 10. nexo-access-owner (7 C06 classes)
// ---------------------------------------------------------------------------
test('C06 [nexo-access-owner]: 401, 403, 409, 503, timeout, disconnect-reconnect, partial-uncertain', async () => {
  const nowS = Math.floor(EPOCH / 1000);
  const catalogPage = {
    observed_at: nowS, next_cursor: null,
    rows: [{ id: UUID_1, subject: 'Hogar', role: 'home', user_id: null, issued_at: nowS,
      expires_at: nowS + 3600, revision: 1, active: true, pending: false, samples: [] }],
  };
  const makeSession = () => {
    const s = state.createSession({ wall: () => EPOCH, monotonic: () => 0 });
    s.accept({ ok: true, role: 'operator', subject: 'op', permissions: ['auth:manage', 'admin:manage'], expires_at: nowS + 300 });
    return s;
  };

  // 401 and 403 trigger verifyIdentity -> onUnauthorized when authority says false
  for (const status of [401, 403]) {
    let unauth = 0;
    const owner = createAccessOwner({
      session: makeSession(),
      fetch: async () => jsonReply(status, {}),
      verifyIdentity: async () => false,
      onUnauthorized: () => { unauth++; },
      newRequestId: () => UUID_2,
    });
    await owner.open();
    assert.equal(unauth >= 1, true, `access-owner/${status}`);
    await owner.clear();
  }

  // 503, timeout, disconnect-reconnect, 409/partial-uncertain
  {
    let mode = '503', reqSeq = 0;
    const reqIds = [UUID_2, '40000000-0000-4000-8000-000000000004'];
    const owner = createAccessOwner({
      session: makeSession(),
      fetch: async (url, init = {}) => {
        if (url === '/api/admin/users') return jsonReply(200, []);
        if (mode === '503') return jsonReply(503, {});
        if (mode === 'timeout') {
          return new Promise((_, reject) => {
            init.signal.addEventListener('abort', () => reject(new DOMException('aborted', 'AbortError')), { once: true });
          });
        }
        if (mode === 'offline') throw new TypeError('Failed to fetch');
        if (url.startsWith('/api/admin/auth/credentials/') && init.method === 'DELETE') {
          if (mode === '409') return jsonReply(409, { detail: 'revision_conflict' });
          throw new TypeError('Failed to fetch');
        }
        if (url.startsWith('/api/admin/auth/credential-attempts/')) {
          if (mode === '409') {
            return jsonReply(200, {
              state: 'terminal',
              receipt: { request_id: UUID_2, action: 'revoke', credential_id: UUID_1, expected_revision: 1,
                outcome: 'no_effect', result_revision: null, reason: 'revision_conflict', completed_at: nowS + 1 },
            });
          }
          return jsonReply(200, { state: 'unconfirmed', receipt: null });
        }
        return jsonReply(200, catalogPage);
      },
      verifyIdentity: async () => true,
      newRequestId: () => reqIds[reqSeq++],
      timeoutMs: 15,
    });
    await owner.open();
    assert.equal(owner.snapshot().status, 'unavailable');
    assertCleanSpanish(owner.snapshot().notice, 'nexo-access-owner/503');

    mode = 'timeout';
    await owner.refresh();
    assert.equal(owner.snapshot().status, 'unavailable');
    assertCleanSpanish(owner.snapshot().notice, 'nexo-access-owner/timeout');

    mode = 'offline';
    await owner.refresh();
    assert.equal(owner.snapshot().status, 'unavailable');
    assertCleanSpanish(owner.snapshot().notice, 'nexo-access-owner/disconnect');

    mode = 'ok';
    await owner.refresh();
    assert.equal(owner.snapshot().status, 'real');
    assert.equal(owner.snapshot().rows.length, 1);

    // 409 conflict reconciled cleanly to terminal no_effect
    mode = '409';
    await owner.revoke(UUID_1, 1);
    assertCleanSpanish(owner.snapshot().notice, 'nexo-access-owner/409');

    // partial-uncertain lost write holds canMutate = false
    mode = 'uncertain';
    await owner.revoke(UUID_1, 1);
    assert.equal(owner.snapshot().canMutate, false);
    assertCleanSpanish(owner.snapshot().notice, 'nexo-access-owner/partial-uncertain');
    await owner.clear();
  }
});

// ---------------------------------------------------------------------------
// 11. nexo-voice-owner (7 C06 classes)
// ---------------------------------------------------------------------------
test('C06 [nexo-voice-owner]: 401, 403, 409, 503, timeout, disconnect-reconnect, partial-uncertain', async () => {
  const RESET = { type: 'live_reset', payload: { id: UUID_1, nonce: UUID_2 }, ts: '' };
  const READY = { type: 'rest_ready', connection_id: UUID_1, channels: ['voices:captures', 'voices:profiles', 'voices:review'], ts: '' };
  const clock = fakeClock(EPOCH);
  const session = state.createSession({ wall: () => EPOCH, monotonic: () => 0 });
  session.accept({ ok: true, role: 'operator', expires_at: EPOCH / 1000 + 300, permissions: ['voices:read', 'voices:review'] });

  let statusMode = 200, unauth = 0, verifyResult = false;
  const sockets = [], controllers = [], statuses = [];
  const owner = createVoiceOwner({
    session,
    socketUrl: () => 'ws://synthetic/ws/voices',
    connect() { const s = new FakeSocket(); sockets.push(s); return s; },
    mount(controller) {
      controllers.push(controller);
      return () => { controller.leave(); for (const k of ['profiles', 'captures', 'review']) controller.state[k].committed(null); };
    },
    onStatus: msg => statuses.push(msg),
    verifyIdentity: async () => verifyResult,
    onUnauthorized: () => { unauth++; },
    schedule: clock.setTimer,
    cancel: clock.clearTimer,
    retryMs: 500,
    fetch: async (url, init = {}) => {
      if (init.method === 'DELETE') {
        if (statusMode === 'timeout') throw new DOMException('timed out', 'TimeoutError');
        if (statusMode === 'network') throw new TypeError('Failed to fetch');
        return jsonReply(statusMode, {});
      }
      return jsonReply(200, { connection_id: UUID_1, channel: url.endsWith('/profiles') ? 'voices:profiles' : 'voices:captures', snapshot: null });
    },
  });

  await owner.open();
  sockets.at(-1).open();
  sockets.at(-1).event('message', RESET);
  await flush();
  sockets.at(-1).event('message', READY);
  await flush();
  const ctrl = controllers.at(-1);

  for (const [label, mode, expected] of [
    ['401', 401, 'denied'],
    ['403', 403, 'denied'],
    ['409', 409, 'conflict'],
    ['503', 503, 'unavailable'],
    ['timeout', 'timeout', 'uncertain'],
    ['partial-uncertain', 'network', 'uncertain'],
  ]) {
    statusMode = mode;
    const outcome = await ctrl.deleteProfileOutcome(PROFILE_ID, 1);
    assert.equal(outcome.state, expected, `voice-owner/${label}`);
    if (expected === 'uncertain') {
      // Clear held uncertain mutation via fresh refresh
      statusMode = 200;
      await ctrl.refresh({ surface: 'obs' });
      await flush();
    }
  }
  await flush();
  assert.ok(unauth >= 1, '401/403 triggers verifyIdentity -> onUnauthorized');

  // disconnect-reconnect via socket close + retry timer
  const beforeSockets = sockets.length;
  sockets.at(-1).close();
  await flush();
  assert.ok(statuses.some(m => /perdió la conexión/.test(m)));
  assertCleanSpanish(statuses.at(-1), 'nexo-voice-owner/disconnect');
  clock.advance(500);
  await flush();
  assert.equal(sockets.length, beforeSockets + 1, 'reconnect opens a fresh WebSocket');
  await owner.clear();
});

// ---------------------------------------------------------------------------
// 12. nexo-evaluation-owner (6 C06 classes)
// ---------------------------------------------------------------------------
test('C06 [nexo-evaluation-owner]: 401, 403, 503, timeout, disconnect-reconnect, expired', async () => {
  const RESET = { type: 'live_reset', payload: { id: UUID_1, nonce: UUID_2 }, ts: '' };
  const READY = { type: 'rest_ready', connection_id: UUID_1, channels: ['evaluation:sample'], ts: '' };
  let wall = EPOCH, mono = 0;
  const clock = fakeClock(EPOCH);
  const session = state.createSession({ wall: () => wall, monotonic: () => mono });
  session.accept({ ok: true, role: 'evaluator', expires_at: EPOCH / 1000 + 300, permissions: ['evaluation:read'] });

  let fetchMode = 503, unauth = 0, verifyResult = false;
  const sockets = [], controllers = [], statuses = [];
  const owner = createEvaluationOwner({
    session,
    socketUrl: () => 'ws://synthetic/ws/evaluation',
    connect() { const s = new FakeSocket(); sockets.push(s); return s; },
    mount(controller) { controllers.push(controller); return () => {}; },
    onStatus: msg => statuses.push(msg),
    verifyIdentity: async () => verifyResult,
    onUnauthorized: () => { unauth++; },
    monotonic: () => mono, wall: () => wall,
    schedule: clock.setTimer, cancel: clock.clearTimer, retryMs: 500,
    fetch: async () => {
      if (fetchMode === 'timeout') throw new DOMException('timed out', 'TimeoutError');
      if (fetchMode === 'ok') {
        const nowS = wall / 1000;
        return jsonReply(200, {
          observed_at: nowS, expires_at: nowS + 2,
          grants: [{ fingerprint_id: FP, modalities: ['transcript_full'], expires_at: nowS + 2 }],
        });
      }
      return jsonReply(fetchMode, {});
    },
  });

  await owner.open();
  sockets.at(-1).open();
  sockets.at(-1).event('message', RESET);
  await flush();
  sockets.at(-1).event('message', READY);
  await flush();

  // 503
  assert.equal(controllers.at(-1).snapshot().status, 'unavailable');
  assertCleanSpanish(statuses.at(-1), 'nexo-evaluation-owner/503');

  // timeout
  fetchMode = 'timeout';
  await controllers.at(-1).refresh();
  await flush();
  assert.equal(controllers.at(-1).snapshot().status, 'unavailable');
  assertCleanSpanish(statuses.at(-1), 'nexo-evaluation-owner/timeout');

  // 403 and 401
  verifyResult = true;
  fetchMode = 403;
  await controllers.at(-1).refresh();
  await flush();
  assert.equal(controllers.at(-1).snapshot().status, 'unavailable');
  assertCleanSpanish(statuses.at(-1), 'nexo-evaluation-owner/403');

  verifyResult = false;
  fetchMode = 401;
  await controllers.at(-1).refresh();
  await flush();
  assert.ok(unauth >= 1, 'evaluation-owner/401 invokes onUnauthorized');

  // expired catalogue transitions to expired when clock advances
  fetchMode = 'ok';
  await controllers.at(-1).refresh();
  await flush();
  assert.equal(controllers.at(-1).snapshot().status, 'real');
  wall += 2500; mono += 2500;
  assert.equal(controllers.at(-1).snapshot().status, 'expired');

  // disconnect-reconnect via socket close + retry timer
  const beforeSockets = sockets.length;
  sockets.at(-1).close();
  await flush();
  assert.ok(statuses.some(m => /perdió la conexión/.test(m)));
  assertCleanSpanish(statuses.at(-1), 'nexo-evaluation-owner/disconnect');
  clock.advance(500);
  await flush();
  assert.equal(sockets.length, beforeSockets + 1);
  await owner.clear();
});

// ---------------------------------------------------------------------------
// 13. obs-data + obs/src/session.js (all 8 C06 classes, read-only import)
// ---------------------------------------------------------------------------
test('C06 [obs-data]: 401, 403, 409, 503, timeout, disconnect-reconnect, expired, partial-uncertain (read-only)', async t => {
  const clock = fakeClock(EPOCH);
  const store = obsData.createObservationStore({ now: clock.now, setTimer: clock.setTimer, clearTimer: clock.clearTimer });
  t.after(() => store.clear());
  store.setPermissions(['admin:manage', 'diagnostics:read']);

  // 401 on readCollection & runOperation
  await assert.rejects(
    obsData.readCollection({ fetch: async () => jsonReply(401, {}), store, key: 'ZONES', path: '/api/zones' }),
    obsData.AuthRequired
  );
  const op401 = await obsData.runOperation({ fetch: async () => jsonReply(401, {}), store, kind: 'ackAlert', target: 'a1' });
  assert.equal(op401.state, 'auth');
  assertCleanSpanish(op401.message, 'obs-data/401');

  // 403 on readCollection & runOperation
  assert.equal(await obsData.readCollection({ fetch: async () => jsonReply(403, {}), store, key: 'ZONES', path: '/api/zones' }), 'sin permiso');
  assertCleanSpanish(store.entry('ZONES').reason, 'obs-data/403/read');
  const op403 = await obsData.runOperation({ fetch: async () => jsonReply(403, {}), store, kind: 'ackAlert', target: 'a1' });
  assert.equal(op403.state, 'denied');
  assertCleanSpanish(op403.message, 'obs-data/403/op');

  // 409 conflict & demo rejection
  const op409 = await obsData.runOperation({ fetch: async () => jsonReply(409, { detail: 'state_changed' }), store, kind: 'ackAlert', target: 'a1' });
  assert.equal(op409.state, 'conflict');
  assertCleanSpanish(op409.message, 'obs-data/409');

  // 503 on readCollection & runOperation
  assert.equal(await obsData.readCollection({ fetch: async () => jsonReply(503, {}), store, key: 'ALERTS', path: '/api/alerts' }), 'degradada');
  assertCleanSpanish(store.entry('ALERTS').reason, 'obs-data/503/read');
  const op503 = await obsData.runOperation({ fetch: async () => jsonReply(503, {}), store, kind: 'ackAlert', target: 'a1' });
  assert.equal(op503.state, 'unavailable');
  assertCleanSpanish(op503.message, 'obs-data/503/op');

  // timeout on runOperation -> uncertain and held until fresh read
  {
    const opClock = fakeClock(EPOCH);
    const hung = obsData.runOperation({
      fetch: (_u, { signal }) => new Promise((_, reject) => {
        signal.addEventListener('abort', () => reject(new DOMException('aborted', 'AbortError')), { once: true });
      }),
      store, kind: 'restartService', target: 'kza-voice',
      setTimer: opClock.setTimer, clearTimer: opClock.clearTimer,
    });
    await Promise.resolve();
    opClock.advance(obsData.OPERATION_TIMEOUT_MS);
    const outcome = await hung;
    assert.equal(outcome.state, 'uncertain');
    assert.match(outcome.message, /no respondió a tiempo/);
    assertCleanSpanish(outcome.message, 'obs-data/timeout');
  }

  // disconnect-reconnect: network fail marks degraded, liveCloseStatus retries, fresh read restores real + clears held op
  {
    assert.equal(await obsData.readCollection({
      fetch: async () => { throw new TypeError('Failed to fetch'); },
      store, key: 'SERVICES', path: '/api/system/services',
    }), 'degradada');
    assertCleanSpanish(store.entry('SERVICES').reason, 'obs-data/disconnect');
    const wsRetry = await obsSession.liveCloseStatus(async () => { throw new TypeError('Failed to fetch'); }, 1006);
    assert.equal(wsRetry.retry, true);
    assertCleanSpanish(wsRetry.message, 'obs-data/ws-reconnect');
    assert.equal(await obsData.readCollection({
      fetch: async () => jsonReply(200, [{ name: 'kza-voice', status: 'active' }], realObsHeaders()),
      store, key: 'SERVICES', path: '/api/system/services',
    }), 'real');
    assert.equal(store.operationOutcome('restartService', 'kza-voice'), undefined);
  }

  // expired: time-driven freshness transitions vigente -> vencida
  {
    assert.equal(store.freshness('SERVICES'), 'vigente');
    clock.advance(7000);
    assert.equal(store.freshness('SERVICES'), 'vencida');
  }

  // partial-uncertain: 504 holds uncertain; confirmed write + 503 refresh reports confirmed with refresh warning
  {
    const op504 = await obsData.runOperation({
      fetch: async () => jsonReply(504, { detail: { effect: 'uncertain' } }),
      store, kind: 'clearCooldown', target: 'fast',
    });
    assert.equal(op504.state, 'uncertain');
    assertCleanSpanish(op504.message, 'obs-data/partial-uncertain/504');
    const opPartial = await obsData.runOperation({
      fetch: async url => url.endsWith('/ack') ? jsonReply(200, { ok: true }) : jsonReply(503, {}),
      store, kind: 'ackAlert', target: 'a-partial',
    });
    assert.equal(opPartial.state, 'confirmed');
    assert.match(opPartial.message, /No se pudo actualizar la lista/);
    assertCleanSpanish(opPartial.message, 'obs-data/partial-uncertain/refresh');
  }
});

// ---------------------------------------------------------------------------
// 14. JSON evidence matrix integrity check (C02 simulation labels & counts)
// ---------------------------------------------------------------------------
test('C06 [artifact]: docs/research/2026-11-20-autonomous-t045-error-matrix.json matches C02 simulation labeling and counts', () => {
  const doc = JSON.parse(fs.readFileSync(MATRIX_PATH, 'utf8'));
  assert.equal(doc.task, 'T045');
  assert.equal(doc.unit, 'T045-U3');
  assert.equal(doc.criterion, 'C06');
  assert.equal(doc.evidence_class, 'C02 simulation');
  assert.deepEqual(doc.error_classes, [...C06_CLASSES]);
  assert.ok(Array.isArray(doc.entries) && doc.entries.length > 0);

  const bySurface = {};
  const byClass = Object.fromEntries(C06_CLASSES.map(c => [c, 0]));
  const seenPairs = new Set();

  for (const entry of doc.entries) {
    assert.equal(entry.evidence_class, 'C02 simulation', `entry ${entry.id} must be labeled C02 simulation`);
    assert.ok(C06_CLASSES.includes(entry.error_class), `unknown class ${entry.error_class}`);
    assert.ok(typeof entry.surface === 'string' && entry.surface.length > 0);
    assert.ok(typeof entry.expected_behavior === 'string' && entry.expected_behavior.length > 0);
    assert.doesNotMatch(entry.expected_behavior, /Failed to fetch|AbortError|TimeoutError|NetworkError/);
    const pair = `${entry.surface}::${entry.error_class}`;
    assert.equal(seenPairs.has(pair), false, `duplicate matrix pair ${pair}`);
    seenPairs.add(pair);
    bySurface[entry.surface] = (bySurface[entry.surface] || 0) + 1;
    byClass[entry.error_class] += 1;
  }

  assert.equal(doc.total_entries, doc.entries.length);
  assert.deepEqual(doc.counts_by_surface, bySurface);
  assert.deepEqual(doc.counts_by_class, byClass);
  for (const c of C06_CLASSES) assert.ok(byClass[c] > 0, `class ${c} must have at least one entry`);
  for (const surface of [
    'home-createApi', 'nexo-login', 'nexo-notices-owner', 'nexo-tasks-owner', 'nexo-people',
    'nexo-routine-owner', 'nexo-memory-owner', 'nexo-my-memory-owner', 'nexo-my-voice-owner',
    'nexo-voice-consents-owner', 'nexo-access-owner', 'nexo-voice-owner', 'nexo-evaluation-owner', 'obs-data',
  ]) {
    assert.ok(bySurface[surface] > 0, `missing surface ${surface}`);
  }
});
