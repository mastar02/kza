const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { test } = require('node:test');
const data = require('../../src/dashboard/frontend/obs/src/obs-data.js');
const Babel = require('../../src/dashboard/frontend/obs/vendor/babel-standalone-7.29.0.min.js');

const OBS = path.resolve(__dirname, '../../src/dashboard/frontend/obs');

function headers(values = {}) {
  const lower = Object.fromEntries(Object.entries(values).map(([key, value]) => [key.toLowerCase(), value]));
  return { get: name => lower[name.toLowerCase()] ?? null };
}

function reply(status, body, extra = {}) {
  return { ok: status >= 200 && status < 300, status, headers: headers(extra), json: async () => body };
}

function realHeaders(extra = {}) {
  return { 'X-KZA-Source': 'real', 'X-KZA-Source-Id': 'alert_manager',
    'X-KZA-Observed-At': '2026-11-10T12:00:00+00:00', 'X-KZA-Expires-At': '2026-11-10T12:00:06+00:00', ...extra };
}

function sourceFiles() {
  const out = [];
  (function walk(dir) {
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
      const full = path.join(dir, entry.name);
      if (entry.isDirectory()) { if (entry.name !== 'vendor') walk(full); }
      else if (/\.(jsx?|html)$/.test(entry.name)) out.push(full);
    }
  })(OBS);
  return out;
}

function adminStore(permissions = ['admin:manage', 'diagnostics:read']) {
  const store = data.createObservationStore();
  store.setPermissions(permissions);
  return store;
}

test('a real response is kept with its provenance and unobserved fields', () => {
  const store = data.createObservationStore();
  const status = store.accept('ZONES', data.parseObservation(headers(realHeaders({
    'X-KZA-Unobserved-Fields': 'user,last_utterance' }))), [{ id: 'sala' }]);
  assert.equal(status, 'real');
  assert.deepEqual(store.get('ZONES'), [{ id: 'sala' }]);
  assert.equal(store.entry('ZONES').sourceId, 'alert_manager');
  assert.equal(store.unobserved('ZONES', 'user'), true);
  assert.equal(store.unobserved('ZONES', 'spotify_playback'), false);
  assert.equal(data.freshness(store.entry('ZONES'), new Date('2026-11-10T12:00:03Z')), 'vigente');
  assert.equal(data.freshness(store.entry('ZONES'), new Date('2026-11-10T12:00:07Z')), 'vencida');
});

test('degraded, missing or unproven responses keep no data', () => {
  const store = data.createObservationStore();
  for (const source of ['degraded', '', 'something-else']) {
    assert.equal(store.accept('ALERTS', data.parseObservation(headers({ 'X-KZA-Source': source })), [{ id: 'x' }]), 'degradada');
    assert.deepEqual(store.get('ALERTS'), []);
    assert.equal(store.available('ALERTS'), false);
  }
  assert.equal(store.accept('ALERTS', data.parseObservation(headers(realHeaders())), null), 'degradada');
  assert.deepEqual(store.get('ALERTS'), []);
});

test('server demo mode is kept only as an explicitly labelled demonstration', () => {
  const store = data.createObservationStore();
  assert.equal(store.accept('GPUS', data.parseObservation(headers({ 'X-KZA-Source': 'mock' })), [{ id: 0 }]), 'demostración');
  assert.equal(store.available('GPUS'), true);
  assert.deepEqual(store.summary().demo, ['GPUS']);
});

test('hydrate reads permissions, marks failures unavailable and never invents data', async () => {
  const store = data.createObservationStore();
  const fetch = async (url) => {
    if (url === '/api/admin/auth/whoami') return reply(200, { permissions: ['diagnostics:read'] });
    if (url === '/api/alerts') return reply(200, [{ id: 'a1', acked: false }], realHeaders());
    if (url === '/api/system/gpus') throw new TypeError('network');
    return reply(500, { detail: 'boom' });
  };
  await data.hydrate({ fetch, store });
  assert.equal(store.can('diagnostics:read'), true);
  assert.equal(store.can('admin:manage'), false);
  assert.equal(store.status('ALERTS'), 'real');
  for (const [key] of data.COLLECTIONS.filter(([key]) => key !== 'ALERTS')) {
    assert.equal(store.status(key), 'degradada', key);
    assert.deepEqual(store.get(key), [], key);
  }
});

test('hydrate turns an expired or denied session into a login request', async () => {
  for (const status of [401, 403]) {
    const store = data.createObservationStore();
    const fetch = async (url) => url === '/api/admin/auth/whoami' ? reply(200, { permissions: [] }) : reply(status, {});
    await assert.rejects(data.hydrate({ fetch, store }), data.AuthRequired);
  }
  const store = data.createObservationStore();
  await assert.rejects(data.hydrate({ fetch: async () => reply(401, {}), store }), data.AuthRequired);
});

test('clear drops data, permissions and channel state but keeps the shared sources object', () => {
  const store = adminStore();
  const sources = store.sources;
  store.accept('ALERTS', data.parseObservation(headers(realHeaders())), [{ id: 'a' }]);
  store.setSource('CONVERSATIONS', 'real');
  store.setChannel('ws', 'conectado');
  store.setNotice({ state: 'confirmed', message: 'x' });
  store.clear();
  assert.equal(store.notice(), null);
  assert.equal(store.sources, sources);
  assert.equal(sources.CONVERSATIONS, undefined);
  assert.equal(sources.ALERTS, 'sin consultar');
  assert.deepEqual(store.get('ALERTS'), []);
  assert.equal(store.can('admin:manage'), false);
  assert.equal(store.channel('ws'), 'sin conectar');
});

test('operations are disabled with a reason without administrative permission', async () => {
  const store = adminStore(['diagnostics:read']);
  for (const kind of ['ackAlert', 'dismissAlert', 'restartService', 'clearCooldown']) {
    const allowed = data.availability(store, kind, 'kza-voice');
    assert.equal(allowed.enabled, false);
    assert.match(allowed.reason, /administración/);
  }
  let called = false;
  const outcome = await data.runOperation({ fetch: async () => { called = true; }, store, kind: 'ackAlert', target: 'a1' });
  assert.equal(called, false);
  assert.equal(outcome.state, 'denied');
});

test('restart is offered only for services in the server allowlist', () => {
  const store = adminStore();
  assert.deepEqual([...data.RESTARTABLE_SERVICES].sort(), ['chromadb', 'kza-llm-fast', 'kza-llm-ik', 'kza-voice']);
  assert.equal(data.availability(store, 'restartService', 'kza-voice').enabled, true);
  const ha = data.availability(store, 'restartService', 'home-assistant');
  assert.equal(ha.enabled, false);
  assert.match(ha.reason, /no se reinicia/);
  assert.equal(data.availability(store, 'unknownOperation', 'x').enabled, false);
});

test('admin python allowlist matches the panel mirror', () => {
  const source = fs.readFileSync(path.resolve(__dirname, '../../src/dashboard/admin.py'), 'utf8');
  const block = source.match(/RESTARTABLE_SERVICES = frozenset\(\{([^}]*)\}\)/);
  assert.ok(block);
  const names = [...block[1].matchAll(/"([^"]+)"/g)].map(match => match[1]).sort();
  assert.deepEqual(names, [...data.RESTARTABLE_SERVICES].sort());
});

test('each operation calls its existing endpoint and confirms only after the server and a fresh read', async () => {
  const cases = [
    ['ackAlert', 'al 1', 'POST', '/api/admin/alerts/al%201/ack', '/api/alerts', 'ALERTS'],
    ['dismissAlert', 'al_2', 'DELETE', '/api/admin/alerts/al_2', '/api/alerts', 'ALERTS'],
    ['restartService', 'kza-voice', 'POST', '/api/admin/services/kza-voice/restart', '/api/system/services', 'SERVICES'],
    ['clearCooldown', 'fast', 'POST', '/api/llm/endpoints/fast/clear-cooldown', '/api/llm/endpoints', 'LLM_ENDPOINTS'],
  ];
  for (const [kind, target, method, url, refresh, key] of cases) {
    const store = adminStore();
    const calls = [];
    const fetch = async (requested, options = {}) => {
      calls.push([options.method || 'GET', requested]);
      if (requested === url) return reply(200, { ok: true });
      return reply(200, [{ id: 'fresh' }], realHeaders());
    };
    const outcome = await data.runOperation({ fetch, store, kind, target });
    assert.deepEqual(calls, [[method, url], ['GET', refresh]], kind);
    assert.equal(outcome.state, 'confirmed', kind);
    assert.equal(data.OPERATIONS[kind](target).done, outcome.message, kind);
    assert.deepEqual(store.get(key), [{ id: 'fresh' }], kind);
  }
});

test('a confirmed operation says so when the refreshed list is unavailable', async () => {
  const store = adminStore();
  const fetch = async (url) => url.endsWith('/ack') ? reply(200, { ok: true }) : reply(503, {});
  const outcome = await data.runOperation({ fetch, store, kind: 'ackAlert', target: 'a1' });
  assert.equal(outcome.state, 'confirmed');
  assert.match(outcome.message, /No se pudo actualizar la lista/);
  assert.equal(store.status('ALERTS'), 'degradada');
});

test('failures are reported honestly and never as applied', async () => {
  const expectations = [
    [reply(403, { detail: 'dashboard_permission_denied' }), 'denied'],
    [reply(401, {}), 'auth'],
    [reply(404, { detail: 'alert_not_found' }), 'not_found'],
    [reply(503, { detail: 'alert_manager no inyectado' }), 'unavailable'],
    [reply(409, { detail: 'observability_demo_mode' }), 'rejected'],
    [reply(409, { detail: 'other_conflict' }), 'conflict'],
    [reply(504, { detail: { reason: 'restart_timeout', effect: 'uncertain' } }), 'uncertain'],
    [reply(500, { detail: { reason: 'restart_command_failed', effect: 'uncertain' } }), 'uncertain'],
    [reply(500, { detail: 'other' }), 'error'],
  ];
  for (const [response, state] of expectations) {
    const store = adminStore();
    const calls = [];
    const outcome = await data.runOperation({ fetch: async (url) => { calls.push(url); return response; },
      store, kind: 'restartService', target: 'kza-voice' });
    assert.equal(outcome.state, state);
    assert.equal(calls.length, 1, 'no refresh after a failed operation');
    assert.doesNotMatch(outcome.message, /completó|marcó|descartó|quitó/);
  }
  const outcome = await data.runOperation({ fetch: async () => { throw new TypeError('offline'); },
    store: adminStore(), kind: 'clearCooldown', target: 'fast' });
  assert.equal(outcome.state, 'uncertain');
});

function fakeClock(start) {
  let current = new Date(start).getTime();
  const timers = [];
  return {
    now: () => new Date(current),
    setTimer(callback, ms) { const timer = { at: current + ms, callback }; timers.push(timer); return timer; },
    clearTimer(timer) { const index = timers.indexOf(timer); if (index >= 0) timers.splice(index, 1); },
    advance(ms) {
      current += ms;
      for (;;) {
        const due = timers.filter(timer => timer.at <= current).sort((x, y) => x.at - y.at)[0];
        if (!due) break;
        timers.splice(timers.indexOf(due), 1);
        due.callback();
      }
    },
    pending: () => timers.length,
  };
}

test('freshness is time-driven: expiry relabels the observation and wakes subscribers without a new read', () => {
  const clock = fakeClock('2026-11-10T12:00:01Z');
  const store = data.createObservationStore(clock);
  store.accept('ALERTS', data.parseObservation(headers(realHeaders())), [{ id: 'a' }]);
  let renders = 0;
  store.subscribe(() => { renders++; });
  assert.equal(store.freshness('ALERTS'), 'vigente');
  clock.advance(4000);
  assert.equal(renders, 0);
  assert.equal(store.freshness('ALERTS'), 'vigente');
  clock.advance(1500);
  assert.equal(renders, 1, 'a redraw is scheduled at expiry');
  assert.equal(store.freshness('ALERTS'), 'vencida');
  assert.equal(clock.pending(), 0);
  store.accept('GPUS', data.parseObservation(headers(realHeaders({ 'X-KZA-Expires-At': '2026-11-10T12:00:20+00:00' }))), []);
  store.clear();
  assert.equal(clock.pending(), 0, 'clear cancels the expiry timer');
});

test('every operation is disabled in server demo mode and a mocked reply is never a success', async () => {
  const store = adminStore();
  for (const [key, kind, target] of [['ALERTS', 'ackAlert', 'a'], ['ALERTS', 'dismissAlert', 'a'],
      ['SERVICES', 'restartService', 'kza-voice'], ['LLM_ENDPOINTS', 'clearCooldown', 'fast']]) {
    store.accept(key, data.parseObservation(headers({ 'X-KZA-Source': 'mock' })), [{ id: target }]);
    const allowed = data.availability(store, kind, target);
    assert.equal(allowed.enabled, false, kind);
    assert.match(allowed.reason, /modo demostración/);
    let called = false;
    const outcome = await data.runOperation({ fetch: async () => { called = true; }, store, kind, target });
    assert.equal(called, false, kind);
    assert.notEqual(outcome.state, 'confirmed');
  }
  const real = adminStore();
  const outcome = await data.runOperation({ store: real, kind: 'clearCooldown', target: 'fast',
    fetch: async () => reply(200, { ok: true, endpoint_id: 'fast', mocked: true }) });
  assert.equal(outcome.state, 'demo');
  assert.match(outcome.message, /no se aplicó/);
});

test('a hung operation is aborted on the client and reported as uncertain', async () => {
  const clock = fakeClock('2026-11-10T12:00:00Z');
  let signal;
  const fetch = (url, options) => new Promise((resolve, reject) => {
    signal = options.signal;
    options.signal.addEventListener('abort', () => reject(new DOMException('aborted', 'AbortError')));
  });
  const running = data.runOperation({ fetch, store: adminStore(), kind: 'restartService', target: 'kza-voice',
    setTimer: clock.setTimer, clearTimer: clock.clearTimer });
  await Promise.resolve();
  assert.equal(data.OPERATION_TIMEOUT_MS > 15000, true, 'client bound stays above the server restart timeout');
  clock.advance(data.OPERATION_TIMEOUT_MS);
  const outcome = await running;
  assert.equal(signal.aborted, true);
  assert.equal(outcome.state, 'uncertain');
  assert.match(outcome.message, /no respondió a tiempo/);
  assert.equal(clock.pending(), 0);
});

test('an applied operation whose refresh finds an expired session still says the server accepted it', async () => {
  const fetch = async (url) => url.endsWith('/restart') ? reply(200, { ok: true, command_completed: true }) : reply(401, {});
  const outcome = await data.runOperation({ fetch, store: adminStore(), kind: 'restartService', target: 'kza-voice' });
  assert.equal(outcome.state, 'confirmed');
  assert.equal(outcome.sessionExpired, true);
  assert.match(outcome.message, /^systemctl completó el reinicio de kza-voice/);
  assert.match(outcome.message, /No se pudo actualizar la lista porque la sesión venció/);
  assert.doesNotMatch(outcome.message, /reintent/i);
});

test('cooldown countdown needs a zone-explicit instant and is independent of the browser zone', () => {
  const now = new Date('2026-11-10T14:32:31Z');
  assert.equal(data.remainingUntil('2026-11-10T14:35:02+00:00', now), 151);
  assert.equal(data.remainingUntil('2026-11-10T11:35:02-03:00', now), 151);
  assert.equal(data.remainingUntil('2026-11-10T14:30:00Z', now), 0);
  for (const value of ['14:35:02', '2026-11-10T14:35:02', null, 'bad']) assert.equal(data.remainingUntil(value, now), null);
  assert.equal(data.formatInstant('2026-11-10T14:35:02'), 'no disponible (sin zona)');
  assert.match(data.formatInstant('2026-11-10T14:35:02+00:00'), /\d{2}:\d{2}:\d{2}\s?\S+/);
});

test('every panel script compiles with the vendored Babel', () => {
  for (const file of sourceFiles().filter(name => name.endsWith('.jsx'))) {
    assert.doesNotThrow(() => Babel.transform(fs.readFileSync(file, 'utf8'), { presets: ['react'], filename: file }), file);
  }
});

test('no visible button in the panel is left without a handler', () => {
  const findings = [];
  for (const file of sourceFiles().filter(name => name.endsWith('.jsx'))) {
    Babel.transform(fs.readFileSync(file, 'utf8'), {
      presets: ['react'], filename: file,
      plugins: [() => ({ visitor: { JSXOpeningElement(node) {
        if (node.node.name.type !== 'JSXIdentifier' || node.node.name.name !== 'button') return;
        const attrs = new Map(node.node.attributes.filter(a => a.type === 'JSXAttribute').map(a => [a.name.name, a.value]));
        const submit = attrs.get('type')?.value === 'submit';
        const alwaysDisabled = attrs.has('disabled') && attrs.get('disabled') === null;
        if (!attrs.has('onClick') && !submit && !alwaysDisabled) {
          findings.push(`${path.relative(OBS, file)}:${node.node.loc.start.line}`);
        }
      } } })],
    });
  }
  for (const file of sourceFiles().filter(name => name.endsWith('.js'))) {
    const text = fs.readFileSync(file, 'utf8');
    for (const match of text.matchAll(/createElement\('button'\)/g)) {
      const line = text.slice(0, match.index).split('\n').length;
      const window = text.slice(match.index, match.index + 600);
      if (!/(onclick|addEventListener\('click'|\.type\s*=\s*'submit'|disabled\s*=\s*true)/.test(window)) {
        findings.push(`${path.relative(OBS, file)}:${line}`);
      }
    }
  }
  assert.deepEqual(findings, []);
});

test('no inputs without a handler and no decorative search remain', () => {
  for (const file of sourceFiles().filter(name => name.endsWith('.jsx'))) {
    Babel.transform(fs.readFileSync(file, 'utf8'), {
      presets: ['react'], filename: file,
      plugins: [() => ({ visitor: { JSXOpeningElement(node) {
        const name = node.node.name.name;
        if (!['input', 'select', 'textarea'].includes(name)) return;
        const attrs = new Set(node.node.attributes.filter(a => a.type === 'JSXAttribute').map(a => a.name.name));
        assert.ok(attrs.has('onChange') || attrs.has('disabled') || attrs.has('readOnly'), `${file}:${node.node.loc.start.line}`);
      } } })],
    });
  }
});

test('no example data or wireframe tooling remains in operational mode', () => {
  assert.equal(fs.existsSync(path.join(OBS, 'src/mocks.jsx')), false);
  assert.equal(fs.existsSync(path.join(OBS, 'tweaks-panel.jsx')), false);
  const forbidden = [
    'window.MOCKS', 'MOCKS', 'mocks.jsx', 'tweaks-panel', 'EDITMODE', 'TweaksPanel', 'wireframe',
    '192.168.1.2', '21d 08h', 'kza@a3f9c2e', 'v0.4', '14:32:31', '14:35:02', 'turn_88', 'u_marco',
    'Marco', 'Lucía', 'vllm-7b', 'ik-30b', 'Khruangbin', 'Threadripper', '2x RTX 3070',
    'LATENCY_FAST', 'LATENCY_BREAKDOWN', 'WAKE_EVENTS_24H', 'COOLDOWN_HISTORY_7D', 'LOG_LINES', 'PIPELINE_STAGES',
    'i % 7', 'cooldown · ejemplo', 'sistema · ejemplo', 'datos de ejemplo del wireframe',
  ];
  const hits = [];
  for (const file of sourceFiles()) {
    const text = fs.readFileSync(file, 'utf8');
    for (const token of forbidden) if (text.includes(token)) hits.push(`${path.relative(OBS, file)}: ${token}`);
  }
  assert.deepEqual(hits, []);
});

test('the panel loads the observation module before the views and no example scripts', () => {
  const html = fs.readFileSync(path.join(OBS, 'index.html'), 'utf8');
  const scripts = [...html.matchAll(/<script[^>]*src="([^"]+)"/g)].map(match => match[1]);
  assert.ok(scripts.includes('src/obs-data.js'));
  assert.ok(scripts.indexOf('src/obs-data.js') < scripts.indexOf('src/primitives.jsx'));
  for (const file of scripts.filter(name => !name.startsWith('vendor/'))) assert.ok(fs.existsSync(path.join(OBS, file)), file);
  assert.match(html, /<title>KZA — Panel técnico<\/title>/);
});

test('panel views use only existing authenticated endpoints for writes', () => {
  const routes = new Set();
  for (const [kind] of Object.entries(data.OPERATIONS)) {
    const op = data.OPERATIONS[kind]('x');
    routes.add(`${op.method} ${op.path.replace(/\/x(\/|$)/, '/{id}$1')}`);
  }
  assert.deepEqual([...routes].sort(), [
    'DELETE /api/admin/alerts/{id}',
    'POST /api/admin/alerts/{id}/ack',
    'POST /api/admin/services/{id}/restart',
    'POST /api/llm/endpoints/{id}/clear-cooldown',
  ]);
  const admin = fs.readFileSync(path.resolve(__dirname, '../../src/dashboard/admin.py'), 'utf8');
  const observability = fs.readFileSync(path.resolve(__dirname, '../../src/dashboard/observability.py'), 'utf8');
  assert.match(admin, /@router\.post\("\/alerts\/\{alert_id\}\/ack", dependencies=\[Depends\(require_admin\)\]\)/);
  assert.match(admin, /@router\.delete\("\/alerts\/\{alert_id\}", dependencies=\[Depends\(require_admin\)\]\)/);
  assert.match(admin, /@router\.post\("\/services\/\{name\}\/restart", dependencies=\[Depends\(require_admin\)\]\)/);
  assert.match(observability, /@router\.post\("\/api\/llm\/endpoints\/\{endpoint_id\}\/clear-cooldown"\)/);
});
