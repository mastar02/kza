const assert = require('node:assert/strict');
const { test } = require('node:test');
const { createSession, SessionRequired, loginAccess, liveCloseStatus } =
  require('../../src/dashboard/frontend/obs/src/session.js');

function fixture(options = {}) {
  const states = [], failures = [], calls = { hydrate: 0, connect: 0, dispose: 0, clear: 0 };
  const session = createSession({
    fetch: async () => ({ ok: true, status: 200 }),
    hydrate: async () => { calls.hydrate++; },
    connect: () => { calls.connect++; return () => { calls.dispose++; }; },
    clear: () => { calls.clear++; },
    onState: (state, failure) => { states.push(state); failures.push(failure); }, ...options,
  });
  return { session, states, failures, calls };
}

const whoami = '/api/admin/auth/whoami';
const cases = [
  ['401', 401, 'login', 'login', 'La credencial es inválida o la sesión venció.'],
  ['403', 403, 'forbidden', 'forbidden', 'No tenés permiso para este acceso.'],
  ['503', 503, 'unavailable', 'unavailable', 'La autoridad de acceso no está disponible.'],
  ['network', new TypeError(), 'network', 'unavailable', 'No se pudo conectar con el servidor.'],
  ['timeout', new DOMException('', 'TimeoutError'), 'timeout', 'unavailable', 'El servidor no respondió a tiempo.'],
  ['abort', new DOMException('', 'AbortError'), 'aborted', 'unavailable', 'Se canceló la solicitud.'],
];

function failedFetch(expectedPath, failure) {
  return async (path, options) => {
    assert.equal(path, expectedPath);
    assert.equal(options.credentials, 'same-origin');
    assert.equal(options.cache, 'no-store');
    assert.ok(options.signal instanceof AbortSignal);
    if (typeof failure !== 'number') throw failure;
    return { ok: false, status: failure };
  };
}

for (const [name, failure, kind, state, message] of cases) {
  test(`WEB-07 login classifies ${name}`, async () => {
    const path = '/api/admin/auth/login';
    const fail = failedFetch(path, failure);
    await assert.rejects(loginAccess((url, options) => {
      assert.equal(options.method, 'POST');
      assert.deepEqual(JSON.parse(options.body), { token: 'synthetic' });
      return fail(url, options);
    }, 'synthetic'), { kind, message });
  });

  test(`WEB-07 whoami classifies ${name} before hydration`, async () => {
    const { session, states, failures, calls } = fixture({ fetch: failedFetch(whoami, failure) });
    await session.start();
    assert.deepEqual(states, ['checking', state]);
    assert.equal(failures.at(-1).kind, kind);
    assert.equal(failures.at(-1).message, message);
    assert.equal(calls.hydrate, 0);
    assert.equal(calls.connect, 0);
  });

  test(`WEB-07 live close rechecks ${name} even after a policy close`, async () => {
    const result = await liveCloseStatus(failedFetch(whoami, failure), 1008);
    const active = fixture();
    await active.session.start();
    active.session.fail(result.failure);
    assert.equal(active.states.at(-1), state);
    assert.equal(active.failures.at(-1).kind, kind);
    assert.equal(active.failures.at(-1).message, message);
    assert.equal(active.calls.dispose, 1);
    assert.equal(active.calls.clear, 2);
  });
}

test('a socket close with valid authority distinguishes policy, busy service and transport', async () => {
  const fetch = async () => ({ ok: true, status: 200 });
  assert.deepEqual(await liveCloseStatus(fetch, 1008),
    { kind: 'forbidden', message: 'No tenés permiso para este acceso.', retry: false });
  assert.deepEqual(await liveCloseStatus(fetch, 1013),
    { kind: 'unavailable', message: 'La autoridad de acceso no está disponible.', retry: true });
  assert.deepEqual(await liveCloseStatus(fetch, 1006),
    { kind: 'disconnected', message: 'desconectado', retry: true });
});

test('a hung login is canceled at the deadline and reported as a timeout', async t => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  let signal;
  const fetch = async (_url, options) => new Promise((_, reject) => {
    signal = options.signal;
    signal.addEventListener('abort', () => reject(signal.reason), { once: true });
  });
  const request = loginAccess(fetch, 'synthetic');
  t.mock.timers.tick(11999);
  assert.equal(signal.aborted, false);
  t.mock.timers.tick(1);
  await assert.rejects(request, { kind: 'timeout', message: 'El servidor no respondió a tiempo.' });
  assert.equal(signal.reason.name, 'TimeoutError');
});

test('canceling the owning login aborts the request without a timeout', async () => {
  const owner = new AbortController();
  const request = loginAccess(async (_url, { signal }) => new Promise((_, reject) => {
    signal.addEventListener('abort', () => reject(signal.reason), { once: true });
  }), 'synthetic', owner.signal);
  owner.abort();
  await assert.rejects(request, { kind: 'aborted', message: 'Se canceló la solicitud.' });
});

test('missing session never hydrates data or opens a socket', async () => {
  const { session, states, calls } = fixture({ fetch: async () => ({ ok: false, status: 401 }) });
  await session.start();
  assert.deepEqual(states, ['checking', 'login']);
  assert.equal(calls.hydrate, 0);
  assert.equal(calls.connect, 0);
});

test('unconfigured service never shows a simulated authenticated dashboard', async () => {
  const { session, states, calls } = fixture({ fetch: async () => ({ ok: false, status: 503 }) });
  await session.start();
  assert.equal(states.at(-1), 'unavailable');
  assert.equal(calls.hydrate, 0);
  assert.equal(calls.connect, 0);
});

test('logout disposes the socket and clears private state', async () => {
  const { session, states, calls } = fixture();
  await session.start();
  assert.deepEqual(states, ['checking', 'loading', 'ready']);
  session.expire();
  assert.equal(states.at(-1), 'login');
  assert.equal(calls.dispose, 1);
  assert.equal(calls.clear, 2);
});

test('a hydration result arriving after logout cannot restore the session', async () => {
  let finish, started, signal;
  const pending = new Promise(resolve => { started = resolve; });
  const { session, states, calls } = fixture({ hydrate: args => {
    signal = args.signal;
    started();
    return new Promise(resolve => { finish = resolve; });
  }});
  const checking = session.start();
  await pending;
  session.expire();
  assert.equal(signal.aborted, true);
  finish();
  await checking;
  assert.equal(states.at(-1), 'login');
  assert.equal(calls.connect, 0);
  assert.equal(states.includes('ready'), false);
});

test('authentication loss while loading is a login state, not mock data', async () => {
  const { session, states, calls } = fixture({ hydrate: async () => { throw new SessionRequired(); } });
  await session.start();
  assert.equal(states.at(-1), 'login');
  assert.equal(calls.connect, 0);
  assert.equal(calls.clear, 2);
});

test('an old credential check cannot override a newer login attempt', async () => {
  let finish;
  let reads = 0;
  const { session, states, calls } = fixture({ fetch: async () => {
    reads++;
    if (reads === 1) return new Promise(resolve => { finish = resolve; });
    return { ok: true, status: 200 };
  }});
  const old = session.start();
  await session.start();
  finish({ ok: false, status: 401 });
  await old;
  assert.equal(states.at(-1), 'ready');
  assert.equal(calls.hydrate, 1);
  assert.equal(calls.connect, 1);
  session.stop();
  assert.equal(calls.dispose, 1);
});
