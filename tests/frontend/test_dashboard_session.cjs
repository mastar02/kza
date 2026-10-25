const assert = require('node:assert/strict');
const { test } = require('node:test');
const { createSession, SessionRequired } = require('../../src/dashboard/frontend/obs/src/session.js');

function fixture(options = {}) {
  const states = [], calls = { hydrate: 0, connect: 0, dispose: 0, clear: 0 };
  const session = createSession({
    fetch: async () => ({ ok: true, status: 200 }),
    hydrate: async () => { calls.hydrate++; },
    connect: () => { calls.connect++; return () => { calls.dispose++; }; },
    clear: () => { calls.clear++; },
    onState: state => states.push(state), ...options,
  });
  return { session, states, calls };
}

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
