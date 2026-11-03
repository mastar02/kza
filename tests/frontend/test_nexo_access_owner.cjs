'use strict';

// Synthetic transport tests for the real NEXO access owner.
const assert = require('node:assert/strict');
const { test: nodeTest } = require('node:test');
const { createAccessOwner, parseReceipt, parseCatalog, parsePeople, normalizeGrant } = require('../../src/dashboard/frontend/home/nexo/access-owner.js');
const test = (name, run) => nodeTest(name, { timeout: 5000 }, run);

const BASE = '/api/admin/auth/credentials';
const ATTEMPTS = '/api/admin/auth/credential-attempts/';
const PEOPLE = '/api/admin/users';
const SECRET = 'kza_' + 's'.repeat(43);
const uuid = value => `10000000-0000-4000-8000-${String(value).padStart(12, '0')}`;
const TARGET = uuid(200);
const NOW = Math.floor(Date.now() / 1000);
const deferred = () => {
  let resolve;
  const promise = new Promise(yes => { resolve = yes; });
  return { promise, resolve };
};
const turn = async () => {
  for (let index = 0; index < 8; index++) await new Promise(resolve => setImmediate(resolve));
};

function session() {
  let generation = 1, role = 'operator', permissions = new Set(['auth:manage', 'admin:manage']);
  return {
    generation: () => generation,
    role: () => role,
    active: () => role !== null,
    can: permission => role !== null && permissions.has(permission),
    accept(nextRole = 'operator', nextPermissions = ['auth:manage', 'admin:manage']) {
      generation++; role = nextRole; permissions = new Set(nextPermissions);
    },
    clear() { generation++; role = null; permissions.clear(); },
  };
}

function row(id = TARGET, changes = {}) {
  return { id, subject: 'Synthetic access', role: 'home', user_id: null,
    issued_at: NOW, expires_at: NOW + 3600, revision: 3, active: true,
    pending: false, samples: [], ...changes };
}
function page(rows = [row()], next = null) {
  return { rows, observed_at: NOW, next_cursor: next };
}
function issueSpec(changes = {}) {
  return { subject: 'Synthetic created access', role: 'home', expires_at: NOW + 3600, ...changes };
}
function receipt(requestId, action = 'issue', changes = {}) {
  return { request_id: requestId, action,
    credential_id: action === 'issue' ? requestId : TARGET,
    expected_revision: action === 'issue' ? null : 3,
    outcome: 'committed', result_revision: action === 'issue' ? 1 : action === 'revise' ? 4 : 3,
    reason: 'applied', completed_at: NOW + 1, ...changes };
}
function response(document, status = 200, hooks = {}) {
  return { ok: status >= 200 && status < 300, status,
    body: { cancel: hooks.cancel || (async () => {}) },
    json: hooks.json || (async () => document),
    arrayBuffer: async () => new ArrayBuffer(0),
  };
}

test('revision receipts must confirm exactly one CAS increment', () => {
  assert.equal(parseReceipt(receipt(uuid(1), 'revise')).result_revision, 4);
  for (const result_revision of [2, 3, 5, 2147483647]) {
    assert.throws(() => parseReceipt(receipt(uuid(1), 'revise', { result_revision })));
  }
});

test('catalogue names and credential subjects count Unicode codepoints', () => {
  const name = '😀'.repeat(128);
  assert.doesNotThrow(() => normalizeGrant(issueSpec({ subject: name })));
  assert.doesNotThrow(() => parseCatalog(page([row(TARGET, { subject: name })])));
  assert.doesNotThrow(() => parsePeople([{ id: 'person', name, permission_level: 'guest', is_active: true } ]));
  for (const subject of [name + 'a', '\ud800', '\udfff']) assert.throws(() => normalizeGrant(issueSpec({ subject })));
});

function fixture(t, options = {}) {
  const identity = options.session || session();
  const requests = [], snapshots = [], cleanup = [], ids = [];
  let count = 0, unauthorized = 0, checks = 0;
  const fetch = async (path, init = {}) => {
    requests.push({ path, init });
    return options.fetch ? options.fetch(path, init) : response(page());
  };
  const owner = createAccessOwner({ session: identity, fetch,
    newRequestId: () => { const value = uuid(++count); ids.push(value); return value; },
    verifyIdentity: async () => { checks++; return options.verifyIdentity ? options.verifyIdentity() : true; },
    onUnauthorized: () => { unauthorized++; options.onUnauthorized?.(); },
  });
  const unsubscribe = owner.subscribe(value => snapshots.push(value));
  t.after(async () => {
    for (const release of cleanup) release();
    await owner.clear();
    unsubscribe?.();
  });
  return { owner, session: identity, requests, snapshots, cleanup, ids,
    state: () => snapshots.at(-1), unauthorized: () => unauthorized, checks: () => checks };
}

function noSecret(g, ...results) {
  assert.ok(!JSON.stringify(g.snapshots).includes(SECRET), 'safe subscriptions retained the secret');
  for (const result of results) {
    assert.equal(typeof result, 'boolean');
    assert.ok(!JSON.stringify(result).includes(SECRET));
  }
}

test('home/evaluator and an operator without auth:manage cannot admit access requests', async t => {
  const g = fixture(t);
  for (const [role, permissions] of [
    ['home', ['home:read']], ['evaluator', ['evaluation:read']], ['operator', ['admin:manage']],
    ['home', ['auth:manage', 'admin:manage']], ['evaluator', ['auth:manage', 'admin:manage']],
  ]) {
    g.session.accept(role, permissions);
    assert.equal(await g.owner.open(), false);
    assert.equal(await g.owner.refresh(), false);
    assert.equal(await g.owner.issue(issueSpec(), { acceptSecret() { assert.fail('secret without authority'); } }), false);
    assert.equal(await g.owner.reconcile(uuid(1)), false);
  }
  assert.deepEqual(g.requests, []);
});

test('catalogue, people and attempt requests stay on closed same-origin routes', async t => {
  const g = fixture(t, { fetch: async path => response(path === PEOPLE ? [] : page()) });
  assert.equal(await g.owner.open(), true);
  const before = g.requests.length;
  for (const bad of ['', '\u0000', 'x'.repeat(129)]) {
    assert.equal(await g.owner.revise(bad, 3, { role: 'home', expires_at: NOW + 3600 }), false);
    assert.equal(await g.owner.revoke(bad, 3), false);
    assert.equal(await g.owner.reconcile(bad), false);
  }
  assert.equal(g.requests.length, before);
  for (const bad of ['../../users', 'https://elsewhere.invalid/token', TARGET + '?role=operator']) {
    assert.equal(await g.owner.reconcile(bad), false);
  }
  assert.equal(await g.owner.loadPeople(), true);
  for (const { path, init } of g.requests) {
    const parsed = new URL(path, 'http://synthetic.invalid');
    assert.equal(parsed.origin, 'http://synthetic.invalid');
    assert.ok([BASE, PEOPLE].includes(parsed.pathname) || parsed.pathname.startsWith(ATTEMPTS));
    assert.equal(init.credentials, 'same-origin');
    assert.equal(init.cache, 'no-store');
  }
});

test('an arbitrary canonical target is encoded as one credential path segment', async t => {
  const target = '../../users?role=operator#fragment';
  const g = fixture(t, { fetch: async (path, init) => {
    if (init.method === 'DELETE') {
      const command = JSON.parse(init.body);
      return response({ receipt: receipt(command.request_id, 'revoke', { credential_id: target,
        outcome: 'no_effect', result_revision: null, reason: 'target_missing' }), applied: false });
    }
    return response(page());
  } });
  await g.owner.open();
  assert.equal(await g.owner.revoke(target, 3), true);
  const sent = g.requests.find(request => request.init.method === 'DELETE');
  assert.equal(sent.path, BASE + '/' + encodeURIComponent(target));
  const parsed = new URL(sent.path, 'http://synthetic.invalid');
  assert.equal(parsed.origin, 'http://synthetic.invalid');
  assert.equal(parsed.search, ''); assert.equal(parsed.hash, '');
});

test('duplicate click admits one frozen issue and returns the secret only to its current synchronous consumer', async t => {
  const entered = deferred(), body = deferred();
  let original;
  const g = fixture(t, { fetch: async (path, init) => {
    if ((init.method || 'GET') === 'POST') {
      original = JSON.parse(init.body); entered.resolve();
      return response(null, 200, { json: () => body.promise });
    }
    return response(page());
  } });
  g.cleanup.push(() => body.resolve({ receipt: receipt(original?.request_id || uuid(1)), applied: true, token: SECRET }));
  await g.owner.open();
  const spec = issueSpec(), secrets = [];
  const first = g.owner.issue(spec, { acceptSecret: value => secrets.push(value) });
  await entered.promise;
  spec.subject = 'Caller changed subject'; spec.role = 'operator';
  const duplicate = await g.owner.issue(issueSpec(), { acceptSecret() { assert.fail('duplicate consumer'); } });
  assert.equal(duplicate, false);
  assert.equal(g.requests.filter(request => request.init.method === 'POST').length, 1);
  assert.equal(g.ids.length, 1);
  assert.equal(original.subject, 'Synthetic created access');
  assert.equal(original.role, 'home');
  body.resolve({ receipt: receipt(original.request_id), applied: true, token: SECRET });
  const accepted = await first;
  assert.equal(accepted, true); assert.deepEqual(secrets, [SECRET]);
  noSecret(g, accepted, duplicate);
});

test('revise and revoke capture the selected target and CAS revision before asynchronous work', async t => {
  const commands = [];
  const g = fixture(t, { fetch: async (path, init) => {
    if (init.method === 'PUT' || init.method === 'DELETE') {
      const sent = JSON.parse(init.body); commands.push({ path, method: init.method, sent });
      return response({ receipt: receipt(sent.request_id, init.method === 'PUT' ? 'revise' : 'revoke'), applied: true });
    }
    return response(page());
  } });
  await g.owner.open();
  const spec = { role: 'home', expires_at: NOW + 3600, user_id: null, selections: [] };
  const revising = g.owner.revise(TARGET, 3, spec);
  spec.role = 'operator'; spec.selections.push({ fingerprint_id: 'vf_' + 'a'.repeat(64), modalities: ['audio_preview'], expires_at: NOW + 10 });
  assert.equal(await revising, true);
  assert.equal(await g.owner.revoke(TARGET, 3), true);
  assert.deepEqual(commands.map(command => command.path), [BASE + '/' + TARGET, BASE + '/' + TARGET]);
  assert.equal(commands[0].sent.role, 'home');
  assert.deepEqual(commands[0].sent.selections, []);
  assert.equal(commands[0].sent.expected_revision, 3);
  assert.equal(commands[1].sent.expected_revision, 3);
  assert.notEqual(commands[0].sent.request_id, commands[1].sent.request_id);
  noSecret(g);
});

test('an unknown issue reconciles through its original attempt ID without reposting or recovering a secret', async t => {
  let command, terminal = false;
  const g = fixture(t, { fetch: async (path, init) => {
    if (init.method === 'POST') {
      command = JSON.parse(init.body);
      return response(null, 200, { json: async () => { throw Error('Synthetic truncated body'); } });
    }
    if (path === ATTEMPTS + command?.request_id) return response(terminal
      ? { state: 'terminal', receipt: receipt(command.request_id) } : { state: 'pending', receipt: null });
    return response(page());
  } });
  await g.owner.open();
  let secrets = 0;
  assert.equal(await g.owner.issue(issueSpec(), { acceptSecret() { secrets++; } }), false);
  assert.ok(JSON.stringify(g.state()).includes(command.request_id));
  assert.equal(g.state().pending[0].state, 'pending');
  assert.equal(await g.owner.issue(issueSpec(), { acceptSecret() { secrets++; } }), false);
  await g.owner.leave(); await g.owner.open();
  assert.equal(g.requests.filter(request => request.init.method === 'POST').length, 1);
  assert.equal(g.state().pending[0].request_id, command.request_id);
  terminal = true;
  assert.equal(await g.owner.reconcile(command.request_id), true);
  assert.equal(g.requests.filter(request => request.init.method === 'POST').length, 1);
  assert.ok(g.requests.some(request => request.path === ATTEMPTS + command.request_id && (request.init.method || 'GET') === 'GET'));
  assert.equal(secrets, 0); noSecret(g);
});

test('logout and login B wait for an old decoder and cannot receive its issue secret', async t => {
  const entered = deferred(), body = deferred(); let command, secrets = 0;
  const g = fixture(t, { fetch: async (path, init) => {
    if (init.method === 'POST') {
      command = JSON.parse(init.body);
      return response(null, 200, { json: () => { entered.resolve(); return body.promise; } });
    }
    return response(page());
  } });
  g.cleanup.push(() => body.resolve({ receipt: receipt(command?.request_id || uuid(1)), applied: true, token: SECRET }));
  await g.owner.open();
  const pending = g.owner.issue(issueSpec(), { acceptSecret() { secrets++; } });
  await entered.promise;
  let closed = false;
  const leaving = g.owner.leave().then(value => { closed = true; return value; });
  g.session.clear(); g.session.accept();
  const before = g.requests.length, opening = g.owner.open();
  await turn(); assert.equal(closed, false); assert.equal(g.requests.length, before);
  body.resolve({ receipt: receipt(command.request_id), applied: true, token: SECRET });
  assert.equal(await pending, false); assert.equal(await leaving, true); assert.equal(await opening, true);
  assert.equal(secrets, 0); assert.deepEqual(g.state().pending, []); noSecret(g);
});

test('opening login B without an explicit leave still drains the predecessor decoder before B fetches', async t => {
  const entered = deferred(), body = deferred(); let commands = 0;
  const g = fixture(t, { fetch: async (path, init) => {
    if (init.method === 'POST') {
      commands++;
      const command = JSON.parse(init.body);
      return response(null, 200, { json: () => { entered.resolve(); return body.promise.then(() => ({
        receipt: receipt(command.request_id), applied: true, token: SECRET })); } });
    }
    return response(page());
  } });
  g.cleanup.push(() => body.resolve());
  await g.owner.open();
  let secrets = 0;
  const issuing = g.owner.issue(issueSpec(), { acceptSecret() { secrets++; } });
  await entered.promise;
  g.session.accept();
  const before = g.requests.length, opening = g.owner.open();
  await turn();
  assert.equal(g.requests.length, before, 'login B admitted a fetch before A actually released its decoder');
  body.resolve();
  assert.equal(await issuing, false); assert.equal(await opening, true);
  assert.equal(commands, 1); assert.equal(secrets, 0); assert.deepEqual(g.state().pending, []);
  noSecret(g);
});

test('a target-specific 403 verifies the original identity and preserves a valid operator session', async t => {
  const g = fixture(t, { fetch: async (path, init) => init.method === 'DELETE'
    ? response({ detail: 'credential_request_unavailable' }, 403) : response(page()) });
  await g.owner.open();
  assert.equal(await g.owner.revoke(TARGET, 3), false);
  await g.owner.settled();
  assert.equal(g.checks(), 1); assert.equal(g.unauthorized(), 0); assert.equal(g.session.active(), true);
  assert.equal(g.requests.filter(request => request.init.method === 'DELETE').length, 1);
});

test('failed identity verification retires only the still-current login', async t => {
  const check = deferred(), entered = deferred();
  const g = fixture(t, { fetch: async () => response({ detail: 'dashboard_session_retired' }, 403),
    verifyIdentity: () => { entered.resolve(); return check.promise; } });
  g.cleanup.push(() => check.resolve(false));
  const opening = g.owner.open(); await entered.promise;
  g.session.accept();
  check.resolve(false);
  assert.equal(await opening, false); await g.owner.settled();
  assert.equal(g.unauthorized(), 0, 'a result from login A retired login B');
});

test('failed identity verification for the current login invokes retirement once', async t => {
  const g = fixture(t, { fetch: async () => response({ detail: 'dashboard_session_retired' }, 403), verifyIdentity: async () => false });
  assert.equal(typeof await g.owner.open(), 'boolean'); await g.owner.settled();
  assert.equal(g.checks(), 1); assert.equal(g.unauthorized(), 1);
});

test('people choices preserve canonical IDs for homonyms and expose only the promised metadata', async t => {
  const people = [
    { id: 'canonical-person-a', name: 'Same name', permission_level: 'adult', is_active: true },
    { id: 'canonical-person-b', name: 'Same name', permission_level: 'child', is_active: false },
  ];
  const g = fixture(t, { fetch: async path => response(path === PEOPLE ? people : page()) });
  await g.owner.open(); assert.equal(await g.owner.loadPeople(), true);
  assert.deepEqual(g.state().people, people.map(({ id, name, is_active }) => ({ id, name, is_active })));
  people[0].name = 'Changed caller value';
  assert.equal(g.state().people[0].name, 'Same name');
  assert.deepEqual(Object.keys(g.state().people[0]).sort(), ['id', 'is_active', 'name']);
  g.session.accept('operator', ['auth:manage']);
  await g.owner.open(); const before = g.requests.length;
  assert.equal(await g.owner.loadPeople(), false); assert.equal(g.requests.length, before);
});

test('pagination uses server cursors and previous returns to the original page', async t => {
  const NEXT = uuid(300);
  const g = fixture(t, { fetch: async path => {
    const cursor = new URL(path, 'http://synthetic.invalid').searchParams.get('after');
    return response(cursor === TARGET ? page([row(NEXT)], null) : page([row()], TARGET));
  } });
  await g.owner.open(); assert.deepEqual(g.state().rows.map(value => value.id), [TARGET]);
  assert.equal(await g.owner.next(), true); assert.deepEqual(g.state().rows.map(value => value.id), [NEXT]);
  assert.equal(await g.owner.previous(), true); assert.deepEqual(g.state().rows.map(value => value.id), [TARGET]);
  assert.equal(g.requests.filter(request => new URL(request.path, 'http://synthetic.invalid').searchParams.get('after') === TARGET).length, 1);
});

test('late headers keep cancellation owned until the real body cancellation settles', async t => {
  const headers = deferred(), entered = deferred(), cancel = deferred(); let decodes = 0;
  const g = fixture(t, { fetch: async () => { entered.resolve(); return headers.promise; } });
  g.cleanup.push(() => { headers.resolve(response(page(), 200, { cancel: () => cancel.promise, json: async () => { decodes++; return page(); } })); cancel.resolve(); });
  const opening = g.owner.open(); await entered.promise;
  let closed = false;
  const closing = g.owner.leave().then(value => { closed = true; return value; });
  headers.resolve(response(page(), 200, { cancel: () => cancel.promise, json: async () => { decodes++; return page(); } }));
  await turn(); assert.equal(closed, false); assert.equal(decodes, 0);
  cancel.resolve(); assert.equal(await opening, false); assert.equal(await closing, true);
  assert.equal(await g.owner.settled(), true);
});

test('a malformed catalogue cannot copy a secret or claim known rows', async t => {
  const g = fixture(t, { fetch: async () => response(page([row(TARGET, { token: SECRET })])) });
  await g.owner.open();
  assert.equal(g.state().known, false); assert.deepEqual(g.state().rows, []);
  assert.equal(g.state().canMutate, false); noSecret(g);
});

test('clear synchronously removes people, attempts and visible rows before its actual drain finishes', async t => {
  const entered = deferred(), body = deferred(); let command;
  const g = fixture(t, { fetch: async (path, init) => {
    if (path === PEOPLE) return response([{ id: 'canonical-person-a', name: 'Synthetic person', permission_level: 'adult', is_active: true }]);
    if (init.method === 'POST') {
      command = JSON.parse(init.body);
      return response(null, 200, { json: () => { entered.resolve(); return body.promise; } });
    }
    return response(page());
  } });
  g.cleanup.push(() => body.resolve({ receipt: receipt(command?.request_id || uuid(1)), applied: true, token: SECRET }));
  await g.owner.open(); await g.owner.loadPeople();
  let secrets = 0;
  const issuing = g.owner.issue(issueSpec(), { acceptSecret() { secrets++; } });
  await entered.promise;
  const closing = g.owner.clear();
  assert.equal(g.state().visible, false);
  assert.deepEqual(g.state().rows, []); assert.deepEqual(g.state().people, []); assert.deepEqual(g.state().pending, []);
  let drained = false; closing.then(() => { drained = true; });
  await turn(); assert.equal(drained, false);
  body.resolve({ receipt: receipt(command.request_id), applied: true, token: SECRET });
  assert.equal(await issuing, false); assert.equal(await closing, true); assert.equal(secrets, 0);
  noSecret(g);
});

test('a throwing one-time secret consumer cannot leak into subscriptions or authorize another delivery', async t => {
  let command, deliveries = 0;
  const g = fixture(t, { fetch: async (path, init) => {
    if (init.method === 'POST') {
      command = JSON.parse(init.body);
      return response({ receipt: receipt(command.request_id), applied: true, token: SECRET });
    }
    if (path === ATTEMPTS + command?.request_id) return response({ state: 'terminal', receipt: receipt(command.request_id) });
    return response(page());
  } });
  await g.owner.open();
  const result = await g.owner.issue(issueSpec(), { acceptSecret(value) { deliveries++; assert.equal(value, SECRET); throw Error(SECRET); } });
  assert.equal(deliveries, 1); noSecret(g, result);
  await g.owner.reconcile(command.request_id);
  assert.equal(deliveries, 1); assert.equal(g.requests.filter(request => request.init.method === 'POST').length, 1);
  noSecret(g);
});
