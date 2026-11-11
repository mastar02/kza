const assert = require('node:assert/strict');
const { test } = require('node:test');
const memory = require('../../src/dashboard/frontend/home/nexo/memory-owner.js');

const { createMemoryOwner, parseSnapshot, classifyWrite, writeDisposition, resolveMarker, observe, dependents, PREFIX, ENDED_KEY, CAP } = memory;
const ID = char => 'fact_' + char.repeat(64);
const hex = n => 'fact_' + n.toString(16).padStart(64, '0');
const deferred = () => { let resolve, reject; const promise = new Promise((yes, no) => { resolve = yes; reject = no; }); return { promise, resolve, reject }; };
const tick = () => new Promise(resolve => setImmediate(resolve));
async function settle() { for (let index = 0; index < 12; index++) await tick(); }

function source(depends = []) {
  return { capture_id: 'c', conversation_id: 'v', captured_at: 90, expires_at: 1000, origin: 'voice',
    ...(depends.length ? { fact_dependencies: depends.map(id => ({ fact_id: id, revision: 1, expires_at: null })) } : {}) };
}
function fact(id, overrides = {}) {
  const state = overrides.state || 'automatic';
  return { id, content: 'Dato sintético ' + id.slice(5, 9), category: 'fact', confidence: 1, state, created_at: 90, extracted_at: 90,
    expires_at: state === 'confirmed' ? null : 500, revision: 1, confirmed_at: state === 'confirmed' ? 95 : null,
    confirmed_by: state === 'confirmed' ? 'synthetic-a' : null, sources: [source()], owner: null, owner_revision: 0, owner_change: null, ...overrides };
}
const proposal = (id, overrides = {}) => fact(id, { state: 'proposed', ...overrides });
const snap = (facts = [], proposals = [], extra = {}) => ({ observed_at: 100, facts, proposals, legacy_policy: 'quarantined', pending_index: 0, pending_index_cleanup: 0,
  facts_next: null, proposals_next: null, limit: 100, assign_notice: 'Aviso sintético de asignación.', ...extra });
const filler = (count, offset = 0) => Array.from({ length: count }, (_, index) => fact(hex(offset + index + 1)));

function storageMock(fail = false) {
  const data = new Map();
  return { data,
    get length() { return data.size; }, key: index => [...data.keys()][index] ?? null,
    getItem: key => data.has(key) ? data.get(key) : null,
    setItem: (key, value) => { if (fail) throw Error('quota'); data.set(key, String(value)); },
    removeItem: key => { data.delete(key); } };
}
function sessionMock(subject = 'synthetic-a', permissions = ['memory:read', 'memory:review']) {
  return { subject: () => subject, role: () => 'operator', can: permission => permissions.includes(permission) };
}
const whoami = (subject = 'synthetic-a', role = 'operator', permissions = ['memory:read', 'memory:review']) => ({ ok: true, subject, role, permissions });

/** A scripted server: each handler gets (path, options) and returns or throws. */
function fixture({ storage = storageMock(), session = sessionMock(), identity = () => whoami(), onAccountChanged = () => {}, maxPages, extra = {} } = {}) {
  const calls = [];
  let reads = [], writes = [];
  let time = 0;
  const api = (path, options = {}) => {
    const method = options.method || 'GET';
    calls.push(method + ' ' + path);
    if (path === '/api/admin/auth/whoami') return Promise.resolve().then(identity);
    if (method === 'GET') {
      for (const [prefix, handler] of Object.entries(extra)) {
        if (path === prefix || path.startsWith(prefix + '?')) return Promise.resolve().then(() => handler(path));
      }
    }
    if (path === '/api/memory' || path.startsWith('/api/memory?')) {
      const handler = reads.shift();
      if (!handler) return Promise.reject(Object.assign(Error('no read scripted'), { status: 599 }));
      const result = typeof handler === 'function' ? handler(options) : handler;
      if (options.signal) {
        return new Promise((resolve, reject) => {
          options.signal.addEventListener('abort', () => reject(Object.assign(Error('aborted'), { name: 'AbortError' })));
          Promise.resolve(result).then(resolve, reject);
        });
      }
      return Promise.resolve(result);
    }
    assert.equal(options.signal, undefined, 'writes are never abortable');
    const handler = writes.shift();
    if (!handler) return Promise.reject(Object.assign(Error('no write scripted'), { status: 599 }));
    return Promise.resolve().then(() => typeof handler === 'function' ? handler(path, options) : handler);
  };
  const owner = createMemoryOwner({ api, session, storage, schedule: () => 0, cancel: () => {}, monotonic: () => time, wall: () => time, onAccountChanged, ...(maxPages ? { maxPages } : {}) });
  return { owner, calls, storage, read: (...handlers) => reads.push(...handlers), write: (...handlers) => writes.push(...handlers),
    advance(ms) { time += ms; }, writes: () => calls.filter(call => !call.startsWith('GET')) };
}
const fail = (status, detail, extra = {}) => () => { throw Object.assign(Error('http'), { status, detail, ...extra }); };
const network = () => { throw new TypeError('Failed to fetch'); };

test('open confirms the account before the first read and shows the snapshot', async () => {
  const f = fixture();
  f.read(snap([fact(ID('a'))], [proposal(ID('b'))]));
  await f.owner.open(); await settle();
  assert.deepEqual(f.calls, ['GET /api/admin/auth/whoami', 'GET /api/memory']);
  const state = f.owner.getState();
  assert.deepEqual([state.facts.length, state.proposals.length, state.loaded], [1, 1, true]);
});

test('F5: another account on open sends no memory read and calls onAccountChanged once', async () => {
  let changed = 0;
  const f = fixture({ identity: () => whoami('synthetic-b'), onAccountChanged: () => changed++ });
  await f.owner.open(); await settle();
  assert.deepEqual(f.calls, ['GET /api/admin/auth/whoami']);
  assert.equal(changed, 1);
  // Unreachable whoami: nothing is read or sent, markers stay.
  const storage = storageMock(); storage.setItem(PREFIX + ID('c'), JSON.stringify({ op: 'delete', subject: 'synthetic-a', sent_at: 1, state: 'sending' }));
  const g = fixture({ storage, identity: () => { throw new TypeError('offline'); } });
  await g.owner.open(); await settle();
  assert.deepEqual(g.calls, ['GET /api/admin/auth/whoami']);
  assert.ok(storage.getItem(PREFIX + ID('c')));
  assert.match(g.owner.getState().read.error, /No se pudo confirmar la cuenta/);
  // Same subject but another role or without memory:read is also a change.
  for (const identity of [() => whoami('synthetic-a', 'home', []), () => whoami('synthetic-a', 'operator', ['home:read'])]) {
    let count = 0; const h = fixture({ identity, onAccountChanged: () => count++ });
    await h.owner.open(); await settle();
    assert.equal(count, 1); assert.ok(!h.calls.includes('GET /api/memory'));
  }
});

test('read errors are distinct; a status-less failure keeps content only for the same account (F9)', async () => {
  const f = fixture();
  f.read(snap([fact(ID('a'))]), network);
  await f.owner.open(); await settle();
  await f.owner.refresh(); await settle();
  let state = f.owner.getState();
  assert.equal(state.facts.length, 1); assert.equal(state.read.error, 'No pudimos consultar la memoria.');
  for (const [status, detail, text] of [[503, 'memory_unavailable', /no está disponible en este servidor/], [403, 'dashboard_permission_denied', /no permite consultar/], [500, undefined, /no pudo completar/]]) {
    const g = fixture(); g.read(snap([fact(ID('a'))]), fail(status, detail));
    await g.owner.open(); await settle(); await g.owner.refresh(); await settle();
    state = g.owner.getState();
    assert.equal(state.loaded, false, 'content dropped on ' + status); assert.match(state.read.error, text);
  }
  // Unknown account after a status-less failure: content dropped.
  let first = true;
  const h = fixture({ identity: () => { if (first) { first = false; return whoami(); } throw new TypeError('offline'); } });
  h.read(snap([fact(ID('a'))]), network);
  await h.owner.open(); await settle(); await h.owner.refresh(); await settle();
  assert.equal(h.owner.getState().loaded, false);
  // Changed account after a status-less failure: owner wipes and reports the change.
  let changed = 0, firstB = true;
  const k = fixture({ identity: () => { if (firstB) { firstB = false; return whoami(); } return whoami('synthetic-b'); }, onAccountChanged: () => changed++ });
  k.read(snap([fact(ID('a'))]), network);
  await k.owner.open(); await settle(); await k.owner.refresh(); await settle();
  assert.equal(changed, 1); assert.equal(k.owner.getState().loaded, false);
});

test('server-clock expiry hides items and an expired proposal cannot start a write', async () => {
  const f = fixture();
  f.read(snap([], [proposal(ID('b'), { expires_at: 101 })]));
  await f.owner.open(); await settle();
  assert.equal(f.owner.getState().proposals.length, 1);
  f.advance(1500);
  assert.equal(f.owner.getState().proposals.length, 0);
  assert.deepEqual(await f.owner.confirm(ID('b')), { sent: false });
  assert.deepEqual(f.writes(), []);
});

test('confirm 200 moves the card; delete and reject 200 deleted:false say it was already gone', async () => {
  const f = fixture();
  const p = proposal(ID('b')), confirmed = fact(ID('b'), { state: 'confirmed', revision: 2 });
  f.read(snap([fact(ID('a'))], [p, proposal(ID('c'))]), snap([confirmed, fact(ID('a'))], [proposal(ID('c'))]), snap([confirmed, fact(ID('a'))], []), snap([confirmed], []));
  await f.owner.open(); await settle();
  f.write({ fact: confirmed, status: {} });
  await f.owner.confirm(ID('b')); await settle();
  let state = f.owner.getState();
  assert.ok(state.facts.some(item => item.id === ID('b') && item.state === 'confirmed'));
  assert.equal(state.notices[ID('b')], 'Recuerdo conservado sin vencimiento.');
  f.write({ deleted: false, status: {} });
  await f.owner.reject(ID('c')); await settle();
  assert.equal(f.owner.getState().notices[ID('c')], 'Ya no estaba guardada.');
  f.write({ deleted: false, status: {} });
  await f.owner.remove(ID('a'), { confirmed: true }); await settle();
  state = f.owner.getState();
  assert.equal(state.notices[ID('a')], 'Ya no estaba guardado.');
  assert.ok(f.calls.includes('DELETE /api/memory/facts/' + ID('a')));
  assert.deepEqual(Object.keys(state.markers), []);
});

test('delete needs an explicit confirmation and a whoami for the same account; double tap sends once', async () => {
  const f = fixture();
  f.read(snap([fact(ID('a'))]), snap([]));
  await f.owner.open(); await settle();
  assert.deepEqual(await f.owner.remove(ID('a')), { sent: false });
  const hold = deferred();
  f.write(() => hold.promise);
  const first = f.owner.remove(ID('a'), { confirmed: true });
  await settle();
  assert.deepEqual(await f.owner.remove(ID('a'), { confirmed: true }), { sent: false });
  hold.resolve({ deleted: true, status: {} }); await first; await settle();
  assert.equal(f.writes().length, 1);
  const whoamis = f.calls.filter(call => call.endsWith('whoami')).length;
  assert.equal(whoamis, 2, 'open and delete each confirm the account');
});

test('stored markers never carry content and storage or subject failure sends nothing', async () => {
  const f = fixture();
  const p = proposal(ID('b'), { content: 'Contenido privado sintético' });
  const hold = deferred();
  f.read(snap([], [p]));
  await f.owner.open(); await settle();
  f.write(() => hold.promise);
  void f.owner.confirm(ID('b')); await settle();
  const raw = f.storage.getItem(PREFIX + ID('b'));
  assert.deepEqual(Object.keys(JSON.parse(raw)).sort(), ['op', 'sent_at', 'state', 'subject']);
  assert.ok(!raw.includes('privado') && !raw.includes('capture') && !raw.includes('conversation'));
  for (const [storage, session, text] of [[storageMock(true), sessionMock(), /no permite guardar/], [storageMock(), sessionMock(null), /cuenta/]]) {
    const g = fixture({ storage, session, identity: () => whoami(session.subject()) });
    g.read(snap([], [p]));
    await g.owner.open(); await settle();
    // A session without subject is treated as another account on open; force the view open for the write check.
    if (!session.subject()) { assert.equal(g.owner.getState().loaded, false); continue; }
    assert.deepEqual(await g.owner.confirm(ID('b')), { sent: false });
    assert.match(g.owner.getState().notices[ID('b')], text);
    assert.deepEqual(g.writes(), []);
  }
});

test('F2: a lost delete is never reported as not applied by a read; the later commit resolves it', async () => {
  const f = fixture();
  const a = fact(ID('a'));
  const early = deferred();
  f.read(snap([a]));
  await f.owner.open(); await settle();
  // A read that started before the loss resolves after it: it must not decide.
  f.read(() => early.promise);
  void f.owner.refresh(); await settle();
  f.write(network);
  f.read(snap([a]), snap([]));
  await f.owner.remove(ID('a'), { confirmed: true });
  early.resolve(snap([])); await settle();
  let state = f.owner.getState();
  assert.equal(state.markers[ID('a')].state, 'unresolved');
  assert.equal(state.notices[ID('a')], 'Todavía figura guardado; puede que el cambio no se haya aplicado.');
  assert.ok(!Object.values(state.notices).some(text => /no se eliminó/.test(text)));
  // The server thread commits later; the next read shows the effect.
  await f.owner.refresh(); await settle();
  state = f.owner.getState();
  assert.deepEqual(state.markers, {});
  assert.equal(state.notices[ID('a')], 'Comprobado: ya no está guardado.');
});

test('F3: a 404 on confirm is provisional and a read showing the fact confirmed reports applied', async () => {
  const f = fixture();
  const p = proposal(ID('b'));
  f.read(snap([], [p]), snap([fact(ID('b'), { state: 'confirmed' })], []));
  await f.owner.open(); await settle();
  f.write(fail(404, 'memory_proposal_unavailable'));
  await f.owner.confirm(ID('b')); await settle();
  const state = f.owner.getState();
  assert.deepEqual(state.markers, {});
  assert.equal(state.notices[ID('b')], 'Comprobado: el recuerdo quedó conservado sin vencimiento.');
  // A read that agrees makes the 409 final, with the server reason.
  const g = fixture();
  g.read(snap([], [proposal(ID('c'))]), snap([fact(ID('c'), { state: 'confirmed' })], []));
  await g.owner.open(); await settle();
  g.write(fail(409, 'memory_proposal_state_changed'));
  await g.owner.reject(ID('c')); await settle();
  assert.equal(g.owner.getState().notices[ID('c')], 'Otra sesión ya la confirmó; no se descartó.');
  assert.deepEqual(g.owner.getState().markers, {});
});

// U3b-3: a read walks every page; absence is inconclusive only when the walk stopped before the end.
const CUR = n => 'k1.' + n.toString(16).padStart(16, '0') + '.' + hex(n);
const INCOMPLETE_TEXT = 'No pudimos comprobarlo: la lista de recuerdos no se pudo recorrer completa.';

test('F1: an incomplete walk makes absence inconclusive for confirm, reject and delete; resend settles it', async () => {
  const full = filler(CAP);
  for (const [op, start, after] of [
    ['confirm', snap(filler(CAP), [proposal(ID('b'))], { facts_next: CUR(1) }), snap(full, [], { facts_next: CUR(1) })],
    ['reject', snap(filler(CAP), [proposal(ID('b'))], { facts_next: CUR(1) }), snap(full, [], { facts_next: CUR(1) })],
    ['delete', snap([fact(ID('b')), ...filler(CAP - 1)], [], { facts_next: CUR(1) }), snap(full, [], { facts_next: CUR(1) })],
  ]) {
    const f = fixture({ maxPages: 1 });
    f.read(start, after);
    await f.owner.open(); await settle();
    f.write(network);
    await (op === 'delete' ? f.owner.remove(ID('b'), { confirmed: true }) : f.owner[op](ID('b'))); await settle();
    let state = f.owner.getState();
    assert.equal(state.markers[ID('b')]?.state, 'unresolved', op);
    assert.equal(state.notices[ID('b')], INCOMPLETE_TEXT);
    assert.equal(state.factsFull, true); assert.equal(state.factsComplete, false);
    // Reintentar: idempotent resend after the account check; its answer is final.
    f.write(op === 'confirm' ? { fact: fact(ID('b'), { state: 'confirmed' }) } : { deleted: false });
    f.read(snap(full, [], { facts_next: CUR(1) }));
    await f.owner.retry(ID('b')); await settle();
    state = f.owner.getState();
    assert.deepEqual(state.markers, {}, op);
  }
});

test('F1: a proposal confirmed elsewhere that sank beyond the walked part is never reported as rejected', async () => {
  const f = fixture({ maxPages: 1 });
  f.read(snap(filler(CAP - 1), [proposal(ID('b'))]), snap(filler(CAP), [], { facts_next: CUR(1) }));
  await f.owner.open(); await settle();
  f.write(network);
  await f.owner.reject(ID('b')); await settle();
  const state = f.owner.getState();
  assert.equal(state.markers[ID('b')].state, 'unresolved');
  assert.ok(!/descartada|Comprobado/.test(state.notices[ID('b')]));
});

test('F1: a confirmed card stays visible while an incomplete walk omits it, and goes once the walk is complete', async () => {
  const f = fixture({ maxPages: 1 });
  const confirmed = fact(ID('b'), { state: 'confirmed' });
  f.read(snap(filler(CAP), [proposal(ID('b'))], { facts_next: CUR(1) }), snap(filler(CAP), [], { facts_next: CUR(1) }), snap(filler(CAP - 1), []));
  await f.owner.open(); await settle();
  f.write({ fact: confirmed });
  await f.owner.confirm(ID('b')); await settle();
  let state = f.owner.getState();
  assert.deepEqual(state.kept.map(item => item.id), [ID('b')]);
  assert.equal(state.factsFull, true);
  await f.owner.refresh(); await settle();
  state = f.owner.getState();
  assert.deepEqual(state.kept, []);
});

test('the walk follows every cursor: proposals first, then facts from the last proposals page; a moved proposal is listed once', async () => {
  const f = fixture();
  const moved = ID('f');
  const page1 = snap(filler(CAP, 0), [proposal(moved), ...filler(CAP - 1, 500).map(item => ({ ...item, state: 'proposed' }))], { facts_next: CUR(1), proposals_next: CUR(2) });
  // The proposals page 2 call also returns a fresh facts page 1, where the moved proposal is now confirmed.
  const page2 = snap([fact(moved, { state: 'confirmed', revision: 2 }), ...filler(CAP - 1, 0)], filler(3, 900).map(item => ({ ...item, state: 'proposed' })), { facts_next: CUR(3) });
  const page3 = snap(filler(40, 200), [], { facts_next: null });
  f.read(page1, page2, page3);
  await f.owner.open(); await settle();
  assert.equal(f.owner.getState().read.error, null);
  assert.deepEqual(f.calls, ['GET /api/admin/auth/whoami', 'GET /api/memory', 'GET /api/memory?proposals_after=' + CUR(2), 'GET /api/memory?facts_after=' + CUR(3)]);
  const state = f.owner.getState();
  assert.equal(state.facts.length, CAP + 40); assert.equal(state.proposals.length, CAP - 1 + 3);
  assert.ok(state.facts.some(item => item.id === moved && item.state === 'confirmed') && !state.proposals.some(item => item.id === moved));
  assert.equal(state.factsComplete, true); assert.equal(state.proposalsComplete, true); assert.equal(state.factsFull, false);
});

test('absence after a complete multi-page walk resolves a lost delete; a fact beyond page one is still found', async () => {
  const f = fixture();
  const deep = ID('d');
  f.read(snap(filler(CAP), [], { facts_next: CUR(1) }), snap([fact(deep)], []));
  await f.owner.open(); await settle();
  assert.ok(f.owner.getState().facts.some(item => item.id === deep));
  f.write(network);
  f.read(snap(filler(CAP), [], { facts_next: CUR(1) }), snap([fact(ID('e'))], []));
  await f.owner.remove(deep, { confirmed: true }); await settle();
  const state = f.owner.getState();
  assert.deepEqual(state.markers, {});
  assert.equal(state.notices[deep], 'Comprobado: ya no está guardado.');
});

test('a repeated cursor, a duplicate id across pages or an oversized page fails the read closed', async () => {
  for (const pages of [
    [snap(filler(2), [], { facts_next: CUR(1) }), snap(filler(2, 10), [], { facts_next: CUR(1) })],
    [snap(filler(2), [], { facts_next: CUR(1) }), snap(filler(2), [])],
    [snap(filler(CAP + 1))],
    [snap(filler(2), [], { facts_next: 'k1.zz' })],
  ]) {
    const f = fixture();
    f.read(...pages);
    await f.owner.open(); await settle();
    const state = f.owner.getState();
    assert.equal(state.loaded, false);
    assert.equal(state.read.error, 'El servidor devolvió la memoria con un formato desconocido.');
  }
});

test('401/403 on the first send and on a retry are uncertain; answered:true is uncertain', async () => {
  for (const handler of [fail(401), fail(403, 'dashboard_session_retired'), fail(403, 'dashboard_permission_denied'), fail(403, 'x', { answered: true, sent: true })]) {
    const f = fixture();
    f.read(snap([], [proposal(ID('b'))]), snap([], [proposal(ID('b'))]), snap([], [proposal(ID('b'))]));
    await f.owner.open(); await settle();
    f.write(handler);
    await f.owner.confirm(ID('b')); await settle();
    assert.equal(f.owner.getState().markers[ID('b')].state, 'unresolved');
    f.write(fail(403, 'dashboard_session_retired'));
    await f.owner.retry(ID('b')); await settle();
    assert.equal(f.owner.getState().markers[ID('b')].state, 'unresolved');
    assert.equal(f.writes().length, 2);
  }
});

test('retry asks whoami first: another account sends nothing; a foreign marker is never resent', async () => {
  let subject = 'synthetic-a', changed = 0;
  const f = fixture({ identity: () => whoami(subject), onAccountChanged: () => changed++ });
  f.read(snap([], [proposal(ID('b'))]), snap([], [proposal(ID('b'))]));
  await f.owner.open(); await settle();
  f.write(network);
  await f.owner.confirm(ID('b')); await settle();
  subject = 'synthetic-b';
  assert.equal((await f.owner.retry(ID('b'))).accountChanged, true);
  assert.equal(f.writes().length, 1); assert.equal(changed, 1);
  assert.equal(f.storage.getItem(PREFIX + ID('b')), null);
  // Reload into another account: the stored marker is dropped with a reminder, never resent.
  const storage = storageMock(); storage.setItem(PREFIX + ID('c'), JSON.stringify({ op: 'delete', subject: 'synthetic-a', sent_at: 1, state: 'unresolved' }));
  const g = fixture({ storage, session: sessionMock('synthetic-b'), identity: () => whoami('synthetic-b') });
  g.read(snap([fact(ID('c'))]));
  await g.owner.open(); await settle();
  assert.equal(storage.getItem(PREFIX + ID('c')), null);
  assert.match(g.owner.getState().notices.session, /otra cuenta/);
  assert.deepEqual(g.owner.getState().markers, {});
  assert.deepEqual(g.writes(), []);
});

test('reload restores an own marker as unresolved and only a read decides', async () => {
  const storage = storageMock();
  storage.setItem(PREFIX + ID('c'), JSON.stringify({ op: 'delete', subject: 'synthetic-a', sent_at: 1, state: 'sending' }));
  const f = fixture({ storage });
  const hold = deferred();
  f.read(() => hold.promise);
  await f.owner.open(); await settle();
  assert.equal(f.owner.getState().markers[ID('c')].state, 'unresolved');
  hold.resolve(snap([]));
  await settle();
  assert.deepEqual(f.owner.getState().markers, {});
  assert.equal(storage.getItem(PREFIX + ID('c')), null);
});

test('leave never aborts a write; its late answer is applied and shown on the next open', async () => {
  const f = fixture();
  const hold = deferred(), readHold = deferred();
  f.read(snap([fact(ID('a'))]), () => readHold.promise, snap([]));
  await f.owner.open(); await settle();
  f.write(() => hold.promise);
  const pending = f.owner.remove(ID('a'), { confirmed: true }); await settle();
  void f.owner.refresh(); await settle();
  assert.equal(f.owner.leave() instanceof Promise, true);
  assert.equal(f.owner.getState().loaded, false, 'leave drops private content');
  hold.resolve({ deleted: true }); await pending; await settle();
  assert.deepEqual(f.owner.getState().markers, {});
  assert.equal(f.owner.getState().notices[ID('a')], 'Recuerdo eliminado.');
  await f.owner.open(); await settle();
  assert.equal(f.owner.getState().facts.length, 0);
});

test('F11: a session end during a send leaves ENDED, wipes keys, and a late 200 re-persists nothing', async () => {
  const f = fixture();
  const hold = deferred();
  f.read(snap([], [proposal(ID('b'))]));
  await f.owner.open(); await settle();
  f.write(() => hold.promise);
  const pending = f.owner.confirm(ID('b')); await settle();
  assert.equal(JSON.parse(f.storage.getItem(PREFIX + ID('b'))).state, 'sending');
  await f.owner.clear({ wipe: true });
  assert.equal(f.storage.getItem(ENDED_KEY), '1');
  assert.equal(f.storage.getItem(PREFIX + ID('b')), null);
  hold.resolve({ fact: fact(ID('b'), { state: 'confirmed' }) }); await pending; await settle();
  assert.deepEqual([...f.storage.data.keys()], [ENDED_KEY]);
  assert.deepEqual(f.owner.getState().markers, {}, 'a late answer never re-adopts a wiped marker');
  // A late uncertain answer of the ended session re-persists nothing either.
  const k = fixture(); k.read(snap([], [proposal(ID('c'))]));
  await k.owner.open(); await settle();
  const late = deferred(); k.write(() => late.promise);
  const sent = k.owner.confirm(ID('c')); await settle();
  await k.owner.clear({ wipe: true });
  late.reject(new TypeError('lost')); await sent; await settle();
  assert.deepEqual([...k.storage.data.keys()], [ENDED_KEY]);
  assert.deepEqual(k.owner.getState().markers, {});
  // pagehide keeps markers; the next open shows the ENDED reminder once.
  const g = fixture({ storage: f.storage });
  g.read(snap([]));
  await g.owner.open(); await settle();
  assert.match(g.owner.getState().notices.session, /sesión anterior/);
  assert.equal(f.storage.getItem(ENDED_KEY), null);
  const h = fixture(); h.read(snap([], [proposal(ID('d'))]));
  await h.owner.open(); await settle();
  h.write(() => new Promise(() => {}));
  void h.owner.confirm(ID('d')); await settle();
  await h.owner.clear({ wipe: false });
  assert.ok(h.storage.getItem(PREFIX + ID('d')));
  assert.equal(h.storage.getItem(ENDED_KEY), null);
});

test('F6: the visible cascade is listed, blocks deletes while marked, and vanishes on the re-read', async () => {
  const a = fact(ID('a')), b = fact(ID('b'), { sources: [source([ID('a')])] }), c = fact(ID('c'), { sources: [source([ID('b')])] }), d = fact(ID('d'));
  assert.deepEqual(dependents(parseSnapshot(snap([a, b, c, d])).facts, ID('a')).map(item => item.id), [ID('b'), ID('c')]);
  const f = fixture();
  f.read(snap([a, b, c, d]), snap([d]));
  await f.owner.open(); await settle();
  assert.deepEqual(f.owner.dependents(ID('a')).map(item => item.id), [ID('b'), ID('c')]);
  const hold = deferred();
  f.write(() => hold.promise);
  const pending = f.owner.remove(ID('c'), { confirmed: true }); await settle();
  assert.equal(f.owner.deleteBlocked(ID('a')), true);
  assert.deepEqual(await f.owner.remove(ID('a'), { confirmed: true }), { sent: false });
  hold.reject(new TypeError('lost')); await pending; await settle();
  assert.equal(f.owner.getState().facts.length, 1, 'the re-read removed the whole cascade');
  assert.deepEqual(f.owner.getState().markers, {});
});

test('a read started before an applied write cannot restore the deleted card', async () => {
  const f = fixture();
  const a = fact(ID('a'));
  const early = deferred();
  f.read(snap([a]));
  await f.owner.open(); await settle();
  f.read(() => early.promise, snap([]));
  void f.owner.refresh(); await settle();
  f.write({ deleted: true });
  await f.owner.remove(ID('a'), { confirmed: true }); await settle();
  early.resolve(snap([a])); await settle();
  assert.equal(f.owner.getState().facts.length, 0);
});

test('N2: the stale pre-write snapshot is never applied, even when the follow-up read fails', async () => {
  const f = fixture();
  const a = fact(ID('a'));
  const early = deferred();
  f.read(snap([a]));
  await f.owner.open(); await settle();
  // The pre-write read, then the follow-up read is offline (status-less; whoami says same account: content kept).
  f.read(() => early.promise, network);
  void f.owner.refresh(); await settle();
  f.write({ deleted: true });
  await f.owner.remove(ID('a'), { confirmed: true }); await settle();
  assert.equal(f.owner.getState().facts.length, 0);
  early.resolve(snap([a])); await settle();
  const state = f.owner.getState();
  assert.deepEqual(state.facts.map(item => item.id), [], 'a deleted memory must not reappear from a read that predates the delete');
  assert.equal(state.read.error, 'No pudimos consultar la memoria.');
});

test('N3: a whoami timeout is reported, and an earlier visit never writes into a later one', async () => {
  // (a) The 12 s client timeout raises AbortError while the view is open: 'Actualizar' must say so.
  let mode = 'ok';
  const f = fixture({ identity: () => { if (mode === 'timeout') throw Object.assign(Error('timeout'), { name: 'AbortError' }); return whoami(); } });
  f.read(snap([fact(ID('a'))]));
  await f.owner.open(); await settle();
  mode = 'timeout';
  await f.owner.reload(); await settle();
  assert.equal(f.owner.getState().read.error, 'No se pudo confirmar la cuenta de esta sesión; no se consultó ni se envió nada.');
  assert.equal(f.owner.getState().loaded, false);
  // The same timeout on open is reported too.
  const g = fixture({ identity: () => { throw Object.assign(Error('timeout'), { name: 'AbortError' }); } });
  await g.owner.open(); await settle();
  assert.match(g.owner.getState().read.error, /No se pudo confirmar la cuenta/);
  // (b) open -> leave -> open: the first visit's aborted whoami stays silent.
  const first = deferred(), second = deferred(); let calls = 0;
  const h = fixture({ identity: () => (++calls === 1 ? first.promise : second.promise) });
  h.read(snap([fact(ID('a'))]));
  const one = h.owner.open(); await settle();
  h.owner.leave();
  const two = h.owner.open(); await settle();
  first.reject(Object.assign(Error('aborted'), { name: 'AbortError' })); await one; await settle();
  assert.equal(h.owner.getState().read.error, null, 'no transient account error from the earlier visit');
  second.resolve(whoami()); await two; await settle();
  assert.equal(h.owner.getState().facts.length, 1);
  // A late 'changed' from the earlier visit cannot wipe the later one either.
  let changed = 0; const late = deferred(); let n = 0;
  const k = fixture({ identity: () => (++n === 1 ? late.promise : whoami()), onAccountChanged: () => changed++ });
  k.read(snap([fact(ID('a'))]));
  const old = k.owner.open(); await settle();
  k.owner.leave(); await k.owner.open(); await settle();
  late.resolve(whoami('synthetic-b')); await old; await settle();
  assert.equal(changed, 0); assert.equal(k.owner.getState().facts.length, 1);
});

test('N6: a retry whose account cannot be confirmed keeps the provisional reason and its final resolution', async () => {
  // A provisional marker (restored from storage) whose retry cannot confirm the account.
  const storage = storageMock();
  storage.setItem(PREFIX + ID('c'), JSON.stringify({ op: 'reject', subject: 'synthetic-a', sent_at: 1, state: 'provisional', reason: 'Otra sesión ya la confirmó; no se descartó.' }));
  let m = 'ok';
  const g = fixture({ storage, identity: () => { if (m === 'down') throw new TypeError('offline'); return whoami(); } });
  const hold = deferred();
  g.read(() => hold.promise);
  await g.owner.open(); await settle();
  m = 'down';
  await g.owner.retry(ID('c')); await settle();
  assert.equal(g.owner.getState().markers[ID('c')].state, 'provisional');
  assert.equal(JSON.parse(storage.getItem(PREFIX + ID('c'))).reason, 'Otra sesión ya la confirmó; no se descartó.');
  hold.resolve(snap([], [proposal(ID('c'))])); await settle();
  assert.deepEqual(g.owner.getState().markers, {}, 'the agreeing read settles it as final');
  assert.equal(g.owner.getState().notices[ID('c')], 'Otra sesión ya la confirmó; no se descartó.');
});

// Read-ordering guards, table-driven: per op, a read that started before the loss never
// resolves; one that started after does. And an applied write discards older reads.
const ORDERING = [];
for (const op of ['confirm', 'reject', 'delete']) {
  for (const timing of ['before', 'after']) ORDERING.push({ op, timing, guard: 'loss' });
  ORDERING.push({ op, timing: 'before', guard: 'floor' });
}
for (const row of ORDERING) {
  test(`ordering guard ${row.guard} / ${row.op} / read started ${row.timing}`, async () => {
    const f = fixture();
    const target = row.op === 'delete' ? fact(ID('a')) : proposal(ID('a'));
    const start = row.op === 'delete' ? snap([target]) : snap([], [target]);
    const effect = row.op === 'confirm' ? snap([fact(ID('a'), { state: 'confirmed' })]) : snap([]);
    f.read(start);
    await f.owner.open(); await settle();
    const call = () => row.op === 'delete' ? f.owner.remove(ID('a'), { confirmed: true }) : f.owner[row.op](ID('a'));
    if (row.guard === 'loss') {
      if (row.timing === 'before') {
        const early = deferred();
        f.read(() => early.promise, start);
        void f.owner.refresh(); await settle();
        f.write(network);
        await call();
        early.resolve(effect); await settle();
        // The early read showed the effect but started before the loss: still unresolved.
        assert.equal(f.owner.getState().markers[ID('a')]?.state, 'unresolved');
      } else {
        f.write(network);
        f.read(effect);
        await call(); await settle();
        assert.deepEqual(f.owner.getState().markers, {});
        assert.match(f.owner.getState().notices[ID('a')], /^Comprobado/);
      }
    } else {
      const early = deferred(), offline = network;
      f.read(() => early.promise, offline);
      void f.owner.refresh(); await settle();
      f.write(row.op === 'confirm' ? { fact: fact(ID('a'), { state: 'confirmed' }) } : { deleted: true });
      await call(); await settle();
      early.resolve(start); await settle();
      const state = f.owner.getState();
      if (row.op === 'confirm') assert.ok(!state.proposals.some(item => item.id === ID('a')) && state.facts.some(item => item.id === ID('a')));
      else assert.ok(![...state.facts, ...state.proposals].some(item => item.id === ID('a')));
    }
  });
}

test('household or read-only capabilities never write', async () => {
  const f = fixture({ session: sessionMock('synthetic-a', ['memory:read']), identity: () => whoami('synthetic-a', 'operator', ['memory:read']) });
  f.read(snap([fact(ID('a'))], [proposal(ID('b'))]));
  await f.owner.open(); await settle();
  for (const result of [await f.owner.confirm(ID('b')), await f.owner.reject(ID('b')), await f.owner.remove(ID('a'), { confirmed: true })]) assert.deepEqual(result, { sent: false });
  assert.deepEqual(f.writes(), []);
});

test('F15: bidi control characters are kept verbatim as data', () => {
  const text = 'Sintético ‮etnemlaicini‬ fin';
  assert.equal(parseSnapshot(snap([fact(ID('a'), { content: text })])).facts[0].content, text);
});

test('parseSnapshot fails closed on every malformed shape (fuzz)', () => {
  const good = snap([fact(ID('a')), fact(ID('c'), { state: 'confirmed' })], [proposal(ID('b'))]);
  assert.doesNotThrow(() => parseSnapshot(good));
  const mutations = [
    s => null, s => [], s => ({ ...s, observed_at: 'x' }), s => ({ ...s, observed_at: Infinity }), s => ({ ...s, facts: null }), s => ({ ...s, proposals: {} }),
    s => ({ ...s, facts: filler(CAP + 1) }), s => ({ ...s, proposals: [proposal(ID('b')), proposal(ID('b'))] }), s => ({ ...s, facts: [fact(ID('b'))] }),
    s => ({ ...s, pending_index: -1 }), s => ({ ...s, pending_index_cleanup: 1.5 }), s => ({ ...s, pending_index: '2' }),
  ];
  const itemMutations = [
    i => ({ ...i, id: 'fact_x' }), i => ({ ...i, id: ID('A') }), i => ({ ...i, id: 7 }), i => ({ ...i, content: null }), i => ({ ...i, category: 1 }),
    i => ({ ...i, state: 'deleted' }), i => ({ ...i, state: 'proposed' }), i => ({ ...i, expires_at: 'never' }), i => ({ ...i, expires_at: null }),
    i => ({ ...i, revision: 0 }), i => ({ ...i, revision: 1.5 }), i => ({ ...i, confirmed_at: 'x' }), i => ({ ...i, confirmed_by: 4 }), i => ({ ...i, sources: null }),
    i => ({ ...i, sources: [{ origin: 'radio', captured_at: 1 }] }), i => ({ ...i, sources: [{ origin: 'voice', captured_at: NaN }] }),
    i => ({ ...i, sources: [{ origin: 'voice', captured_at: 1, fact_dependencies: [{ fact_id: 'nope' }] }] }), i => ({ ...i, sources: [null] }), () => null,
  ];
  let rows = 0;
  for (const mutate of mutations) { assert.throws(() => parseSnapshot(mutate(good))); rows++; }
  for (const mutate of itemMutations) {
    assert.throws(() => parseSnapshot({ ...good, facts: [mutate(good.facts[0])] })); rows++;
    const p = mutate(good.proposals[0]);
    // A proposal with state 'proposed' is valid; the 'proposed' mutation only breaks facts.
    if (!(p && p.state === 'proposed' && p.id === good.proposals[0].id && p.expires_at !== null && typeof p.expires_at === 'number' && Array.isArray(p.sources)
        && p.sources.every(item => item && ['ambient', 'voice', 'interaction'].includes(item.origin) && Number.isFinite(item.captured_at) && (item.fact_dependencies || []).every(dep => /^fact_[0-9a-f]{64}$/.test(dep.fact_id)))
        && typeof p.content === 'string' && typeof p.category === 'string' && Number.isSafeInteger(p.revision) && p.revision >= 1
        && (p.confirmed_at === null || Number.isFinite(p.confirmed_at)) && (p.confirmed_by === null || typeof p.confirmed_by === 'string'))) {
      assert.throws(() => parseSnapshot({ ...good, proposals: [p] })); rows++;
    }
  }
  assert.ok(rows >= 40, rows);
});

// ---------------------------------------------------------------------------
// C02 failure matrix: op x write answer x post-answer observation (F13).
// Expected classes are written out here, not derived from the implementation.
const OPS = ['confirm', 'reject', 'delete'];
const target = ID('e');
const RESPONSES = [
  ['200 valid', op => ({ value: op === 'confirm' ? { fact: fact(target, { state: 'confirmed' }) } : { deleted: true } }), { confirm: 'applied', reject: 'applied', delete: 'applied' }],
  ['200 already', op => ({ value: op === 'confirm' ? { fact: fact(target, { state: 'confirmed', confirmed_by: 'other' }) } : { deleted: false } }), { confirm: 'applied', reject: 'applied', delete: 'applied' }],
  ['200 malformed', () => ({ value: {} }), { confirm: 'uncertain', reject: 'uncertain', delete: 'uncertain' }],
  ['200 wrong receipt', op => ({ value: op === 'confirm' ? { fact: fact(ID('f'), { state: 'confirmed' }) } : { deleted: 'yes' } }), { confirm: 'uncertain', reject: 'uncertain', delete: 'uncertain' }],
  ['200 null body', () => ({ value: null }), { confirm: 'uncertain', reject: 'uncertain', delete: 'uncertain' }],
  ['network', () => ({ error: new TypeError('Failed to fetch') }), { confirm: 'uncertain', reject: 'uncertain', delete: 'uncertain' }],
  ['timeout', () => ({ error: Object.assign(Error('aborted'), { name: 'AbortError' }) }), { confirm: 'uncertain', reject: 'uncertain', delete: 'uncertain' }],
  ['401', () => ({ error: { status: 401, detail: 'auth required' } }), { confirm: 'uncertain', reject: 'uncertain', delete: 'uncertain' }],
  ['403 permission', () => ({ error: { status: 403, detail: 'dashboard_permission_denied' } }), { confirm: 'uncertain', reject: 'uncertain', delete: 'uncertain' }],
  ['403 retired', () => ({ error: { status: 403, detail: 'dashboard_session_retired' } }), { confirm: 'uncertain', reject: 'uncertain', delete: 'uncertain' }],
  ['403 review denied', () => ({ error: { status: 403, detail: 'memory_review_denied' } }), { confirm: 'uncertain', reject: 'uncertain', delete: 'uncertain' }],
  ['404 unavailable', () => ({ error: { status: 404, detail: 'memory_proposal_unavailable' } }), { confirm: 'provisional', reject: 'uncertain', delete: 'uncertain' }],
  ['404 other', () => ({ error: { status: 404, detail: 'Not Found' } }), { confirm: 'uncertain', reject: 'uncertain', delete: 'uncertain' }],
  ['409 proposal required', () => ({ error: { status: 409, detail: 'memory_proposal_required' } }), { confirm: 'provisional', reject: 'uncertain', delete: 'uncertain' }],
  ['409 state changed', () => ({ error: { status: 409, detail: 'memory_proposal_state_changed' } }), { confirm: 'uncertain', reject: 'provisional', delete: 'uncertain' }],
  ['409 unknown', () => ({ error: { status: 409, detail: 'invalid_fact_dependency' } }), { confirm: 'uncertain', reject: 'uncertain', delete: 'uncertain' }],
  ['422', () => ({ error: { status: 422, detail: [{ loc: ['path'] }] } }), { confirm: 'provisional', reject: 'provisional', delete: 'provisional' }],
  ['500', () => ({ error: { status: 500 } }), { confirm: 'uncertain', reject: 'uncertain', delete: 'uncertain' }],
  ['502', () => ({ error: { status: 502 } }), { confirm: 'uncertain', reject: 'uncertain', delete: 'uncertain' }],
  ['503 memory unavailable', () => ({ error: { status: 503, detail: 'memory_unavailable' } }), { confirm: 'provisional', reject: 'provisional', delete: 'provisional' }],
  ['503 unknown', () => ({ error: { status: 503, detail: 'upstream' } }), { confirm: 'uncertain', reject: 'uncertain', delete: 'uncertain' }],
  ['answered', () => ({ error: { status: 403, detail: 'dashboard_session_retired', sent: true, answered: true } }), { confirm: 'uncertain', reject: 'uncertain', delete: 'uncertain' }],
  ['local refusal', () => ({ error: { status: 403, sent: false } }), { confirm: 'not_applied', reject: 'not_applied', delete: 'not_applied' }],
];
const OBSERVATIONS = {
  present_unchanged: op => op === 'delete' ? snap([fact(target)]) : snap([], [proposal(target)]),
  present_other: () => snap([fact(target, { state: 'confirmed' })]),
  present_automatic: () => snap([fact(target)]),
  absent_below: () => snap(filler(3), filler(2, 50).map(item => ({ ...item, state: 'proposed' }))),
  // An incomplete walk (page bound reached) on either list.
  absent_facts_incomplete: () => ({ ...parseSnapshot(snap(filler(CAP), [], { facts_next: CUR(1) })), factsComplete: false }),
  absent_proposals_incomplete: () => ({ ...parseSnapshot(snap([], filler(CAP).map(item => ({ ...item, state: 'proposed' })), { proposals_next: CUR(1) })), proposalsComplete: false }),
};
const STILL_P = 'Todavía figura como propuesta; puede que el cambio no se haya aplicado.';
const STILL_F = 'Todavía figura guardado; puede que el cambio no se haya aplicado.';
const NOPE = INCOMPLETE_TEXT;
const NOT_PROPOSAL_C = 'Este recuerdo no es una propuesta; no se conservó.';
const NOT_PROPOSAL_R = 'Este recuerdo no es una propuesta; no se descartó.';
const CONFIRMED_ELSEWHERE = 'Otra sesión ya la confirmó; no se descartó.';
const GONE = 'Ya no está guardada: venció o se eliminó.';
// Finals a read may give for an uncertain write: only states that make the operation impossible from now on.
const IMPOSSIBLE = new Set([NOT_PROPOSAL_C, NOT_PROPOSAL_R, CONFIRMED_ELSEWHERE, GONE]);
// [result, text] for unresolved and provisional markers ('REASON' = the server reason).
const EXPECTED = {
  confirm: {
    present_unchanged: { unresolved: ['pending', STILL_P], provisional: ['final', 'REASON'] },
    present_other: { unresolved: ['applied', 'Comprobado: el recuerdo quedó conservado sin vencimiento.'], provisional: ['applied', 'Comprobado: el recuerdo quedó conservado sin vencimiento.'] },
    present_automatic: { unresolved: ['final', NOT_PROPOSAL_C], provisional: ['final', 'REASON'] },
    absent_below: { unresolved: ['final', GONE], provisional: ['final', GONE] },
    absent_facts_incomplete: { unresolved: ['inconclusive', NOPE], provisional: ['inconclusive', NOPE] },
    absent_proposals_incomplete: { unresolved: ['inconclusive', NOPE], provisional: ['final', 'REASON'] },
  },
  reject: {
    present_unchanged: { unresolved: ['pending', STILL_P], provisional: ['final', 'REASON'] },
    present_other: { unresolved: ['final', CONFIRMED_ELSEWHERE], provisional: ['final', CONFIRMED_ELSEWHERE] },
    present_automatic: { unresolved: ['final', NOT_PROPOSAL_R], provisional: ['final', NOT_PROPOSAL_R] },
    absent_below: { unresolved: ['applied', 'Comprobado: la propuesta ya no está guardada.'], provisional: ['applied', 'Comprobado: la propuesta ya no está guardada.'] },
    absent_facts_incomplete: { unresolved: ['inconclusive', NOPE], provisional: ['inconclusive', NOPE] },
    absent_proposals_incomplete: { unresolved: ['inconclusive', NOPE], provisional: ['inconclusive', NOPE] },
  },
  delete: {
    present_unchanged: { unresolved: ['pending', STILL_F], provisional: ['final', 'REASON'] },
    present_other: { unresolved: ['pending', STILL_F], provisional: ['final', 'REASON'] },
    present_automatic: { unresolved: ['pending', STILL_F], provisional: ['final', 'REASON'] },
    absent_below: { unresolved: ['applied', 'Comprobado: ya no está guardado.'], provisional: ['applied', 'Comprobado: ya no está guardado.'] },
    absent_facts_incomplete: { unresolved: ['inconclusive', NOPE], provisional: ['inconclusive', NOPE] },
    absent_proposals_incomplete: { unresolved: ['inconclusive', NOPE], provisional: ['inconclusive', NOPE] },
  },
};
// Marker disposition per class and send: written out, not derived.
const DISPOSITION = { applied: { first: 'drop', retry: 'drop' }, not_applied: { first: 'final', retry: 'unresolved' },
  provisional: { first: 'provisional', retry: 'provisional' }, uncertain: { first: 'unresolved', retry: 'unresolved' } };
const NOT_APPLIED_CLAIM = /no se (eliminó|conservó|aplicó|descartó)\.?$/;

test('the F2 claim pattern can fire (self-check of the matrix assertion)', () => {
  for (const text of ['no se eliminó.', 'Este recuerdo no es una propuesta; no se conservó.', 'no se aplicó']) assert.match(text, NOT_APPLIED_CLAIM);
});

test('C02 failure matrix: classification, first/retry disposition and post-answer resolution', () => {
  const rows = [], distinct = new Set();
  for (const op of OPS) {
    for (const [name, make, expected] of RESPONSES) {
      const verdict = classifyWrite(op, target, make(op));
      assert.equal(verdict.kind, expected[op], `${op} / ${name}`);
      assert.equal(typeof verdict.text, 'string'); assert.ok(verdict.text.length > 8, `${op} / ${name} has copy`);
      assert.ok(!/no se eliminó|borrado total/.test(verdict.text));
      for (const send of ['first', 'retry']) {
        const disposition = writeDisposition(verdict.kind, send === 'first');
        assert.equal(disposition, DISPOSITION[verdict.kind][send], `${op} / ${name} / ${send}`);
        rows.push({ op, response: name, send, class: verdict.kind, disposition });
        distinct.add([op, verdict.kind, send, disposition].join('|'));
        if (disposition === 'drop' || disposition === 'final') continue;
        const provisional = disposition === 'provisional';
        for (const [observation, build] of Object.entries(OBSERVATIONS)) {
          const built = build(op);
          const resolved = resolveMarker(op, observe('factsComplete' in built || 'proposalsComplete' in built ? built : parseSnapshot(built), target), provisional, provisional ? verdict.text : '');
          const [result, text] = EXPECTED[op][observation][disposition];
          assert.equal(resolved.result, result, `${op} / ${name} / ${send} / ${observation}`);
          assert.equal(resolved.text, text === 'REASON' ? verdict.text : text, `${op} / ${name} / ${send} / ${observation} copy`);
          // F2: for an unresolved write a read may only end it when the operation became impossible.
          if (!provisional && resolved.result === 'final') assert.ok(IMPOSSIBLE.has(resolved.text), `${op} / ${name} / ${observation}: ${resolved.text}`);
          if (!provisional && NOT_APPLIED_CLAIM.test(resolved.text)) assert.ok(IMPOSSIBLE.has(resolved.text), resolved.text);
          rows.push({ op, response: name, send, observation, class: verdict.kind, disposition, result });
          distinct.add([op, disposition, observation, result].join('|'));
        }
      }
    }
  }
  assert.ok(rows.length >= 200, 'matrix rows: ' + rows.length);
  if (process.env.KZA_MEMORY_MATRIX_OUT) require('node:fs').writeFileSync(process.env.KZA_MEMORY_MATRIX_OUT, JSON.stringify({ rows: rows.length, distinct_cases: distinct.size, entries: rows }, null, 1) + '\n');
});

// The Memoria copy must match what this release does (u3b design v2 P2/P4): facts are not used to answer,
// automatic facts come from ambient listening and may be processed in the cloud (code default 'cloud').
test('Memoria copy: no use-in-answers claim, no never-leaves-the-house claim, required statements present', () => {
  const view = require('node:fs').readFileSync(require.resolve('../../src/dashboard/frontend/home/nexo/memory.js'), 'utf8');
  for (const phrase of ['se usa en las respuestas', 'usa en sus respuestas', 'no se usan en las respuestas', 'deja de usarlo en sus respuestas',
    'Cualquier persona que hable con KZA', 'deduce de las conversaciones', 'Sólo tu confirmación', 'no sale de', 'nunca sale', 'sin salir de', 'borrado total']) {
    assert.ok(!view.includes(phrase), phrase);
  }
  for (const phrase of ['deduce de lo que escucha en el ambiente', 'puede procesarse en la nube', 'como máximo 30 días',
    'En esta versión KZA no usa estos recuerdos para responder', 'La memoria es compartida por todo el hogar', 'las cuentas de administración ven y revisan los recuerdos de todos']) {
    assert.ok(view.includes(phrase), phrase);
  }
});

// ---------------------------------------------------------------------------
// U3b-3: owner assignment, 'Devueltos', counts and people (operator with memory:assign).
const OPERATOR = ['memory:read', 'memory:review', 'memory:assign', 'admin:manage'];
const A = 'user_' + 'a'.repeat(32), B = 'user_' + 'b'.repeat(32), RETIRED = 'user_' + 'c'.repeat(32);
const PEOPLE = [{ id: A, name: 'Persona A', permission_level: 'adult', is_active: true }, { id: B, name: 'Persona B', permission_level: 'adult', is_active: true },
  { id: RETIRED, name: 'Persona inactiva', permission_level: 'adult', is_active: false }];
const COUNTS = { currently_owned: 1, returned: 1, ever_assigned_live: 2, owned_proposals: 0, retired_owner_prune_pending: false };
const owned = (id, person, revision = 1, overrides = {}) => fact(id, { state: 'confirmed', revision: 2, owner: { person_id: person, method: 'operator_assignment', revision, assigned_at: 99 },
  owner_revision: revision, owner_change: revision === 1 ? 'assigned' : 'reassigned', ...overrides });
const returnedItem = id => fact(id, { state: 'confirmed', revision: 2, owner: null, owner_revision: 2, owner_change: 'owner_returned' });
function operatorFixture({ returned = () => ({ facts: [], next: null, limit: 100 }), counts = () => COUNTS, people = () => PEOPLE, ...options } = {}) {
  return fixture({ session: sessionMock('synthetic-a', OPERATOR), identity: () => whoami('synthetic-a', 'operator', OPERATOR),
    extra: { '/api/memory/returned': returned, '/api/memory/owners/counts': counts, '/api/admin/users': people }, ...options });
}
const ASSIGN_KEYS = ['delete_dependents', 'expected_fact_revision', 'expected_owner_revision', 'op', 'person_id', 'sent_at', 'state', 'subject'];

test('U3b-3: extras are read only with memory:assign (people only with admin:manage) and each fails on its own', async () => {
  const f = operatorFixture();
  f.read(snap([fact(ID('a'))]));
  await f.owner.open(); await settle();
  assert.deepEqual(f.calls, ['GET /api/admin/auth/whoami', 'GET /api/memory', 'GET /api/memory/returned', 'GET /api/memory/owners/counts', 'GET /api/admin/users']);
  let state = f.owner.getState();
  assert.deepEqual(state.counts, { currently_owned: 1, returned: 1, ever_assigned_live: 2, owned_proposals: 0, prune_pending: false });
  assert.deepEqual(state.people.map(item => [item.id, item.active]), [[A, true], [B, true], [RETIRED, false]]);
  assert.ok(!('permission_level' in state.people[0]), 'only id, name and activity are kept');
  // Read-only operator (memory:read): no owner routes at all.
  const g = fixture();
  g.read(snap([fact(ID('a'))]));
  await g.owner.open(); await settle();
  assert.deepEqual(g.calls, ['GET /api/admin/auth/whoami', 'GET /api/memory']);
  // memory:assign without admin:manage: no people catalog, so nothing is assignable.
  const permissions = ['memory:read', 'memory:review', 'memory:assign'];
  const h = fixture({ session: sessionMock('synthetic-a', permissions), identity: () => whoami('synthetic-a', 'operator', permissions),
    extra: { '/api/memory/returned': () => ({ facts: [], next: null, limit: 100 }), '/api/memory/owners/counts': () => COUNTS } });
  h.read(snap([fact(ID('a'))]));
  await h.owner.open(); await settle();
  assert.ok(!h.calls.includes('GET /api/admin/users'));
  assert.equal(h.owner.assignable(ID('a')), false);
  assert.deepEqual(await h.owner.assign(ID('a'), A, { confirmed: true }), { sent: false });
  // A failing 'Devueltos' read or counts read keeps the main list and says so.
  const k = operatorFixture({ returned: fail(503, 'memory_unavailable'), counts: () => ({ ...COUNTS, returned: -1 }) });
  k.read(snap([fact(ID('a'))]));
  await k.owner.open(); await settle();
  state = k.owner.getState();
  assert.equal(state.loaded, true); assert.equal(state.facts.length, 1);
  assert.equal(state.returnedError, 'No pudimos consultar los recuerdos devueltos.');
  assert.equal(state.countsError, 'No pudimos consultar los totales de asignación.');
  // The prune flag is carried through.
  const m = operatorFixture({ counts: () => ({ ...COUNTS, retired_owner_prune_pending: true }) });
  m.read(snap([]));
  await m.owner.open(); await settle();
  assert.equal(m.owner.getState().counts.prune_pending, true);
});

test('U3b-3: Devueltos walks every page through ?after= and is listed separately from the owner rows', async () => {
  const pages = [{ facts: [returnedItem(ID('a'))], next: CUR(7), limit: 1 }, { facts: [returnedItem(ID('b'))], next: null, limit: 1 }];
  const f = operatorFixture({ returned: () => pages.shift() });
  f.read(snap([returnedItem(ID('a')), returnedItem(ID('b')), owned(ID('c'), A)]));
  await f.owner.open(); await settle();
  assert.ok(f.calls.includes('GET /api/memory/returned?after=' + CUR(7)));
  const state = f.owner.getState();
  assert.deepEqual(state.returned.map(item => item.id), [ID('a'), ID('b')]);
  assert.equal(state.returnedComplete, true);
  // A returned row that still has an owner is malformed: the queue fails closed, the list stays.
  const g = operatorFixture({ returned: () => ({ facts: [owned(ID('c'), A)], next: null, limit: 100 }) });
  g.read(snap([owned(ID('c'), A)]));
  await g.owner.open(); await settle();
  assert.equal(g.owner.getState().returnedError, 'No pudimos consultar los recuerdos devueltos.');
  assert.equal(g.owner.getState().facts.length, 1);
});

test('U3b-3: assign sends the expected revisions after a whoami, stores no content and applies the receipt', async () => {
  const f = operatorFixture();
  f.read(snap([fact(ID('a'), { revision: 3 })]), snap([owned(ID('a'), A)]));
  await f.owner.open(); await settle();
  assert.equal(f.owner.assignable(ID('a')), true);
  // Without the dialog's confirmation, to an inactive or unknown person: nothing is sent.
  for (const [person, options] of [[A, {}], [RETIRED, { confirmed: true }], ['user_' + 'd'.repeat(32), { confirmed: true }]]) {
    assert.deepEqual(await f.owner.assign(ID('a'), person, options), { sent: false });
  }
  assert.deepEqual(f.writes(), []);
  let body = null;
  f.write((path, options) => { body = options.body; assert.equal(path, '/api/memory/facts/' + ID('a') + '/owner'); assert.equal(options.method, 'POST');
    const stored = JSON.parse(f.storage.getItem(PREFIX + ID('a')));
    assert.deepEqual(Object.keys(stored).sort(), ASSIGN_KEYS);
    assert.ok(!JSON.stringify(stored).includes('Dato sintético'));
    return { fact_revision: 4, owner_revision: 1, deleted_dependents: 0, confirmed_now: true, derived_records_retired: true, notice: 'Aviso A16 sintético.' }; });
  const before = f.calls.length;
  const result = await f.owner.assign(ID('a'), A, { confirmed: true }); await settle();
  assert.deepEqual(f.calls.slice(before, before + 2), ['GET /api/admin/auth/whoami', 'POST /api/memory/facts/' + ID('a') + '/owner']);
  assert.deepEqual(body, { person_id: A, expected_fact_revision: 3, expected_owner_revision: 0 });
  assert.equal(result.applied, true);
  const state = f.owner.getState();
  assert.equal(state.notices[ID('a')], 'Recuerdo asignado. Quedó confirmado y no vence. Aviso A16 sintético.');
  assert.equal(state.facts[0].owner.person_id, A); assert.equal(state.facts[0].state, 'confirmed');
  assert.equal(f.storage.getItem(PREFIX + ID('a')), null);
});

test('U3b-3: a derived memory, a missing assign notice or a missing people catalog is never assignable', async () => {
  for (const [name, start, options] of [
    ['derived', snap([fact(ID('a')), fact(ID('b'), { sources: [source([ID('a')])] })]), {}],
    ['no notice', snap([fact(ID('b'))], [], { assign_notice: undefined }), {}],
    ['no people', snap([fact(ID('b'))]), { people: fail(503, 'user_catalog_unavailable') }],
  ]) {
    const f = operatorFixture(options);
    f.read(start);
    await f.owner.open(); await settle();
    assert.equal(f.owner.assignable(ID('b')), false, name);
    assert.deepEqual(await f.owner.assign(ID('b'), A, { confirmed: true }), { sent: false }, name);
    assert.deepEqual(f.writes(), [], name);
  }
});

test('U3b-3: 409 has_dependents returns ids and states, drops the marker, and the confirmed resend deletes them with the same revisions', async () => {
  const f = operatorFixture();
  f.read(snap([fact(ID('a'), { revision: 2 }), fact(ID('b'), { sources: [source([ID('a')])] })]), snap([owned(ID('a'), A)]));
  await f.owner.open(); await settle();
  const bodies = [];
  f.write((path, options) => { bodies.push(options.body); throw Object.assign(Error('http'), { status: 409, detail: { code: 'memory_owner_has_dependents', notice: 'Tiene derivados (sintético).', dependents: [{ id: ID('b'), state: 'automatic' }, { id: ID('c'), state: 'confirmed' }] } }); },
    (path, options) => { bodies.push(options.body); return { fact_revision: 3, owner_revision: 1, deleted_dependents: 2, confirmed_now: true, derived_records_retired: true, notice: null }; });
  const first = await f.owner.assign(ID('a'), A, { confirmed: true }); await settle();
  assert.deepEqual(first.dependents, [{ id: ID('b'), state: 'automatic' }, { id: ID('c'), state: 'confirmed' }]);
  assert.equal(first.notice, 'Tiene derivados (sintético).');
  assert.deepEqual(f.owner.getState().markers, {}); assert.equal(f.storage.getItem(PREFIX + ID('a')), null);
  const second = await f.owner.assign(ID('a'), A, { confirmed: true, deleteDependents: true, expected: first.expected }); await settle();
  assert.equal(second.applied, true); assert.equal(second.receipt.deleted_dependents, 2);
  assert.deepEqual(bodies, [{ person_id: A, expected_fact_revision: 2, expected_owner_revision: 0 },
    { person_id: A, expected_fact_revision: 2, expected_owner_revision: 0, delete_dependents: true }]);
  assert.match(f.owner.getState().notices[ID('a')], /Se eliminaron 2 recuerdos derivados/);
  // A malformed dependents body is not trusted as a list: provisional, a read decides.
  const g = operatorFixture();
  g.read(snap([fact(ID('a'))]), snap([fact(ID('a'))]));
  await g.owner.open(); await settle();
  g.write(fail(409, { code: 'memory_owner_has_dependents', dependents: [{ id: 'nope', state: 'x' }] }));
  await g.owner.assign(ID('a'), A, { confirmed: true }); await settle();
  assert.equal(g.owner.getState().notices[ID('a')], 'El servidor informó recuerdos derivados con un formato desconocido.');
});

test('U3b-3: a lost assign is judged only by a later read: owner reached, CAS moved elsewhere, or still pending', async () => {
  // Applied: the read shows the chosen person.
  let f = operatorFixture();
  f.read(snap([fact(ID('a'))]), snap([owned(ID('a'), A)]));
  await f.owner.open(); await settle();
  f.write(network);
  await f.owner.assign(ID('a'), A, { confirmed: true }); await settle();
  assert.deepEqual(f.owner.getState().markers, {});
  assert.equal(f.owner.getState().notices[ID('a')], 'Comprobado: el recuerdo quedó asignado y confirmado.');
  // Another session assigned it to B meanwhile: the CAS can never match again; the copy claims nothing about ours.
  f = operatorFixture();
  f.read(snap([fact(ID('a'))]), snap([owned(ID('a'), B)]));
  await f.owner.open(); await settle();
  f.write(network);
  await f.owner.assign(ID('a'), A, { confirmed: true }); await settle();
  assert.deepEqual(f.owner.getState().markers, {});
  assert.equal(f.owner.getState().notices[ID('a')], 'La asignación de este recuerdo cambió desde otra sesión. Revisá a quién quedó asignado.');
  assert.ok(!/no se asignó/.test(f.owner.getState().notices[ID('a')]));
  // Unchanged: still pending, never "not applied"; the marker survives a reload without content.
  f = operatorFixture();
  f.read(snap([fact(ID('a'))]), snap([fact(ID('a'))]));
  await f.owner.open(); await settle();
  f.write(network);
  await f.owner.assign(ID('a'), A, { confirmed: true }); await settle();
  assert.equal(f.owner.getState().markers[ID('a')].state, 'unresolved');
  assert.equal(f.owner.getState().notices[ID('a')], 'Todavía figura sin esa asignación; puede que el cambio no se haya aplicado.');
  const stored = JSON.parse(f.storage.getItem(PREFIX + ID('a')));
  assert.deepEqual(Object.keys(stored).sort(), ASSIGN_KEYS); assert.equal(stored.person_id, A);
  // Retry resends the identical body; a 409 conflict is provisional and the read settles it as applied.
  let body = null;
  f.write((path, options) => { body = options.body; throw Object.assign(Error('http'), { status: 409, detail: 'memory_owner_conflict' }); });
  f.read(snap([owned(ID('a'), A)]));
  await f.owner.retry(ID('a')); await settle();
  assert.deepEqual(body, { person_id: A, expected_fact_revision: 1, expected_owner_revision: 0 });
  assert.deepEqual(f.owner.getState().markers, {});
  assert.equal(f.owner.getState().notices[ID('a')], 'Comprobado: el recuerdo quedó asignado y confirmado.');
});

test('U3b-3: unassign sends person_id null with the owner revision; a lost unassign resolves from the read', async () => {
  const f = operatorFixture();
  f.read(snap([owned(ID('a'), A, 2)]), snap([fact(ID('a'), { state: 'confirmed', revision: 2, owner: null, owner_revision: 3, owner_change: 'operator_unassigned' })]));
  await f.owner.open(); await settle();
  assert.deepEqual(await f.owner.unassign(ID('a'), {}), { sent: false }, 'needs the dialog');
  let body = null;
  f.write((path, options) => { body = options.body; throw new TypeError('Failed to fetch'); });
  await f.owner.unassign(ID('a'), { confirmed: true }); await settle();
  assert.deepEqual(body, { person_id: null, expected_owner_revision: 2 });
  assert.deepEqual(f.owner.getState().markers, {});
  assert.equal(f.owner.getState().notices[ID('a')], 'Comprobado: el recuerdo quedó sin asignar.');
  // An unowned memory has nothing to unassign.
  const g = operatorFixture();
  g.read(snap([fact(ID('a'))]));
  await g.owner.open(); await settle();
  assert.deepEqual(await g.owner.unassign(ID('a'), { confirmed: true }), { sent: false });
  assert.deepEqual(g.writes(), []);
});

test('U3b-3: an owner marker of another account is never resent; an account switch before assign sends nothing', async () => {
  const storage = storageMock();
  storage.setItem(PREFIX + ID('a'), JSON.stringify({ op: 'assign', subject: 'synthetic-b', sent_at: 1, state: 'unresolved', person_id: A, expected_fact_revision: 1, expected_owner_revision: 0, delete_dependents: false }));
  const f = operatorFixture({ storage });
  f.read(snap([fact(ID('a'))]));
  await f.owner.open(); await settle();
  assert.equal(storage.getItem(PREFIX + ID('a')), null);
  assert.deepEqual(f.owner.getState().markers, {});
  assert.match(f.owner.getState().notices.session, /otra cuenta/);
  assert.deepEqual(f.writes(), []);
  let changed = 0, first = true;
  const g = operatorFixture({ identity: () => { if (first) { first = false; return whoami('synthetic-a', 'operator', OPERATOR); } return whoami('synthetic-b', 'operator', OPERATOR); }, onAccountChanged: () => changed++ });
  g.read(snap([fact(ID('a'))]));
  await g.owner.open(); await settle();
  assert.deepEqual(await g.owner.assign(ID('a'), A, { confirmed: true }), { sent: false, accountChanged: true });
  assert.equal(changed, 1); assert.deepEqual(g.writes(), []);
  // A malformed stored owner marker is dropped, not restored.
  const broken = storageMock();
  broken.setItem(PREFIX + ID('a'), JSON.stringify({ op: 'assign', subject: 'synthetic-a', sent_at: 1, state: 'unresolved', person_id: '', expected_fact_revision: 0 }));
  const h = operatorFixture({ storage: broken });
  h.read(snap([fact(ID('a'))]));
  await h.owner.open(); await settle();
  assert.equal(broken.getItem(PREFIX + ID('a')), null); assert.deepEqual(h.owner.getState().markers, {});
});

// Owner failure matrix: op x answer x send, and every non-final disposition against post-answer reads.
// Expected classes and copy are written out here, not derived from the implementation.
const OWNER_OPS = ['assign', 'unassign'];
const OWNER_RESPONSES = [
  ['200 valid', op => ({ value: op === 'assign' ? { fact_revision: 2, owner_revision: 1, deleted_dependents: 0, confirmed_now: true, derived_records_retired: true, notice: 'n' } : { fact_revision: 2, owner_revision: 2 } }), { assign: 'applied', unassign: 'applied' }],
  ['200 malformed', () => ({ value: { fact_revision: 'x' } }), { assign: 'uncertain', unassign: 'uncertain' }],
  ['200 null', () => ({ value: null }), { assign: 'uncertain', unassign: 'uncertain' }],
  ['network', () => ({ error: new TypeError('Failed to fetch') }), { assign: 'uncertain', unassign: 'uncertain' }],
  ['timeout', () => ({ error: Object.assign(Error('aborted'), { name: 'AbortError' }) }), { assign: 'uncertain', unassign: 'uncertain' }],
  ['401', () => ({ error: { status: 401 } }), { assign: 'uncertain', unassign: 'uncertain' }],
  ['403 assign denied', () => ({ error: { status: 403, detail: 'memory_assign_denied' } }), { assign: 'uncertain', unassign: 'uncertain' }],
  ['403 retired', () => ({ error: { status: 403, detail: 'dashboard_session_retired' } }), { assign: 'uncertain', unassign: 'uncertain' }],
  ['404 unavailable', () => ({ error: { status: 404, detail: 'memory_fact_unavailable' } }), { assign: 'provisional', unassign: 'provisional' }],
  ['404 other', () => ({ error: { status: 404, detail: 'Not Found' } }), { assign: 'uncertain', unassign: 'uncertain' }],
  ['409 conflict', () => ({ error: { status: 409, detail: 'memory_owner_conflict' } }), { assign: 'provisional', unassign: 'provisional' }],
  ['409 derived', () => ({ error: { status: 409, detail: 'memory_owner_derived_fact' } }), { assign: 'provisional', unassign: 'provisional' }],
  ['409 dependents', () => ({ error: { status: 409, detail: { code: 'memory_owner_has_dependents', notice: 'Tiene derivados (sintético).', dependents: [{ id: ID('f'), state: 'automatic' }] } } }), { assign: 'dependents', unassign: 'uncertain' }],
  ['409 unknown', () => ({ error: { status: 409, detail: 'memory_proposal_required' } }), { assign: 'uncertain', unassign: 'uncertain' }],
  ['422 not applicable', () => ({ error: { status: 422, detail: 'memory_owner_dependents_not_applicable' } }), { assign: 'provisional', unassign: 'provisional' }],
  ['422 retired', () => ({ error: { status: 422, detail: 'memory_owner_retired' } }), { assign: 'provisional', unassign: 'provisional' }],
  ['422 inactive', () => ({ error: { status: 422, detail: 'memory_owner_inactive' } }), { assign: 'provisional', unassign: 'provisional' }],
  ['422 invalid person', () => ({ error: { status: 422, detail: 'memory_owner_invalid_person' } }), { assign: 'provisional', unassign: 'provisional' }],
  ['422 invalid request', () => ({ error: { status: 422, detail: 'memory_owner_invalid_request' } }), { assign: 'provisional', unassign: 'provisional' }],
  ['422 pydantic list', () => ({ error: { status: 422, detail: [{ loc: ['body'] }] } }), { assign: 'uncertain', unassign: 'uncertain' }],
  ['500', () => ({ error: { status: 500 } }), { assign: 'uncertain', unassign: 'uncertain' }],
  ['503 memory unavailable', () => ({ error: { status: 503, detail: 'memory_unavailable' } }), { assign: 'provisional', unassign: 'provisional' }],
  ['503 authority missing', () => ({ error: { status: 503, detail: 'memory_owner_authority_missing' } }), { assign: 'provisional', unassign: 'provisional' }],
  ['503 authority unavailable', () => ({ error: { status: 503, detail: 'memory_owner_authority_unavailable' } }), { assign: 'provisional', unassign: 'provisional' }],
  ['503 unknown', () => ({ error: { status: 503, detail: 'upstream' } }), { assign: 'uncertain', unassign: 'uncertain' }],
  ['answered', () => ({ error: { status: 403, detail: 'dashboard_session_retired', sent: true, answered: true } }), { assign: 'uncertain', unassign: 'uncertain' }],
  ['local refusal', () => ({ error: { status: 403, sent: false } }), { assign: 'not_applied', unassign: 'not_applied' }],
];
const OWNER_DISPOSITION = { ...DISPOSITION, dependents: { first: 'final', retry: 'final' } };
// Sent: assign to A with fact revision 1 / owner revision 0; unassign from A with owner revision 1.
const EXPECTED_OWNER = { assign: { person_id: A, expected_fact_revision: 1, expected_owner_revision: 0 }, unassign: { expected_owner_revision: 1 } };
const OWNER_OBSERVATIONS = {
  reached: op => snap([op === 'assign' ? owned(target, A) : fact(target, { state: 'confirmed', revision: 2, owner_revision: 2, owner_change: 'operator_unassigned' })]),
  unchanged: op => snap([op === 'assign' ? fact(target) : owned(target, A)]),
  moved_elsewhere: op => snap([op === 'assign' ? owned(target, B) : owned(target, B, 2)]),
  absent_complete: () => snap(filler(3)),
  absent_incomplete: () => ({ ...parseSnapshot(snap(filler(CAP), [], { facts_next: CUR(1) })), factsComplete: false }),
};
const MOVED = 'La asignación de este recuerdo cambió desde otra sesión. Revisá a quién quedó asignado.';
const OWNER_GONE = 'Ya no está guardado: venció o se eliminó.';
const EXPECTED_OWNER_RESULT = {
  assign: { reached: ['applied', 'Comprobado: el recuerdo quedó asignado y confirmado.'], unchanged: { unresolved: ['pending', 'Todavía figura sin esa asignación; puede que el cambio no se haya aplicado.'], provisional: ['final', 'REASON'] },
    moved_elsewhere: ['final', MOVED], absent_complete: ['final', OWNER_GONE], absent_incomplete: ['inconclusive', NOPE] },
  unassign: { reached: ['applied', 'Comprobado: el recuerdo quedó sin asignar.'], unchanged: { unresolved: ['pending', 'Todavía figura asignado; puede que el cambio no se haya aplicado.'], provisional: ['final', 'REASON'] },
    moved_elsewhere: ['final', MOVED], absent_complete: ['final', OWNER_GONE], absent_incomplete: ['inconclusive', NOPE] },
};
const OWNER_IMPOSSIBLE = new Set([MOVED, OWNER_GONE]);

test('U3b-3 owner failure matrix: classification, first/retry disposition and post-answer resolution for assign and unassign', () => {
  const rows = [], distinct = new Set();
  for (const op of OWNER_OPS) {
    for (const [name, make, expected] of OWNER_RESPONSES) {
      const verdict = classifyWrite(op, target, make(op));
      assert.equal(verdict.kind, expected[op], `${op} / ${name}`);
      assert.ok(typeof verdict.text === 'string' && verdict.text.length > 8, `${op} / ${name} has copy`);
      assert.ok(!/no se asignó|borrado total/.test(verdict.text));
      for (const send of ['first', 'retry']) {
        const disposition = writeDisposition(verdict.kind, send === 'first');
        assert.equal(disposition, OWNER_DISPOSITION[verdict.kind][send], `${op} / ${name} / ${send}`);
        rows.push({ op, response: name, send, class: verdict.kind, disposition });
        if (disposition === 'drop' || disposition === 'final') continue;
        const provisional = disposition === 'provisional';
        for (const [observation, build] of Object.entries(OWNER_OBSERVATIONS)) {
          const built = build(op);
          const snapshot = 'factsComplete' in built ? built : parseSnapshot(built);
          const resolved = resolveMarker(op, observe(snapshot, target), provisional, provisional ? verdict.text : '', EXPECTED_OWNER[op]);
          const spec = EXPECTED_OWNER_RESULT[op][observation];
          const [result, text] = Array.isArray(spec) ? spec : spec[disposition];
          assert.equal(resolved.result, result, `${op} / ${name} / ${send} / ${observation}`);
          assert.equal(resolved.text, text === 'REASON' ? verdict.text : text, `${op} / ${name} / ${send} / ${observation} copy`);
          // F2: an unresolved owner write ends on a read only when it became impossible (never "not applied").
          if (!provisional && resolved.result === 'final') assert.ok(OWNER_IMPOSSIBLE.has(resolved.text), resolved.text);
          rows.push({ op, response: name, send, observation, class: verdict.kind, disposition, result });
          distinct.add([op, disposition, observation, result].join('|'));
        }
      }
    }
  }
  assert.ok(rows.length >= 200, 'owner matrix rows: ' + rows.length);
  if (process.env.KZA_MEMORY_OWNER_MATRIX_OUT) require('node:fs').writeFileSync(process.env.KZA_MEMORY_OWNER_MATRIX_OUT, JSON.stringify({ rows: rows.length, distinct_cases: distinct.size, entries: rows }, null, 1) + '\n');
});

test('U3b-3 parse: owner fields, cursors, returned pages, counts and people fail closed', () => {
  const { parseReturnedPage, parseCounts, parsePeople } = memory;
  const good = snap([owned(ID('a'), A)]);
  assert.doesNotThrow(() => parseSnapshot(good));
  const ownerMutations = [
    i => ({ ...i, owner: { ...i.owner, person_id: '' } }), i => ({ ...i, owner: { ...i.owner, person_id: 'x\u0007' } }), i => ({ ...i, owner: { ...i.owner, method: 'voice' } }),
    i => ({ ...i, owner: { ...i.owner, revision: 0 } }), i => ({ ...i, owner: { ...i.owner, assigned_at: 'now' } }), i => ({ ...i, owner: 'A' }),
    i => ({ ...i, owner_revision: -1 }), i => ({ ...i, owner_revision: 1.5 }), i => ({ ...i, owner_change: 'stolen' }), i => ({ ...i, owner_revision: 0 }),
    i => { const { owner_revision, ...rest } = i; return rest; }, i => ({ ...i, owner_change: null }),
  ];
  for (const mutate of ownerMutations) assert.throws(() => parseSnapshot({ ...good, facts: [mutate(good.facts[0])] }));
  for (const next of ['', 'k1.', CUR(1) + 'x', 'K1' + CUR(1).slice(2), 7]) assert.throws(() => parseSnapshot({ ...good, facts_next: next }));
  assert.equal(parseSnapshot({ ...good, assign_notice: 3 }).assign_notice, null);
  assert.doesNotThrow(() => parseReturnedPage({ facts: [returnedItem(ID('b'))], next: null, limit: 100 }));
  for (const value of [null, { facts: null, next: null }, { facts: [returnedItem(ID('b')), returnedItem(ID('b'))], next: null }, { facts: [owned(ID('b'), A)], next: null },
    { facts: filler(CAP + 1).map(item => ({ ...item, owner_change: 'owner_returned', owner_revision: 1 })), next: null }, { facts: [], next: 'x' }]) assert.throws(() => parseReturnedPage(value));
  assert.throws(() => parseCounts({ ...COUNTS, retired_owner_prune_pending: 'no' }));
  assert.throws(() => parseCounts({ ...COUNTS, currently_owned: undefined }));
  for (const value of [{}, [{ id: A, name: 'x' }], [{ id: A, name: 'x', is_active: true }, { id: A, name: 'y', is_active: true }], [{ id: '', name: 'x', is_active: true }]]) assert.throws(() => parsePeople(value));
});

// --- U3b-3 review fixes (R1, R2, R5, R7, R8) ---------------------------------------------------

test('R2: a proposal confirmed mid-walk may end both lists with the same cursor; each list keeps its own cursor set', async () => {
  // Reproduces probe_cursor_collision.cjs: confirm keeps extracted_at, so the keyset cursor is identical.
  const P = 1000, cursor = CUR(P);
  const proposals = filler(99, 0).map(item => ({ ...item, state: 'proposed' })).concat([proposal(hex(P))]);
  const facts = filler(99, 2000).concat([fact(hex(P), { state: 'confirmed', expires_at: null, confirmed_at: 95, confirmed_by: 'synthetic-a' })]);
  const f = fixture();
  f.read(snap([], proposals, { proposals_next: cursor }), snap(facts, [], { facts_next: cursor }), snap([fact(hex(3000))]));
  await f.owner.open(); await settle();
  assert.deepEqual(f.calls.slice(1), ['GET /api/memory', 'GET /api/memory?proposals_after=' + encodeURIComponent(cursor), 'GET /api/memory?facts_after=' + encodeURIComponent(cursor)]);
  const state = f.owner.getState();
  assert.equal(state.read.error, null); assert.equal(state.loaded, true);
  assert.deepEqual([state.facts.length, state.proposals.length], [101, 99]);
  // A cursor repeated within one list is still a cycle.
  const g = fixture();
  g.read(snap([], [], { facts_next: CUR(1) }), snap([], [], { facts_next: CUR(2) }), snap([], [], { facts_next: CUR(1) }));
  await g.owner.open(); await settle();
  assert.equal(g.owner.getState().loaded, false); assert.ok(g.owner.getState().read.error);
});

test('R7/m18: Devueltos fails closed on a repeated cursor and on a longer cursor cycle', async () => {
  for (const [name, cycle] of [['one step', [CUR(1), CUR(1)]], ['two steps', [CUR(1), CUR(2), CUR(1)]], ['three steps', [CUR(1), CUR(2), CUR(3), CUR(2)]]]) {
    let index = 0;
    const f = operatorFixture({ returned: () => ({ facts: [returnedItem(hex(500 + index))], next: cycle[index++] ?? null, limit: 1 }) });
    f.read(snap([fact(ID('a'))]));
    await f.owner.open(); await settle();
    const state = f.owner.getState();
    assert.equal(state.returnedError, 'No pudimos consultar los recuerdos devueltos.', name);
    assert.equal(state.returnedLoaded, false, name);
    assert.equal(index, cycle.length, name + ': stops at the first repeated cursor');
    assert.equal(state.facts.length, 1, name + ': the main list stays');
  }
});

test('R8/m15: the dependents resend uses the revisions the 409 refused on, even after a read changed the card', async () => {
  const f = operatorFixture();
  f.read(snap([fact(ID('a'), { revision: 2 }), fact(ID('b'), { sources: [source([ID('a')])] })]),
    snap([fact(ID('a'), { revision: 5 }), fact(ID('b'), { sources: [source([ID('a')])] })]), snap([owned(ID('a'), A)]));
  await f.owner.open(); await settle();
  const bodies = [];
  f.write((path, options) => { bodies.push(options.body); throw Object.assign(Error('http'), { status: 409, detail: { code: 'memory_owner_has_dependents', notice: 'Tiene derivados (sintético).', dependents: [{ id: ID('b'), state: 'automatic' }] } }); },
    (path, options) => { bodies.push(options.body); return { fact_revision: 3, owner_revision: 1, deleted_dependents: 1, confirmed_now: true, derived_records_retired: true, notice: null }; });
  const first = await f.owner.assign(ID('a'), A, { confirmed: true }); await settle();
  // A refresh between the 409 and the dialog's answer moves the card to revision 5.
  await f.owner.refresh(); await settle();
  assert.equal(f.owner.getState().facts.find(item => item.id === ID('a')).revision, 5);
  await f.owner.assign(ID('a'), A, { confirmed: true, deleteDependents: true, expected: first.expected, expectedDependents: 1 }); await settle();
  assert.deepEqual(bodies[1], { person_id: A, expected_fact_revision: 2, expected_owner_revision: 0, delete_dependents: true });
});

test('R1: the confirmed dependents count travels with the marker and a different receipt count is said on the card', async () => {
  for (const [deleted, mismatch] of [[2, false], [3, true], [0, true]]) {
    const f = operatorFixture();
    f.read(snap([fact(ID('a'), { revision: 2 }), fact(ID('b'), { sources: [source([ID('a')])] })]), snap([owned(ID('a'), A)]));
    await f.owner.open(); await settle();
    let stored = null;
    f.write(fail(409, { code: 'memory_owner_has_dependents', notice: 'Tiene derivados (sintético).', dependents: [{ id: ID('b'), state: 'automatic' }, { id: ID('c'), state: 'confirmed' }] }),
      () => { stored = JSON.parse(f.storage.getItem(PREFIX + ID('a'))); return { fact_revision: 3, owner_revision: 1, deleted_dependents: deleted, confirmed_now: true, derived_records_retired: true, notice: null }; });
    const first = await f.owner.assign(ID('a'), A, { confirmed: true }); await settle();
    await f.owner.assign(ID('a'), A, { confirmed: true, deleteDependents: true, expected: first.expected, expectedDependents: first.dependents.length }); await settle();
    assert.equal(stored.expected_dependents, 2, 'a count, never ids or content');
    assert.deepEqual(Object.keys(stored).sort(), [...ASSIGN_KEYS, 'expected_dependents'].sort());
    const text = f.owner.getState().notices[ID('a')];
    if (mismatch) {
      assert.ok(text.includes(`Confirmaste 2 recuerdos derivados y se eliminaron ${deleted}: la lista cambió mientras decidías.`), text);
      assert.ok(text.includes('El servidor no informa cuáles se eliminaron.'), text);
    } else assert.ok(!text.includes('Confirmaste'), text);
  }
  // The pre-confirm dialog copy says the deleted set is decided by the server at apply time.
  const view = require('../../src/dashboard/frontend/home/nexo/memory.js');
  assert.match(view.DEPENDENTS_SCOPE, /que existan al momento de asignar/);
  assert.match(view.DEPENDENTS_SCOPE, /solo cuántos se eliminaron, no cuáles/);
});

test('R5: a retried assign refused for dependents says on the card that nothing was assigned and how to review them', async () => {
  const f = operatorFixture();
  f.read(snap([fact(ID('a'), { revision: 2 })]), snap([fact(ID('a'), { revision: 2 })]));
  await f.owner.open(); await settle();
  f.write(network, fail(409, { code: 'memory_owner_has_dependents', notice: 'Tiene derivados (sintético).', dependents: [{ id: ID('b'), state: 'automatic' }] }));
  await f.owner.assign(ID('a'), A, { confirmed: true }); await settle();
  assert.equal(f.owner.getState().markers[ID('a')].state, 'unresolved');
  const result = await f.owner.retry(ID('a')); await settle();
  assert.deepEqual(result.dependents, [{ id: ID('b'), state: 'automatic' }]);
  const state = f.owner.getState();
  assert.equal(state.notices[ID('a')], 'Tiene derivados (sintético). ' + memory.RETRY_DEPENDENTS);
  assert.deepEqual(state.markers, {});
  assert.match(memory.RETRY_DEPENDENTS, /^No se asignó\./);
});

test('R8: the /obs account removal confirmation states that assigned memories are deleted irreversibly', () => {
  const fs = require('node:fs'), path = require('node:path');
  const admin = fs.readFileSync(path.join(__dirname, '../../src/dashboard/frontend/obs/src/views/admin.jsx'), 'utf8');
  const people = require('../../src/dashboard/frontend/home/nexo/people.js');
  assert.ok(people.MEMORY_REMOVAL.includes('sin posibilidad de recuperarlos'));
  const at = admin.indexOf('confirm(`¿Eliminar ${uid}?'), confirmCall = admin.slice(at, admin.indexOf(')) return;', at));
  assert.ok(at >= 0, 'the removal asks for confirmation');
  assert.ok(confirmCall.includes(people.MEMORY_REMOVAL), 'the /obs confirm carries the same sentence as NEXO');
});
