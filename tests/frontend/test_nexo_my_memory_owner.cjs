const assert = require('node:assert/strict');
const { test } = require('node:test');
const fs = require('node:fs');
const mine = require('../../src/dashboard/frontend/home/nexo/my-memory-owner.js');

const { createMyMemoryOwner, parsePage, classifyWrite, writeDisposition, resolveMarker, PREFIX, ENDED_KEY, RETURN_NOTICE } = mine;
const ID = char => 'fact_' + char.repeat(64);
const hex = n => 'fact_' + n.toString(16).padStart(64, '0');
const CUR = n => 'k1.' + n.toString(16).padStart(16, '0') + '.' + hex(n);
const deferred = () => { let resolve, reject; const promise = new Promise((yes, no) => { resolve = yes; reject = no; }); return { promise, resolve, reject }; };
const tick = () => new Promise(resolve => setImmediate(resolve));
async function settle() { for (let index = 0; index < 12; index++) await tick(); }

function item(id, overrides = {}) {
  return { id, content: 'Recuerdo sintético ' + id.slice(5, 9), category: 'preference', state: 'confirmed', created_at: 90, expires_at: null,
    owner_method: 'operator_assignment', owner_assigned_at: 95, owned_dependents: [], has_hidden_dependents: false, ...overrides };
}
const page = (facts = [], next = null, limit = 100) => ({ facts, next, limit });
const filler = (count, offset = 0) => Array.from({ length: count }, (_, index) => item(hex(offset + index + 1)));
function storageMock() {
  const data = new Map();
  return { data, get length() { return data.size; }, key: index => [...data.keys()][index] ?? null,
    getItem: key => data.has(key) ? data.get(key) : null, setItem: (key, value) => { data.set(key, String(value)); }, removeItem: key => { data.delete(key); } };
}
const PERSON = ['home:read', 'memory:own:read', 'memory:own:write'];
function sessionMock(subject = 'synthetic-home-a', permissions = PERSON, role = 'home') {
  return { subject: () => subject, role: () => role, can: permission => permissions.includes(permission) };
}
const whoami = (subject = 'synthetic-home-a', role = 'home', permissions = PERSON) => ({ ok: true, subject, role, permissions });

function fixture({ storage = storageMock(), session = sessionMock(), identity = () => whoami(), onAccountChanged = () => {}, maxPages } = {}) {
  const calls = [], reads = [], writes = [];
  let time = 100000;
  const api = (path, options = {}) => {
    const method = options.method || 'GET';
    calls.push(method + ' ' + path);
    if (path === '/api/admin/auth/whoami') return Promise.resolve().then(identity);
    if (method === 'GET' && (path === '/api/memory/mine' || path.startsWith('/api/memory/mine?'))) {
      const handler = reads.shift();
      if (!handler) return Promise.reject(Object.assign(Error('no read scripted'), { status: 599 }));
      const result = typeof handler === 'function' ? handler(options) : handler;
      return new Promise((resolve, reject) => {
        options.signal?.addEventListener('abort', () => reject(Object.assign(Error('aborted'), { name: 'AbortError' })));
        Promise.resolve(result).then(resolve, reject);
      });
    }
    assert.equal(options.signal, undefined, 'writes are never abortable');
    const handler = writes.shift();
    if (!handler) return Promise.reject(Object.assign(Error('no write scripted'), { status: 599 }));
    return Promise.resolve().then(() => typeof handler === 'function' ? handler(path, options) : handler);
  };
  const owner = createMyMemoryOwner({ api, session, storage, schedule: () => 0, cancel: () => {}, monotonic: () => time, wall: () => time, onAccountChanged, ...(maxPages ? { maxPages } : {}) });
  return { owner, calls, storage, read: (...handlers) => reads.push(...handlers), write: (...handlers) => writes.push(...handlers),
    writes: () => calls.filter(call => !call.startsWith('GET')) };
}
const fail = (status, detail, extra = {}) => () => { throw Object.assign(Error('http'), { status, detail, ...extra }); };
const network = () => { throw new TypeError('Failed to fetch'); };

test('open asks whoami, then walks every page of /api/memory/mine through ?after=', async () => {
  const f = fixture();
  f.read(page(filler(2), CUR(1), 2), page(filler(1, 10), null, 2));
  await f.owner.open(); await settle();
  assert.deepEqual(f.calls, ['GET /api/admin/auth/whoami', 'GET /api/memory/mine', 'GET /api/memory/mine?after=' + CUR(1)]);
  const state = f.owner.getState();
  assert.equal(state.facts.length, 3); assert.equal(state.complete, true); assert.equal(state.loaded, true);
});

test('a household account without a person, an operator or an evaluator never reads or writes', async () => {
  for (const [session, identity] of [
    [sessionMock('synthetic-home', ['home:read']), whoami('synthetic-home', 'home', ['home:read'])],
    [sessionMock('synthetic-operator', ['memory:read', 'memory:review', 'memory:assign'], 'operator'), whoami('synthetic-operator', 'operator', ['memory:read', 'memory:review', 'memory:assign'])],
    [sessionMock('synthetic-evaluator', ['evaluation:read'], 'evaluator'), whoami('synthetic-evaluator', 'evaluator', ['evaluation:read'])],
  ]) {
    let changed = 0;
    const f = fixture({ session, identity: () => identity, onAccountChanged: () => changed++ });
    await f.owner.open(); await settle();
    assert.ok(!f.calls.some(call => call.includes('/api/memory')), f.calls.join());
    assert.deepEqual(await f.owner.remove(ID('a'), { confirmed: true }), { sent: false });
    assert.deepEqual(await f.owner.giveBack(ID('a'), { confirmed: true }), { sent: false });
    assert.deepEqual(f.writes(), []);
    assert.equal(changed, 1, 'whoami without memory:own:read is a different account for this view');
  }
});

test('F5: whoami of another subject, role or without memory:own:read is a change; unknown sends nothing', async () => {
  for (const identity of [() => whoami('synthetic-home-b'), () => whoami('synthetic-home-a', 'operator'), () => whoami('synthetic-home-a', 'home', ['home:read'])]) {
    let changed = 0;
    const f = fixture({ identity, onAccountChanged: () => changed++ });
    await f.owner.open(); await settle();
    assert.equal(changed, 1); assert.deepEqual(f.calls, ['GET /api/admin/auth/whoami']);
  }
  const g = fixture({ identity: () => { throw new TypeError('offline'); } });
  await g.owner.open(); await settle();
  assert.deepEqual(g.calls, ['GET /api/admin/auth/whoami']);
  assert.match(g.owner.getState().read.error, /No se pudo confirmar la cuenta/);
});

test('parsePage fails closed on any extra field (sources, subjects, counters) and on malformed shapes', () => {
  assert.doesNotThrow(() => parsePage(page([item(ID('a'))])));
  const pageMutations = [p => null, p => ({ ...p, status: {} }), p => ({ ...p, next: 'x' }), p => ({ ...p, limit: 0 }), p => ({ ...p, limit: 101 }),
    p => ({ ...p, facts: [item(ID('a')), item(ID('a'))] }), p => ({ ...p, facts: filler(3), limit: 2 }), p => { const { next, ...rest } = p; return rest; }];
  const itemMutations = [i => ({ ...i, sources: [] }), i => ({ ...i, confirmed_by: 'operator' }), i => ({ ...i, confidence: 1 }), i => ({ ...i, revision: 2 }),
    i => ({ ...i, pending_index: 0 }), i => ({ ...i, id: 'fact_x' }), i => ({ ...i, content: 1 }), i => ({ ...i, state: 'deleted' }), i => ({ ...i, expires_at: 5 }),
    i => ({ ...i, state: 'automatic' }), i => ({ ...i, owner_method: 'guess' }), i => ({ ...i, owner_assigned_at: null }), i => ({ ...i, has_hidden_dependents: 'no' }),
    i => ({ ...i, owned_dependents: ['nope'] }), i => ({ ...i, owned_dependents: [i.id] }), i => ({ ...i, owned_dependents: [ID('b'), ID('b')] }),
    i => { const { has_hidden_dependents, ...rest } = i; return rest; }, () => null];
  let rows = 0;
  for (const mutate of pageMutations) { assert.throws(() => parsePage(mutate(page([item(ID('a'))])))); rows++; }
  for (const mutate of itemMutations) { assert.throws(() => parsePage(page([mutate(item(ID('a')))]))); rows++; }
  assert.ok(rows >= 25, rows);
});

test('a malformed page drops nothing silently: the read reports an unknown format and keeps no new content', async () => {
  const f = fixture();
  f.read(page([item(ID('a'), { sources: [] })]));
  await f.owner.open(); await settle();
  assert.equal(f.owner.getState().loaded, false);
  assert.equal(f.owner.getState().read.error, 'El servidor devolvió tus recuerdos con un formato desconocido.');
  const g = fixture();
  g.read(page(filler(1), CUR(1)), page(filler(1, 5), CUR(1)));
  await g.owner.open(); await settle();
  assert.equal(g.owner.getState().read.error, 'El servidor devolvió tus recuerdos con un formato desconocido.', 'a repeated cursor never loops');
});

test('delete needs the dialog and a whoami; the stored marker has no content; 200 removes it with its owned dependents', async () => {
  const f = fixture();
  f.read(page([item(ID('a'), { owned_dependents: [ID('b')], has_hidden_dependents: true }), item(ID('b')), item(ID('c'))]), page([item(ID('c'))]));
  await f.owner.open(); await settle();
  assert.deepEqual(await f.owner.remove(ID('a')), { sent: false });
  assert.deepEqual(f.writes(), []);
  assert.deepEqual(f.owner.ownedDependents(ID('a')).map(entry => entry.id), [ID('b')]);
  f.write((path, options) => {
    assert.equal(path, '/api/memory/mine/facts/' + ID('a')); assert.equal(options.method, 'DELETE');
    const stored = JSON.parse(f.storage.getItem(PREFIX + ID('a')));
    assert.deepEqual(Object.keys(stored).sort(), ['op', 'sent_at', 'state', 'subject']);
    assert.ok(!JSON.stringify([...f.storage.data.values()]).includes('Recuerdo sintético'));
    return { deleted: true };
  });
  const before = f.calls.length;
  const first = f.owner.remove(ID('a'), { confirmed: true }), second = f.owner.remove(ID('a'), { confirmed: true });
  assert.equal((await first).applied, true); assert.deepEqual(await second, { sent: false }); await settle();
  const sentCalls = f.calls.slice(before);
  assert.equal(sentCalls[0], 'GET /api/admin/auth/whoami');
  assert.ok(sentCalls.indexOf('DELETE /api/memory/mine/facts/' + ID('a')) > 0, 'the whoami precedes the write');
  assert.equal(f.calls.filter(call => call.startsWith('DELETE')).length, 1, 'a double tap sends once');
  const state = f.owner.getState();
  assert.deepEqual(state.facts.map(entry => entry.id), [ID('c')]);
  assert.equal(state.notices[ID('a')], 'Recuerdo eliminado.');
  // deleted:false (foreign, missing, already gone) never claims a deletion.
  const g = fixture();
  g.read(page([item(ID('a'))]), page([]));
  await g.owner.open(); await settle();
  g.write({ deleted: false });
  await g.owner.remove(ID('a'), { confirmed: true }); await settle();
  assert.equal(g.owner.getState().notices[ID('a')], 'Ya no figuraba entre tus recuerdos.');
});

test("'No es mío' posts /return and keeps the notice visible after the card leaves", async () => {
  const f = fixture();
  f.read(page([item(ID('a'))]), page([]));
  await f.owner.open(); await settle();
  assert.deepEqual(await f.owner.giveBack(ID('a')), { sent: false });
  f.write((path, options) => { assert.equal(path, '/api/memory/mine/facts/' + ID('a') + '/return'); assert.equal(options.method, 'POST'); return { returned: true, notice: RETURN_NOTICE }; });
  await f.owner.giveBack(ID('a'), { confirmed: true }); await settle();
  const state = f.owner.getState();
  assert.equal(state.returnNotice, RETURN_NOTICE);
  assert.deepEqual(state.facts, []);
  assert.match(RETURN_NOTICE, /No se borró: sigue guardado/);
  // returned:false is "no longer yours", never a return receipt.
  const g = fixture();
  g.read(page([item(ID('a'))]), page([]));
  await g.owner.open(); await settle();
  g.write({ returned: false });
  await g.owner.giveBack(ID('a'), { confirmed: true }); await settle();
  assert.equal(g.owner.getState().returnNotice, null);
  assert.equal(g.owner.getState().notices[ID('a')], 'Ya no figuraba entre tus recuerdos.');
  // The notice belongs to the visit: leaving drops it.
  f.owner.leave();
  assert.equal(f.owner.getState().returnNotice, null);
});

test('F2: a lost write is judged only by a read that started after the loss', async () => {
  for (const op of ['delete', 'return']) {
    const f = fixture();
    const early = deferred();
    f.read(page([item(ID('a'))]), () => early.promise, page([]));
    await f.owner.open(); await settle();
    void f.owner.refresh(); await settle();
    f.write(network);
    await (op === 'delete' ? f.owner.remove(ID('a'), { confirmed: true }) : f.owner.giveBack(ID('a'), { confirmed: true })); await settle();
    // The early read shows the item gone but started before the loss: it decides nothing.
    early.resolve(page([])); await settle();
    let state = f.owner.getState();
    // The follow-up read (after the loss) is what resolves it.
    assert.deepEqual(state.markers, {}, op);
    assert.equal(state.notices[ID('a')], 'Comprobado: ya no figura entre tus recuerdos.', op);
    // Present after the loss: pending, never "not applied".
    const g = fixture();
    g.read(page([item(ID('a'))]), page([item(ID('a'))]));
    await g.owner.open(); await settle();
    g.write(network);
    await (op === 'delete' ? g.owner.remove(ID('a'), { confirmed: true }) : g.owner.giveBack(ID('a'), { confirmed: true })); await settle();
    state = g.owner.getState();
    assert.equal(state.markers[ID('a')].state, 'unresolved');
    assert.ok(!/no se (eliminó|devolvió)/.test(state.notices[ID('a')]), state.notices[ID('a')]);
    // Retry resends the same operation; its answer settles it.
    g.write(op === 'delete' ? { deleted: false } : { returned: false });
    g.read(page([]));
    await g.owner.retry(ID('a')); await settle();
    assert.deepEqual(g.owner.getState().markers, {});
  }
});

test('ordering: a read that started before the loss never resolves the marker', async () => {
  const f = fixture();
  const early = deferred(), late = deferred();
  f.read(page([item(ID('a'))]), () => early.promise, () => late.promise);
  await f.owner.open(); await settle();
  void f.owner.refresh(); await settle();
  f.write(network);
  const sent = f.owner.remove(ID('a'), { confirmed: true }); await settle();
  early.resolve(page([])); await settle();
  assert.equal(f.owner.getState().markers[ID('a')]?.state, 'unresolved', 'the pre-loss read is ignored');
  late.resolve(page([item(ID('a'))])); await sent; await settle();
  assert.equal(f.owner.getState().markers[ID('a')]?.state, 'unresolved');
  assert.equal(f.owner.getState().notices[ID('a')], 'Todavía figura entre tus recuerdos; puede que el cambio no se haya aplicado.');
});

test('an incomplete walk keeps a lost write inconclusive', async () => {
  const f = fixture({ maxPages: 1 });
  f.read(page([item(ID('a'))], CUR(1)), page(filler(1), CUR(1)));
  await f.owner.open(); await settle();
  assert.equal(f.owner.getState().complete, false);
  f.write(network);
  await f.owner.remove(ID('a'), { confirmed: true }); await settle();
  assert.equal(f.owner.getState().markers[ID('a')].state, 'unresolved');
  assert.equal(f.owner.getState().notices[ID('a')], 'No pudimos comprobarlo: tu lista de recuerdos no se pudo recorrer completa.');
});

test('401/403 on a write are uncertain; a local refusal is final only on the first send', async () => {
  for (const answer of [fail(401), fail(403, 'memory_owner_actor_denied'), fail(403, 'dashboard_session_retired'), fail(403, 'memory_person_required')]) {
    const f = fixture();
    f.read(page([item(ID('a'))]), page([item(ID('a'))]));
    await f.owner.open(); await settle();
    f.write(answer);
    await f.owner.remove(ID('a'), { confirmed: true }); await settle();
    assert.equal(f.owner.getState().markers[ID('a')].state, 'unresolved');
  }
});

test('leave never aborts a write; account switches never resend or resolve another account\'s marker', async () => {
  const f = fixture();
  f.read(page([item(ID('a'))]));
  await f.owner.open(); await settle();
  const held = deferred();
  f.write(() => held.promise);
  const sent = f.owner.giveBack(ID('a'), { confirmed: true }); await settle();
  f.owner.leave();
  held.resolve({ returned: true, notice: RETURN_NOTICE });
  assert.equal((await sent).applied, true);
  // A marker stored by another subject is wiped on open, never resent.
  const storage = storageMock();
  storage.setItem(PREFIX + ID('b'), JSON.stringify({ op: 'delete', subject: 'synthetic-home-b', sent_at: 1, state: 'unresolved' }));
  const g = fixture({ storage });
  g.read(page([item(ID('b'))]));
  await g.owner.open(); await settle();
  assert.equal(storage.getItem(PREFIX + ID('b')), null);
  assert.deepEqual(g.writes(), []);
  assert.match(g.owner.getState().notices.session, /otra cuenta/);
  // Retry after another account took the cookie: nothing is sent.
  let changed = 0, first = 0;
  const h = fixture({ identity: () => (++first <= 1 ? whoami() : whoami('synthetic-home-b')), onAccountChanged: () => changed++ });
  h.read(page([item(ID('a'))]));
  await h.owner.open(); await settle();
  assert.deepEqual(await h.owner.remove(ID('a'), { confirmed: true }), { sent: false, accountChanged: true });
  assert.equal(changed, 1); assert.deepEqual(h.writes(), []);
});

test('F11: a session end wipes markers and leaves only a content-free reminder; a late answer re-persists nothing', async () => {
  const f = fixture();
  f.read(page([item(ID('a'))]));
  await f.owner.open(); await settle();
  const held = deferred();
  f.write(() => held.promise);
  const sent = f.owner.remove(ID('a'), { confirmed: true }); await settle();
  assert.ok(f.storage.getItem(PREFIX + ID('a')));
  await f.owner.clear({ wipe: true });
  assert.equal(f.storage.getItem(PREFIX + ID('a')), null);
  assert.equal(f.storage.getItem(ENDED_KEY), '1');
  held.resolve({ deleted: true });
  assert.deepEqual(await sent, { stale: true });
  assert.deepEqual([...f.storage.data.keys()], [ENDED_KEY]);
});

// Household failure matrix: op x answer x send x post-answer observation.
const OPS = ['delete', 'return'];
const RESPONSES = [
  ['200 done', op => ({ value: op === 'delete' ? { deleted: true } : { returned: true, notice: RETURN_NOTICE } }), { delete: 'applied', return: 'applied' }],
  ['200 already', op => ({ value: op === 'delete' ? { deleted: false } : { returned: false } }), { delete: 'applied', return: 'applied' }],
  ['200 malformed', () => ({ value: { ok: true } }), { delete: 'uncertain', return: 'uncertain' }],
  ['200 null', () => ({ value: null }), { delete: 'uncertain', return: 'uncertain' }],
  ['network', () => ({ error: new TypeError('Failed to fetch') }), { delete: 'uncertain', return: 'uncertain' }],
  ['timeout', () => ({ error: Object.assign(Error('aborted'), { name: 'AbortError' }) }), { delete: 'uncertain', return: 'uncertain' }],
  ['401', () => ({ error: { status: 401 } }), { delete: 'uncertain', return: 'uncertain' }],
  ['403 permission', () => ({ error: { status: 403, detail: 'dashboard_permission_denied' } }), { delete: 'uncertain', return: 'uncertain' }],
  ['403 person required', () => ({ error: { status: 403, detail: 'memory_person_required' } }), { delete: 'uncertain', return: 'uncertain' }],
  ['403 actor denied', () => ({ error: { status: 403, detail: 'memory_owner_actor_denied' } }), { delete: 'uncertain', return: 'uncertain' }],
  ['403 retired', () => ({ error: { status: 403, detail: 'dashboard_session_retired' } }), { delete: 'uncertain', return: 'uncertain' }],
  ['404', () => ({ error: { status: 404, detail: 'Not Found' } }), { delete: 'uncertain', return: 'uncertain' }],
  ['405', () => ({ error: { status: 405 } }), { delete: 'uncertain', return: 'uncertain' }],
  ['409', () => ({ error: { status: 409, detail: 'x' } }), { delete: 'uncertain', return: 'uncertain' }],
  ['422 fact invalid', () => ({ error: { status: 422, detail: 'memory_fact_invalid' } }), { delete: 'provisional', return: 'provisional' }],
  ['500', () => ({ error: { status: 500 } }), { delete: 'uncertain', return: 'uncertain' }],
  ['503 memory unavailable', () => ({ error: { status: 503, detail: 'memory_unavailable' } }), { delete: 'provisional', return: 'provisional' }],
  ['503 authority unavailable', () => ({ error: { status: 503, detail: 'dashboard_authority_unavailable' } }), { delete: 'provisional', return: 'provisional' }],
  ['503 unknown', () => ({ error: { status: 503, detail: 'upstream' } }), { delete: 'uncertain', return: 'uncertain' }],
  ['answered', () => ({ error: { status: 403, detail: 'dashboard_session_retired', sent: true, answered: true } }), { delete: 'uncertain', return: 'uncertain' }],
  ['local refusal', () => ({ error: { status: 403, sent: false } }), { delete: 'not_applied', return: 'not_applied' }],
];
const DISPOSITION = { applied: { first: 'drop', retry: 'drop' }, not_applied: { first: 'final', retry: 'unresolved' },
  provisional: { first: 'provisional', retry: 'provisional' }, uncertain: { first: 'unresolved', retry: 'unresolved' } };
const OBSERVATIONS = { present: { present: true, complete: true }, present_incomplete: { present: true, complete: false },
  absent_complete: { present: false, complete: true }, absent_incomplete: { present: false, complete: false } };
const STILL = { delete: 'Todavía figura entre tus recuerdos; puede que el cambio no se haya aplicado.', return: 'Todavía figura entre tus recuerdos; puede que no se haya devuelto.' };
const GONE = 'Comprobado: ya no figura entre tus recuerdos.', NOPE = 'No pudimos comprobarlo: tu lista de recuerdos no se pudo recorrer completa.';
const EXPECTED = {
  present: { unresolved: op => ['pending', STILL[op]], provisional: () => ['final', 'REASON'] },
  present_incomplete: { unresolved: op => ['pending', STILL[op]], provisional: () => ['final', 'REASON'] },
  absent_complete: { unresolved: () => ['applied', GONE], provisional: () => ['applied', GONE] },
  absent_incomplete: { unresolved: () => ['inconclusive', NOPE], provisional: () => ['inconclusive', NOPE] },
};

test('household failure matrix: classification, disposition and resolution for delete and No es mío', () => {
  const rows = [], distinct = new Set();
  for (const op of OPS) {
    for (const [name, make, expected] of RESPONSES) {
      const verdict = classifyWrite(op, make(op));
      assert.equal(verdict.kind, expected[op], `${op} / ${name}`);
      assert.ok(typeof verdict.text === 'string' && verdict.text.length > 8);
      assert.ok(!/no se (eliminó|devolvió)|borrado total/.test(verdict.text), verdict.text);
      for (const send of ['first', 'retry']) {
        const disposition = writeDisposition(verdict.kind, send === 'first');
        assert.equal(disposition, DISPOSITION[verdict.kind][send], `${op} / ${name} / ${send}`);
        rows.push({ op, response: name, send, class: verdict.kind, disposition });
        if (disposition === 'drop' || disposition === 'final') continue;
        const provisional = disposition === 'provisional';
        for (const [observation, value] of Object.entries(OBSERVATIONS)) {
          const resolved = resolveMarker(op, value, provisional, provisional ? verdict.text : '');
          const [result, text] = EXPECTED[observation][disposition](op);
          assert.equal(resolved.result, result, `${op} / ${name} / ${send} / ${observation}`);
          assert.equal(resolved.text, text === 'REASON' ? verdict.text : text);
          // Absence never claims which of delete / return / reassignment happened.
          if (observation.startsWith('absent')) assert.ok(!/eliminado|devuelto|reasignado/.test(resolved.text), resolved.text);
          rows.push({ op, response: name, send, observation, class: verdict.kind, disposition, result });
          distinct.add([op, disposition, observation, result].join('|'));
        }
      }
    }
  }
  assert.ok(rows.length >= 200, 'household matrix rows: ' + rows.length);
  if (process.env.KZA_MY_MEMORY_MATRIX_OUT) fs.writeFileSync(process.env.KZA_MY_MEMORY_MATRIX_OUT, JSON.stringify({ rows: rows.length, distinct_cases: distinct.size, entries: rows }, null, 1) + '\n');
});

test('Mis recuerdos copy: what is visible, administration sees everything, not used in answers, No es mío is not a deletion', () => {
  const view = fs.readFileSync(require.resolve('../../src/dashboard/frontend/home/nexo/my-memory.js'), 'utf8');
  for (const phrase of ['Recuerdos que la administración del hogar te asignó. Ves solo los tuyos.',
    'Las cuentas de administración del hogar ven todos los recuerdos, también estos',
    'En esta versión KZA no usa estos recuerdos para responder.',
    'vuelve a la administración y no se borra',
    'KZA no reconoce voces para asignarte recuerdos automáticamente.',
    'También se eliminan datos derivados de este recuerdo que no se muestran acá.',
    'No borra la conversación de origen (vence sola en 48 horas como máximo) ni el texto que ya se haya enviado a la nube',
    // R6: unconditional, the cascade is decided when the server deletes and no count comes back.
    'Se elimina además todo lo que dependa de este recuerdo al momento de borrarlo, aunque haya aparecido después de abrir este aviso.',
    'No se informa cuántos datos derivados se eliminaron.']) {
    assert.ok(view.includes(phrase), phrase);
  }
  for (const phrase of ['se usa en las respuestas', 'usa en sus respuestas', 'borrado total', 'nunca sale', 'Solo vos', 'Sólo vos', 'nadie más']) assert.ok(!view.includes(phrase), phrase);
  // Same sentence before acting as the server returns after (v4 person_notice).
  const api = fs.readFileSync(require.resolve('../../src/dashboard/memory_api.py'), 'utf8').replace(/"\s*\n\s*"/g, '');
  assert.ok(api.includes(RETURN_NOTICE), 'RETURN_NOTICE matches memory_api.RETURN_NOTICE');
});

const BIND_A = 'a'.repeat(32), BIND_B = 'b'.repeat(32);
const bound = (binding, subject = 'synthetic-home-a') => () => ({ ...whoami(subject), account_binding: binding });

test('R3 binding: a restored marker with the same account_binding resolves like this page\'s own (Comprobado, retry)', async () => {
  for (const op of ['delete', 'return']) {
    const storage = storageMock();
    storage.setItem(PREFIX + ID('a'), JSON.stringify({ op, subject: 'synthetic-home-a', binding: BIND_A, sent_at: 1, state: 'unresolved' }));
    const f = fixture({ storage, identity: bound(BIND_A) });
    f.read(page([item(ID('b'))]));
    await f.owner.open(); await settle();
    const state = f.owner.getState();
    assert.equal(state.notices[ID('a')], 'Comprobado: ya no figura entre tus recuerdos.', op);
    assert.deepEqual(state.markers, {}); assert.equal(storage.getItem(PREFIX + ID('a')), null);
    assert.deepEqual(f.writes(), []);
  }
  // Present after the adopt: a one-click retry is offered and resends (same grant, same person).
  const storage = storageMock();
  storage.setItem(PREFIX + ID('a'), JSON.stringify({ op: 'delete', subject: 'synthetic-home-a', binding: BIND_A, sent_at: 1, state: 'unresolved' }));
  const g = fixture({ storage, identity: bound(BIND_A) });
  g.read(page([item(ID('a'))]), page([]));
  await g.owner.open(); await settle();
  assert.equal(g.owner.getState().markers[ID('a')].restored, false);
  g.write({ deleted: true });
  const result = await g.owner.retry(ID('a')); await settle();
  assert.equal(result.applied, true); assert.deepEqual(g.writes(), ['DELETE /api/memory/mine/facts/' + ID('a')]);
});

test('R3 binding: a restored marker of another grant or person is dropped unsent; one without a binding stays restored', async () => {
  const storage = storageMock();
  storage.setItem(PREFIX + ID('a'), JSON.stringify({ op: 'delete', subject: 'synthetic-home-a', binding: BIND_B, sent_at: 1, state: 'unresolved' }));
  storage.setItem(PREFIX + ID('c'), JSON.stringify({ op: 'return', subject: 'synthetic-home-a', sent_at: 1, state: 'unresolved' }));
  const f = fixture({ storage, identity: bound(BIND_A) });
  f.read(page([item(ID('b'))]));
  await f.owner.open();
  await settle();
  const state = f.owner.getState();
  assert.equal(state.notices.session, 'Una operación de otra cuenta quedó sin confirmar; no se reintentó. Revisá tu lista.');
  assert.equal(state.notices[ID('a')], undefined, 'never said to be Comprobado or still listed');
  assert.equal(storage.getItem(PREFIX + ID('a')), null);
  assert.equal(state.notices[ID('c')], mine.RESTORED_ABSENT, 'no binding stored: the pre-binding R3 rule');
  assert.deepEqual(f.writes(), []);
  // A server that names no binding never upgrades a stored one.
  const other = storageMock();
  other.setItem(PREFIX + ID('a'), JSON.stringify({ op: 'delete', subject: 'synthetic-home-a', binding: BIND_A, sent_at: 1, state: 'unresolved' }));
  const g = fixture({ storage: other });
  g.read(page([item(ID('b'))]));
  await g.owner.open(); await settle();
  assert.equal(g.owner.getState().notices[ID('a')], mine.RESTORED_ABSENT);
  // A malformed stored binding is ignored (the marker stays restored), never matched.
  const third = storageMock();
  third.setItem(PREFIX + ID('a'), JSON.stringify({ op: 'delete', subject: 'synthetic-home-a', binding: 'A'.repeat(32), sent_at: 1, state: 'unresolved' }));
  const h = fixture({ storage: third, identity: bound('A'.repeat(32).toLowerCase()) });
  h.read(page([item(ID('b'))]));
  await h.owner.open(); await settle();
  assert.equal(h.owner.getState().notices[ID('a')], mine.RESTORED_ABSENT);
});

test('R3 binding: the stored marker carries the binding (never a person id); a later whoami with another or a malformed binding is a change', async () => {
  for (const [name, later] of [['another grant', BIND_B], ['none', null], ['malformed', 'x'.repeat(32)], ['not a string', 7]]) {
    let calls = 0, changed = 0;
    const f = fixture({ identity: () => (++calls === 1 ? bound(BIND_A)() : { ...whoami(), account_binding: later }), onAccountChanged: () => changed++ });
    f.read(page([item(ID('a'))]));
    await f.owner.open(); await settle();
    assert.equal(changed, 0, name);
    assert.deepEqual(await f.owner.remove(ID('a'), { confirmed: true }), { sent: false, accountChanged: true }, name);
    assert.equal(changed, 1, name); assert.deepEqual(f.writes(), [], name);
  }
  // The same binding: sent, and the stored marker has exactly the binding added.
  const g = fixture({ identity: bound(BIND_A) });
  g.read(page([item(ID('a'))]), page([]));
  await g.owner.open(); await settle();
  let stored = null;
  g.write(() => { stored = JSON.parse(g.storage.getItem(PREFIX + ID('a'))); return { deleted: true }; });
  assert.equal((await g.owner.remove(ID('a'), { confirmed: true })).applied, true);
  assert.deepEqual(Object.keys(stored).sort(), ['binding', 'op', 'sent_at', 'state', 'subject']);
  assert.equal(stored.binding, BIND_A);
  // A first whoami with a malformed binding is a change before anything is read.
  let changed = 0;
  const h = fixture({ identity: bound('Z'.repeat(32)), onAccountChanged: () => changed++ });
  await h.owner.open(); await settle();
  assert.equal(changed, 1); assert.deepEqual(h.calls, ['GET /api/admin/auth/whoami']);
});

test('R3 binding: the end of a session forgets the binding, so the next login of the same subject starts fresh', async () => {
  let current = BIND_A, changed = 0;
  const f = fixture({ identity: () => bound(current)(), onAccountChanged: () => changed++ });
  f.read(page([item(ID('a'))]), page([item(ID('a'))]), page([]));
  await f.owner.open(); await settle();
  await f.owner.clear(); current = BIND_B;
  await f.owner.open(); await settle();
  assert.equal(changed, 0, 'a new session is not a change of the previous one');
  f.write({ deleted: true });
  assert.equal((await f.owner.remove(ID('a'), { confirmed: true })).applied, true);
});

test('R3: a marker restored from an earlier page is never Comprobado and never resent with one click', async () => {
  // whoami names only a subject: the same subject may be another login or another bound person.
  for (const [name, list, complete, expected] of [
    ['absent, complete walk', [item(ID('b'))], true, mine.RESTORED_ABSENT],
    ['present', [item(ID('a'))], true, mine.RESTORED_STILL],
  ]) {
    for (const op of ['delete', 'return']) {
      const storage = storageMock();
      storage.setItem(PREFIX + ID('a'), JSON.stringify({ op, subject: 'synthetic-home-a', sent_at: 1, state: 'unresolved' }));
      const f = fixture({ storage });
      f.read(page(list));
      await f.owner.open();
      // Before any read: the marker is shown as restored and retry sends nothing.
      assert.equal(f.owner.getState().markers[ID('a')].restored, true, name);
      assert.deepEqual(await f.owner.retry(ID('a')), { sent: false, restored: true }, name);
      await settle();
      const state = f.owner.getState();
      assert.equal(state.notices[ID('a')], expected, `${op} / ${name}`);
      assert.ok(!/Comprobado/.test(state.notices[ID('a')]), `${op} / ${name}`);
      assert.deepEqual(state.markers, {}, `${op} / ${name}: settled without a resend`);
      assert.equal(storage.getItem(PREFIX + ID('a')), null);
      assert.deepEqual(f.writes(), [], `${op} / ${name}`);
      assert.equal(f.calls.filter(call => call.endsWith('/whoami')).length, 1, 'only the open asked whoami; the refused retry asked nothing');
    }
  }
  // Incomplete walk: still inconclusive, still no one-click retry.
  const storage = storageMock();
  storage.setItem(PREFIX + ID('a'), JSON.stringify({ op: 'delete', subject: 'synthetic-home-a', sent_at: 1, state: 'unresolved' }));
  const g = fixture({ storage, maxPages: 1 });
  g.read(page([item(ID('b'))], CUR(2), 1));
  await g.owner.open(); await settle();
  assert.equal(g.owner.getState().markers[ID('a')].restored, true);
  assert.match(g.owner.getState().notices[ID('a')], /no se pudo recorrer completa/);
  assert.deepEqual(await g.owner.retry(ID('a')), { sent: false, restored: true });
  assert.deepEqual(g.writes(), []);
  // A marker of this page (not restored) still resolves as Comprobado from a later complete read.
  for (const observation of [{ present: false, complete: true }, { present: true, complete: true }, { present: false, complete: false }]) {
    for (const op of ['delete', 'return']) {
      const restored = resolveMarker(op, observation, false, '', true);
      assert.notEqual(restored.result, 'applied'); assert.ok(!/Comprobado/.test(restored.text));
      assert.notEqual(resolveMarker(op, observation, true, 'Motivo sintético.', true).result, 'applied');
    }
  }
  assert.equal(resolveMarker('delete', { present: false, complete: true }).text, 'Comprobado: ya no figura entre tus recuerdos.');
});
