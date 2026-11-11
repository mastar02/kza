const assert = require('node:assert/strict');
const { test } = require('node:test');
const mine = require('../../src/dashboard/frontend/home/nexo/my-memory-owner.js');
const voiceOwner = require('../../src/dashboard/frontend/home/nexo/my-voice-owner.js');
const { mountMyMemory, COPY } = require('../../src/dashboard/frontend/home/nexo/my-memory.js');
const { mountMyVoice } = require('../../src/dashboard/frontend/home/nexo/my-voice.js');
const { createDocument, icon, visibleText, buttons } = require('../mocks/dom_shim.cjs');

const { createMyMemoryOwner, parsePage, PREFIX, VOICE_RETURN_NOTICE } = mine;
const ID = char => 'fact_' + char.repeat(64);
const HOUR = 3600;
const tick = () => new Promise(resolve => setImmediate(resolve));
async function settle() { for (let index = 0; index < 12; index++) await tick(); }
const pending = (id, created = 1_700_000_000 - (1_700_000_000 % HOUR), expires = created + 47 * HOUR) => ({ id, kind: 'voice_proposal_pending', created_at: created, expires_at: expires });
const item = (id, overrides = {}) => ({ id, content: 'Recuerdo sintético ' + id.slice(5, 9), category: 'preference', state: 'confirmed', created_at: 90, expires_at: null,
  owner_method: 'voice_proposal', owner_assigned_at: 95, owned_dependents: [], has_hidden_dependents: false, ...overrides });
const page = (facts = []) => ({ facts, next: null, limit: 100 });
function storageMock() {
  const data = new Map();
  return { data, get length() { return data.size; }, key: index => [...data.keys()][index] ?? null,
    getItem: key => data.has(key) ? data.get(key) : null, setItem: (key, value) => { data.set(key, String(value)); }, removeItem: key => { data.delete(key); } };
}
const PERMS = ['home:read', 'memory:own:read', 'memory:own:write', 'memory:own:voice'];
const session = (permissions = PERMS) => ({ subject: () => 'synthetic-home-a', role: () => 'home', can: permission => permissions.includes(permission) });
const whoami = () => ({ ok: true, subject: 'synthetic-home-a', role: 'home', permissions: PERMS });

function memoryFixture({ now = 1_700_000_000 } = {}) {
  const reads = [], writes = [], calls = [];
  let time = now * 1000;
  const api = (path, options = {}) => {
    const method = options.method || 'GET'; calls.push(method + ' ' + path);
    if (path === '/api/admin/auth/whoami') return Promise.resolve(whoami());
    const queue = method === 'GET' ? reads : writes, handler = queue.shift();
    if (!handler) return Promise.reject(Object.assign(Error('unscripted'), { status: 599 }));
    return Promise.resolve().then(() => typeof handler === 'function' ? handler(path, options) : handler);
  };
  const storage = storageMock();
  const owner = createMyMemoryOwner({ api, session: session(), storage, schedule: () => 0, cancel: () => {}, monotonic: () => time, wall: () => time });
  return { owner, storage, calls, read: (...h) => reads.push(...h), write: (...h) => writes.push(...h), advance: s => { time += s * 1000; } };
}

test('the content-free pending entry parses exactly; any content, category, evidence or a non-hour time fails closed', () => {
  const parsed = parsePage(page([pending(ID('a'))])).facts[0];
  assert.deepEqual({ pending: parsed.pending, content: parsed.content, category: parsed.category, method: parsed.method, owned: parsed.owned_dependents }, { pending: true, content: null, category: null, method: 'voice_proposal', owned: [] });
  const rows = [
    p => ({ ...p, content: 'x' }), p => ({ ...p, category: 'fact' }), p => ({ ...p, evidence: {} }), p => ({ ...p, state: 'proposed' }), p => ({ ...p, kind: 'voice_proposal' }),
    p => ({ ...p, created_at: p.created_at + 1 }), p => ({ ...p, expires_at: p.expires_at + 59 }), p => ({ ...p, expires_at: null }), p => ({ ...p, created_at: '0' }),
    p => ({ ...p, expires_at: p.created_at - HOUR }), p => ({ ...p, id: 'fact_x' }), p => ({ ...p, owner_method: 'voice_proposal' }), p => { const { kind, ...rest } = p; return rest; },
    p => ({ ...p, profile_id: '00000000-0000-4000-8000-00000000000a' }), p => ({ ...p, sources: [] }), p => ({ ...p, created_at: Infinity }),
  ];
  for (const mutate of rows) assert.throws(() => parsePage(page([mutate(pending(ID('a')))])), /invalid_my_memory/, String(mutate));
  assert.ok(rows.length >= 15);
  // Mixed pages: a confirmed fact keeps its full projection next to a pending entry.
  assert.equal(parsePage(page([pending(ID('a')), item(ID('b'))])).facts[1].content, 'Recuerdo sintético bbbb');
});

test('«Descartar» deletes through the person route with a content-free marker; it says «Propuesta descartada.»', async () => {
  const f = memoryFixture();
  f.read(page([pending(ID('a'))]), page([]));
  await f.owner.open(); await settle();
  f.write((path, options) => {
    assert.equal(path, '/api/memory/mine/facts/' + ID('a')); assert.equal(options.method, 'DELETE');
    assert.deepEqual(Object.keys(JSON.parse(f.storage.getItem(PREFIX + ID('a')))).sort(), ['op', 'sent_at', 'state', 'subject']);
    return { deleted: true };
  });
  assert.equal((await f.owner.remove(ID('a'), { confirmed: true })).applied, true); await settle();
  assert.equal(f.owner.getState().notices[ID('a')], 'Propuesta descartada.');
  assert.equal(f.owner.getState().returnNotice, null, 'a discard never shows a return notice');
});

test('«No es mío» on a pending entry shows the server N5 notice and keeps it after the card leaves', async () => {
  const f = memoryFixture();
  f.read(page([pending(ID('a'))]), page([]));
  await f.owner.open(); await settle();
  f.write((path, options) => { assert.equal(path, '/api/memory/mine/facts/' + ID('a') + '/return'); assert.equal(options.method, 'POST'); return { returned: true, notice: VOICE_RETURN_NOTICE }; });
  await f.owner.giveBack(ID('a'), { confirmed: true }); await settle();
  assert.equal(f.owner.getState().returnNotice, VOICE_RETURN_NOTICE);
  assert.match(VOICE_RETURN_NOTICE, /vence sola en menos de 48 horas/);
});

test('N7: a pending entry stays listed until the latest possible real expiry (its hour plus one)', async () => {
  const created = 1_700_000_000 - (1_700_000_000 % HOUR), expires = created + HOUR;
  const f = memoryFixture({ now: expires + 10 });
  f.read(page([pending(ID('a'), created, expires)]));
  await f.owner.open(); await settle();
  assert.equal(f.owner.getState().facts.length, 1, 'past the rounded expiry, inside the hour');
  f.advance(HOUR);
  assert.equal(f.owner.getState().facts.length, 0);
});

function mountMemory(f, { voiceEnabled = () => false } = {}) {
  const doc = createDocument(), container = doc.createElement('section'), dialog = doc.createElement('dialog');
  const dispose = mountMyMemory({ container, confirmDialog: dialog, owner: f.owner, icon: icon(doc), returnNotice: mine.RETURN_NOTICE, voiceReturnNotice: VOICE_RETURN_NOTICE, voiceEnabled });
  return { doc, container, dialog, dispose };
}

test('view: the pending card shows the title, hour times and exactly «Descartar» and «No es mío»; «Eliminar» only after confirmation', async () => {
  const f = memoryFixture();
  f.read(page([pending(ID('a')), item(ID('b'))]));
  const view = mountMemory(f);
  await f.owner.open(); await settle();
  const card = view.container.querySelector(`[data-my-memory-id="${ID('a')}"]`), confirmed = view.container.querySelector(`[data-my-memory-id="${ID('b')}"]`);
  assert.ok(card && confirmed);
  const text = visibleText(card);
  assert.match(text, /Propuesta de voz pendiente de revisión/);
  assert.match(text, /Todavía no se muestra el texto/);
  assert.match(text, /hora aproximada/);
  assert.match(text, /:00/);
  const format = seconds => new Date(seconds * 1000).toLocaleString('es-UY', { timeZone: 'America/Montevideo', dateStyle: 'medium', timeStyle: 'short' });
  const created = pending(ID('a')).created_at, expires = pending(ID('a')).expires_at;
  assert.ok(text.includes('Recibida: ' + format(created) + ' (hora aproximada)'), 'received at its hour');
  assert.ok(text.includes('Vence antes de: ' + format(expires + HOUR)), 'N7: the latest possible real expiry, one hour after the rounded one');
  assert.ok(!text.includes('Recuerdo sintético'), 'no content on the pending card');
  assert.deepEqual(buttons(card).map(button => button.textContent), ['Descartar', 'No es mío']);
  assert.ok(buttons(card).every(button => button.getAttribute('aria-disabled') === 'false' && !button.disabled), 'aria-disabled, never disabled');
  assert.equal(card.getAttribute('aria-label'), 'Propuesta de voz pendiente de revisión');
  assert.deepEqual(buttons(confirmed).map(button => button.textContent), ['No es mío…', 'Eliminar…']);
  // «Descartar» opens its own confirmation; cancel sends nothing; accept deletes.
  buttons(card)[0].click(); await settle();
  assert.ok(view.dialog.open);
  assert.match(visibleText(view.dialog), /Se va a borrar esta propuesta\. La administración no la va a ver\./);
  assert.ok(!visibleText(view.dialog).includes('Recuerdo sintético'));
  view.dialog.querySelector('[data-key="confirm-cancel"]').click(); await settle();
  assert.equal(view.dialog.open, false);
  assert.ok(!f.calls.some(call => call.startsWith('DELETE')));
  // «No es mío» states the N5 notice before acting.
  buttons(card)[1].click(); await settle();
  assert.match(visibleText(view.dialog), /Si no fuiste vos, la devolvemos a la administración/);
  assert.ok(visibleText(view.dialog).includes(VOICE_RETURN_NOTICE));
  view.dialog.querySelector('[data-key="confirm-cancel"]').click(); await settle();
  view.dispose();
});

test('view: a blocked pending card keeps its buttons focusable with aria-disabled and ignores clicks', async () => {
  const f = memoryFixture();
  let release;
  f.read(page([pending(ID('a'))]), page([]));
  const view = mountMemory(f);
  await f.owner.open(); await settle();
  f.write(() => new Promise(resolve => { release = () => resolve({ deleted: true }); }));
  const sending = f.owner.remove(ID('a'), { confirmed: true }); await settle();
  const card = view.container.querySelector(`[data-my-memory-id="${ID('a')}"]`);
  const retry = buttons(card).find(button => button.textContent.startsWith('Reintentar'));
  assert.ok(retry && retry.getAttribute('aria-disabled') === 'true' && !retry.disabled);
  retry.click(); await settle();
  assert.equal(f.calls.filter(call => call.startsWith('DELETE')).length, 1);
  release(); await sending; await settle();
  view.dispose();
});

test('view: the empty text keeps the stage-1 copy with the flag off and switches with the flag', async () => {
  let enabled = false;
  const f = memoryFixture();
  f.read(page([]));
  const view = mountMemory(f, { voiceEnabled: () => enabled });
  await f.owner.open(); await settle();
  assert.equal(view.container.querySelector('#my-memory-empty').textContent, COPY.empty);
  enabled = true; f.read(page([])); await f.owner.refresh(); await settle();
  assert.equal(view.container.querySelector('#my-memory-empty').textContent, COPY.emptyVoice);
  view.dispose();
});

function voiceFixture(statuses, { permissions = PERMS } = {}) {
  const calls = [], writes = [];
  const api = (path, options = {}) => {
    const method = options.method || 'GET'; calls.push(method + ' ' + path);
    if (path === '/api/admin/auth/whoami') return Promise.resolve(whoami());
    if (path === '/api/home/catalog') return Promise.resolve({ rooms: [{ room_id: 'living', name: 'Living', entity_ids: [] }] });
    if (method === 'GET') return Promise.resolve(statuses.length > 1 ? statuses.shift() : statuses[0]);
    const handler = writes.shift();
    return Promise.resolve().then(() => typeof handler === 'function' ? handler(path, options) : handler);
  };
  const owner = voiceOwner.createMyVoiceOwner({ api, session: session(permissions), storage: storageMock(), schedule: () => 0, cancel: () => {}, wall: () => 1_000_000 });
  const doc = createDocument(), container = doc.createElement('div'), dialog = doc.createElement('dialog');
  const dispose = mountMyVoice({ container, confirmDialog: dialog, owner, icon: icon(doc) });
  return { owner, container, dialog, calls, dispose, write: (...h) => writes.push(...h) };
}
const OFF = (state, revision) => ({ state, revision, enabled: false, window: null });
const ON = (state, revision, extra = {}) => ({ state, revision, enabled: true, window: null, failures_left: state === 'pending_verification' ? 2 : null, message: null, disclosure: 'x', ...extra });

test('voice view, flag off: status and revocation only (no opt-in, window or disclosure)', async () => {
  for (const [status, expectedButtons] of [[OFF('active', 2), ['Darme de baja…']], [OFF(null, 0), []], [OFF('revoked', 3), []], [OFF('pending_verification', 1), ['Darme de baja…']]]) {
    const v = voiceFixture([status]);
    await v.owner.open(); await settle();
    assert.deepEqual(buttons(v.container).map(button => button.textContent), expectedButtons, JSON.stringify(status));
    const text = visibleText(v.container);
    assert.match(text, /desactivada en este hogar/);
    assert.ok(!text.includes('48 horas'), 'no disclosure with the flag off');
    assert.ok(!v.calls.includes('GET /api/home/catalog'));
    v.dispose();
  }
});

test('voice view, flag on: opt-in dialog carries the full disclosure; pending shows rooms, failures left and the window', async () => {
  const v = voiceFixture([ON(null, 0)]);
  await v.owner.open(); await settle();
  assert.deepEqual(buttons(v.container).map(button => button.textContent), ['Inscribirme…']);
  buttons(v.container)[0].click(); await settle();
  assert.ok(v.dialog.open && visibleText(v.dialog).includes(voiceOwner.OPT_IN_DISCLOSURE));
  v.dialog.querySelector('[data-key="voice-confirm-cancel"]').click(); await settle();
  assert.ok(!v.calls.some(call => call.startsWith('POST')));
  v.dispose();
  const w = voiceFixture([ON('pending_verification', 1), ON('pending_verification', 1, { window: { room_id: 'living', phrase: 'mesa verde lluvia río', expires_at: 1060 } })]);
  await w.owner.open(); await settle();
  assert.match(visibleText(w.container), /Te quedan 2 intentos/);
  assert.ok(w.container.querySelector('#my-voice-room'));
  const open = buttons(w.container).find(button => button.textContent === 'Abrir ventana de verificación');
  assert.equal(open.getAttribute('aria-disabled'), 'false');
  w.write({ window: { room_id: 'living', phrase: 'mesa verde lluvia río', expires_at: 1060 } });
  open.click(); await settle();
  assert.match(visibleText(w.container), /Ventana abierta en Living/);
  assert.match(visibleText(w.container), /«mesa verde lluvia río»/);
  w.dispose();
  const n = voiceFixture([ON('pending_verification', 1, { window: { room_id: 'living', phrase: null, expires_at: 1060 } })]);
  await n.owner.open(); await settle();
  assert.match(visibleText(n.container), /todavía no está disponible en esta versión/);
  n.dispose();
});

test('voice view: a person without the voice grant sees no voice section', async () => {
  const v = voiceFixture([ON(null, 0)], { permissions: ['home:read', 'memory:own:read', 'memory:own:write'] });
  await v.owner.open(); await settle();
  assert.equal(v.container.querySelector('.memory-voice').hidden, true);
  assert.deepEqual(v.calls, []);
  v.dispose();
});

test('voice view: a control marked aria-disabled ignores clicks (no dialog while a write is in flight)', async () => {
  const v = voiceFixture([ON(null, 0)]);
  await v.owner.open(); await settle();
  let release;
  v.write(() => new Promise(resolve => { release = () => resolve({ state: 'pending_verification', revision: 1 }); }));
  const sending = v.owner.requestConsent({ confirmed: true }); await settle();
  const request = buttons(v.container).find(button => button.textContent === 'Inscribirme…');
  assert.ok(request && request.getAttribute('aria-disabled') === 'true' && !request.disabled, 'focusable, aria-disabled');
  request.click(); await settle();
  assert.equal(v.dialog.open, false, 'an aria-disabled control opens nothing');
  release(); await sending; await settle();
  v.dispose();
});

test('T/K: a locked person\'s revoke dialog keeps the «sigue bloqueada» warning; the flag-off line offers baja only with a consent', async () => {
  const v = voiceFixture([ON('verification_locked', 3)]);
  await v.owner.open(); await settle();
  buttons(v.container).find(button => button.textContent === 'Darme de baja…').click(); await settle();
  assert.match(visibleText(v.dialog), /la verificación sigue bloqueada hasta que la administración la habilite/);
  v.dialog.querySelector('[data-key="voice-confirm-cancel"]').click(); await settle();
  v.dispose();
  const w = voiceFixture([ON('active', 2)]);
  await w.owner.open(); await settle();
  buttons(w.container).find(button => button.textContent === 'Darme de baja…').click(); await settle();
  assert.ok(!visibleText(w.dialog).includes('sigue bloqueada'));
  w.dialog.querySelector('[data-key="voice-confirm-cancel"]').click(); await settle();
  w.dispose();
  for (const [status, offers] of [[OFF('active', 2), true], [OFF(null, 0), false], [OFF('revoked', 3), false]]) {
    const x = voiceFixture([status]);
    await x.owner.open(); await settle();
    assert.equal(x.container.querySelector('#my-voice-status').textContent.includes('Podés darte de baja.'), offers, JSON.stringify(status));
    x.dispose();
  }
});
