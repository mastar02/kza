const assert = require('node:assert/strict');
const { test } = require('node:test');
const admin = require('../../src/dashboard/frontend/home/nexo/voice-consents-owner.js');
const { mountVoiceConsents } = require('../../src/dashboard/frontend/home/nexo/voice-consents.js');
const { createDocument, icon, visibleText, buttons } = require('../mocks/dom_shim.cjs');

const { createVoiceConsentsOwner, parseSwitch, parseConsents, classifyWrite, resolveMarker, PREFIX, VOICES_DISCLOSURE, MULTI_PROFILE_NOTICE } = admin;
const tick = () => new Promise(resolve => setImmediate(resolve));
async function settle() { for (let index = 0; index < 12; index++) await tick(); }
const deferred = () => { let resolve, reject; const promise = new Promise((yes, no) => { resolve = yes; reject = no; }); return { promise, resolve, reject }; };
const PA = 'user_' + 'a'.repeat(32), PB = 'user_' + 'b'.repeat(32), PC = 'user_' + 'c'.repeat(32);
const P1 = '00000000-0000-4000-8000-00000000000a', P2 = '00000000-0000-4000-8000-00000000000b';
const SWITCH = (enabled = true, revision = 1, configured = true) => ({ configured, enabled, revision, effective: configured && enabled });
const row = (person, state, revision, verified = null, linked = 1) => ({ person_id: person, state, revision, changed_at: 1, verified_profile: verified, linked_live_profiles: linked });
const LIST = (overrides = {}) => ({ consents: [row(PA, 'active', 2, { id: P1, name: 'Voz de A' }), row(PB, 'verification_locked', 3), row(PC, 'pending_verification', 1, null, 2)],
  suspended_profiles: [{ profile_id: P2, revision: 1 }], multiple_profiles: [{ person_id: PC, linked_live_profiles: 2 }], truncated: false,
  multiple_profiles_notice: MULTI_PROFILE_NOTICE, disclosure: VOICES_DISCLOSURE, ...overrides });
const USERS = [{ id: PA, name: 'Ana', permission_level: 2, is_active: true }, { id: PB, name: 'Beto', permission_level: 2, is_active: true }];
const OPERATOR = ['memory:read', 'memory:review', 'memory:assign', 'admin:manage', 'voices:read', 'voices:review'];
function storageMock() {
  const data = new Map();
  return { data, get length() { return data.size; }, key: index => [...data.keys()][index] ?? null,
    getItem: key => data.has(key) ? data.get(key) : null, setItem: (key, value) => { data.set(key, String(value)); }, removeItem: key => { data.delete(key); } };
}
const session = (permissions = OPERATOR, role = 'operator', subject = 'synthetic-operator') => ({ subject: () => subject, role: () => role, can: permission => permissions.includes(permission) });
const whoami = (subject = 'synthetic-operator', role = 'operator', permissions = OPERATOR) => ({ ok: true, subject, role, permissions });
const fail = (status, detail) => () => { throw Object.assign(Error('http'), { status, detail }); };
const network = () => { throw new TypeError('Failed to fetch'); };

function fixture({ session: current = session(), identity = () => whoami(), storage = storageMock(), onAccountChanged = () => {} } = {}) {
  const calls = [], bodies = [], writes = [], switches = [], lists = [];
  const api = (path, options = {}) => {
    const method = options.method || 'GET'; calls.push(method + ' ' + path);
    if (path === '/api/admin/auth/whoami') return Promise.resolve().then(identity);
    if (path === '/api/admin/users') return Promise.resolve(USERS);
    if (method === 'GET') {
      const queue = path === '/api/memory/voice/switch' ? switches : lists, handler = queue.length > 1 ? queue.shift() : queue[0];
      if (!handler) return Promise.reject(Object.assign(Error('unscripted'), { status: 599 }));
      return new Promise((resolve, reject) => {
        options.signal?.addEventListener('abort', () => reject(Object.assign(Error('aborted'), { name: 'AbortError' })));
        Promise.resolve().then(() => typeof handler === 'function' ? handler() : handler).then(resolve, reject);
      });
    }
    assert.equal(options.signal, undefined, 'writes are never abortable');
    bodies.push(options.body);
    const handler = writes.shift();
    if (!handler) return Promise.reject(Object.assign(Error('unscripted'), { status: 599 }));
    return Promise.resolve().then(() => typeof handler === 'function' ? handler(path, options) : handler);
  };
  const owner = createVoiceConsentsOwner({ api, session: current, storage, onAccountChanged, wall: () => 1000 });
  return { owner, calls, bodies, storage, switches, lists, write: (...h) => writes.push(...h), writes: () => calls.filter(call => !call.startsWith('GET')) };
}

test('parseSwitch and parseConsents accept the 4a-2 shapes and fail closed on anything else', () => {
  assert.deepEqual(parseSwitch(SWITCH()), { configured: true, enabled: true, effective: true, revision: 1 });
  for (const value of [null, { ...SWITCH(), extra: 1 }, { ...SWITCH(), effective: false }, { ...SWITCH(), revision: -1 }, { ...SWITCH(), enabled: 'yes' },
    { ...SWITCH(true, 1, false), effective: true }]) assert.throws(() => parseSwitch(value), /invalid_voice_admin/);
  const parsed = parseConsents(LIST());
  assert.equal(parsed.consents[0].verified.name, 'Voz de A'); assert.equal(parsed.truncated, false);
  const rows = [
    l => ({ ...l, extra: 1 }), l => ({ ...l, truncated: 'no' }), l => { const { truncated, ...rest } = l; return rest; }, l => ({ ...l, consents: {} }),
    l => ({ ...l, consents: [{ ...l.consents[0], score: 0.9 }] }), l => ({ ...l, consents: [{ ...l.consents[0], attempts: [] }] }),
    l => ({ ...l, consents: [{ ...l.consents[0], state: 'x' }] }), l => ({ ...l, consents: [{ ...l.consents[0], person_id: 'a b' }] }),
    l => ({ ...l, consents: [l.consents[0], l.consents[0]] }), l => ({ ...l, consents: [{ ...l.consents[0], verified_profile: { id: 'x', name: 'y' } }] }),
    l => ({ ...l, consents: [{ ...l.consents[0], verified_profile: { id: P1, name: 'y', revision: 1 } }] }), l => ({ ...l, consents: [{ ...l.consents[0], linked_live_profiles: -1 }] }),
    l => ({ ...l, suspended_profiles: [{ profile_id: P2, revision: 0 }] }), l => ({ ...l, suspended_profiles: [{ profile_id: P2, revision: 1, at: 1 }] }),
    l => ({ ...l, suspended_profiles: [{ profile_id: P2, revision: 1 }, { profile_id: P2, revision: 1 }] }), l => ({ ...l, multiple_profiles: [{ person_id: PC, linked_live_profiles: 1 }] }),
    l => ({ ...l, consents: Array.from({ length: 257 }, (_, index) => row('user_' + String(index).padStart(32, '0'), 'active', 1)) }), l => ({ ...l, disclosure: null }),
  ];
  for (const mutate of rows) assert.throws(() => parseConsents(mutate(LIST())), /invalid_voice_admin/, String(mutate));
  assert.equal(parseConsents(LIST({ consents: Array.from({ length: 256 }, (_, index) => row('user_' + String(index).padStart(32, '0'), 'active', 1)), truncated: true })).truncated, true);
});

test('classifyWrite and resolveMarker: refusals are final, 401/403 uncertain, only the right revision step is Comprobado', () => {
  const rows = [
    ['switch', { value: SWITCH(false, 2) }, 'applied'], ['switch', { value: { enabled: false } }, 'uncertain'], ['resume', { value: { resumed: true } }, 'applied'],
    ['resume', { value: { resumed: false } }, 'uncertain'], ['reopen', { value: { state: 'pending_verification', revision: 4 } }, 'applied'], ['reopen', { value: {} }, 'uncertain'],
    ['switch', { error: { status: 409, detail: 'memory_voice_not_configured' } }, 'refused'], ['reopen', { error: { status: 409, detail: 'memory_voice_consent_state' } }, 'refused'],
    ['resume', { error: { status: 409, detail: 'memory_voice_conflict' } }, 'refused'], ['resume', { error: { status: 422, detail: 'memory_voice_invalid_request' } }, 'refused'],
    ['switch', { error: { status: 403, detail: 'memory_voice_switch_denied' } }, 'uncertain'], ['reopen', { error: { status: 401 } }, 'uncertain'],
    ['resume', { error: new TypeError('x') }, 'uncertain'], ['resume', { error: { sent: false } }, 'not_applied'], ['resume', { error: { answered: true } }, 'uncertain'],
  ];
  for (const [op, outcome, kind] of rows) assert.equal(classifyWrite(op, outcome).kind, kind, JSON.stringify([op, outcome]));
  const list = parseConsents(LIST());
  assert.equal(resolveMarker({ op: 'switch', expected_revision: 1, enabled: false }, { switch: { ...parseSwitch(SWITCH(false, 2)) }, list }).result, 'applied');
  assert.equal(resolveMarker({ op: 'switch', expected_revision: 1, enabled: false, restored: true }, { switch: parseSwitch(SWITCH(false, 2)), list }).result, 'final');
  assert.match(resolveMarker({ op: 'switch', expected_revision: 1, enabled: false }, { switch: parseSwitch(SWITCH(true, 1)), list }).text, /Todavía no cambió/);
  assert.equal(resolveMarker({ op: 'reopen', target: PB, expected_revision: 2 }, { switch: null, list: parseConsents(LIST({ consents: [row(PB, 'pending_verification', 3)] })) }).result, 'applied');
  assert.equal(resolveMarker({ op: 'reopen', target: PB, expected_revision: 3 }, { switch: null, list }).result, 'final');
  assert.notEqual(resolveMarker({ op: 'reopen', target: PB, expected_revision: 2, restored: true }, { switch: null, list: parseConsents(LIST({ consents: [row(PB, 'pending_verification', 3)] })) }).result, 'applied',
    'a restored reopen marker is never Comprobado');
  assert.equal(resolveMarker({ op: 'resume', target: P2, expected_revision: 1 }, { switch: null, list: parseConsents(LIST({ suspended_profiles: [] })) }).result, 'applied');
  assert.equal(resolveMarker({ op: 'resume', target: P2, expected_revision: 1 }, { switch: null, list: null }).result, 'pending');
});

test('open: whoami (operator with memory:assign and memory:read), switch, consents, names', async () => {
  const f = fixture();
  f.switches.push(SWITCH()); f.lists.push(LIST());
  await f.owner.open(); await settle();
  assert.deepEqual(f.calls, ['GET /api/admin/auth/whoami', 'GET /api/memory/voice/switch', 'GET /api/memory/voice/consents', 'GET /api/admin/users']);
  const state = f.owner.getState();
  assert.equal(state.list.consents[0].name, 'Ana'); assert.equal(state.list.consents[2].name, 'Persona sin nombre visible');
  assert.equal(f.owner.profileNote(P1), 'Verificada por Ana');
  assert.equal(f.owner.profileNote(P2), 'Propuestas por voz en pausa');
  assert.equal(f.owner.profileNote('00000000-0000-4000-8000-0000000000ff'), null);
});

test('with the settings flag off the switch still answers and the list says why it is missing', async () => {
  const f = fixture();
  f.switches.push(SWITCH(true, 1, false)); f.lists.push(fail(409, 'memory_voice_disabled'));
  await f.owner.open(); await settle();
  const state = f.owner.getState();
  assert.equal(state.list, null); assert.equal(state.listReason, 'disabled'); assert.equal(state.read.error, null);
  assert.equal(f.owner.profileNote(P1), null);
  // Switching on without the settings flag is never sent; switching off is.
  assert.deepEqual(await f.owner.setSwitch(true, { confirmed: true }), { sent: false });
  f.write(SWITCH(false, 2, false));
  assert.equal((await f.owner.setSwitch(false, { confirmed: true })).applied, true);
  assert.deepEqual(f.bodies.at(-1), { enabled: false, expected_revision: 1 });
});

test('reopen, resume and switch: dialog confirmation, whoami first, CAS body, content-free markers', async () => {
  const f = fixture();
  f.switches.push(SWITCH()); f.lists.push(LIST());
  await f.owner.open(); await settle();
  for (const call of [() => f.owner.reopen(PB), () => f.owner.resume(P2), () => f.owner.setSwitch(false)]) assert.deepEqual(await call(), { sent: false });
  assert.deepEqual(await f.owner.reopen(PA, { confirmed: true }), { sent: false }, 'only a locked consent');
  assert.deepEqual(await f.owner.resume(P1, { confirmed: true }), { sent: false }, 'only a suspended profile');
  f.write((path, options) => {
    assert.equal(path, `/api/memory/voice/consents/${PB}/reopen`); assert.deepEqual(options.body, { expected_revision: 3 });
    const stored = JSON.parse(f.storage.getItem(PREFIX + 'reopen:' + PB));
    assert.deepEqual(Object.keys(stored).sort(), ['expected_revision', 'op', 'sent_at', 'subject']);
    return { state: 'pending_verification', revision: 4 };
  });
  const before = f.calls.length;
  assert.equal((await f.owner.reopen(PB, { confirmed: true })).applied, true);
  assert.equal(f.calls[before], 'GET /api/admin/auth/whoami');
  f.write((path, options) => { assert.equal(path, `/api/voices/profiles/${P2}/attribution/resume`); assert.deepEqual(options.body, { expected_revision: 1 }); return { resumed: true }; });
  assert.equal((await f.owner.resume(P2, { confirmed: true })).applied, true);
  assert.ok(![...f.storage.data.values()].some(value => value.includes('Ana') || value.includes('Voz de A')), 'no names in storage');
});

test('an uncertain write keeps a marker until a read that started after the loss; 403 never reads as a denial', async () => {
  const f = fixture();
  const later = deferred();
  f.switches.push(SWITCH(true, 1), () => later.promise); f.lists.push(LIST());
  await f.owner.open(); await settle();
  f.write(fail(403, 'dashboard_permission_denied'));
  assert.deepEqual(await f.owner.setSwitch(false, { confirmed: true }), { uncertain: true });
  assert.match(f.owner.getState().notices['switch:switch'], /^Sin confirmar\./);
  assert.ok(f.storage.getItem(PREFIX + 'switch:switch'));
  later.resolve(SWITCH(false, 2)); await settle();
  assert.equal(f.owner.getState().notices['switch:switch'], 'Comprobado: quedó apagada.');
  assert.equal(f.storage.getItem(PREFIX + 'switch:switch'), null);
});

test('accounts: household, evaluator or an operator without memory:assign never read; a changed whoami sends nothing', async () => {
  for (const current of [session(['home:read', 'memory:own:voice'], 'home'), session(['evaluation:read'], 'evaluator'), session(['memory:read', 'voices:read'])]) {
    const f = fixture({ session: current });
    await f.owner.open(); await settle();
    assert.deepEqual(f.calls, []);
  }
  for (const identity of [() => whoami('synthetic-other'), () => whoami('synthetic-operator', 'home'), () => whoami('synthetic-operator', 'operator', ['memory:read'])]) {
    let changed = 0;
    const f = fixture({ identity, onAccountChanged: () => changed++ });
    await f.owner.open(); await settle();
    assert.equal(changed, 1); assert.deepEqual(f.calls, ['GET /api/admin/auth/whoami']);
  }
  const storage = storageMock();
  storage.setItem(PREFIX + 'resume:' + P2, JSON.stringify({ op: 'resume', subject: 'synthetic-other', sent_at: 1, expected_revision: 1 }));
  const g = fixture({ storage });
  g.switches.push(SWITCH()); g.lists.push(LIST());
  await g.owner.open(); await settle();
  assert.equal(storage.getItem(PREFIX + 'resume:' + P2), null);
  assert.deepEqual(g.writes(), []);
});

test('panel: disclosure, verified profile, multi-profile notice, truncated note, reopen only for locked, resume, 44px-ready buttons', async () => {
  const f = fixture();
  f.switches.push(SWITCH()); f.lists.push(LIST({ truncated: true }));
  const doc = createDocument(), container = doc.createElement('div'), dialog = doc.createElement('dialog');
  const dispose = mountVoiceConsents({ container, confirmDialog: dialog, owner: f.owner, icon: icon(doc) });
  await f.owner.open(); await settle();
  const text = visibleText(container);
  assert.ok(text.includes(VOICES_DISCLOSURE));
  assert.match(text, /Verificada con «Voz de A»/);
  assert.ok(text.includes(MULTI_PROFILE_NOTICE));
  assert.match(text, /Se muestran las primeras 256 inscripciones/);
  assert.match(text, /Interruptor encendido · funcionando/);
  const labels = buttons(container).map(button => button.textContent);
  assert.deepEqual(labels, ['Actualizar', 'Apagar…', 'Habilitar un nuevo intento…', 'Reanudar…']);
  assert.ok(buttons(container).every(button => !button.disabled));
  assert.equal(container.querySelectorAll('[data-voice-consent]').length, 3);
  // Reopen: dialog names the person; cancel sends nothing.
  buttons(container).find(button => button.textContent.startsWith('Habilitar')).click(); await settle();
  assert.ok(dialog.open); assert.match(visibleText(dialog), /^Habilitar un nuevo intentoBeto va a poder volver a verificar su voz/);
  dialog.querySelector('[data-key="voice-admin-cancel"]').click(); await settle();
  assert.deepEqual(f.writes(), []);
  // Switch off through the dialog.
  f.write(SWITCH(false, 2));
  buttons(container).find(button => button.textContent === 'Apagar…').click(); await settle();
  assert.match(visibleText(dialog), /se cierran las ventanas de verificación abiertas/);
  dialog.querySelector('[data-key="voice-admin-accept"]').click(); await settle();
  assert.deepEqual(f.writes(), ['POST /api/memory/voice/switch']);
  dispose();
});

test('panel: with the flag off the list is replaced by the reason; no scores or attempts are ever shown', async () => {
  const f = fixture();
  f.switches.push(SWITCH(true, 1, false)); f.lists.push(fail(409, 'memory_voice_disabled'));
  const doc = createDocument(), container = doc.createElement('div'), dialog = doc.createElement('dialog');
  const dispose = mountVoiceConsents({ container, confirmDialog: dialog, owner: f.owner, icon: icon(doc) });
  await f.owner.open(); await settle();
  const text = visibleText(container);
  assert.match(text, /no habilita la memoria por voz/);
  assert.match(text, /Interruptor encendido · sin efecto/);
  assert.ok(!/puntaje|score|intentos fallidos:/i.test(text));
  assert.deepEqual(buttons(container).map(button => button.textContent), ['Actualizar', 'Apagar…']);
  dispose();
});

test('profileNote: «Verificada por» only for an active consent; the pause only for a suspended profile', async () => {
  const f = fixture();
  f.switches.push(SWITCH());
  f.lists.push(LIST({ consents: [row(PA, 'active', 2, { id: P1, name: 'Voz de A' }), row(PB, 'revoked', 4, { id: P2, name: 'Voz de B' })], multiple_profiles: [] }));
  await f.owner.open(); await settle();
  assert.equal(f.owner.profileNote(P1), 'Verificada por Ana');
  assert.equal(f.owner.profileNote(P2), 'Propuestas por voz en pausa', 'a revoked consent names nobody');
  assert.equal(f.owner.profileNote('00000000-0000-4000-8000-00000000000c'), null);
});

test('panel: a control marked aria-disabled ignores clicks while its write is in flight', async () => {
  const f = fixture();
  f.switches.push(SWITCH()); f.lists.push(LIST());
  const doc = createDocument(), container = doc.createElement('div'), dialog = doc.createElement('dialog');
  const dispose = mountVoiceConsents({ container, confirmDialog: dialog, owner: f.owner, icon: icon(doc) });
  await f.owner.open(); await settle();
  const hold = deferred();
  f.write(() => hold.promise);
  const sending = f.owner.setSwitch(false, { confirmed: true }); await settle();
  const control = buttons(container).find(button => button.dataset.key === 'voice-switch');
  assert.ok(control && control.getAttribute('aria-disabled') === 'true' && !control.disabled);
  control.click(); await settle();
  assert.equal(dialog.open, false, 'an aria-disabled control opens nothing');
  hold.resolve(SWITCH(false, 2)); await sending; await settle();
  dispose();
});

test('S: pagehide (wipe false) keeps an uncertain admin marker; the next load resolves it without «Comprobado»; the end of the session wipes it', async () => {
  const storage = storageMock();
  const f = fixture({ storage });
  f.switches.push(SWITCH(true, 1)); f.lists.push(LIST());
  await f.owner.open(); await settle();
  f.write(fail(403, 'dashboard_permission_denied'));
  assert.deepEqual(await f.owner.setSwitch(false, { confirmed: true }), { uncertain: true });
  await f.owner.clear({ wipe: false });
  assert.ok(storage.getItem(PREFIX + 'switch:switch'), 'the marker survives a pagehide');
  assert.equal(storage.getItem(admin.ENDED_KEY), null);
  const g = fixture({ storage });
  g.switches.push(SWITCH(false, 2)); g.lists.push(LIST());
  await g.owner.open(); await settle();
  assert.ok(g.owner.getState().notices['switch:switch'] && !/Comprobado/.test(g.owner.getState().notices['switch:switch']), g.owner.getState().notices['switch:switch']);
  assert.deepEqual(g.writes(), [], 'never resent');
  const h = fixture({ storage });
  h.switches.push(SWITCH(true, 2)); h.lists.push(LIST());
  await h.owner.open(); await settle();
  h.write(network);
  await h.owner.reopen(PB, { confirmed: true });
  await h.owner.clear();
  assert.deepEqual([...storage.data.keys()].filter(key => key.startsWith(PREFIX)), [], 'the end of the session wipes every marker');
  assert.equal(storage.getItem(admin.ENDED_KEY), '1');
  const k = fixture({ storage });
  k.switches.push(SWITCH(true, 2)); k.lists.push(LIST());
  await k.owner.open(); await settle();
  assert.match(k.owner.getState().notices.session, /quedó sin confirmar al terminar la sesión anterior/);
});

test('T: a restored resume marker is never «Comprobado»; the switch needs admin:manage', async () => {
  const gone = parseConsents(LIST({ suspended_profiles: [] }));
  assert.notEqual(resolveMarker({ op: 'resume', target: P2, expected_revision: 1, restored: true }, { switch: null, list: gone }).result, 'applied');
  assert.ok(!/Comprobado/.test(resolveMarker({ op: 'resume', target: P2, expected_revision: 1, restored: true }, { switch: null, list: gone }).text));
  const f = fixture({ session: session(OPERATOR.filter(permission => permission !== 'admin:manage')) });
  assert.equal(f.owner.capabilities().toggle, false);
  assert.equal(fixture().owner.capabilities().toggle, true);
  f.switches.push(SWITCH()); f.lists.push(LIST());
  await f.owner.open(); await settle();
  assert.deepEqual(await f.owner.setSwitch(false, { confirmed: true }), { sent: false });
  assert.deepEqual(f.writes(), []);
});
