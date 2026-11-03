/* Evaluator transport, payload and native media ownership regressions. */
const assert = require('node:assert/strict');
const { test } = require('node:test');
global.KzaLivePrivacy = require('../../src/dashboard/frontend/obs/src/live-privacy.js');
global.KzaConversationChannel = require('../../src/dashboard/frontend/obs/src/conversation-channel.js');
global.KzaVoicePrivacy = require('../../src/dashboard/frontend/obs/src/voice-privacy.js');
const { createEvaluationOwner, permittedPath, validateEnvelope } = require('../../src/dashboard/frontend/home/nexo/evaluation-owner.js');
const { createSession } = require('../../src/dashboard/frontend/home/nexo/state.js');
const ID = '10000000-0000-4000-8000-000000000001';
const NONCE = '20000000-0000-4000-8000-000000000002';
const EVENT = '30000000-0000-4000-8000-000000000003';
const TICKET = '40000000-0000-4000-8000-000000000004';
const FP = 'vf_' + 'a'.repeat(64);
const RESET = { type: 'live_reset', payload: { id: ID, nonce: NONCE }, ts: '' };
const READY = { type: 'rest_ready', connection_id: ID, channels: ['evaluation:sample'], ts: '' };
const deferred = () => { let resolve; const promise = new Promise(done => { resolve = done; }); return { promise, resolve }; };
const turn = async () => { for (let index = 0; index < 16; index++) await new Promise(resolve => setImmediate(resolve)); };
class Socket extends EventTarget {
  constructor() { super(); this.readyState = 0; this.sent = []; }
  event(type, value) { const event = new Event(type); if (value !== undefined) event.data = JSON.stringify(value); this.dispatchEvent(event); }
  open() { this.readyState = 1; this.event('open'); }
  send(value) { assert.equal(this.readyState, 1); this.sent.push(JSON.parse(value)); }
  close() { this.readyState = 3; this.event('close'); }
}
class Player extends EventTarget {
  constructor() {
    super(); this.isConnected = true; this.paused = false; this.networkState = 1; this.readyState = 2;
    this.currentSrc = 'blob:synthetic'; this.srcObject = null; this.buffered = { length: 1 }; this.duration = .8;
    this.attributes = new Set(['src']);
  }
  pause() { this.paused = true; }
  hasAttribute(value) { return this.attributes.has(value); }
  removeAttribute(value) { this.attributes.delete(value); }
  querySelector() { return null; }
  remove() { this.isConnected = false; }
  load() { /* A native decoder is deliberately retained until empty() below. */ }
  empty() {
    this.readyState = this.networkState = 0; this.buffered = { length: 0 }; this.duration = NaN;
    this.dispatchEvent(new Event('emptied'));
  }
}
function wav() {
  const size = Math.round(.8 * 16000) * 2, value = Buffer.alloc(size + 44);
  value.write('RIFF', 0); value.writeUInt32LE(size + 36, 4); value.write('WAVEfmt ', 8);
  value.writeUInt32LE(16, 16); value.writeUInt16LE(1, 20); value.writeUInt16LE(1, 22);
  value.writeUInt32LE(16000, 24); value.writeUInt32LE(32000, 28); value.writeUInt16LE(2, 32);
  value.writeUInt16LE(16, 34); value.write('data', 36); value.writeUInt32LE(size, 40); return value.toString('base64');
}
function catalogue() {
  return { observed_at: 1000, expires_at: 1100,
    grants: [{ fingerprint_id: FP, modalities: ['audio_preview', 'transcript_full'], expires_at: 1100 }] };
}
function sample(mode = 'audio_preview') {
  const row = mode === 'audio_preview'
    ? { id: FP, modality: mode, audio_mime: 'audio/wav', clip_duration_s: .8, audio_base64: wav() }
    : { id: FP, modality: mode, transcript_scope: 'full_segment', transcript: 'Synthetic granted text' };
  return { connection_id: ID, channel: 'evaluation:sample', snapshot: { event_id: EVENT, observed_at: 1000,
    expires_at: 1050, receipt: { ticket: TICKET, nonce: NONCE }, channel: 'evaluation:sample', rows: [row] } };
}

test('transcripts use backend codepoint and UTF-8 limits, including supplementary characters', () => {
  const selection = { fingerprint_id: FP, modality: 'transcript_full', expires_at: 1100 };
  for (const transcript of ['a'.repeat(8192), '😀'.repeat(4097), '😀'.repeat(8192)]) {
    const value = sample('transcript_full'); value.snapshot.rows[0].transcript = transcript;
    assert.doesNotThrow(() => validateEnvelope(value, selection));
  }
  for (const transcript of ['a'.repeat(8193), '😀'.repeat(8193), '\ud800', '\udfff']) {
    const value = sample('transcript_full'); value.snapshot.rows[0].transcript = transcript;
    assert.throws(() => validateEnvelope(value, selection));
  }
});
function setup(options = {}) {
  let now = 1000000, serial = 0;
  const session = createSession({ wall: () => now, monotonic: () => now });
  session.accept({ ok: true, role: 'evaluator', expires_at: 2000, permissions: ['evaluation:read'] });
  const sockets = [], controllers = [], requests = [], timers = new Map(), revoked = [], allocated = [], mounted = new Set();
  const url = { createObjectURL() { const value = 'blob:' + ++serial; allocated.push(value); return value; }, revokeObjectURL(value) { revoked.push(value); } };
  const owner = createEvaluationOwner({ session, monotonic: () => now, wall: () => now, url, pageId: () => ID,
    schedule(callback, delay) { const id = ++serial; timers.set(id, { callback, delay }); return id; }, cancel(id) { timers.delete(id); },
    connect() { const socket = new Socket(); sockets.push(socket); return socket; }, socketUrl: () => 'ws://synthetic/ws/evaluation',
    mount(controller) {
      controllers.push(controller); mounted.add(controller);
      const stop = controller.store.subscribe(() => controller.store.committed(controller.store.snapshot().length ? controller.store.deliveryToken() : null));
      return () => { stop(); mounted.delete(controller); controller.store.detachMedia(); controller.store.committed(null); };
    },
    fetch: async (requestPath, init) => {
      requests.push([requestPath, init]);
      return { ok: true, status: 200, json: async () => requestPath === '/api/evaluation/grants'
        ? catalogue() : sample(requestPath.includes('transcript_full') ? 'transcript_full' : 'audio_preview') };
    }, ...options });
  return { session, sockets, controllers, requests, timers, revoked, allocated, mounted, owner,
    setNow(value) { now = value; } };
}
async function ready(g) {
  assert.equal(await g.owner.open(), true); const socket = g.sockets.at(-1); socket.open();
  socket.event('message', RESET); await turn(); assert.equal(socket.sent.at(-1).type, 'live_reset_ack');
  socket.event('message', READY); await turn(); return socket;
}

test('the transport is GET-only and permits only exact evaluator routes', () => {
  assert.equal(permittedPath('/api/evaluation/grants'), true);
  assert.equal(permittedPath('/api/evaluation/samples/' + FP + '?modality=audio_preview'), true);
  for (const route of ['/api/home/catalog', '/api/admin/users', '/api/voices', '/api/conversations',
    '/api/evaluation/grants?all=true', '/api/evaluation/samples/' + FP + '?modality=audio_preview&modality=transcript_full']) assert.equal(permittedPath(route), false);
  assert.equal(permittedPath('/api/evaluation/grants', 'POST'), false);
});
test('evaluator role, not the operator permission union, precedes socket and fetch', async () => {
  for (const role of ['home', 'operator']) {
    const g = setup(); g.session.accept({ ok: true, role, expires_at: 2000, permissions: ['evaluation:read'] });
    assert.equal(await g.owner.open(), false); assert.equal(g.sockets.length, 0); assert.equal(g.requests.length, 0);
    await g.owner.leave();
  }
});
test('protocol3 reset/ready precedes automatic catalogue and never auto-downloads payload', async () => {
  const g = setup(); await g.owner.open(); const socket = g.sockets[0]; socket.open();
  assert.equal(socket.sent[0].version, 3); socket.event('message', READY); await turn(); assert.equal(g.requests.length, 0);
  socket.event('message', RESET); await turn();
  socket.event('message', { ...READY, channels: ['voices:review', 'evaluation:sample'] }); await turn(); assert.equal(g.requests.length, 0);
  socket.event('message', READY); await turn();
  assert.deepEqual(g.requests.map(([route]) => route), ['/api/evaluation/grants']);
  assert.equal(g.controllers.at(-1).snapshot().grants[0].fingerprint_id, FP);
  await g.owner.leave();
});
test('one exact modality is validated before any Blob allocation', async () => {
  const g = setup(); await ready(g); const controller = g.controllers.at(-1);
  assert.equal(controller.load(), true); await turn();
  assert.equal(g.allocated.length, 1); assert.equal(controller.store.snapshot()[0].modality, 'audio_preview');
  const mixed = sample('transcript_full'); mixed.snapshot.rows[0].audio_base64 = wav();
  assert.throws(() => validateEnvelope(mixed, { fingerprint_id: FP, modality: 'transcript_full', expires_at: 1100 }));
  const extra = sample(); extra.snapshot.rows.push({ ...extra.snapshot.rows[0] });
  assert.throws(() => validateEnvelope(extra, { fingerprint_id: FP, modality: 'audio_preview', expires_at: 1100 }));
  await g.owner.leave(); assert.deepEqual(g.revoked, g.allocated);
});
test('ACK waits for actual decoder emptying after pause, detach and Blob revocation', async () => {
  const g = setup(), socket = await ready(g), controller = g.controllers.at(-1);
  controller.load(); await turn(); const player = new Player(); controller.store.attachMedia(player);
  const before = socket.sent.length; controller.clearSample(); await turn();
  assert.equal(player.paused, true); assert.equal(player.isConnected, false); assert.deepEqual(g.revoked, g.allocated);
  assert.equal(socket.sent.length, before);
  player.empty(); await turn(); assert.equal(socket.sent.at(-1).type, 'live_ack');
  assert.equal(socket.sent.at(-1).receipts[0].ticket, TICKET); await g.owner.leave();
});
test('reset rebuilds the selector only after old decoder retirement, without a timer ACK', async () => {
  const g = setup(), socket = await ready(g), controller = g.controllers.at(-1);
  controller.load(); await turn(); const player = new Player(); controller.store.attachMedia(player);
  const before = socket.sent.length; socket.event('message', RESET); await turn();
  assert.equal(g.mounted.size, 0); assert.equal(socket.sent.length, before);
  player.empty(); await turn(); assert.equal(socket.sent.at(-1).type, 'live_reset_ack');
  socket.event('message', READY); await turn(); assert.equal(g.controllers.at(-1).snapshot().status, 'real');
  assert.equal(g.requests.filter(([route]) => route === '/api/evaluation/grants').length, 2); await g.owner.leave();
});
test('late JSON is drained before a new login can open; the old connection cannot ACK it', async () => {
  const body = deferred(), entered = deferred();
  const g = setup({ fetch: async requestPath => ({ ok: true, status: 200, json: () => {
    if (requestPath === '/api/evaluation/grants') return Promise.resolve(catalogue());
    entered.resolve(); return body.promise;
  } }) });
  const first = await ready(g), controller = g.controllers.at(-1); controller.load(); await entered.promise;
  const count = first.sent.length, closing = g.owner.leave();
  g.session.accept({ ok: true, role: 'evaluator', expires_at: 2000, permissions: ['evaluation:read'] });
  const opening = g.owner.open(); await turn(); assert.equal(g.sockets.length, 1);
  body.resolve(sample()); assert.equal(await closing, true); assert.equal(await opening, true);
  assert.equal(first.sent.length, count); assert.equal(g.allocated.length, 0); await g.owner.leave();
});
test('functional sample403 verifies the original identity and refreshes grants without global logout', async () => {
  let verifies = 0, retired = 0;
  const g = setup({ verifyIdentity: async () => { verifies++; return true; }, onUnauthorized: () => retired++,
    fetch: async requestPath => requestPath === '/api/evaluation/grants'
      ? { ok: true, status: 200, json: async () => catalogue() } : { ok: false, status: 403 } });
  await ready(g); g.controllers.at(-1).load(); await turn();
  assert.equal(verifies, 1); assert.equal(retired, 0); assert.equal(g.session.active(), true);
  assert.deepEqual(g.controllers.at(-1).store.snapshot(), []); await g.owner.leave();
});
test('a late verification is drained and cannot log out the following login', async () => {
  const check = deferred(), entered = deferred(); let retired = 0;
  const g = setup({ verifyIdentity: () => { entered.resolve(); return check.promise; }, onUnauthorized: () => retired++,
    fetch: async requestPath => requestPath === '/api/evaluation/grants'
      ? { ok: true, status: 200, json: async () => catalogue() } : { ok: false, status: 403 } });
  const first = await ready(g); g.controllers.at(-1).load(); await entered.promise;
  const closing = g.owner.leave(); g.session.accept({ ok: true, role: 'evaluator', expires_at: 2000, permissions: ['evaluation:read'] });
  const opening = g.owner.open(); await turn(); assert.equal(g.sockets.length, 1);
  const before = first.sent.length; check.resolve(false);
  await closing; await opening; assert.equal(retired, 0); assert.equal(first.sent.length, before); await g.owner.leave();
});
test('catalogue expiry removes choices while keeping a still-current evaluator session', async () => {
  const g = setup(); await ready(g); const controller = g.controllers.at(-1);
  g.setNow(1100000); const value = controller.snapshot();
  assert.equal(value.status, 'expired'); assert.deepEqual(value.grants, []); assert.equal(value.selected, null);
  assert.equal(g.session.active(), true); assert.equal(controller.load(), false); await g.owner.leave();
});

test('catalogue refresh enables explicit consultation only after the previous decoder retires', async () => {
  const g = setup(); await ready(g); const controller = g.controllers.at(-1);
  assert.equal(controller.snapshot().busy, false); assert.equal(controller.store.status(), 'waiting');
  assert.equal(g.requests.length, 1);
  controller.load(); await turn();
  const player = new Player(); controller.store.attachMedia(player);
  const refresh = controller.refresh(); await turn();
  assert.equal(controller.snapshot().busy, true); assert.equal(controller.snapshot().selected, null);
  player.empty(); assert.equal(await refresh, true); await turn();
  assert.equal(controller.snapshot().busy, false); assert.equal(controller.store.status(), 'waiting');
  assert.deepEqual(controller.store.snapshot(), []); await g.owner.leave();
});

test('a throwing view cleanup still retires the decoder and socket, and blocks reopening', async () => {
  let controller, fail = false;
  const g = setup({ mount(value) {
    controller = value;
    const stop = value.store.subscribe(() => value.store.committed(value.store.snapshot().length ? value.store.deliveryToken() : null));
    return () => { stop(); if (fail) throw Error('synthetic renderer failure'); value.store.detachMedia(); value.store.committed(null); };
  } });
  const socket = await ready(g);
  controller.load(); await turn();
  const player = new Player(); controller.store.attachMedia(player); fail = true;
  const count = socket.sent.length, closing = g.owner.leave(); let settled = false;
  closing.then(() => { settled = true; }); await turn();
  assert.equal(socket.readyState, 3); assert.equal(player.paused, true); assert.equal(player.isConnected, false);
  assert.deepEqual(g.revoked, g.allocated); assert.equal(settled, false); assert.equal(socket.sent.length, count);
  player.empty(); assert.equal(await closing, false);
  assert.equal(await g.owner.open(), false); assert.equal(g.sockets.length, 1);
});
