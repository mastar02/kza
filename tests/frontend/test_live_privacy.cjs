const assert = require('node:assert/strict');
const { test } = require('node:test');
const { createLiveState } = require('../../src/dashboard/frontend/obs/src/live-privacy.js');
const A = '00000000-0000-4000-8000-000000000001';
const B = '00000000-0000-4000-8000-000000000002';
function fixture() {
  let now = 2000000000000, next = 0;
  const timers = new Map(), changes = [], acks = [], deferred = [];
  let transport = true;
  const live = createLiveState({ onChange: f => changes.push(f), emitAck: message => { if (!transport) return false; acks.push(message); return true; },
    defer: f => deferred.push(f), now: () => now,
    schedule: (f, ms) => { timers.set(++next, { f, at: now + ms }); return next; }, cancel: id => timers.delete(id) });
  function advance(ms) { now += ms; for (const [id, t] of [...timers]) if (t.at <= now) { timers.delete(id); t.f(); } }
  const frame = (id = A) => ({ type: 'turn', event_id: id, expires_at: (now + 1000) / 1000,
    ts: '12:00:00', receipt:{ ticket:id, nonce:B }, payload: { stt: 'Synthetic private turn' } });
  return { live, changes, timers, advance, frame, acks, flush: () => { while (deferred.length) deferred.shift()(); }, transport: flag => { transport = flag; } };
}

test('idle ticker expires at original deadline without another frame', () => {
  const f = fixture();
  f.live.receive(f.frame());
  f.advance(1000);
  assert.equal(f.live.snapshot(), null);
  assert.equal(f.changes.at(-1), null);
  assert.equal(f.timers.size, 0);
});

test('matching retirement clears the ticker while another token remains live', () => {
  const f = fixture();
  f.live.receive(f.frame());
  f.live.receive({ type: 'retire', payload: { event_ids: [B] } });
  assert.equal(f.live.snapshot().event_id, A);
  f.live.receive({ type: 'retire', payload: { event_ids: [A] } });
  assert.equal(f.live.snapshot(), null);
});

test('duplicate delivery cannot extend the original deadline', () => {
  const f = fixture();
  f.live.receive(f.frame());
  f.advance(500);
  f.live.receive(f.frame());
  f.advance(500);
  assert.equal(f.live.snapshot(), null);
});

test('unscoped, expired or unsupported frames do not replace a live one', () => {
  const f = fixture();
  f.live.receive(f.frame(A));
  const expired = { ...f.frame(B), expires_at: 1 };
  assert.equal(f.live.receive(expired), false);
  assert.equal(f.live.receive({ type:'turn', payload:{ stt:'No deadline' } }), false);
  assert.equal(f.live.receive({ ...f.frame(B), payload:{ stt:{ raw:'Unknown object' } } }), false);
  assert.equal(f.live.snapshot().event_id, A);
});

test('projection is detached and dispose clears payload and timers', () => {
  const f = fixture(), frame = f.frame();
  f.live.receive(frame);
  frame.payload.stt = 'Mutated producer';
  assert.equal(f.live.snapshot().payload.stt, 'Synthetic private turn');
  f.live.dispose();
  assert.equal(f.live.snapshot(), null);
  assert.equal(f.timers.size, 0);
  assert.equal(f.live.receive(f.frame()), false);
});

test('retirement ACK waits for DOM retirement and the render stack to return', () => {
  const f = fixture();
  f.live.receive(f.frame()); f.live.committed(A); f.flush();
  assert.equal(f.acks.length, 0);
  f.live.clear(); f.flush();
  assert.equal(f.acks.length, 0);
  f.live.committed(null);
  assert.equal(f.acks.length, 0);
  f.flush();
  assert.deepEqual(f.acks, [{ type:'live_ack', receipts:[{ ticket:A, nonce:B }] }]);
  assert.equal(JSON.stringify(f.acks).includes('Synthetic private'), false);
});

test('replacement retires the old ticket only after the new DOM commits', () => {
  const f = fixture();
  f.live.receive(f.frame(A)); f.live.committed(A); f.flush();
  f.live.receive(f.frame(B)); f.flush();
  assert.equal(f.acks.length, 0);
  f.live.committed(B); f.flush();
  assert.deepEqual(f.acks[0].receipts, [{ ticket:A, nonce:B }]);
  assert.equal(f.live.snapshot().event_id, B);
});

test('reconnect reset covers old capabilities and requires an empty cache and DOM', () => {
  const f = fixture();
  f.live.receive(f.frame()); f.live.committed(A); f.flush();
  f.live.receive({ type:'live_reset', payload:{ id:B, nonce:A } }); f.flush();
  assert.equal(f.live.snapshot(), null);
  assert.equal(f.acks.length, 0);
  f.live.committed(null); f.flush();
  assert.deepEqual(f.acks, [{ type:'live_reset_ack', id:B, nonce:A }]);
});

test('unseen retired ticket is acknowledged only after the current view is verified unrelated', () => {
  const f = fixture();
  f.live.receive(f.frame(B));
  f.live.receive({ type:'retire', payload:{ event_ids:[A], receipts:[{ event_id:A, ticket:A, nonce:B }] } }); f.flush();
  assert.equal(f.acks.length, 0);
  f.live.committed(B); f.flush();
  assert.deepEqual(f.acks[0].receipts, [{ ticket:A, nonce:B }]);
});

test('expired frame is never cached but its ticket still needs a clear acknowledgement', () => {
  const f = fixture();
  assert.equal(f.live.receive({ ...f.frame(), expires_at:1 }), false);
  f.flush(); assert.equal(f.acks.length, 0);
  f.live.committed(null); f.flush();
  assert.equal(f.acks[0].receipts[0].ticket, A);
  assert.equal(f.live.snapshot(), null);
});

test('lost transport cannot fabricate a persisted acknowledgement', () => {
  const f = fixture();
  f.live.receive(f.frame()); f.live.committed(A); f.flush();
  f.transport(false); f.live.clear(); f.live.committed(null); f.flush();
  assert.equal(f.acks.length, 0);
  f.transport(true); f.live.committed(null); f.flush();
  assert.equal(f.acks.length, 1);
});

test('retirement capabilities are sent in bounded batches without payloads', () => {
  const f = fixture();
  const ids = Array.from({ length:20 }, (_, i) => `00000000-0000-4000-8000-${String(i + 10).padStart(12, '0')}`);
  f.live.committed(null);
  f.live.receive({ type:'retire', payload:{ event_ids:ids, receipts:ids.map(id => ({ event_id:id, ticket:id, nonce:B })) } });
  f.flush();
  assert.deepEqual(f.acks.map(message => message.receipts.length), [16, 4]);
});
