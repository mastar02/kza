const assert = require('node:assert/strict');
const { test } = require('node:test');
const { createLiveState } = require('../../src/dashboard/frontend/obs/src/live-privacy.js');
const A = '00000000-0000-4000-8000-000000000001';
const B = '00000000-0000-4000-8000-000000000002';
function fixture() {
  let now = 2000000000000, next = 0;
  const timers = new Map(), changes = [];
  const live = createLiveState({ onChange: f => changes.push(f), now: () => now,
    schedule: (f, ms) => { timers.set(++next, { f, at: now + ms }); return next; }, cancel: id => timers.delete(id) });
  function advance(ms) { now += ms; for (const [id, t] of [...timers]) if (t.at <= now) { timers.delete(id); t.f(); } }
  const frame = (id = A) => ({ type: 'turn', event_id: id, expires_at: (now + 1000) / 1000,
    ts: '12:00:00', payload: { stt: 'Synthetic private turn' } });
  return { live, changes, timers, advance, frame };
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
