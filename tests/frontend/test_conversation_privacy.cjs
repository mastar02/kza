const assert = require('node:assert/strict');
const { test } = require('node:test');
const { createConversationState } = require('../../src/dashboard/frontend/obs/src/conversation-privacy.js');
function fixture() {
  let mono = 0, wall = 2000000000000, next = 0;
  const timers = new Map(), notices = [];
  const store = createConversationState({ monotonic: () => mono, wall: () => wall,
    schedule: (f, delay) => { timers.set(++next, { f, at: mono + delay }); return next; }, cancel: id => timers.delete(id) });
  store.subscribe(revision => notices.push(revision));
  const row = (index = 1) => ({ id:`turn_1_00000000-0000-4000-8000-${String(index).padStart(12, '0')}`,
    ts:'12:00:00', user:'Synthetic voice', zone:'Synthetic room', path:'fast', stt:'Synthetic private turn', tts:'Synthetic response',
    intent:'light.turn_on', target:'light.synthetic', latency_ms:20, success:null,
    observed_at:wall/1000, expires_at:(wall+1000)/1000 });
  function advance(ms, wallMs = ms) { mono += ms; wall += wallMs; for (const [id, timer] of [...timers]) if (timer.at <= mono) { timers.delete(id); timer.f(); } }
  return { store, timers, notices, row, advance, start: () => ({ mono, wall }) };
}
test('an idle conversation snapshot expires without a new response or user action', () => {
  const f = fixture(); f.store.receive([f.row()], f.start());
  f.advance(1000); assert.deepEqual(f.store.snapshot(), []); assert.equal(f.timers.size, 0);
  assert(f.notices.every(Number.isInteger));
});
test('request and JSON decoding delay consume the original deadline', () => {
  const f = fixture(), started = f.start(), row = f.row(); f.advance(900);
  f.store.receive([row], started); assert.equal(f.store.snapshot().length, 1);
  f.advance(100); assert.deepEqual(f.store.snapshot(), []);
});
test('a response arriving after expiry is never admitted', () => {
  const f = fixture(), started = f.start(), row = f.row(); f.advance(1000);
  f.store.receive([row], started); assert.deepEqual(f.store.snapshot(), []);
});
test('wall-clock rollback cannot extend a received snapshot', () => {
  const f = fixture(); f.store.receive([f.row()], f.start()); f.advance(1000, -10000);
  assert.deepEqual(f.store.snapshot(), []);
});
test('wall-clock advancement retires snapshots on resume checks', () => {
  const f = fixture(); f.store.receive([f.row()], f.start()); f.advance(0, 1001); f.store.check();
  assert.deepEqual(f.store.snapshot(), []); assert.equal(f.timers.size, 0);
});
test('repeated delivery of an ID cannot extend its earliest local deadline', () => {
  const f = fixture(); f.store.receive([f.row()], f.start()); f.advance(500);
  f.store.receive([f.row()], f.start()); f.advance(500); assert.deepEqual(f.store.snapshot(), []);
});
test('raw snapshots are detached and unknown nested fields are never cached', () => {
  const f = fixture(), row = { ...f.row(), args:{ private:'Extra nested text' }, unknown:'Other text' };
  f.store.receive([row], f.start()); row.stt = 'Changed input';
  const [cached] = f.store.snapshot(); assert.equal(cached.stt, 'Synthetic private turn');
  assert.equal(cached.args, undefined); assert.equal(cached.unknown, undefined); assert(Object.isFrozen(cached));
});
test('unscoped, oversized, duplicate or malformed responses clear previous private data', () => {
  for (const bad of [[{...fixture().row(), expires_at:null}], [{...fixture().row(), stt:{raw:'Bad'}}],
    Array.from({length:51}, (_, i) => fixture().row(i)), [fixture().row(), fixture().row()]]) {
    const f = fixture(); f.store.receive([f.row()], f.start());
    assert.equal(f.store.receive(bad, f.start()), false); assert.deepEqual(f.store.snapshot(), []);
    assert.equal(f.timers.size, 0);
  }
});
test('session disposal clears snapshots and prevents late hydration', () => {
  const f = fixture(); f.store.receive([f.row()], f.start()); f.store.dispose();
  assert.deepEqual(f.store.snapshot(), []); assert.equal(f.timers.size, 0);
  assert.equal(f.store.receive([f.row()], f.start()), false);
});
