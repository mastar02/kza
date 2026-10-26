const assert = require('node:assert/strict');
const { test } = require('node:test');
const { createMemoryController } = require('../../src/dashboard/frontend/home/memory.js');

function fixture(apiFetch) {
  let time = 0; const timers = new Map(); let seq = 0;
  const controller = createMemoryController({ apiFetch, monotonic: () => time, wall: () => time,
    schedule(callback) { timers.set(++seq, callback); return seq; }, cancel(id) { timers.delete(id); } });
  return { controller, timers, advance(ms) { time += ms; controller.tick(); } };
}
const proposal = { id: 'fact_a', content: 'Un dato.', state: 'proposed', expires_at: 101 };
const snapshot = () => ({ observed_at: 100, proposals: [proposal], facts: [], pending_index: 0 });

test('expired proposals disappear and cannot initiate confirmation', async () => {
  const calls = []; const { controller, advance } = fixture(async (...args) => { calls.push(args); return snapshot(); });
  await controller.refresh(); advance(1000);
  assert.deepEqual(controller.getState().proposals, []);
  await controller.act(proposal.id, 'confirm');
  assert.equal(calls.length, 1); controller.destroy();
});

test('confirmation is explicit and refreshes authoritative state', async () => {
  const calls = []; let confirmed = false;
  const { controller } = fixture(async (url, options) => {
    calls.push([url, options.method]);
    if (options.method === 'POST') { confirmed = true; return {}; }
    return confirmed ? { observed_at: 100, proposals: [], facts: [{ ...proposal, state: 'confirmed', expires_at: null }] } : snapshot();
  });
  await controller.refresh(); assert.equal(calls.length, 1);
  await controller.act(proposal.id, 'confirm');
  assert.deepEqual(calls.map(item => item[1]), [undefined, 'POST', undefined]);
  assert.equal(controller.getState().facts[0].state, 'confirmed'); controller.destroy();
});

test('a stale rejection reports conflict and clears the private snapshot', async () => {
  const { controller } = fixture(async (_url, options) => {
    if (options.method) throw Object.assign(new Error('changed'), { status: 409 });
    return snapshot();
  });
  await controller.refresh(); await controller.act(proposal.id, 'reject');
  assert.equal(controller.getState().ready, false);
  assert.deepEqual(controller.getState().proposals, []); controller.destroy();
});

test('logout/navigation discards late requests and owns no timer or private rows', async () => {
  let release; const pending = new Promise(resolve => { release = resolve; });
  const { controller, timers } = fixture(async () => pending);
  const request = controller.refresh(); controller.destroy(); release(snapshot()); await request;
  assert.deepEqual(controller.getState().proposals, []); assert.equal(timers.size, 0);
});

test('a slower old GET cannot replace a newer snapshot', async () => {
  let release; const first = new Promise(resolve => { release = resolve; }); let count = 0;
  const { controller } = fixture(async () => ++count === 1 ? first : { observed_at: 102, facts: [], proposals: [] });
  const older = controller.refresh(); await controller.refresh(); release(snapshot()); await older;
  assert.deepEqual(controller.getState().proposals, []); controller.destroy();
});

test('unavailable memory is not an empty successful list or a mock', async () => {
  const { controller } = fixture(async () => { throw Object.assign(new Error('offline'), { status: 503 }); });
  await controller.refresh(); assert.equal(controller.getState().ready, false);
  assert.match(controller.getState().error, /no está disponible/); controller.destroy();
});
