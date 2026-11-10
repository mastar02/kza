const assert = require('node:assert/strict');
const { test } = require('node:test');
const home = require('../../src/dashboard/frontend/home/home.js');

// Server observations expire STATE_OBSERVATION_TTL_S after the HA read (ha_client.py).
const TTL_MS = 6000;
const T0 = Date.parse('2026-11-10T00:00:00Z');
const ac = { entity_id: 'climate.test_ac', domain: 'climate', available: true, state: 'cool', capabilities: ['set_temperature'], control_limits: { temperature_min: 18, temperature_max: 25, temperature_step: 0.5 } };

// Drives the real poller, snapshot clock and control gate on one injected clock.
// roundTrip(i) gives the i-th catalog round trip in ms, or null for a request that never answers.
// The server reads HA halfway through the round trip and stamps that instant.
async function simulate({ roundTrip, until = 60000, step = 10 }) {
  let t = T0, nextId = 0, loads = 0, data = null, clock = null;
  const timers = new Map(), starts = [], samples = [];
  const schedule = (fn, ms) => { const id = ++nextId; timers.set(id, { at: t + ms, fn }); return id; };
  const cancel = id => { timers.delete(id); };
  const flush = () => new Promise(resolve => setImmediate(resolve));
  function catalog(read) {
    const observation = { state: 'fresh', observed_at: read / 1000, expires_at: (read + TTL_MS) / 1000 };
    return { observation, rooms: [], unassigned_entities: [], entities: [{ ...ac, observation: { ...observation } }] };
  }
  const poller = home.createPoller({
    load: () => {
      const index = loads++, started = t, trip = roundTrip(index);
      starts.push(started);
      return new Promise(resolve => { if (trip !== null) schedule(() => resolve(catalog(started + trip / 2)), trip); });
    },
    onData(value) { data = value; clock = home.createSnapshotClock(value, { monotonic: () => t - T0, wallClock: () => t }); },
    onError(error) { throw error; },
    deadline: () => clock?.read().remainingMs ?? Infinity,
    now: () => t, schedule, cancel
  });
  poller.start(); await flush();
  for (let sample = T0; sample <= T0 + until; sample += step) {
    for (;;) {
      const due = [...timers].filter(([, timer]) => timer.at <= sample).sort((a, b) => a[1].at - b[1].at)[0];
      if (!due) break;
      timers.delete(due[0]); t = due[1].at; due[1].fn(); await flush();
    }
    t = sample;
    if (!data) continue;
    const entity = data.entities[0];
    const enabled = home.entityCanControl(entity, 'set_temperature', clock.read().stale, clock.now());
    samples.push({ t: t - T0, enabled, expires: entity.observation.expires_at * 1000 - T0 });
  }
  poller.stop();
  return { samples, starts: starts.map(start => start - T0) };
}

function disabledAfterFirstData(samples) {
  const first = samples.findIndex(sample => sample.enabled);
  assert.ok(first >= 0 && samples.length > 1000, 'the simulation produced controllable data');
  return samples.slice(first).filter(sample => !sample.enabled);
}

// The gate must never present expired data as controllable, nor hide current data.
function assertHonest(samples) {
  for (const sample of samples) assert.equal(sample.enabled, sample.t < sample.expires, `t=${sample.t} expires=${sample.expires}`);
}

test('a slow round trip within budget refreshes before expiry: controls never flicker disabled', async () => {
  // 1.5 s per catalog: the old 5 s-after-completion cadence lets each observation lapse for ~1 s.
  const { samples, starts } = await simulate({ roundTrip: () => 1500 });
  assertHonest(samples);
  assert.deepEqual(disabledAfterFirstData(samples).map(sample => sample.t), []);
  // Early refresh does not degenerate into a busy loop.
  for (let i = 1; i < starts.length; i++) assert.ok(starts[i] - starts[i - 1] >= 1000 + 1500, `poll ${i} started ${starts[i] - starts[i - 1]} ms after the previous one`);
});

test('a jittery round trip up to twice the recent worst still lands before expiry', async () => {
  const trips = [400, 900, 1200, 700, 1500, 1100, 600, 1400, 1000, 1300];
  const { samples } = await simulate({ roundTrip: index => trips[index % trips.length] });
  assertHonest(samples);
  assert.deepEqual(disabledAfterFirstData(samples).map(sample => sample.t), []);
});

test('a fast round trip keeps the 5 s cadence', async () => {
  const { samples, starts } = await simulate({ roundTrip: () => 50 });
  assertHonest(samples);
  assert.deepEqual(disabledAfterFirstData(samples).map(sample => sample.t), []);
  for (let i = 1; i < starts.length; i++) assert.ok(starts[i] - starts[i - 1] >= 4800, `poll ${i} started after ${starts[i] - starts[i - 1]} ms`);
});

test('genuine staleness still disables the controls while a refresh is in flight', async () => {
  // The third catalog request never answers: the held observation must lapse, not be extended.
  const { samples, starts } = await simulate({ roundTrip: index => index < 2 ? 300 : null, until: 20000 });
  assertHonest(samples);
  assert.equal(starts.length, 3);
  const lastExpiry = samples.at(-1).expires;
  assert.ok(lastExpiry < 20000);
  assert.ok(samples.filter(sample => sample.t >= lastExpiry).every(sample => !sample.enabled));
  assert.ok(samples.some(sample => sample.t < lastExpiry && sample.enabled));
});

test('a round trip beyond the TTL budget is shown stale exactly while the data is expired', async () => {
  // 4 s per catalog cannot keep a 6 s observation current: the gap is real and must show.
  const { samples } = await simulate({ roundTrip: () => 4000 });
  assertHonest(samples);
  assert.ok(disabledAfterFirstData(samples).length > 0);
});

test('next poll delay is bounded by cadence, expiry budget and a floor', () => {
  assert.equal(home.nextPollDelay(Infinity, 50), 5000);
  assert.equal(home.nextPollDelay(undefined, 50), 5000);
  assert.equal(home.nextPollDelay(5900, 50), 4900);
  assert.equal(home.nextPollDelay(4500, 1500), 1500);
  assert.equal(home.nextPollDelay(500, 50), 1000);
  assert.equal(home.nextPollDelay(5900, NaN), 4900);
});
