const assert = require('node:assert/strict');
const { test } = require('node:test');
const home = require('../../src/dashboard/frontend/home/home.js');

const ac = { entity_id: 'climate.test_ac', domain: 'climate', available: true, state: 'cool', capabilities: ['set_temperature', 'set_hvac_mode'], control_limits: { temperature_min: 18.5, temperature_max: 25, temperature_step: 0.5, hvac_modes: ['off', 'cool', 'dry'] } };

test('explicit observation expiry survives receipt and cannot be renewed by presenting it again', () => {
  const observed = Date.parse('2026-11-02T00:00:00Z');
  let elapsed = 0;
  const data = { observed_at: '2099-01-01T00:00:00Z', observation: {
    state: 'fresh', observed_at: new Date(observed).toISOString(), expires_at: new Date(observed + 2000).toISOString() } };
  const clock = home.createSnapshotClock(data, { monotonic: () => elapsed, wallClock: () => observed + 1000 });
  assert.equal(clock.read().stale, false);
  elapsed = 1000;
  assert.equal(clock.read().stale, true);
  assert.equal(home.createSnapshotClock(data, { monotonic: () => 0, wallClock: () => observed + 3000 }).read().stale, true);
  assert.equal(home.createSnapshotClock({ ...data, observation: null }).read().stale, true);
  assert.equal(home.createSnapshotClock({ ...data, observation: { ...data.observation, state: 'unavailable' } }).read().stale, true);
});

test('a control cannot outlive its own selection/configuration evidence', () => {
  const now = Date.parse('2026-11-02T00:00:00Z');
  const entity = { ...ac, observation: { state: 'fresh', observed_at: new Date(now).toISOString(), expires_at: new Date(now + 1000).toISOString() } };
  assert.equal(home.entityCanControl(entity, 'set_temperature', false, now), true);
  assert.equal(home.entityCanControl(entity, 'set_temperature', false, now + 1000), false);
  assert.equal(home.entityCanControl(entity, 'set_temperature', false, now - 1), false);
});

test('wall clock rollback cannot extend an entity deadline before the global deadline', () => {
  const received = Date.parse('2026-11-02T00:00:00Z');
  let elapsed = 0, wall = received;
  const observation = { state: 'fresh', observed_at: new Date(received).toISOString(), expires_at: new Date(received + 10000).toISOString() };
  const entity = { ...ac, observation: { ...observation, expires_at: new Date(received + 1000).toISOString() } };
  const clock = home.createSnapshotClock({ observation, entities: [entity] }, { monotonic: () => elapsed, wallClock: () => wall });
  assert.equal(clock.read().remainingMs, 1000);
  assert.equal(home.entityCanControl(entity, 'set_temperature', clock.read().stale, clock.now()), true);
  elapsed = 2000; wall -= 10000;
  assert.equal(clock.read().stale, false);
  assert.equal(home.entityCanControl(entity, 'set_temperature', clock.read().stale, clock.now()), false);
});

test('current domain state and selection determine whether a control is enabled', () => {
  assert.equal(home.entityCanControl(ac, 'set_temperature', false), true);
  assert.equal(home.entityCanControl(ac, 'set_temperature', true), false);
  assert.equal(home.entityCanControl({ ...ac, available: false }, 'set_temperature', false), false);
  assert.equal(home.entityCanControl(ac, 'turn_on', false), false);
  assert.equal(home.entityCanControl({ ...ac, state: 'unavailable' }, 'set_hvac_mode', false), false);
  assert.equal(home.entityCanControl({ entity_id: 'cover.test_blind', state: 'closing', available: true, capabilities: ['stop_cover'] }, 'stop_cover', false), true);
});

test('forms use the selected range, technical step and selected HVAC modes', () => {
  const [temperature] = home.controlFields(ac, 'set_temperature');
  for (const valid of [18.5, 22, 25]) assert.equal(home.validControlValue(temperature, valid), true);
  for (const invalid of [null, '', 18, 25.5, 22.2, NaN, Infinity, '22']) assert.equal(home.validControlValue(temperature, invalid), false);
  const [mode] = home.controlFields(ac, 'set_hvac_mode');
  assert.equal(home.validControlValue(mode, 'dry'), true);
  assert.equal(home.validControlValue(mode, 'heat'), false);
});

test('the action controller preserves typed values and owns the request identity', async () => {
  const calls = [];
  const controller = home.createActionController({ makeId: () => 'server-test-id', api: async (url, options) => { calls.push({ url, ...options }); return { status: 'confirmed' }; } });
  await controller.execute(ac.entity_id, 'set_temperature', { temperature: 23.5, actor: 'spoofed', request_id: 'spoofed' });
  assert.deepEqual(calls[0].body, { entity_id: ac.entity_id, operation: 'set_temperature', request_id: 'server-test-id', temperature: 23.5 });
  assert.equal(controller.get(ac.entity_id, 'set_temperature').status, 'confirmed');
  await controller.execute('light.test_lamp', 'color_temp_kelvin', { color_temp_kelvin: 2700, brightness_pct: 30 });
  assert.equal(calls[1].body.color_temp_kelvin, 2700);
  assert.equal(calls[1].body.brightness_pct, 30);
});

test('reviewed scenes use their capability and retain partial leaf receipts', async () => {
  const scene = { entity_id: 'scene.test_evening', domain: 'scene', state: 'unknown', available: true, capabilities: ['activate_scene'] };
  assert.equal(home.entityCanControl(scene, 'activate_scene', false), true);
  assert.equal(home.entityCanControl({ ...scene, capabilities: [] }, 'activate_scene', false), false);
  assert.equal(home.entityCanControl(scene, 'activate_scene', true), false);
  const effects = [{ status: 'confirmed' }, { status: 'uncertain' }, { status: 'not_executed' }];
  const calls = [];
  const controller = home.createActionController({ makeId: () => 'scene-test', api: async (url, options) => { calls.push(options); return { status: 'uncertain', effects }; } });
  await controller.execute(scene.entity_id, 'activate_scene', { target: 'lock.hidden' });
  assert.deepEqual(calls[0].body, { entity_id: scene.entity_id, operation: 'activate_scene', request_id: 'scene-test' });
  assert.equal(home.effectSummary(controller.get(scene.entity_id, 'activate_scene').effects), '1 confirmado · 1 sin confirmar · 1 sin ejecutar');
  assert.equal(calls.length, 1);
});
