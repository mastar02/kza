const assert = require('node:assert/strict');
const { test } = require('node:test');
const home = require('../../src/dashboard/frontend/home/home.js');
const routines = require('../../src/dashboard/frontend/home/routines.js');

const ac = { entity_id: 'climate.test_ac', domain: 'climate', available: true, state: 'cool', capabilities: ['set_temperature', 'set_hvac_mode'], control_limits: { temperature_min: 18.5, temperature_max: 25, temperature_step: 0.5, hvac_modes: ['off', 'cool', 'dry'] } };

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

test('routine validation retains typed requirements and rejects values outside current limits', () => {
  const draft = { name: 'test', description: '', enabled: true, timezone: 'America/Montevideo', schedule: null, conditions: [], actions: [{ entity_id: ac.entity_id, operation: 'set_temperature', temperature: 23.5 }] };
  assert.equal(routines.validateRoutineDraft(draft, { entities: [ac] }).valid, true);
  for (const temperature of [undefined, null, 30, 22.2]) {
    assert.equal(routines.validateRoutineDraft({ ...draft, actions: [{ ...draft.actions[0], temperature }] }, { entities: [ac] }).valid, false);
  }
  assert.equal(routines.validateRoutineDraft({ ...draft, actions: [{ entity_id: ac.entity_id, operation: 'set_hvac_mode', hvac_mode: 'heat' }] }, { entities: [ac] }).valid, false);
});
