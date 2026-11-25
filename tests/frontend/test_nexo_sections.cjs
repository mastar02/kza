const { test } = require('node:test');
const assert = require('node:assert/strict');
const { SECTIONS, sectionDestination } = require('../../src/dashboard/frontend/home/nexo/sections.js');
const { createSession, resolveViewHash, VIEW_HASHES } = require('../../src/dashboard/frontend/home/nexo/state.js');

const operator = ['home:read', 'tasks:read', 'admin:manage', 'voices:read', 'voices:review', 'memory:read', 'auth:manage'];
const household = ['home:read', 'tasks:read'];
const member = [...household, 'memory:own:read'];
const evaluator = ['evaluation:read'];

function login(role, permissions) {
  const session = createSession({ wall: () => 1000, monotonic: () => 0 });
  assert.equal(session.accept({ ok: true, role, permissions, expires_at: 60 }), true);
  return session;
}

function groupedViews(session) {
  return session.sections().map(section => [section.id, ...section.views.map(view => view.id)]);
}

test('the registry defines the seven sections in order with Activity as a single view', () => {
  assert.deepEqual(SECTIONS.map(({ id, label }) => [id, label]), [
    ['home', 'Casa'], ['routines', 'Rutinas'], ['tasks', 'Tareas'], ['activity', 'Actividad'],
    ['people', 'Personas'], ['memory', 'Memoria'], ['system', 'Sistema'],
  ]);
  const activity = SECTIONS.find(section => section.id === 'activity');
  assert.equal(activity.defaultView, 'activity');
  assert.deepEqual(activity.views, { activity: 'Actividad' });
  const registeredViews = SECTIONS.flatMap(section => Object.keys(section.views)).sort();
  assert.deepEqual(registeredViews, Object.keys(VIEW_HASHES).sort());
});

test('only sections with granted member views appear in each session', () => {
  assert.deepEqual(groupedViews(login('operator', [...operator, 'memory:own:read', ...evaluator])), [
    ['home', 'home', 'actions'], ['routines', 'routines'], ['tasks', 'tasks'], ['activity', 'activity'],
    ['people', 'people', 'voices'], ['memory', 'memory'], ['system', 'access'],
  ]);
  assert.deepEqual(groupedViews(login('home', [...member, 'auth:manage', ...evaluator])), [
    ['home', 'home', 'actions'], ['routines', 'routines'], ['tasks', 'tasks'], ['activity', 'activity'], ['memory', 'mymemory'],
  ]);
  assert.deepEqual(groupedViews(login('home', household)), [
    ['home', 'home', 'actions'], ['routines', 'routines'], ['tasks', 'tasks'], ['activity', 'activity'],
  ]);
  assert.deepEqual(groupedViews(login('evaluator', [...evaluator, 'auth:manage', 'memory:own:read'])), [
    ['system', 'evaluation'],
  ]);
  assert.deepEqual(groupedViews(login('operator', ['voices:read'])), []);
  const session = login('operator', operator);
  session.clear();
  assert.deepEqual(groupedViews(session), []);
});

test('section destinations prefer a permitted last view, then the default, then the first permitted view', () => {
  const people = login('operator', operator).sections().find(section => section.id === 'people');
  assert.equal(sectionDestination(people), 'people');
  assert.equal(sectionDestination(people, 'voices'), 'voices');
  assert.equal(sectionDestination(people, 'access'), 'people');
  const [voicesOnly] = login('operator', ['voices:read', 'voices:review']).sections();
  assert.equal(sectionDestination(voicesOnly, 'people'), 'voices');
  const [system] = login('evaluator', evaluator).sections();
  assert.equal(sectionDestination(system, 'access'), 'evaluation');
});

test('legacy hashes resolve to their existing views only with the original grants', () => {
  const expected = {
    home: '#hogar', actions: '#acciones', routines: '#rutinas', tasks: '#tareas', activity: '#actividad',
    memory: '#memoria', mymemory: '#mis-recuerdos', people: '#personas', voices: '#voces',
    access: '#accesos', evaluation: '#evaluacion',
  };
  assert.deepEqual(VIEW_HASHES, expected);
  for (const [role, permissions] of [['operator', operator], ['home', member], ['evaluator', evaluator]]) {
    const allowed = login(role, permissions).views();
    for (const [view, hash] of Object.entries(expected)) {
      assert.equal(resolveViewHash(hash, allowed), allowed.includes(view) ? view : allowed[0], `${role}: ${hash}`);
    }
  }
  assert.equal(resolveViewHash('#accesos', []), undefined);
});
