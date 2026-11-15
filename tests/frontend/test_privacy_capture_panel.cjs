// R1-B2: the Privacidad panel forgets lost recordings (carried capture owners)
// only after an administrator affirms no tab is recording and confirms the
// count it showed; until then it never says «confirmada».
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');
const Babel = require('../../src/dashboard/frontend/obs/vendor/babel-standalone-7.29.0.min.js');

const FILE = path.resolve(__dirname, '../../src/dashboard/frontend/obs/src/views/privacy.jsx');
const CODE = Babel.transform(fs.readFileSync(FILE, 'utf8'), { presets: ['react'], filename: FILE }).code;

function snapshot(extra = {}) {
  return {
    revision: 4, target_muted: true, persisted_muted: true, phase: 'muted', ingress_sealed: true,
    supported: true, pending_operations: 2, pending_browser_recorders: 2, pending_native_clients: 0,
    live_capture_owners: 0, carried_capture_owners: 2, error_code: null, total_mute_confirmed: false,
    ...extra,
  };
}

const CONFIRMED = snapshot({ pending_operations: 0, pending_browser_recorders: 0, carried_capture_owners: 0,
  total_mute_confirmed: true });

// A minimal hooks renderer for the one function component under test.
function mount(respond) {
  const hooks = [], effects = [], timers = [], requests = [], listeners = {}, history = [];
  let cursor = 0, dirty = false, tree = null, scheduled = false;
  function schedule() {
    if (scheduled) return;
    scheduled = true;
    queueMicrotask(() => { scheduled = false; if (dirty) render(); });
  }
  const React = {
    createElement: (type, props, ...children) => ({ type, props: props || {}, children: children.flat(Infinity) }),
    useState(initial) {
      const index = cursor++;
      if (!(index in hooks)) hooks[index] = { value: initial };
      const slot = hooks[index];
      return [slot.value, next => {
        const value = typeof next === 'function' ? next(slot.value) : next;
        if (!Object.is(value, slot.value)) { slot.value = value; dirty = true; schedule(); }
      }];
    },
    useRef(initial) {
      const index = cursor++;
      if (!(index in hooks)) hooks[index] = { current: initial };
      return hooks[index];
    },
    useEffect(effect, deps) {
      const index = cursor++, previous = hooks[index];
      if (previous && deps && deps.every((dep, k) => Object.is(dep, previous.deps[k]))) return;
      const slot = { deps, cleanup: undefined };
      hooks[index] = slot;
      effects.push(() => { previous?.cleanup?.(); const cleanup = effect(); slot.cleanup = typeof cleanup === 'function' ? cleanup : undefined; });
    },
  };
  async function fetch(url, options = {}) {
    const request = { url, method: options.method || 'GET', options,
      body: options.body === undefined ? undefined : JSON.parse(options.body) };
    requests.push(request);
    const [status, body] = await respond(request);
    return { status, ok: status >= 200 && status < 300, json: async () => body };
  }
  const window = {
    addEventListener: (name, fn) => { (listeners[name] ||= []).push(fn); },
    removeEventListener: (name, fn) => { listeners[name] = (listeners[name] || []).filter(f => f !== fn); },
    dispatchEvent: event => { for (const fn of listeners[event.type] || []) fn(event); },
  };
  const context = vm.createContext({ React, window, fetch, queueMicrotask, AbortController,
    setInterval: fn => { timers.push(fn); return timers.length; }, clearInterval: () => {},
    CustomEvent: class { constructor(type) { this.type = type; } } });
  vm.runInContext(CODE, context);
  function render() {
    for (let round = 0; ; round++) {
      assert.ok(round < 50, 'render loop');
      dirty = false; cursor = 0;
      tree = context.CaptureControlCard();
      history.push(tree);
      for (const effect of effects.splice(0)) effect();
      if (!dirty) return;
    }
  }
  render();
  return {
    requests,
    history,
    get tree() { return tree; },
    poll: () => { for (const fn of timers) fn(); },
  };
}

async function settle() {
  for (let i = 0; i < 20; i++) await new Promise(resolve => setImmediate(resolve));
}

function nodes(node, out = []) {
  if (node && typeof node === 'object') { out.push(node); for (const child of node.children) nodes(child, out); }
  return out;
}

function text(node) {
  if (node === null || node === undefined || typeof node === 'boolean') return '';
  if (typeof node !== 'object') return String(node);
  return node.children.map(text).join('');
}

function button(panel, pattern) {
  const found = nodes(panel.tree).filter(node => node.type === 'button' && pattern.test(text(node)));
  assert.equal(found.length, 1, `one button ${pattern}`);
  return found[0];
}

function buttons(panel, pattern) {
  return nodes(panel.tree).filter(node => node.type === 'button' && pattern.test(text(node)));
}

function checkbox(panel) {
  const found = nodes(panel.tree).filter(node => node.type === 'input' && node.props.type === 'checkbox');
  assert.equal(found.length, 1);
  return found[0];
}

function server(...replies) {
  const queue = [...replies];
  return async request => {
    const next = queue.length > 1 ? queue.shift() : queue[0];
    return typeof next === 'function' ? next(request) : next;
  };
}

test('lost recordings are counted and the mute is never shown as confirmed', async () => {
  const panel = mount(server([200, snapshot()]));
  await settle();
  const all = text(panel.tree);
  assert.match(all, /Grabaciones perdidas: 2\./);
  assert.match(all, /la pausa no se confirma mientras haya grabaciones perdidas/);
  assert.doesNotMatch(all, /confirmada/);
  assert.equal(button(panel, /^Olvidar grabaciones perdidas$/).props.disabled, false);
  assert.equal(nodes(panel.tree).filter(node => node.type === 'input').length, 0);
});

test('no forget control without lost recordings', async () => {
  const panel = mount(server([200, CONFIRMED]));
  await settle();
  assert.match(text(panel.tree), /Captura global pausada y confirmada/);
  assert.deepEqual(buttons(panel, /Olvidar/), []);
});

test('a server claiming confirmation while recordings are lost or live is rejected', async () => {
  for (const extra of [{ carried_capture_owners: 1 }, { live_capture_owners: 1 }]) {
    const panel = mount(server([200, { ...CONFIRMED, ...extra }]));
    await settle();
    assert.match(text(panel.tree), /No se pudo verificar la captura global/);
    assert.doesNotMatch(text(panel.tree), /confirmada/);
  }
});

test('forgetting needs the affirmation and sends exactly the count shown', async () => {
  const panel = mount(server([200, snapshot()], [200, { ...CONFIRMED, forgotten_capture_owners: 2 }]));
  await settle();
  button(panel, /^Olvidar grabaciones perdidas$/).props.onClick();
  await settle();
  assert.match(text(panel.tree), /ninguna pestaña ni herramienta sigue grabando/);
  assert.match(text(panel.tree), /registro sin contenido con la cantidad/);
  const confirm = button(panel, /^Olvidar 2 grabaciones perdidas$/);
  assert.equal(confirm.props.disabled, true);
  assert.equal(checkbox(panel).props.checked, false);
  checkbox(panel).props.onChange({ target: { checked: true } });
  await settle();
  assert.equal(button(panel, /^Olvidar 2 grabaciones perdidas$/).props.disabled, false);
  assert.equal(panel.requests.length, 1);
  button(panel, /^Olvidar 2 grabaciones perdidas$/).props.onClick();
  await settle();
  const sent = panel.requests[1];
  assert.equal(sent.url, '/api/capture/control/forget-carried');
  assert.equal(sent.method, 'POST');
  assert.equal(sent.options.credentials, 'same-origin');
  assert.equal(sent.options.headers['Content-Type'], 'application/json');
  assert.deepEqual(sent.body, { expected_carried: 2, no_capture_running: true });
  const all = text(panel.tree);
  assert.match(all, /Se olvidaron 2 grabaciones perdidas\. Quedó un registro sin contenido/);
  assert.match(all, /Captura global pausada y confirmada/);
  assert.deepEqual(buttons(panel, /Olvidar/), []);
});

test('cancel sends nothing', async () => {
  const panel = mount(server([200, snapshot()]));
  await settle();
  button(panel, /^Olvidar grabaciones perdidas$/).props.onClick();
  await settle();
  checkbox(panel).props.onChange({ target: { checked: true } });
  await settle();
  button(panel, /^Cancelar$/).props.onClick();
  await settle();
  assert.equal(button(panel, /^Olvidar grabaciones perdidas$/).props.disabled, false);
  // Reopened, the affirmation starts unchecked again.
  button(panel, /^Olvidar grabaciones perdidas$/).props.onClick();
  await settle();
  assert.equal(checkbox(panel).props.checked, false);
  assert.ok(panel.requests.every(request => request.method === 'GET'));
});

test('a new count closes an open confirmation; the new one sends the new count', async () => {
  const replies = [[200, snapshot()], [200, snapshot({ carried_capture_owners: 1, pending_browser_recorders: 1 })]];
  const panel = mount(server(...replies));
  await settle();
  button(panel, /^Olvidar grabaciones perdidas$/).props.onClick();
  await settle();
  checkbox(panel).props.onChange({ target: { checked: true } });
  await settle();
  panel.poll();
  await settle();
  assert.match(text(panel.tree), /Grabaciones perdidas: 1\./);
  assert.deepEqual(buttons(panel, /^Olvidar 2/), []);
  button(panel, /^Olvidar grabaciones perdidas$/).props.onClick();
  await settle();
  checkbox(panel).props.onChange({ target: { checked: true } });
  await settle();
  button(panel, /^Olvidar 1 grabación perdida$/).props.onClick();
  await settle();
  const posts = panel.requests.filter(request => request.method === 'POST');
  assert.equal(posts.length, 1);
  assert.deepEqual(posts[0].body, { expected_carried: 1, no_capture_running: true });
});

test('a confirm click from the tree rendered before the count closed it sends the count shown', async () => {
  // R1B2-6: a refresh with a new count renders once with the old confirmation
  // still open before the effect closes it; a click on that tree must send
  // the count it showed, not the new one.
  const panel = mount(server([200, snapshot()], [200, snapshot({ carried_capture_owners: 1, pending_browser_recorders: 1 })],
    [409, { detail: 'capture_carried_count_conflict' }], [200, snapshot({ carried_capture_owners: 1, pending_browser_recorders: 1 })]));
  await settle();
  button(panel, /^Olvidar grabaciones perdidas$/).props.onClick();
  await settle();
  checkbox(panel).props.onChange({ target: { checked: true } });
  await settle();
  const rendered = panel.history.length;
  panel.poll();
  await settle();
  const stale = panel.history.slice(rendered).filter(tree => /Grabaciones perdidas: 1\./.test(text(tree))
    && nodes(tree).some(node => node.type === 'button' && /^Olvidar 2 grabaciones perdidas$/.test(text(node))));
  assert.equal(stale.length, 1, 'one render with the new count and the old confirmation');
  assert.deepEqual(buttons(panel, /^Olvidar 2/), []);
  nodes(stale[0]).find(node => node.type === 'button' && /^Olvidar 2 grabaciones perdidas$/.test(text(node))).props.onClick();
  await settle();
  const posts = panel.requests.filter(request => request.method === 'POST');
  assert.equal(posts.length, 1);
  assert.deepEqual(posts[0].body, { expected_carried: 2, no_capture_running: true });
  assert.match(text(panel.tree), /La cantidad de grabaciones perdidas cambió\. No se olvidó ninguna/);
});

test('a count conflict forgets nothing, says so and refreshes', async () => {
  const panel = mount(server([200, snapshot()], [409, { detail: 'capture_carried_count_conflict' }],
    [200, snapshot({ carried_capture_owners: 3, pending_browser_recorders: 3 })]));
  await settle();
  button(panel, /^Olvidar grabaciones perdidas$/).props.onClick();
  await settle();
  checkbox(panel).props.onChange({ target: { checked: true } });
  await settle();
  button(panel, /^Olvidar 2 grabaciones perdidas$/).props.onClick();
  await settle();
  assert.deepEqual(panel.requests.map(request => request.method), ['GET', 'POST', 'GET']);
  const all = text(panel.tree);
  assert.match(all, /La cantidad de grabaciones perdidas cambió\. No se olvidó ninguna/);
  assert.match(all, /Grabaciones perdidas: 3\./);
  assert.doesNotMatch(all, /confirmada/);
});

test('a forbidden session cannot forget and stops polling', async () => {
  const panel = mount(server([200, snapshot()], [403, { detail: 'capture_control_denied' }]));
  await settle();
  button(panel, /^Olvidar grabaciones perdidas$/).props.onClick();
  await settle();
  checkbox(panel).props.onChange({ target: { checked: true } });
  await settle();
  button(panel, /^Olvidar 2 grabaciones perdidas$/).props.onClick();
  await settle();
  assert.match(text(panel.tree), /no tiene permiso/);
  const count = panel.requests.length;
  panel.poll();
  await settle();
  assert.equal(panel.requests.length, count);
});

test('mute and resume still send the revision-bound policy change', async () => {
  const panel = mount(server([200, snapshot({ phase: 'running', target_muted: false, persisted_muted: false,
    ingress_sealed: false, carried_capture_owners: 0, pending_browser_recorders: 0 })]));
  await settle();
  button(panel, /^Pausar captura global$/).props.onClick();
  await settle();
  assert.equal(panel.requests[1].url, '/api/capture/control');
  assert.deepEqual(panel.requests[1].body, { muted: true, expected_revision: 4 });
});

test('the route the panel calls exists and is administrative only', () => {
  const api = fs.readFileSync(path.resolve(__dirname, '../../src/dashboard/capture_api.py'), 'utf8');
  assert.match(api, /APIRouter\(prefix="\/api\/capture\/control"/);
  assert.match(api, /@router\.post\("\/forget-carried"\)/);
  assert.match(api, /frozenset\(\{"admin:manage"\}\)/);
});
