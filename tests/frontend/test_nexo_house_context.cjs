const assert = require('node:assert/strict');
const { registerHooks } = require('node:module');
const { test } = require('node:test');
const { pathToFileURL } = require('node:url');

const houseURL = new URL('../../src/dashboard/frontend/home/nexo/house.js', pathToFileURL(__filename));
const modelURL = 'data:model/gltf+json,' + encodeURIComponent(JSON.stringify({
  asset: { version: '2.0' }, scene: 0, scenes: [{ nodes: [] }],
}));

test('Casa pauses for native context loss and surfaces render errors with a live context', async t => {
  class Canvas extends EventTarget {
    closest() { return null; }
    getContext() { return { fillRect() {}, strokeRect() {} }; }
    remove() {}
  }
  let lost = false;
  const gl = { isContextLost: () => lost };
  const canvas = new Canvas();
  const render = t.mock.fn();
  class Renderer {
    domElement = canvas;
    shadowMap = {};
    getContext() { return gl; }
    render(scene, camera) { render(scene, camera); }
    setPixelRatio() {}
    setClearColor() {}
    setSize() {}
    dispose() {}
    forceContextLoss() {}
  }
  const status = { setAttribute() {}, remove() {} };
  const document = new EventTarget();
  document.createElement = tag => tag === 'canvas' ? new Canvas() : status;
  const queued = new Map();
  let nextId = 0, time = 0;
  const globals = {
    document, window: { devicePixelRatio: 1 }, contextTestRenderer: Renderer,
    ResizeObserver: class { observe() {} disconnect() {} },
    matchMedia: () => ({ matches: false }),
    requestAnimationFrame(callback) { queued.set(++nextId, callback); return nextId; },
    cancelAnimationFrame(id) { queued.delete(id); },
  };
  Object.assign(globalThis, globals);
  let house;
  t.after(() => {
    try {
      house?.destroy();
    } finally {
      for (const name of Object.keys(globals)) delete globalThis[name];
    }
  });
  // Use the real scene and loader. Substitute just the renderer imported by the viewer.
  const threeURL = new URL('./vendor/three.module.min.js', houseURL);
  const shim = `export * from '${threeURL}'; export const WebGLRenderer = globalThis.contextTestRenderer;`;
  const hooks = registerHooks({ resolve(specifier, context, nextResolve) {
    if (context.parentURL === houseURL.href && specifier === './vendor/three.module.min.js') {
      return { url: `data:text/javascript,${encodeURIComponent(shim)}`, shortCircuit: true };
    }
    return nextResolve(specifier, context);
  } });
  const { createHouse } = await import(houseURL.href).finally(() => hooks.deregister());
  const readiness = [];
  let signalReady;
  const ready = new Promise(resolve => { signalReady = resolve; });
  const container = Object.assign(new EventTarget(), {
    dataset: {}, clientWidth: 480, clientHeight: 320, append() {}, after() {},
  });
  house = createHouse({ container, modelUrl: modelURL, onReady(value) {
    readiness.push(value);
    signalReady(value);
  } });
  assert.equal(await ready, true);

  function frame() {
    assert.equal(queued.size, 1);
    const [[id, callback]] = queued;
    queued.delete(id);
    callback(time += 34);
  }
  function deliverLossAndRestore() {
    const event = new Event('webglcontextlost', { cancelable: true });
    canvas.dispatchEvent(event);
    assert.equal(event.defaultPrevented, true);
    assert.equal(readiness.at(-1), false);
    lost = false;
    canvas.dispatchEvent(new Event('webglcontextrestored'));
    assert.equal(readiness.at(-1), true);
    assert.equal(status.textContent, '');
  }

  lost = true;
  frame();
  assert.equal(render.mock.callCount(), 0);
  assert.deepEqual(readiness, [true, false]);
  assert.match(status.textContent, /El plano 3D se pausó/);
  assert.equal(queued.size, 0);
  deliverLossAndRestore();

  render.mock.mockImplementationOnce(() => {
    lost = true;
    throw new TypeError("Cannot read properties of null (reading 'trim')");
  });
  house.rotate(1);
  frame();
  assert.equal(render.mock.callCount(), 1);
  assert.deepEqual(readiness, [true, false, true, false]);
  assert.equal(queued.size, 0, 'loss cancels the next camera animation frame');
  deliverLossAndRestore();

  frame();
  assert.equal(render.mock.callCount(), 2);
  assert.equal(render.mock.calls[1].arguments[0], render.mock.calls[0].arguments[0], 'restores the same scene');
  const error = new Error('Unexpected render failure');
  render.mock.mockImplementationOnce(() => { throw error; });
  assert.throws(frame, caught => caught === error);
  assert.deepEqual(readiness, [true, false, true, false, true]);
});
