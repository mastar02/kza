const assert = require('node:assert/strict');
const { test } = require('node:test');

const house = import('../../src/dashboard/frontend/home/nexo/house.js');

test('Casa disposes loaded and replaced materials, and closes each shared bitmap once', async () => {
  const { disposeObjects } = await house;
  const disposed = [];
  let closes = 0;
  const bitmap = { close() { closes++; } };
  const resource = name => ({ dispose() {
    assert.equal(closes, 0, 'bitmaps remain open through texture disposal');
    disposed.push(name);
  } });
  const texture = { ...resource('texture'), isTexture: true, source: {data: bitmap} };
  const clone = { ...resource('clone'), isTexture: true, image: bitmap };
  const canvasTexture = { ...resource('canvas'), isTexture: true, image: {} };
  const live = { ...resource('live material'), map: texture, normalMap: canvasTexture };
  const replaced = { ...resource('replaced material'), map: clone, normalMap: texture };
  const mesh = { geometry: resource('geometry'), material: [live, live] };
  const scene = { traverse(visit) { visit(mesh); visit(mesh); } };
  disposeObjects([scene], new Set([replaced]));
  assert.deepEqual(disposed.sort(), ['canvas', 'clone', 'geometry', 'live material', 'replaced material', 'texture']);
  assert.equal(closes, 1);
});

test('Casa revokes an embedded image URL when GLTFLoader cannot decode it', async t => {
  const { modelLoadingManager } = await house;
  const { GLTFLoader } = await import('../../src/dashboard/frontend/home/nexo/vendor/three-addons/loaders/GLTFLoader.js');
  // One triangle with an embedded four-byte buffer that cannot decode as a PNG.
  const model = {
    asset: { version: '2.0' }, scene: 0, scenes: [{ nodes: [0] }], nodes: [{ mesh: 0 }],
    meshes: [{ primitives: [{ attributes: { POSITION: 0 }, material: 0 }] }],
    buffers: [{ byteLength: 40 }],
    bufferViews: [{ buffer: 0, byteLength: 36 }, { buffer: 0, byteOffset: 36, byteLength: 4 }],
    accessors: [{ bufferView: 0, componentType: 5126, count: 3, type: 'VEC3', min: [0, 0, 0], max: [1, 1, 0] }],
    materials: [{ pbrMetallicRoughness: { baseColorTexture: { index: 0 } } }],
    textures: [{ source: 0 }], images: [{ bufferView: 1, mimeType: 'image/png' }],
  };
  const source = JSON.stringify(model);
  const json = Buffer.from(source.padEnd(Math.ceil(source.length / 4) * 4));
  const glb = new ArrayBuffer(12 + 8 + json.length + 8 + 40);
  const header = new DataView(glb);
  header.setUint32(0, 0x46546c67, true);
  header.setUint32(4, 2, true);
  header.setUint32(8, glb.byteLength, true);
  header.setUint32(12, json.length, true);
  header.setUint32(16, 0x4e4f534a, true);
  new Uint8Array(glb, 20, json.length).set(json);
  header.setUint32(20 + json.length, 40, true);
  header.setUint32(24 + json.length, 0x004e4942, true);

  const urls = [], create = URL.createObjectURL;
  globalThis.self = globalThis;
  globalThis.createImageBitmap = async () => { throw new Error('Invalid synthetic PNG'); };
  t.mock.method(URL, 'createObjectURL', blob => {
    const url = create.call(URL, blob);
    urls.push(url);
    return url;
  });
  t.after(() => {
    delete globalThis.createImageBitmap;
    delete globalThis.self;
    for (const url of urls) URL.revokeObjectURL(url);
  });

  const loader = new GLTFLoader(modelLoadingManager());
  const result = await loader.parseAsync(glb, '');
  assert.equal(result.scene.children.length, 1, 'the model survives with an untextured mesh');
  assert.equal(urls.length, 1, 'GLTFLoader created an embedded image URL');
  await assert.rejects(fetch(urls[0]), TypeError);
});
