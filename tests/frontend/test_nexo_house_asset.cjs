const assert = require('node:assert/strict');
const { existsSync, readFileSync } = require('node:fs');
const path = require('node:path');
const { test } = require('node:test');

const modelPath = path.resolve(__dirname, '../../src/dashboard/frontend/home/nexo/private/casa.glb');

test('the house GLB embeds its resources and omits export metadata', {
  skip: !existsSync(modelPath) && 'Private casa.glb is absent',
}, () => {
  const glb = readFileSync(modelPath);
  assert.equal(glb.toString('ascii', 0, 4), 'glTF');
  assert.equal(glb.readUInt32LE(4), 2);
  assert.equal(glb.readUInt32LE(8), glb.length);
  const jsonEnd = 20 + glb.readUInt32LE(12);
  assert.equal(glb.toString('ascii', 16, 20), 'JSON');
  assert.ok(jsonEnd <= glb.length, 'the JSON chunk fits inside the GLB');
  const model = JSON.parse(glb.toString('utf8', 20, jsonEnd));

  assert.equal(model.asset.version, '2.0');
  for (const key of ['generator', 'copyright', 'extras']) {
    assert.ok(!Object.hasOwn(model.asset, key), `asset.${key} must be omitted`);
  }
  // An external URI in a future GLB could make the viewer request resources from third parties.
  for (const [index, buffer] of (model.buffers ?? []).entries()) {
    assert.ok(!Object.hasOwn(buffer, 'uri'), `buffers[${index}].uri must be omitted`);
  }
  for (const [index, image] of (model.images ?? []).entries()) {
    assert.ok(!Object.hasOwn(image, 'uri'), `images[${index}].uri must be omitted`);
    assert.ok(Number.isInteger(image.bufferView) && model.bufferViews?.[image.bufferView],
      `images[${index}] must reference an embedded bufferView`);
  }
});
