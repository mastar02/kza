import * as THREE from '/home/nexo/vendor/three.module.min.js';
import { createHouse } from '/home/nexo/house.js';
import { MAIN_FLOOR } from '/home/nexo/house-map.js';
import { CUT_HEIGHT } from '/home/nexo/house-patches.js';

import { rooms, classifyPatch } from '/home/nexo/private/house-picking-fixture.js';

const catalog = { rooms: rooms.map(([room_id, name]) => ({ room_id, name, entity_ids: [] })), entities: [] };
// Fixture names are unique. Resolve the visible pin label to the expected test identity.
const pinRoom = new Map(rooms.map(([id, name]) => [name, id]));
const roomOf = pin => pinRoom.get(pin?.textContent);
const container = document.querySelector('#house-canvas');
const pins = document.querySelector('#room-pins');
const viewport = container.parentElement;
let scene, camera, selected = 'cocina';
const selections = [];
let frames = 0;
THREE.Scene.prototype.onBeforeRender = function (_renderer, _scene, view) {
  scene = this;
  camera = view;
  frames++;
};
const probe = window.houseProbe = {
  ready: false,
  selections,
  clearSelection() { selections.length = 0; },
  async settle() {
    await new Promise(resolve => setTimeout(resolve, 70));
    await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
  },
  hovered() {
    const cursor = getComputedStyle(container.querySelector('canvas')).cursor;
    return cursor === 'pointer' ? roomOf(pins.querySelector('[data-room-hover]')) ?? null : null;
  },
  unlink() { probe.house.update({ catalog: { ...catalog, rooms: catalog.rooms.slice(0, 3) }, layer: 'climate', selected }); },
  async destroy() {
    probe.house.destroy();
    const before = frames;
    await probe.settle();
    return !container.querySelector('canvas') && !pins.children.length && frames === before;
  },
  measure,
  samples,
  coveredByPadding(point, room) {
    const hit = document.elementFromPoint(point.x, point.y);
    return hit?.matches('.room-pin') && roomOf(hit) !== room;
  },
};
const house = probe.house = createHouse({
  container, pins,
  onSelect(room) {
    selections.push(room);
    selected = room;
    house.update({ catalog, layer: 'climate', selected });
  },
  onReady(ready) {
    document.querySelector('#map-loading').hidden = true;
    const theme = document.querySelector('#theme-slot select');
    theme.value = 'claro';
    theme.dispatchEvent(new Event('change'));
    viewport.scrollIntoView({ block: 'center' });
    probe.ready = ready;
  },
});
document.querySelector('#live-badge').hidden = false;
house.update({ catalog, layer: 'climate', selected });

function canvasRect() {
  return container.querySelector('canvas').getBoundingClientRect();
}

function project(world) {
  const clip = world.clone().project(camera);
  const { left, top, width, height } = canvasRect();
  return { x: left + (clip.x + 1) * width / 2, y: top + (1 - clip.y) * height / 2 };
}

function meshGroups() {
  const meshes = [], floors = new Map(), byName = new Map();
  scene.traverseVisible(object => {
    if (!object.isMesh) return;
    meshes.push(object);
    const room = object.userData.room;
    if (room) {
      if (!floors.has(room)) floors.set(room, []);
      floors.get(room).push(object);
    }
    if (object.name) byName.set(object.name, [object]);
    else if (!room && object.geometry.type === 'BoxGeometry') {
      const box = new THREE.Box3().setFromObject(object);
      const center = box.getCenter(new THREE.Vector3());
      const key = classifyPatch(center);
      if (key) byName.set(key, [...(byName.get(key) ?? []), object]);
    }
  });
  return { meshes, floors, byName };
}

function anchors(floors) {
  return Object.fromEntries([...floors].map(([room, [mesh]]) => {
    const main = MAIN_FLOOR[room] && scene.getObjectByName(MAIN_FLOOR[room]);
    const center = new THREE.Vector3();
    let area = 0;
    for (const triangle of topTriangles(main || mesh)) {
      const size = triangle.getArea();
      center.addScaledVector(triangle.getMidpoint(new THREE.Vector3()), size);
      area += size;
    }
    return [room, center.divideScalar(area)];
  }));
}

function measure() {
  const canvas = container.querySelector('canvas');
  const rect = canvasRect();
  const gl = canvas.getContext('webgl2');
  const pixels = new Uint8Array(canvas.width * canvas.height * 4);
  gl.readPixels(0, 0, canvas.width, canvas.height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
  let left = canvas.width, right = 0, bottom = canvas.height, top = 0;
  for (let y = 0; y < canvas.height; y++) for (let x = 0; x < canvas.width; x++) {
    if (pixels[(y * canvas.width + x) * 4 + 3] < 128) continue;
    left = Math.min(left, x); right = Math.max(right, x);
    bottom = Math.min(bottom, y); top = Math.max(top, y);
  }
  const model = { width: right - left + 1, height: top - bottom + 1 };
  const occupancy = Math.max(model.width / canvas.width, model.height / canvas.height);
  const projected = Object.entries(anchors(meshGroups().floors)).map(([room, center]) => [room, project(center)]);
  const buttons = [...pins.querySelectorAll('button')];
  const positions = buttons.map(button => button.getBoundingClientRect());
  const pinOverlaps = positions.flatMap((a, i) => positions.slice(i + 1).flatMap((b, j) =>
    a.left < b.right && b.left < a.right && a.top < b.bottom && b.top < a.bottom
      ? [[roomOf(buttons[i]), roomOf(buttons[i + j + 1])]] : []));
  const coveredAnchors = positions.flatMap((box, i) => projected.flatMap(([room, { x, y }]) =>
    room !== roomOf(buttons[i]) && x >= box.left && x <= box.right && y >= box.top && y <= box.bottom
      ? [[roomOf(buttons[i]), room]] : []));
  return {
    canvas: { width: rect.width, height: rect.height }, model, occupancy, pinOverlaps, coveredAnchors,
    pins: buttons.map((button, i) => {
      const { left, top, right, bottom, width: w, height: h } = positions[i];
      const center = { x: left + w / 2, y: top + h / 2 };
      const [, anchor] = projected.find(([room]) => room === roomOf(button));
      return {
        room: roomOf(button), name: catalog.rooms.find(room => room.room_id === roomOf(button)).name,
        label: button.textContent, w, h, font: parseFloat(getComputedStyle(button.firstChild).fontSize),
        center, anchorGap: Math.hypot(center.x - anchor.x, center.y - anchor.y),
        visible: getComputedStyle(button).visibility === 'visible',
        inside: left >= rect.left && right <= rect.right && top >= rect.top && bottom <= rect.bottom,
      };
    }),
  };
}

function topTriangles(mesh) {
  const geometry = mesh.geometry;
  const positions = geometry.getAttribute('position');
  const indices = geometry.getIndex();
  const triangles = [];
  for (let i = 0; i < (indices?.count ?? positions.count); i += 3) {
    const points = [i, i + 1, i + 2].map(index =>
      new THREE.Vector3().fromBufferAttribute(positions, indices?.getX(index) ?? index).applyMatrix4(mesh.matrixWorld));
    const tri = new THREE.Triangle(...points);
    const normal = tri.getNormal(new THREE.Vector3()).y;
    if (normal > .85 || (geometry.type === 'ShapeGeometry' && normal < -.85)) triangles.push(tri);
  }
  return triangles;
}

function samples() {
  const { meshes, floors, byName } = meshGroups();
  const rect = canvasRect();
  const centers = anchors(floors);
  const ndc = new THREE.Vector2();
  const ray = new THREE.Raycaster();
  // The decorative ground mesh has no clipping plane and is outside the house.
  const visible = meshes.filter(mesh => mesh.material.clippingPlanes?.length);
  const allFloors = [...floors.values()].flat();
  const pixel = ({ x, y }) => ({ x: Math.round(x), y: Math.round(y) });
  function hitAt({ x, y }, objects = visible) {
    ndc.set((x - rect.left) / rect.width * 2 - 1, 1 - (y - rect.top) / rect.height * 2);
    ray.setFromCamera(ndc, camera);
    return ray.intersectObjects(objects).find(hit => hit.point.y <= CUT_HEIGHT + 1e-6);
  }
  const pointsByRoom = {};
  const floorArea = {};
  const footprints = new Map();
  const screenTriangles = new Map();
  for (const [room, meshes] of floors) {
    const triangles = meshes.flatMap(topTriangles);
    const plan = ({ x, z }) => new THREE.Vector3(x, 0, z);
    footprints.set(room, triangles.map(({ a, b, c }) => new THREE.Triangle(plan(a), plan(b), plan(c))));
    const screen = triangles.map(({ a, b, c }) => new THREE.Triangle(
      ...[a, b, c].map(point => {
        const { x, y } = project(point);
        return new THREE.Vector3(x, y, 0);
      }))).filter(triangle => triangle.getArea() > 1e-6);
    screenTriangles.set(room, screen);
    floorArea[room] = Math.round(screen.reduce((total, tri) => total + tri.getArea(), 0));
    pointsByRoom[room] = triangles.flatMap(({ a, b, c }) => {
      const points = [];
      for (let i = 1; i < 8; i++) for (let j = 1; j < 9 - i; j++) {
        points.push(a.clone().multiplyScalar(1 - (i + j) / 9).addScaledVector(b, i / 9).addScaledVector(c, j / 9));
      }
      return points;
    });
  }
  const closest = new THREE.Vector3();
  function distances(trianglesByRoom, point) {
    return [...trianglesByRoom].map(([room, triangles]) => [
      room, Math.min(...triangles.map(tri => tri.closestPointToPoint(point, closest).distanceTo(point)).filter(Number.isFinite)),
    ]).sort(([, a], [, b]) => a - b);
  }
  function range(a, b, n) {
    return Array.from({ length: n }, (_, i) => a + (b - a) * i / (n - 1));
  }
  function objectPoint(names, room) {
    const candidates = [];
    for (const name of names) for (const mesh of byName.get(name) ?? []) {
      const { min, max } = new THREE.Box3().setFromObject(mesh);
      for (const x of range(min.x, max.x, 5)) for (const z of range(min.z, max.z, 6)) {
        for (const y of range(Math.max(min.y, .07), Math.min(max.y, CUT_HEIGHT), 5)) {
          const point = pixel(project(new THREE.Vector3(x, y, z)));
          const hit = hitAt(point);
          if (hit?.object !== mesh) continue;
          const [[actual, distance], [, next]] = distances(footprints, new THREE.Vector3(hit.point.x, 0, hit.point.z));
          if (actual !== room || distance > .35 || next - distance <= .03) continue;
          const oldFloor = hitAt(point, allFloors)?.object.userData.room ?? null;
          candidates.push({ point, mesh: name, oldFloor });
        }
      }
    }
    // Prefer an observable regression of floor-only picking when the visible face permits it.
    return candidates.find(item => item.oldFloor !== room) ?? candidates[0];
  }
  const samples = {};
  for (const [room, , walls, furniture] of rooms) {
    const center = centers[room];
    samples[room] = [];
    for (const [kind, dx, dz] of [['centro', 0, 0], ['x+0.8', .8, 0], ['x-0.8', -.8, 0], ['z+0.8', 0, .8], ['z-0.8', 0, -.8]]) {
      const desired = center.clone().add(new THREE.Vector3(dx, 0, dz));
      const candidates = [desired, ...pointsByRoom[room]].sort((a, b) => a.distanceToSquared(desired) - b.distanceToSquared(desired));
      for (const candidate of candidates) {
        const point = pixel(project(candidate));
        if (samples[room].some(sample => sample.point.x === point.x && sample.point.y === point.y)) continue;
        const hit = hitAt(point);
        if (!hit) continue;
        const [[actual, distance], [, next]] = distances(footprints, new THREE.Vector3(hit.point.x, 0, hit.point.z));
        const floor = hit.object.userData.room;
        if (floor === room || (!floor && actual === room && distance <= .05 && next - distance > .03)) {
          samples[room].push({
            kind, point, surface: floor ? 'floor' : hit.object.name || 'patch',
            displacement: Number(candidate.distanceTo(desired).toFixed(2)),
          });
          break;
        }
      }
    }
    for (const [kind, names] of [['pared', walls], ['mueble', furniture]]) {
      const sample = objectPoint(names, room);
      if (sample) samples[room].push({ kind, ...sample });
    }
  }
  let miss, nearMiss;
  let airDistance = Infinity;
  for (let y = Math.ceil(rect.top) + 6; y < rect.bottom - 6 && (!miss || !nearMiss); y += 9) {
    for (let x = Math.ceil(rect.left) + 6; x < rect.right - 6; x += 9) {
      if (hitAt({ x, y })) continue;
      const [[room, distance], [, next]] = distances(screenTriangles, new THREE.Vector3(x, y, 0));
      airDistance = Math.min(airDistance, distance);
      if (distance > 50 && !miss) miss = { x, y };
      if (distance >= 10 && distance <= 43 && next - distance > 4 && !nearMiss) {
        nearMiss = { room, distance: Number(distance.toFixed(1)), point: { x, y } };
      }
    }
  }
  return { floorArea, samples, miss, nearMiss, airDistance, unlinked: samples.bano[0].point };
}
