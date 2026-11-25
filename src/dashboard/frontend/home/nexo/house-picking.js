import * as THREE from './vendor/three.module.min.js';

const EDGE_TOLERANCE = .4; // metres in the scan
const CAPTURE_RADIUS = 44; // CSS pixels around projected floors

function cross(a, b, x, y) {
  return (b.x - a.x) * (y - a.y) - (b.y - a.y) * (x - a.x);
}

function segmentDistance(a, b, x, y) {
  const dx = b.x - a.x, dy = b.y - a.y;
  const length = dx * dx + dy * dy;
  const t = length ? THREE.MathUtils.clamp(((x - a.x) * dx + (y - a.y) * dy) / length, 0, 1) : 0;
  return Math.hypot(x - a.x - t * dx, y - a.y - t * dy);
}

function triangleDistance([a, b, c], x, y) {
  const ab = cross(a, b, x, y), bc = cross(b, c, x, y), ca = cross(c, a, x, y);
  if ((ab >= 0 && bc >= 0 && ca >= 0) || (ab <= 0 && bc <= 0 && ca <= 0)) return 0;
  return Math.min(segmentDistance(a, b, x, y), segmentDistance(b, c, x, y), segmentDistance(c, a, x, y));
}

function nearestRoom(rooms, x, y, limit, ambiguity, linked) {
  let nearest, best = Infinity, second = Infinity;
  for (const [scan, triangles] of rooms) {
    if (linked && !linked.has(scan)) continue;
    let distance = Infinity;
    for (const triangle of triangles) {
      distance = Math.min(distance, triangleDistance(triangle, x, y));
      if (!distance) break;
    }
    if (distance < best) {
      second = best;
      best = distance;
      nearest = scan;
    } else {
      second = Math.min(second, distance);
    }
  }
  return best <= limit && second - best > ambiguity ? nearest : undefined;
}

// Use actual floor triangles so that corridors, alcoves and the stair opening keep their shape.
function floorTriangles(mesh) {
  const { geometry, matrixWorld } = mesh;
  const positions = geometry.getAttribute('position');
  const points = Array.from({ length: positions.count }, (_, i) =>
    new THREE.Vector3().fromBufferAttribute(positions, i).applyMatrix4(matrixWorld));
  const index = geometry.getIndex();
  const triangles = [];
  for (let i = 0; i < (index?.count ?? points.length); i += 3) {
    const world = [0, 1, 2].map(offset => points[index ? index.getX(i + offset) : i + offset]);
    const plan = world.map(point => new THREE.Vector2(point.x, point.z));
    // The stair landing is a double-sided shape facing down. The other floors have upper faces.
    const signedArea = cross(plan[0], plan[1], plan[2].x, plan[2].y);
    const area = geometry.type === 'ShapeGeometry' ? signedArea : -signedArea;
    if (area > 1e-7) triangles.push({ world, plan, area });
  }
  return triangles;
}

export function floorCentroid(mesh) {
  const center = new THREE.Vector3(), midpoint = new THREE.Vector3();
  let weight = 0;
  for (const { world: [a, b, c], area } of floorTriangles(mesh)) {
    midpoint.copy(a).add(b).add(c).divideScalar(3);
    center.addScaledVector(midpoint, area);
    weight += area;
  }
  return weight ? center.divideScalar(weight) : new THREE.Box3().setFromObject(mesh).getCenter(center);
}

export function createRoomPicker(meshes, floors, cutPlane) {
  const raycaster = new THREE.Raycaster();
  const pointer = new THREE.Vector2();
  const roomTriangles = new Map();
  for (const floor of floors) {
    const scan = floor.userData.room;
    if (!roomTriangles.has(scan)) roomTriangles.set(scan, []);
    roomTriangles.get(scan).push(...floorTriangles(floor));
  }
  const footprints = new Map([...roomTriangles].map(([scan, triangles]) =>
    [scan, triangles.map(triangle => triangle.plan)]));
  let projected, projectedVersion;

  return function pickRoom(x, y, width, height, camera, projection, linked) {
    pointer.set(x / width * 2 - 1, 1 - y / height * 2);
    raycaster.setFromCamera(pointer, camera);
    // Three.js raycasting does not apply material clipping. Ignore the discarded upper faces.
    const hit = raycaster.intersectObjects(meshes, false).find(({ point }) => cutPlane.distanceToPoint(point) >= -1e-6);
    if (hit) {
      const scan = hit.object.userData.room ?? nearestRoom(footprints, hit.point.x, hit.point.z, EDGE_TOLERANCE, .005);
      return { scan, surface: true };
    }

    if (projectedVersion !== projection) {
      projected = new Map([...roomTriangles].map(([scan, triangles]) => [scan, triangles.map(({ world }) =>
        world.map(point => {
          const ndc = point.clone().project(camera);
          return new THREE.Vector2((ndc.x + 1) * width / 2, (1 - ndc.y) * height / 2);
        }))]));
      projectedVersion = projection;
    }
    const scan = nearestRoom(projected, x, y, CAPTURE_RADIUS, .01, linked);
    return { scan, surface: false };
  };
}
