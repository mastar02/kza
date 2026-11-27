import * as THREE from 'three'

const EDGE_TOLERANCE = 0.4
const CAPTURE_RADIUS = 44

function cross(a: THREE.Vector2, b: THREE.Vector2, x: number, y: number): number {
  return (b.x - a.x) * (y - a.y) - (b.y - a.y) * (x - a.x)
}

function segmentDistance(a: THREE.Vector2, b: THREE.Vector2, x: number, y: number): number {
  const dx = b.x - a.x
  const dy = b.y - a.y
  const length = dx * dx + dy * dy
  const t = length ? THREE.MathUtils.clamp(((x - a.x) * dx + (y - a.y) * dy) / length, 0, 1) : 0
  return Math.hypot(x - a.x - t * dx, y - a.y - t * dy)
}

function triangleDistance(tri: [THREE.Vector2, THREE.Vector2, THREE.Vector2], x: number, y: number): number {
  const [a, b, c] = tri
  const ab = cross(a, b, x, y)
  const bc = cross(b, c, x, y)
  const ca = cross(c, a, x, y)
  if ((ab >= 0 && bc >= 0 && ca >= 0) || (ab <= 0 && bc <= 0 && ca <= 0)) return 0
  return Math.min(segmentDistance(a, b, x, y), segmentDistance(b, c, x, y), segmentDistance(c, a, x, y))
}

function nearestRoom(
  rooms: Map<string, [THREE.Vector2, THREE.Vector2, THREE.Vector2][]>,
  x: number,
  y: number,
  limit: number,
  ambiguity: number,
  linked?: ReadonlyMap<string, unknown>,
): string | undefined {
  let nearest: string | undefined
  let best = Infinity
  let second = Infinity
  for (const [scan, triangles] of rooms) {
    if (linked && !linked.has(scan)) continue
    let distance = Infinity
    for (const triangle of triangles) {
      distance = Math.min(distance, triangleDistance(triangle, x, y))
      if (!distance) break
    }
    if (distance < best) {
      second = best
      best = distance
      nearest = scan
    } else {
      second = Math.min(second, distance)
    }
  }
  return best <= limit && second - best > ambiguity ? nearest : undefined
}

type FloorTriangle = {
  world: [THREE.Vector3, THREE.Vector3, THREE.Vector3]
  plan: [THREE.Vector2, THREE.Vector2, THREE.Vector2]
  area: number
}

function floorTriangles(mesh: THREE.Mesh): FloorTriangle[] {
  const { geometry, matrixWorld } = mesh
  const positions = geometry.getAttribute('position')
  const points = Array.from({ length: positions.count }, (_, i) =>
    new THREE.Vector3().fromBufferAttribute(positions, i).applyMatrix4(matrixWorld))
  const index = geometry.getIndex()
  const triangles: FloorTriangle[] = []
  for (let i = 0; i < (index?.count ?? points.length); i += 3) {
    const p0 = points[index ? index.getX(i) : i]!
    const p1 = points[index ? index.getX(i + 1) : i + 1]!
    const p2 = points[index ? index.getX(i + 2) : i + 2]!
    const plan: [THREE.Vector2, THREE.Vector2, THREE.Vector2] = [
      new THREE.Vector2(p0.x, p0.z),
      new THREE.Vector2(p1.x, p1.z),
      new THREE.Vector2(p2.x, p2.z),
    ]
    const signedArea = cross(plan[0], plan[1], plan[2].x, plan[2].y)
    const area = geometry.type === 'ShapeGeometry' ? signedArea : -signedArea
    if (area > 1e-7) triangles.push({ world: [p0, p1, p2], plan, area })
  }
  return triangles
}

export function floorCentroid(mesh: THREE.Mesh): THREE.Vector3 {
  const center = new THREE.Vector3()
  const midpoint = new THREE.Vector3()
  let weight = 0
  for (const { world: [a, b, c], area } of floorTriangles(mesh)) {
    midpoint.copy(a).add(b).add(c).divideScalar(3)
    center.addScaledVector(midpoint, area)
    weight += area
  }
  return weight ? center.divideScalar(weight) : new THREE.Box3().setFromObject(mesh).getCenter(center)
}

export function createRoomPicker(meshes: THREE.Mesh[], floors: THREE.Mesh[], cutPlane: THREE.Plane) {
  const raycaster = new THREE.Raycaster()
  const pointer = new THREE.Vector2()
  const roomTriangles = new Map<string, FloorTriangle[]>()
  for (const floor of floors) {
    const scan = typeof floor.userData.room === 'string' ? floor.userData.room : ''
    if (!scan) continue
    if (!roomTriangles.has(scan)) roomTriangles.set(scan, [])
    roomTriangles.get(scan)!.push(...floorTriangles(floor))
  }
  const footprints = new Map([...roomTriangles].map(([scan, triangles]) =>
    [scan, triangles.map(triangle => triangle.plan)]))
  let projected = new Map<string, [THREE.Vector2, THREE.Vector2, THREE.Vector2][]>()
  let projectedVersion = -1

  return function pickRoom(
    x: number,
    y: number,
    width: number,
    height: number,
    camera: THREE.Camera,
    projection: number,
    linked: ReadonlyMap<string, unknown>,
  ): { scan: string | undefined; surface: boolean } {
    pointer.set((x / width) * 2 - 1, 1 - (y / height) * 2)
    raycaster.setFromCamera(pointer, camera)
    const hit = raycaster.intersectObjects(meshes, false).find(({ point }) => cutPlane.distanceToPoint(point) >= -1e-6)
    if (hit) {
      const direct = typeof hit.object.userData.room === 'string' ? hit.object.userData.room : undefined
      const scan = direct ?? nearestRoom(footprints, hit.point.x, hit.point.z, EDGE_TOLERANCE, 0.005)
      return { scan, surface: true }
    }
    if (projectedVersion !== projection) {
      projected = new Map([...roomTriangles].map(([scan, triangles]) => [
        scan,
        triangles.map(({ world }) =>
          world.map(point => {
            const ndc = point.clone().project(camera)
            return new THREE.Vector2(((ndc.x + 1) * width) / 2, ((1 - ndc.y) * height) / 2)
          }) as [THREE.Vector2, THREE.Vector2, THREE.Vector2]),
      ]))
      projectedVersion = projection
    }
    const scan = nearestRoom(projected, x, y, CAPTURE_RADIUS, 0.01, linked)
    return { scan, surface: false }
  }
}
