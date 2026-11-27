import * as THREE from 'three'
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js'
import type { GLTF } from 'three/addons/loaders/GLTFLoader.js'
import type { Catalog } from '../lib/home'
import {
  lightReading, matchRooms, parseHousePlan, thermalReading,
} from './house-plan'
import type { BoxSpec, FloorKind, HousePlan, RoomReading } from './house-plan'
import { createRoomPicker, floorCentroid } from './house-picking'
import { floorTexture, planarUV } from './house-textures'

export type LayerMode = 'lights' | 'climate'

export type SceneHandle = {
  update: (args: { catalog: Catalog | null; layer: LayerMode; selectedRoomId: string | null }) => void
  setView: (mode: '3d' | 'top') => void
  rotate: (direction: number) => void
  zoom: (multiplier: number) => void
  reset: () => void
  dispose: () => void
}

const ROTATE_STEP = Math.PI / 12
const DEFAULT_TILT = (Math.PI * 33) / 180
const DEFAULT_ZOOM = 1
const LABEL_HEIGHT = 0.62
const FIT_MARGIN = 1.055
const DRAG_THRESHOLD = 4
const SELECTED_OUTLINE = 0xb3e4aa

function materialsOf(object: THREE.Object3D): THREE.Material[] {
  const mat = (object as THREE.Mesh).material
  if (!mat) return []
  return Array.isArray(mat) ? mat : [mat]
}

function disposeModel(roots: THREE.Object3D[], extraMaterials = new Set<THREE.Material>()) {
  const geometries = new Set<THREE.BufferGeometry>()
  const materials = new Set<THREE.Material>(extraMaterials)
  const textures = new Set<THREE.Texture>()
  const shadows = new Set<{ dispose: () => void }>()

  for (const root of roots) {
    root.traverse(object => {
      if ('geometry' in object && object.geometry instanceof THREE.BufferGeometry) {
        geometries.add(object.geometry)
      }
      for (const material of materialsOf(object)) materials.add(material)
      if (object instanceof THREE.Light) {
        const light = object as THREE.Light & { shadow?: { dispose?: () => void } }
        if (light.shadow && typeof light.shadow.dispose === 'function') {
          shadows.add(light.shadow as { dispose: () => void })
        }
      }
    })
  }
  for (const material of materials) {
    for (const value of Object.values(material)) {
      if (value instanceof THREE.Texture) textures.add(value)
    }
  }
  for (const tex of textures) {
    const data = (tex.source?.data ?? tex.image) as { close?: () => void } | Array<{ close?: () => void }> | undefined
    for (const image of Array.isArray(data) ? data : [data]) {
      if (typeof image?.close === 'function') image.close()
    }
    tex.dispose()
  }
  for (const geom of geometries) geom.dispose()
  for (const mat of materials) mat.dispose()
  for (const shadow of shadows) shadow.dispose()
}

export async function mountScene(
  canvas: HTMLCanvasElement,
  model: ArrayBuffer,
  rawPlan: Record<string, unknown> | null,
  signal: AbortSignal,
  options: {
    pins?: HTMLElement | null
    stage?: HTMLElement | null
    onSelectRoom?: (roomId: string) => void
  } = {},
): Promise<SceneHandle> {
  signal.throwIfAborted()
  const header = new DataView(model)
  if (
    header.byteLength < 20 ||
    header.getUint32(0, true) !== 0x46546c67 ||
    header.getUint32(4, true) !== 2 ||
    header.getUint32(8, true) !== model.byteLength
  ) {
    throw new Error('Invalid GLB')
  }
  const gl = canvas.getContext('webgl2', { antialias: true }) ?? canvas.getContext('webgl', { antialias: true })
  if (!gl) throw new Error('WebGL unavailable')

  const manager = new THREE.LoadingManager()
  manager.setURLModifier(url => {
    if (url.startsWith('blob:') || url.startsWith('data:')) return url
    throw new Error('External GLB resource')
  })
  manager.onError = url => {
    if (url.startsWith('blob:')) URL.revokeObjectURL(url)
  }

  const loseContext = () => gl.getExtension('WEBGL_lose_context')?.loseContext()
  signal.addEventListener('abort', loseContext, { once: true })
  const gltf: GLTF = await new GLTFLoader(manager)
    .parseAsync(model, '')
    .catch(error => {
      loseContext()
      throw error
    })
    .finally(() => signal.removeEventListener('abort', loseContext))

  const plan = parseHousePlan(rawPlan)
  if (!plan) {
    return mountGenericViewer(canvas, gl, gltf, signal, loseContext, rawPlan)
  }
  return mountInteractiveHouse(canvas, gl, gltf, plan, signal, loseContext, options)
}

function mountGenericViewer(
  canvas: HTMLCanvasElement,
  context: WebGLRenderingContext | WebGL2RenderingContext,
  gltf: GLTF,
  signal: AbortSignal,
  loseContext: () => void,
  plan: Record<string, unknown> | null,
): SceneHandle {
  let renderer: THREE.WebGLRenderer | undefined
  let observer: ResizeObserver | undefined
  const scene = new THREE.Scene()
  const camera = new THREE.PerspectiveCamera(42, 1, 0.01, 1000)
  let disposed = false

  function dispose() {
    if (disposed) return
    disposed = true
    observer?.disconnect()
    scene.clear()
    disposeModel(Array.from(new Set([gltf.scene, ...gltf.scenes])))
    renderer?.dispose()
    loseContext()
    canvas.width = canvas.height = 0
    signal.removeEventListener('abort', dispose)
  }

  try {
    signal.throwIfAborted()
    signal.addEventListener('abort', dispose, { once: true })
    const yaw = plan?.DEFAULT_YAW ?? plan?.default_yaw
    if (typeof yaw === 'number' && Number.isFinite(yaw)) gltf.scene.rotation.y = yaw
    const box = new THREE.Box3().setFromObject(gltf.scene)
    const center = box.getCenter(new THREE.Vector3())
    const size = box.getSize(new THREE.Vector3()).length()
    if (box.isEmpty() || !Number.isFinite(size) || size <= 0) throw new Error('Empty GLB')
    camera.near = Math.max(size / 1000, 0.001)
    camera.far = size * 10
    renderer = new THREE.WebGLRenderer({ canvas, context, alpha: true })
    renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 2))
    const sun = new THREE.DirectionalLight(0xffffff, 2)
    sun.position.set(1, 3, 2)
    scene.add(gltf.scene, sun, new THREE.AmbientLight(0xffffff, 1))
    let angle = Math.PI / 4
    let zoom = 1
    let top = false

    const render = () => {
      if (disposed) return
      const width = canvas.clientWidth || 1
      const height = canvas.clientHeight || 1
      renderer?.setSize(width, height, false)
      camera.aspect = width / height
      camera.updateProjectionMatrix()
      const distance = (size * 1.25) / zoom / Math.min(1, camera.aspect)
      camera.position.set(
        center.x + Math.cos(angle) * distance * (top ? 0.01 : 0.8),
        center.y + distance * (top ? 1 : 0.65),
        center.z + Math.sin(angle) * distance * (top ? 0.01 : 0.8),
      )
      camera.lookAt(center)
      renderer?.render(scene, camera)
    }

    observer = new ResizeObserver(render)
    observer.observe(canvas)
    render()
    return {
      update() {},
      setView(mode) { top = mode === 'top'; render() },
      rotate(delta) { angle += delta; render() },
      zoom(multiplier) { zoom = Math.min(2.8, Math.max(0.5, zoom * multiplier)); render() },
      reset() { angle = Math.PI / 4; zoom = 1; top = false; render() },
      dispose,
    }
  } catch (error) {
    dispose()
    throw error
  }
}

function mountInteractiveHouse(
  canvas: HTMLCanvasElement,
  context: WebGLRenderingContext | WebGL2RenderingContext,
  gltf: GLTF,
  plan: HousePlan,
  signal: AbortSignal,
  loseContext: () => void,
  options: {
    pins?: HTMLElement | null
    stage?: HTMLElement | null
    onSelectRoom?: (roomId: string) => void
  },
): SceneHandle {
  const listeners = new AbortController()
  const modelMaterials = new Set<THREE.Material>()
  gltf.scene.traverse(obj => {
    for (const mat of materialsOf(obj)) modelMaterials.add(mat)
  })
  const otherScenes = gltf.scenes.filter(root => root !== gltf.scene)

  const renderer = new THREE.WebGLRenderer({
    canvas,
    context,
    antialias: true,
    alpha: true,
    preserveDrawingBuffer: true,
    powerPreference: 'low-power',
  })
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.8))
  renderer.setClearColor(0x000000, 0)
  renderer.shadowMap.enabled = true
  renderer.shadowMap.autoUpdate = false
  renderer.shadowMap.type = THREE.PCFSoftShadowMap
  renderer.outputColorSpace = THREE.SRGBColorSpace
  renderer.toneMapping = THREE.ACESFilmicToneMapping
  renderer.toneMappingExposure = 0.95
  renderer.localClippingEnabled = true

  const scene = new THREE.Scene()
  const house = new THREE.Group()
  scene.add(house)
  const camera = new THREE.OrthographicCamera(-12, 12, 8, -8, 0.1, 100)
  const target = new THREE.Vector3(0, 0.2, 0)

  let yaw = plan.DEFAULT_YAW
  let tilt = DEFAULT_TILT
  let zoom = DEFAULT_ZOOM
  let currentYaw = yaw
  let currentTilt = tilt
  let currentZoom = zoom
  let width = 0
  let height = 0
  let dirty = true
  let cameraDirty = true
  let pinsDirty = true
  let rendered = false
  let projection = 0
  let rafId = 0
  let disposed = false
  let lastTime = 0

  scene.add(new THREE.HemisphereLight(0xffffff, 0xb9b4aa, 2.35))
  const keyLight = new THREE.DirectionalLight(0xfff3e0, 1.65)
  keyLight.position.set(5, 16, 6)
  keyLight.castShadow = true
  Object.assign(keyLight.shadow.camera, { left: -10, right: 10, top: 10, bottom: -10, near: 1, far: 40 })
  keyLight.shadow.mapSize.set(1024, 1024)
  keyLight.shadow.bias = -0.0004
  keyLight.shadow.normalBias = 0.03
  scene.add(keyLight)

  const focusLight = new THREE.PointLight(0xffd6a0, 0, 9, 1.6)
  focusLight.castShadow = true
  focusLight.shadow.mapSize.set(512, 512)
  focusLight.shadow.bias = -0.003
  Object.assign(focusLight.shadow.camera, { near: 0.2, far: 9 })
  scene.add(focusLight)

  const cutPlane = new THREE.Plane(new THREE.Vector3(0, -1, 0), plan.CUT_HEIGHT)
  const floors: THREE.Mesh[] = []
  const roomFloors: Record<string, THREE.Mesh[]> = {}
  const centers: Record<string, THREE.Vector3> = {}
  const roomLights: Record<string, THREE.PointLight> = {}
  const outlines: Record<string, THREE.LineSegments> = {}
  const floorColors = new Map<THREE.Mesh, THREE.Color>()
  const hoverColor = new THREE.Color(0x7ba6ee)
  const finishes: Partial<Record<FloorKind, THREE.CanvasTexture>> = {}

  const floorMaterial = (kind: FloorKind) => {
    finishes[kind] ??= floorTexture(kind)
    return new THREE.MeshStandardMaterial({
      map: finishes[kind],
      roughness: kind === 'laminado' ? 0.7 : 0.5,
      metalness: 0,
      emissive: 0x000000,
      emissiveIntensity: 0,
      side: THREE.DoubleSide,
      flatShading: true,
    })
  }

  const addFloor = (mesh: THREE.Mesh, scan: string) => {
    const kind: FloorKind = plan.SCAN_ROOMS[scan]?.floor ?? 'porcelanato'
    finishes[kind] ??= floorTexture(kind)
    const tileSize = (finishes[kind]?.userData.size as number | undefined) ?? 2.4
    planarUV(mesh.geometry, tileSize)
    mesh.material = floorMaterial(kind)
    mesh.receiveShadow = true
    mesh.userData.room = scan
    floors.push(mesh)
    ;(roomFloors[scan] ??= []).push(mesh)
  }

  const box = ([x0, x1, y0, y1, z0, z1]: BoxSpec, material: THREE.Material, parent: THREE.Object3D) => {
    const geometry = new THREE.BoxGeometry(x1 - x0, y1 - y0, z1 - z0)
    geometry.translate((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2)
    const mesh = new THREE.Mesh(geometry, material)
    mesh.castShadow = true
    mesh.receiveShadow = true
    parent.add(mesh)
    return mesh
  }

  const root = gltf.scene
  house.add(root)
  const byName: Record<string, THREE.Object3D> = {}
  root.traverse(obj => {
    if (obj.name) byName[obj.name] = obj
  })
  const tintFor = (name: string) => plan.TINTS.find(entry => name.startsWith(entry.prefix))
  const isGlass = (name: string) =>
    plan.GLASS.names.includes(name) || plan.GLASS.prefixes.some(prefix => name.startsWith(prefix))

  root.traverse(obj => {
    if (!(obj instanceof THREE.Mesh)) return
    const name = obj.name
    if (plan.HIDDEN_PREFIXES.some(prefix => name.startsWith(prefix)) || plan.HIDDEN_NAMES.includes(name)) {
      obj.visible = false
      return
    }
    if (plan.FLOOR_ROOM[name]) {
      addFloor(obj, plan.FLOOR_ROOM[name]!)
      return
    }
    if (isGlass(name)) {
      obj.material = new THREE.MeshStandardMaterial({
        color: 0xbcd8ea,
        roughness: 0.1,
        metalness: 0,
        transparent: true,
        opacity: 0.4,
        emissive: 0xcfe6f5,
        emissiveIntensity: 0.5,
        side: THREE.DoubleSide,
        flatShading: true,
      })
      return
    }
    if (name.startsWith(plan.DOOR_PREFIX)) {
      obj.material = new THREE.MeshStandardMaterial({
        color: 0xf1ede2,
        roughness: 0.7,
        side: THREE.DoubleSide,
        flatShading: true,
      })
      obj.castShadow = true
      return
    }
    const baseMat = Array.isArray(obj.material) ? obj.material[0] : obj.material
    const material = (baseMat instanceof THREE.MeshStandardMaterial ? baseMat.clone() : new THREE.MeshStandardMaterial())
    material.metalness = 0
    material.side = THREE.DoubleSide
    material.flatShading = true
    if (obj.parent?.name === plan.WALL_GROUP) {
      material.roughness = 0.95
    } else {
      material.roughness = 0.75
      const entry = tintFor(name)
      if (entry) {
        material.color.set(entry.color)
        if (entry.roughness !== undefined) material.roughness = entry.roughness
        if (entry.metalness !== undefined) material.metalness = entry.metalness
      }
    }
    obj.material = material
    obj.castShadow = true
    obj.receiveShadow = true
  })

  for (const name of plan.LIGHT_CABINETS) {
    const obj = byName[name]
    if (obj instanceof THREE.Mesh && obj.material instanceof THREE.MeshStandardMaterial) {
      obj.material.color.set(0xecebe6)
    }
  }

  const patches = new THREE.Group()
  house.add(patches)
  const patchMaterial = new THREE.MeshStandardMaterial({ color: 0xe6e6e4, roughness: 0.95 })
  for (const wall of plan.WALL_PATCHES) box(wall.box, patchMaterial, patches)
  if (plan.SPLIT) box(plan.SPLIT.box, new THREE.MeshStandardMaterial({ color: 0xf6f6f4, roughness: 0.4 }), patches)
  if (plan.BALCONY) {
    const { slab, rails, room } = plan.BALCONY
    const slabGeometry = new THREE.BoxGeometry(slab.width, slab.thickness, slab.depth)
    slabGeometry.translate(slab.x + slab.width / 2, -slab.thickness / 2, slab.z)
    const slabMesh = new THREE.Mesh(slabGeometry, new THREE.MeshStandardMaterial())
    patches.add(slabMesh)
    addFloor(slabMesh, room)
    const railMaterial = new THREE.MeshStandardMaterial({ color: 0xe8dfc4, roughness: 0.95 })
    for (const rail of rails) {
      const mesh = new THREE.Mesh(new THREE.BoxGeometry(...rail.size), railMaterial)
      mesh.position.set(...rail.centre)
      mesh.castShadow = true
      mesh.receiveShadow = true
      patches.add(mesh)
    }
  }
  if (plan.STAIRS) {
    const stairs = plan.STAIRS
    const { pit } = stairs
    const hole = new THREE.Path([
      new THREE.Vector2(pit.x0, pit.z0),
      new THREE.Vector2(pit.x0, pit.z1),
      new THREE.Vector2(pit.x1, pit.z1),
      new THREE.Vector2(pit.x1, pit.z0),
    ])
    const landing = new THREE.Shape(stairs.landing.map(([x, z]) => new THREE.Vector2(x, z)))
    landing.holes.push(hole)
    const landingGeometry = new THREE.ShapeGeometry(landing)
    landingGeometry.rotateX(Math.PI / 2)
    const landingMesh = new THREE.Mesh(landingGeometry, new THREE.MeshStandardMaterial())
    patches.add(landingMesh)
    addFloor(landingMesh, stairs.room)
    const depth = (pit.z1 - pit.z0) / stairs.steps
    for (let i = 0; i < stairs.steps; i++) {
      const tread = box(
        [pit.x0, pit.x1, stairs.treadBottom, -stairs.rise * (i + 1), pit.z1 - (i + 1) * depth, pit.z1 - i * depth],
        new THREE.MeshStandardMaterial({ color: 0xa9abad, roughness: 0.7, emissive: 0x000000, emissiveIntensity: 0, flatShading: true }),
        patches,
      )
      tread.userData.room = stairs.room
      floors.push(tread)
      ;(roomFloors[stairs.room] ??= []).push(tread)
    }
    const pitMaterial = new THREE.MeshStandardMaterial({ color: 0xd9d9d6, roughness: 1 })
    const wall = stairs.wall
    const sides: [number, number, number, number][] = [
      [pit.x0 - wall, pit.x0, pit.z0, pit.z1],
      [pit.x1, pit.x1 + wall, pit.z0, pit.z1],
      [pit.x0 - wall, pit.x1 + wall, pit.z0 - wall, pit.z0],
      [pit.x0 - wall, pit.x1 + wall, pit.z1, pit.z1 + wall],
    ]
    for (const [x0, x1, z0, z1] of sides) {
      box([x0, x1, stairs.bottom, stairs.wallTop, z0, z1], pitMaterial, patches)
    }
    box(stairs.rail.box, new THREE.MeshStandardMaterial({ color: 0x1c1d20, roughness: 0.5 }), patches)
  }

  house.updateMatrixWorld(true)
  const footprint = new THREE.Box3()
  const world = new THREE.Vector3()
  for (const floor of floors) {
    footprint.expandByObject(floor)
    const mat = floor.material
    if (mat instanceof THREE.MeshStandardMaterial) floorColors.set(floor, mat.color.clone())
  }
  for (const scan of Object.keys(plan.SCAN_ROOMS)) {
    const main = (byName[plan.MAIN_FLOOR[scan] ?? ''] as THREE.Mesh | undefined) ?? roomFloors[scan]?.[0]
    if (!main) continue
    const centre = floorCentroid(main)
    centers[scan] = centre
    const light = new THREE.PointLight(0xffd6a0, 0, 9, 1.6)
    light.position.set(centre.x, 2.25, centre.z)
    scene.add(light)
    roomLights[scan] = light
    const drawn = roomFloors[scan]?.[0]
    if (!drawn) continue
    const outline = new THREE.LineSegments(
      new THREE.EdgesGeometry(drawn.geometry),
      new THREE.LineBasicMaterial({ color: SELECTED_OUTLINE, depthTest: false, transparent: true }),
    )
    outline.renderOrder = 2
    outline.visible = false
    drawn.add(outline)
    outlines[scan] = outline
  }

  house.traverse(obj => {
    if (!(obj instanceof THREE.Mesh)) return
    for (const material of materialsOf(obj)) {
      material.clippingPlanes = [cutPlane]
      material.clipShadows = true
    }
  })

  house.updateMatrixWorld(true)
  footprint.getCenter(target)
  target.y = 0.2
  const pickable: THREE.Mesh[] = []
  const framePoints: THREE.Vector3[] = []
  house.traverseVisible(obj => {
    if (!(obj instanceof THREE.Mesh)) return
    const positions = obj.geometry.getAttribute('position')
    const points: THREE.Vector3[] = []
    for (let index = 0; index < positions.count; index++) {
      world.fromBufferAttribute(positions, index).applyMatrix4(obj.matrixWorld)
      points.push(world.clone())
    }
    if (points.every(point => point.y > plan.CUT_HEIGHT)) return
    pickable.push(obj)
    for (const point of points) {
      point.y = THREE.MathUtils.clamp(point.y, -0.11, plan.CUT_HEIGHT)
      framePoints.push(point)
    }
  })
  const pickRoom = createRoomPicker(pickable, floors, cutPlane)

  const pinsEl = options.pins ?? null
  const stageEl = options.stage ?? canvas.parentElement ?? canvas
  const onSelectRoom = options.onSelectRoom ?? (() => {})
  const pinsObserverTargets = new Set<HTMLElement>()
  type Binding = { id: string; name: string; zone: string; selected: boolean; reading: RoomReading; pin: HTMLButtonElement }
  let bindings: Binding[] = []
  let signature = ''
  let hovered: string | undefined
  let byZone = new Map<string, Binding>()
  const pinRooms = new WeakMap<Element, string>()

  const offsets: [number, number][] = []
  for (const y of [0, -16, 16, -34, 34, -54, 54, -78, 78, -110, 110, -145, 145]) {
    for (const x of [0, -24, 24, -52, 52, -84, 84, -124, 124]) offsets.push([x, y])
  }
  offsets.sort(([x1, y1], [x2, y2]) => Math.hypot(x1, y1) - Math.hypot(x2, y2))

  const pinPosition = new THREE.Vector3()
  const cameraPoint = new THREE.Vector3()
  const gap = 4

  function projectPins() {
    if (!pinsEl || !width || !height) return
    const occupied: { left: number; top: number; right: number; bottom: number }[] = []
    const anchors = Object.entries(centers).map(([zone, centre]) => {
      pinPosition.copy(centre).project(camera)
      return { zone, x: ((pinPosition.x + 1) * width) / 2, y: ((1 - pinPosition.y) * height) / 2 }
    })
    const targets = bindings
      .filter(({ zone }) => Boolean(centers[zone]))
      .map(({ pin, zone }) => {
        pinPosition.copy(centers[zone]!)
        pinPosition.y = LABEL_HEIGHT
        pinPosition.project(camera)
        const { width: w, height: h } = pin.getBoundingClientRect()
        return { pin, zone, w, h, x: ((pinPosition.x + 1) * width) / 2, y: ((1 - pinPosition.y) * height) / 2 }
      })
    const overlaps = (rect: { left: number; top: number; right: number; bottom: number }, zone: string) =>
      occupied.some(o => rect.left < o.right + gap && rect.right > o.left - gap && rect.top < o.bottom + gap && rect.bottom > o.top - gap) ||
      anchors.some(({ zone: other, x, y }) => zone !== other && x >= rect.left && x <= rect.right && y >= rect.top && y <= rect.bottom)

    const neighbours = (targetItem: (typeof targets)[number]) =>
      targets.filter(
        other =>
          targetItem !== other &&
          Math.abs(targetItem.x - other.x) < (targetItem.w + other.w) / 2 + gap &&
          Math.abs(targetItem.y - other.y) < (targetItem.h + other.h) / 2 + gap,
      ).length
    const priorities = new Map(targets.map(item => [item, neighbours(item)]))
    targets.sort((a, b) => (priorities.get(b) ?? 0) - (priorities.get(a) ?? 0))

    for (const { pin, zone, x, y, w, h } of targets) {
      let position: { left: number; top: number; right: number; bottom: number } | undefined
      for (const [dx, dy] of offsets) {
        const left = THREE.MathUtils.clamp(x + dx - w / 2, gap, Math.max(gap, width - w - gap))
        const top = THREE.MathUtils.clamp(y + dy - h / 2, gap, Math.max(gap, height - h - gap))
        const candidate = { left, top, right: left + w, bottom: top + h }
        if (!overlaps(candidate, zone)) {
          position = candidate
          break
        }
      }
      if (!position) {
        const lefts = [gap, ...occupied.map(r => r.right + gap)]
        const tops = [gap, ...occupied.map(r => r.bottom + gap)]
        let nearest = Infinity
        for (const left of lefts) {
          for (const top of tops) {
            const candidate = { left, top, right: left + w, bottom: top + h }
            const distance = Math.hypot(left + w / 2 - x, top + h / 2 - y)
            if (candidate.right <= width - gap && candidate.bottom <= height - gap && distance < nearest && !overlaps(candidate, zone)) {
              position = candidate
              nearest = distance
            }
          }
        }
      }
      pin.style.visibility = position ? '' : 'hidden'
      if (position) {
        occupied.push(position)
        pin.style.translate = `${position.left}px ${position.top}px`
      }
    }
  }

  function setHover(scan: string | undefined) {
    if (scan === hovered) return
    for (const floor of roomFloors[hovered ?? ''] ?? []) {
      const baseColor = floorColors.get(floor)
      if (baseColor && floor.material instanceof THREE.MeshStandardMaterial) floor.material.color.copy(baseColor)
    }
    byZone.get(hovered ?? '')?.pin.removeAttribute('data-room-hover')
    hovered = scan
    for (const floor of roomFloors[scan ?? ''] ?? []) {
      if (floor.material instanceof THREE.MeshStandardMaterial) floor.material.color.lerp(hoverColor, 0.26)
    }
    if (scan) byZone.get(scan)?.pin.setAttribute('data-room-hover', '')
    canvas.toggleAttribute('data-room-hover', !!scan)
    dirty = true
    scheduleFrame()
  }

  const resize = () => {
    if (disposed) return
    width = stageEl.clientWidth
    height = stageEl.clientHeight
    rendered = false
    if (!width || !height) return
    renderer.setSize(width, height, false)
    cameraDirty = true
  }

  const observer = new ResizeObserver(entries => {
    if (entries.some(entry => entry.target === stageEl || entry.target === canvas)) resize()
    pinsDirty = true
    scheduleFrame()
  })
  observer.observe(stageEl)
  resize()

  const motionQuery = matchMedia('(prefers-reduced-motion: reduce)')
  const cameraUpdate = () => {
    const aspect = width / Math.max(1, height)
    camera.position.set(
      target.x + 26 * Math.sin(currentTilt) * Math.sin(currentYaw),
      26 * Math.cos(currentTilt),
      target.z + 26 * Math.sin(currentTilt) * Math.cos(currentYaw),
    )
    camera.lookAt(target)
    camera.updateMatrixWorld()
    let minX = Infinity
    let maxX = -Infinity
    let minY = Infinity
    let maxY = -Infinity
    for (const corner of framePoints) {
      cameraPoint.copy(corner).applyMatrix4(camera.matrixWorldInverse)
      minX = Math.min(minX, cameraPoint.x)
      maxX = Math.max(maxX, cameraPoint.x)
      minY = Math.min(minY, cameraPoint.y)
      maxY = Math.max(maxY, cameraPoint.y)
    }
    const span = (Math.max(maxY - minY, (maxX - minX) / aspect) * FIT_MARGIN) / currentZoom
    const cx = (minX + maxX) / 2
    const cy = (minY + maxY) / 2
    camera.left = cx - (span * aspect) / 2
    camera.right = cx + (span * aspect) / 2
    camera.top = cy + span / 2
    camera.bottom = cy - span / 2
    camera.updateProjectionMatrix()
    projection++
  }

  function scheduleFrame() {
    if (!rafId && !disposed && width && height && !document.hidden) {
      rafId = requestAnimationFrame(frame)
    }
  }

  function frame(time: number) {
    rafId = 0
    if (disposed || !width || !height || document.hidden || renderer.getContext().isContextLost()) return
    if (time - lastTime < 30) {
      scheduleFrame()
      return
    }
    lastTime = time
    const diff = Math.abs(currentYaw - yaw) + Math.abs(currentTilt - tilt) + Math.abs(currentZoom - zoom)
    if (diff > 0.0001) {
      const ease = motionQuery.matches ? 1 : 0.13
      currentYaw += (yaw - currentYaw) * ease
      currentTilt += (tilt - currentTilt) * ease
      currentZoom += (zoom - currentZoom) * ease
      cameraDirty = true
      if (ease < 1) scheduleFrame()
    }
    if (cameraDirty) {
      cameraUpdate()
      cameraDirty = false
      dirty = true
      pinsDirty = true
    }
    if (dirty) {
      renderer.render(scene, camera)
      dirty = false
      rendered = true
    }
    if (pinsDirty) {
      projectPins()
      pinsDirty = false
    }
  }

  type PointerState = { target: Element; x: number; y: number; startX: number; startY: number; travelled: number }
  const pointers = new Map<number, PointerState>()
  let suppressSelection = false
  let pinching = false

  const releasePointer = (id: number) => {
    const ptrTarget = pointers.get(id)?.target
    pointers.delete(id)
    if (ptrTarget && 'hasPointerCapture' in ptrTarget && ptrTarget.hasPointerCapture(id)) {
      ptrTarget.releasePointerCapture(id)
    }
  }

  const pinAt = (el: EventTarget | null) => {
    if (!(el instanceof Element)) return null
    const pin = el.closest('.room-pin')
    return pin && pinsEl?.contains(pin) ? pin : null
  }

  const roomAt = (event: PointerEvent, pin: Element | null) => {
    const pinRoom = pin ? bindings.find(({ pin: button }) => button === pin) : undefined
    if (pinRoom) {
      const label = pin?.firstElementChild
      if (label) {
        const { left, right, top, bottom } = label.getBoundingClientRect()
        if (event.clientX >= left && event.clientX <= right && event.clientY >= top && event.clientY <= bottom) {
          return pinRoom
        }
      }
    }
    if (!rendered || !byZone.size) return pinRoom
    const rect = canvas.getBoundingClientRect()
    const x = event.clientX - rect.left
    const y = event.clientY - rect.top
    if (x < 0 || x >= rect.width || y < 0 || y >= rect.height) return undefined
    const hit = pickRoom(x, y, rect.width, rect.height, camera, projection, byZone)
    return hit.surface || !pinRoom ? byZone.get(hit.scan ?? '') : pinRoom
  }

  stageEl.addEventListener(
    'pointerdown',
    event => {
      if (disposed || event.button !== 0) return
      const targetEl = pinAt(event.target) ?? (event.target === canvas ? canvas : null)
      if (!targetEl) return
      if (!pointers.size) {
        suppressSelection = false
        pinching = false
      } else {
        suppressSelection = true
        pinching = true
      }
      pointers.set(event.pointerId, {
        target: targetEl,
        x: event.clientX,
        y: event.clientY,
        startX: event.clientX,
        startY: event.clientY,
        travelled: 0,
      })
      targetEl.setPointerCapture(event.pointerId)
      setHover(undefined)
    },
    { signal: listeners.signal },
  )

  const movePointer = (event: PointerEvent, pointer: PointerState) => {
    const { clientX: x, clientY: y } = event
    const distance = Math.hypot(x - pointer.x, y - pointer.y)
    if (!distance) return
    if (pinching && pointers.size === 2) {
      const other = [...pointers.values()].find(item => item !== pointer)
      if (other) {
        const before = Math.hypot(pointer.x - other.x, pointer.y - other.y)
        const after = Math.hypot(x - other.x, y - other.y)
        if (before > 0) {
          zoom = THREE.MathUtils.clamp((zoom * after) / before, 0.65, 1.65)
          scheduleFrame()
        }
      }
    } else if (!pinching) {
      const before = pointer.travelled
      pointer.travelled += distance
      if (pointer.travelled > DRAG_THRESHOLD) {
        suppressSelection = true
        yaw -= (x - (before > DRAG_THRESHOLD ? pointer.x : pointer.startX)) * 0.006
        tilt = THREE.MathUtils.clamp(tilt + (y - (before > DRAG_THRESHOLD ? pointer.y : pointer.startY)) * 0.003, 0.12, 1.18)
        scheduleFrame()
      }
    }
    pointer.x = x
    pointer.y = y
  }

  stageEl.addEventListener(
    'pointermove',
    event => {
      const pointer = pointers.get(event.pointerId)
      if (pointer) movePointer(event, pointer)
      else if (event.pointerType === 'mouse' && !pointers.size) setHover(roomAt(event, pinAt(event.target))?.zone)
    },
    { signal: listeners.signal },
  )
  stageEl.addEventListener('pointerleave', () => setHover(undefined), { signal: listeners.signal })
  stageEl.addEventListener(
    'pointerup',
    event => {
      const pointer = pointers.get(event.pointerId)
      if (!pointer) return
      movePointer(event, pointer)
      releasePointer(event.pointerId)
      if (suppressSelection || disposed) return
      const room = roomAt(event, pinAt(pointer.target))
      if (room) onSelectRoom(room.id)
    },
    { signal: listeners.signal },
  )
  for (const type of ['pointercancel', 'lostpointercapture'] as const) {
    stageEl.addEventListener(
      type,
      event => {
        if (!pointers.has(event.pointerId)) return
        suppressSelection = true
        releasePointer(event.pointerId)
      },
      { signal: listeners.signal },
    )
  }
  stageEl.addEventListener(
    'wheel',
    event => {
      if (pointers.size) suppressSelection = true
      setHover(undefined)
      if (event.ctrlKey || event.metaKey) {
        event.preventDefault()
        zoom = THREE.MathUtils.clamp(zoom - event.deltaY * 0.002, 0.65, 1.65)
        scheduleFrame()
      }
    },
    { passive: false, signal: listeners.signal },
  )
  pinsEl?.addEventListener(
    'click',
    event => {
      const pin = pinAt(event.target)
      if (pin && pinRooms.has(pin) && event.detail === 0) onSelectRoom(pinRooms.get(pin)!)
    },
    { signal: listeners.signal },
  )
  document.addEventListener('visibilitychange', scheduleFrame, { signal: listeners.signal })

  function apply({ catalog, layer, selectedRoomId }: { catalog: Catalog | null; layer: LayerMode; selectedRoomId: string | null }) {
    if (disposed) return
    const entitiesById = new Map((catalog?.entities ?? []).map(entity => [entity.id, entity]))
    const climate = layer === 'climate'
    const catalogRooms = catalog?.rooms ?? []
    const matched = matchRooms(catalogRooms, plan.SCAN_ROOMS)
    const rooms = catalogRooms
      .filter(room => Boolean(centers[matched.get(room.id) ?? '']))
      .map(room => {
        const devices = room.entityIds.map(id => entitiesById.get(id)).filter((e): e is NonNullable<typeof e> => Boolean(e))
        return {
          id: room.id,
          name: room.name,
          zone: matched.get(room.id)!,
          selected: room.id === selectedRoomId,
          reading: climate ? thermalReading(room, devices) : lightReading(room, devices),
        }
      })
    const nextSignature = JSON.stringify([climate, rooms])
    if (nextSignature === signature) return
    signature = nextSignature

    const previous = new Map(bindings.map(b => [b.id, b]))
    bindings = rooms.map(room => {
      const old = previous.get(room.id)
      let pin = old?.pin
      if (old?.zone !== room.zone) pinsDirty = true
      previous.delete(room.id)
      if (!pin) {
        pin = document.createElement('button')
        pin.type = 'button'
        pin.className = 'room-pin'
        pinRooms.set(pin, room.id)
        const tag = document.createElement('span')
        tag.className = 'tag'
        const title = document.createElement('span')
        title.className = 'pin-name'
        const meta = document.createElement('span')
        meta.className = 'pin-meta'
        tag.append(title, meta)
        pin.append(tag)
        if (pinsEl) {
          pinsEl.append(pin)
          observer.observe(pin, { box: 'border-box' })
          pinsObserverTargets.add(pin)
          pinsDirty = true
        }
      }
      pin.setAttribute('aria-pressed', String(room.selected))
      const tag = pin.firstElementChild
      const title = tag?.firstElementChild
      const meta = tag?.lastElementChild
      if (title && title.textContent !== room.name) {
        title.textContent = room.name
        pinsDirty = true
      }
      const badgeText = climate
        ? [room.reading.actual, room.reading.status].filter(Boolean).join(' · ')
        : room.reading.fraction > 0
          ? `${Math.round(room.reading.fraction * 100)}%`
          : ''
      if (meta && meta.textContent !== badgeText) {
        meta.textContent = badgeText
        pinsDirty = true
      }
      pin.toggleAttribute('data-room-hover', room.zone === hovered)
      pin.toggleAttribute('data-room-lit', !climate && room.reading.fraction > 0)
      return { ...room, pin }
    })

    for (const { pin } of previous.values()) {
      for (const [id, pointer] of pointers) {
        if (pointer.target === pin) releasePointer(id)
      }
      if (pinsObserverTargets.has(pin)) {
        observer.unobserve(pin)
        pinsObserverTargets.delete(pin)
      }
      pin.remove()
      pinsDirty = true
    }

    byZone = new Map(bindings.map(room => [room.zone, room]))
    if (hovered && !byZone.has(hovered)) setHover(undefined)

    let focus: { light: THREE.PointLight; fraction: number; color: number } | null = null
    for (const scan of Object.keys(plan.SCAN_ROOMS)) {
      const room = byZone.get(scan)
      const reading = room?.reading
      const fraction = climate ? 0 : (reading?.fraction ?? 0)
      const color = reading?.color ?? 0xffffff
      const light = roomLights[scan]
      if (light) {
        light.color.set(color)
        light.intensity = 0
      }
      if (fraction > 0 && room?.selected && light) {
        focus = { light, fraction, color }
      } else if (fraction > 0 && light) {
        light.intensity = 16 * fraction + 3
      }
      const thermal = climate && Boolean(room)
      for (const floor of roomFloors[scan] ?? []) {
        if (floor.material instanceof THREE.MeshStandardMaterial) {
          floor.material.emissive.set(color)
          floor.material.emissiveIntensity = thermal ? (reading?.active ? 0.15 : 0.03) : fraction * 0.2
        }
      }
      const outline = outlines[scan]
      if (outline && outline.material instanceof THREE.LineBasicMaterial) {
        outline.visible = Boolean(room) && (climate || Boolean(room?.selected))
        outline.material.color.set(room?.selected ? SELECTED_OUTLINE : color)
      }
    }
    focusLight.intensity = focus ? 16 * focus.fraction + 3 : 0
    if (focus) {
      focusLight.color.set(focus.color)
      focusLight.position.copy(focus.light.position)
    }
    renderer.shadowMap.needsUpdate = true
    dirty = true
    scheduleFrame()
  }

  function dispose() {
    if (disposed) return
    disposed = true
    cancelAnimationFrame(rafId)
    observer.disconnect()
    listeners.abort()
    for (const id of pointers.keys()) releasePointer(id)
    pinsEl?.replaceChildren()
    disposeModel([scene, ...otherScenes], modelMaterials)
    renderer.renderLists.dispose()
    renderer.dispose()
    loseContext()
    canvas.width = canvas.height = 0
    signal.removeEventListener('abort', dispose)
  }

  signal.addEventListener('abort', dispose, { once: true })
  cameraDirty = true
  renderer.shadowMap.needsUpdate = true
  scheduleFrame()

  return {
    update: apply,
    setView(mode) {
      if (disposed) return
      tilt = mode === 'top' ? 0.025 : DEFAULT_TILT
      yaw = plan.DEFAULT_YAW
      zoom = DEFAULT_ZOOM
      setHover(undefined)
      scheduleFrame()
    },
    rotate(direction) {
      if (disposed) return
      yaw += (direction < 0 ? 1 : -1) * ROTATE_STEP
      if (motionQuery.matches) {
        currentYaw = yaw
        cameraDirty = true
      }
      setHover(undefined)
      scheduleFrame()
    },
    zoom(multiplier) {
      if (disposed) return
      zoom = THREE.MathUtils.clamp(zoom * multiplier, 0.65, 1.65)
      setHover(undefined)
      scheduleFrame()
    },
    reset() {
      if (disposed) return
      yaw = plan.DEFAULT_YAW
      tilt = DEFAULT_TILT
      zoom = DEFAULT_ZOOM
      setHover(undefined)
      scheduleFrame()
    },
    dispose,
  }
}
