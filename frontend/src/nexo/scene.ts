import {
  AmbientLight, Box3, DirectionalLight, LoadingManager, Mesh, Object3D,
  PerspectiveCamera, Scene, Texture, Vector3, WebGLRenderer,
} from 'three'
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js'

function disposeModel(root: Object3D) {
  const textures = new Set<Texture>()
  root.traverse(object => {
    if (!(object instanceof Mesh)) return
    object.geometry.dispose()
    for (const material of Array.isArray(object.material) ? object.material : [object.material]) {
      for (const value of Object.values(material)) {
        if (value instanceof Texture) textures.add(value)
      }
      material.dispose()
    }
  })
  for (const texture of textures) {
    texture.dispose()
    if (typeof ImageBitmap !== 'undefined' && texture.image instanceof ImageBitmap) texture.image.close()
  }
  root.clear()
}

export async function mountScene(canvas: HTMLCanvasElement, model: ArrayBuffer, plan: Record<string, unknown> | null,
  signal: AbortSignal) {
  signal.throwIfAborted()
  const header = new DataView(model)
  if (header.byteLength < 20 || header.getUint32(0, true) !== 0x46546c67 ||
      header.getUint32(4, true) !== 2 || header.getUint32(8, true) !== model.byteLength) {
    throw new Error('Invalid GLB')
  }
  const gl = canvas.getContext('webgl2', { antialias: true })
  if (!gl) throw new Error('WebGL unavailable')
  const context = gl

  // Parse the authenticated GLB buffer. Only bundled resources may be resolved.
  const manager = new LoadingManager()
  manager.setURLModifier(url => {
    if (url.startsWith('blob:') || url.startsWith('data:')) return url
    throw new Error('External GLB resource')
  })
  const loseContext = () => context.getExtension('WEBGL_lose_context')?.loseContext()
  signal.addEventListener('abort', loseContext, { once: true })
  const gltf = await new GLTFLoader(manager).parseAsync(model, '').catch(error => {
    loseContext()
    throw error
  }).finally(() => signal.removeEventListener('abort', loseContext))
  let renderer: WebGLRenderer | undefined
  let observer: ResizeObserver | undefined
  const scene = new Scene()
  const camera = new PerspectiveCamera(42, 1, 0.01, 1000)
  let disposed = false
  function dispose() {
    if (disposed) return
    disposed = true
    observer?.disconnect()
    scene.clear()
    for (const root of new Set([gltf.scene, ...gltf.scenes])) disposeModel(root)
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
    const box = new Box3().setFromObject(gltf.scene)
    const center = box.getCenter(new Vector3())
    const size = box.getSize(new Vector3()).length()
    if (box.isEmpty() || !Number.isFinite(size) || size <= 0) throw new Error('Empty GLB')
    camera.near = Math.max(size / 1000, 0.001)
    camera.far = size * 10
    renderer = new WebGLRenderer({ canvas, context, alpha: true })
    renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 2))
    const sun = new DirectionalLight(0xffffff, 2)
    sun.position.set(1, 3, 2)
    scene.add(gltf.scene, sun, new AmbientLight(0xffffff, 1))
    let angle = Math.PI / 4
    let zoom = 1
    let top = false
    function render() {
      if (disposed) return
      const width = canvas.clientWidth || 1
      const height = canvas.clientHeight || 1
      renderer?.setSize(width, height, false)
      camera.aspect = width / height
      camera.updateProjectionMatrix()
      const distance = size * 1.25 / zoom / Math.min(1, camera.aspect)
      camera.position.set(center.x + Math.cos(angle) * distance * (top ? 0.01 : 0.8),
        center.y + distance * (top ? 1 : 0.65), center.z + Math.sin(angle) * distance * (top ? 0.01 : 0.8))
      camera.lookAt(center)
      renderer?.render(scene, camera)
    }
    observer = new ResizeObserver(render)
    observer.observe(canvas)
    render()
    return {
      rotate(delta: number) { angle += delta; render() },
      zoom(multiplier: number) { zoom = Math.min(2.8, Math.max(0.5, zoom * multiplier)); render() },
      top() { top = !top; render() },
      dispose,
    }
  } catch (error) {
    dispose()
    throw error
  }
}
