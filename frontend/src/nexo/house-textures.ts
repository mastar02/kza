import * as THREE from 'three'
import type { FloorKind } from './house-plan'

type FinishSpec = {
  base: [number, number, number]
  vary: number
  cols: number
  rows: number
  grout: string
  size: number
}

const FINISHES: Record<FloorKind, FinishSpec> = {
  porcelanato: { base: [176, 180, 183], vary: 10, cols: 2, rows: 8, grout: 'rgba(120,124,128,0.55)', size: 2.4 },
  laminado: { base: [212, 207, 198], vary: 9, cols: 2, rows: 12, grout: 'rgba(150,143,132,0.5)', size: 2.4 },
  banio: { base: [150, 153, 156], vary: 6, cols: 4, rows: 4, grout: 'rgba(95,98,102,0.6)', size: 2.4 },
  balcon: { base: [150, 78, 62], vary: 12, cols: 8, rows: 8, grout: 'rgba(90,50,40,0.7)', size: 1.6 },
}

export function floorTexture(kind: FloorKind): THREE.CanvasTexture {
  const spec = FINISHES[kind] ?? FINISHES.porcelanato
  const canvas = document.createElement('canvas')
  canvas.width = canvas.height = 512
  const g = canvas.getContext('2d')
  if (g) {
    const cw = 512 / spec.cols
    const rh = 512 / spec.rows
    let seed = kind.length * 7919
    const rnd = () => {
      seed = (seed * 16807) % 2147483647
      return seed / 2147483647
    }
    for (let r = 0; r < spec.rows; r++) {
      const offset = (r % 2) * cw * 0.5
      for (let col = -1; col <= spec.cols; col++) {
        const v = (rnd() - 0.5) * spec.vary * 2
        g.fillStyle = `rgb(${(spec.base[0] + v) | 0},${(spec.base[1] + v) | 0},${(spec.base[2] + v) | 0})`
        g.fillRect(col * cw + offset, r * rh, cw, rh)
        for (let k = 0; k < 14; k++) {
          g.fillStyle = `rgba(255,255,255,${rnd() * 0.05})`
          g.fillRect(col * cw + offset + rnd() * cw, r * rh + rnd() * rh, rnd() * cw * 0.6, 1 + rnd() * 2)
        }
        g.strokeStyle = spec.grout
        g.lineWidth = 1.5
        g.strokeRect(col * cw + offset, r * rh, cw, rh)
      }
    }
  }
  const texture = new THREE.CanvasTexture(canvas)
  texture.wrapS = texture.wrapT = THREE.RepeatWrapping
  texture.colorSpace = THREE.SRGBColorSpace
  texture.anisotropy = 8
  texture.userData.size = spec.size
  return texture
}

export function planarUV(geometry: THREE.BufferGeometry, size: number) {
  const p = geometry.attributes.position
  if (!p) return
  const uv = new Float32Array(p.count * 2)
  for (let i = 0; i < p.count; i++) {
    uv[i * 2] = p.getX(i) / size
    uv[i * 2 + 1] = p.getZ(i) / size
  }
  geometry.setAttribute('uv', new THREE.BufferAttribute(uv, 2))
}
