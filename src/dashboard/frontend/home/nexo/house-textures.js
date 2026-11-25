// Procedural floor finishes for the scanned rooms (the scan carries geometry, not materials).
import * as THREE from './vendor/three.module.min.js';

const FINISHES = {
  porcelanato: { base: [176, 180, 183], vary: 10, cols: 2, rows: 8, grout: 'rgba(120,124,128,0.55)', size: 2.4 },
  laminado: { base: [212, 207, 198], vary: 9, cols: 2, rows: 12, grout: 'rgba(150,143,132,0.5)', size: 2.4 },
  banio: { base: [150, 153, 156], vary: 6, cols: 4, rows: 4, grout: 'rgba(95,98,102,0.6)', size: 2.4 },
  balcon: { base: [150, 78, 62], vary: 12, cols: 8, rows: 8, grout: 'rgba(90,50,40,0.7)', size: 1.6 }
};

/** Build the repeating canvas texture for one finish; `userData.size` is its edge in metres. */
export function floorTexture(kind) {
  const spec = FINISHES[kind];
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = 512;
  const g = canvas.getContext('2d');
  const cw = 512 / spec.cols, rh = 512 / spec.rows;
  let seed = kind.length * 7919;
  const rnd = () => { seed = (seed * 16807) % 2147483647; return seed / 2147483647; };
  for (let r = 0; r < spec.rows; r++) {
    const offset = (r % 2) * cw * 0.5;
    for (let col = -1; col <= spec.cols; col++) {
      const v = (rnd() - 0.5) * spec.vary * 2;
      g.fillStyle = `rgb(${spec.base[0] + v | 0},${spec.base[1] + v | 0},${spec.base[2] + v | 0})`;
      g.fillRect(col * cw + offset, r * rh, cw, rh);
      for (let k = 0; k < 14; k++) {
        g.fillStyle = `rgba(255,255,255,${rnd() * 0.05})`;
        g.fillRect(col * cw + offset + rnd() * cw, r * rh + rnd() * rh, rnd() * cw * 0.6, 1 + rnd() * 2);
      }
      g.strokeStyle = spec.grout; g.lineWidth = 1.5;
      g.strokeRect(col * cw + offset, r * rh, cw, rh);
    }
  }
  const texture = new THREE.CanvasTexture(canvas);
  texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.anisotropy = 8;
  texture.userData.size = spec.size;
  return texture;
}

/** Top-down UVs in metres so a finish tiles at real scale whatever the floor's shape. */
export function planarUV(geometry, size) {
  const p = geometry.attributes.position;
  const uv = new Float32Array(p.count * 2);
  for (let i = 0; i < p.count; i++) { uv[i * 2] = p.getX(i) / size; uv[i * 2 + 1] = p.getZ(i) / size; }
  geometry.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
}
