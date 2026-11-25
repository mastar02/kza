# Vendored front-end libraries (no bundler)

| Archivo | Origen | Versión | Licencia |
|---|---|---|---|
| `three.module.min.js` | npm `three` `build/three.module.min.js` | 0.170.0 | `THREE-LICENSE.txt` (MIT) |
| `three-addons/controls/OrbitControls.js`, `loaders/GLTFLoader.js`, `utils/BufferGeometryUtils.js` | npm `three` `examples/jsm/…` | 0.170.0 | `three-addons/LICENSE.txt` (MIT) |
| `inter-latin.woff2` | Inter | — | `INTER-LICENSE.txt` |
| `space-grotesk-latin.woff2` | npm `@fontsource-variable/space-grotesk` `latin-wght-normal` | 5.3.0 | `SPACE-GROTESK-LICENSE.txt` (OFL 1.1) |

Los tres addons de three son copias exactas de r170 salvo una línea: el specifier
`'three'` se reescribió a `'../../three.module.min.js'` para que carguen como módulos ES
sin importmap ni bundler (así no dependen de una CSP que permita scripts inline).

Para actualizar, bajar la misma versión de `three` para los cuatro archivos a la vez
(`three.module.min.js` + addons) y repetir en `vendor/`:

    sed -i '' "s#} from 'three';#} from '../../three.module.min.js';#" three-addons/*/*.js
