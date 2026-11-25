# Nexa v2

Nexa recibe voz por habitación, interpreta comandos domésticos en español y controla Home Assistant. El procesamiento de audio y la memoria son locales. La salida a un proveedor cloud pasa por un control explícito de consentimiento. Una coincidencia de voz no autoriza acciones administrativas.

## Cómo se conectan los servicios

| Componente | Responsabilidad |
| --- | --- |
| `nexa-audiod` | Captura ALSA por habitación, procesa el wake en CPU y conserva el sello de captura. |
| `nexa-ml` | Worker Python 3.13 sin estado de negocio. Transcribe, sintetiza voz y calcula vectores de voz y texto. |
| `nexa-memoryd` | Custodia memoria y datos ambientales en SQLite WAL con sqlite-vec y registra los borrados. |
| `nexa-brain` | Coordina NLU, Home Assistant, rutinas, recordatorios y proveedores LLM. |
| `nexa-web` | Sirve la SPA autenticada de NEXO en `/` y observabilidad en `/obs`, con permisos y sesiones propios. |

Los cinco daemons corren como servicios nativos de usuario de `nexa` en Linux. Se comunican por sockets Unix privados. Dos servicios auxiliares proveen inferencia local de Qwen 7B y acceso HTTPS al dashboard. El servidor tiene dos RTX 3070 de 8 GiB. La asignación por dispositivo y los micrófonos están en el [inventario de hardware](docs/architecture/HARDWARE.md).

## Desarrollo y puesta en servicio

El workspace Rust está en [crates](crates/), el worker en [workers/nexa-ml](workers/nexa-ml/) y la interfaz Svelte en [frontend](frontend/). La [CI de v2](.github/workflows/v2-ci.yml) verifica las piezas por separado y prueba su integración. Para contribuir, empezar por las [reglas del checkout](AGENTS.md) y la [guía operativa](docs/referencia/guia-operativa.md).

La Ola 4 sigue en curso. Consultar el [roadmap y manifiesto de purga](docs/architecture/NEXA_V2_ROADMAP.md) antes del cutover o del retiro del monolito. El [procedimiento de despliegue](docs/architecture/DEPLOYMENT.md) distingue el paquete reproducible, la migración privada y la promoción autorizada. El [índice](docs/INDEX.md) identifica los documentos vigentes.
