# Guía operativa de Nexa v2

## Seguir una petición

El XVF3800 de la habitación entra por ALSA (`mic_channel = 0`, beamformed AEC+NS) a audiod con watchdog isócrono USB. Silero VAD y OpenWakeWord corren en CPU (veto en sombra Parakeet deshabilitado por defecto). El worker ML comparte una instancia `whisper-v3-turbo` (`float16`) para STT de comando y ambiental con prioridad para comandos en `cuda:0`, además de TTS en streaming, identificación de hablante y embeddings (`BAAI/bge-m3` en CPU). Brain coordina la interpretación en español con el estado de HA, los motores del hogar, la destilación conversacional local y el enrutamiento LLM (`Qwen3.5-9B` local en `cuda:1`). Verificar el consentimiento en CloudConsentGate antes de toda salida cloud. Memoryd conserva hechos, datos ambientales y personalidad por perfil de voz bajo políticas de borrado. Identificar una voz no concede permisos de cuenta.

| Área | Dónde trabajar | Qué comprobar |
| --- | --- | --- |
| IPC y configuración | [nexa-proto](../../crates/nexa-proto/), [nexa-config](../../crates/nexa-config/), [TOML](../../config/nexa/) | Contratos compatibles, permisos de sockets y rutas de estado. |
| Audio y modelos | [audiod](../../crates/nexa-audiod/) y [worker ML](../../workers/nexa-ml/) | Puerto USB, sello de captura, modelos precargados y GPU física según el [inventario](../architecture/HARDWARE.md). |
| Control del hogar | [brain](../../crates/nexa-brain/), [NLU](../../crates/nexa-nlu/), [HA](../../crates/nexa-ha/), [household](../../crates/nexa-household/), [LLM](../../crates/nexa-llm/) | Estado HA fresco, horarios persistentes y revisiones CAS. Verificar consentimiento también al cambiar de proveedor. |
| Memoria y migración | [memoryd](../../crates/nexa-memoryd/) y [migrador](../../tools/nexa-migrate/) | Revisar tombstones, ledger, retiros y reportes privados antes de abrir el destino. |
| Web y UI | [web](../../crates/nexa-web/) y [SPA](../../frontend/) | Sesión, rol, revalidación WS v3, acceso autenticado a GLB y separación D1. Activar test-mode solo en el harness. |

## Desarrollo y pruebas

Trabajar desde el checkout confirmado. Para Rust, ejecutar `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets -- -D warnings` y `cargo nextest run --workspace`. En Python usar el entorno del proyecto, DTO con dataclasses, tipos explícitos, logging y operaciones asíncronas. Evitar I/O bloqueante en handlers. Mantener voz e interfaz en español y código y logs en inglés.

La [CI de v2](../../.github/workflows/v2-ci.yml) especifica las dependencias y comandos por plataforma. Hay pruebas ML sin GPU, fixtures de replay L3, controles de tipos y pruebas de la SPA, navegador real L4 y smoke de release L5. Para packaging y migración, con nexa-memoryd de pruebas ya compilado, ejecutar `.venv/bin/pytest tests-v2/smoke tools/nexa-migrate/tests`. El smoke comprueba en Linux las notificaciones systemd que el entorno Mac no puede acreditar.

## Diagnóstico y límites del asiento

Como usuario de ejecución, `systemctl --user status nexa.target` y `journalctl --user -u 'nexa-*'` permiten identificar la unit afectada. Comprobar permisos y propietario del UDS, caché offline de ML y rutas escribibles del servicio antes de considerar un reinicio. Los puertos y permisos figuran en el [contrato de despliegue](../architecture/DEPLOYMENT.md).

Para cambios de infraestructura compartida rigen las [convenciones del servidor](../SERVER_CONVENTIONS.md). Notion pág. 8 es la referencia del contrato compartido y se consulta bajo demanda. Su MCP está retirado de KZA. No leer memorias de homelab-infra ni homelab-services.

Cada especialista usa su propio worktree. Verificar contenido real al comparar ramas con historia reescrita o squash. El worktree bench-ambient tiene ámbito independiente y requiere autorización explícita para intervenirlo. Respetar roles, gate por familia real, diálogos visibles y protocolo de reporte de [CLAUDE](../../CLAUDE.md). Mantener checkpoints en el sistema del proyecto y contrastar el HEAD exacto antes de aceptar una entrega.
