# Convenciones del servidor compartido — espejo local

> Espejo de Notion KZA **pág. 8** ("Convenciones del servidor compartido") y **pág. 9** ("Onboarding — cómo sumar un proyecto nuevo"), extraído el **2026-06-09** vía MCP.
> La fuente canónica sigue siendo Notion; si este doc y Notion difieren, **gana Notion**. Root del workspace: page_id `345ab24f-c493-80b2-b6f4-ef917e865f26`.
> Nota: la pág. 8 es un log append-only con actualizaciones fechadas; ante conflicto entre secciones, prevalece la más reciente.
> Inconsistencias conocidas DEL ORIGINAL (espejadas fielmente, no errores de este doc): la pág. 9 dice "reglas R1-R12" pero la pág. 8 define hasta R14; el mapa de puertos ubica `:9501` (Kibana, obs) dentro del sub-rango 9500-9599 reclamado por KZA; el sub-rango de thermouy (9320-9329) quedó dentro del de konsi (9300-9399). El propio original (auditoría 2026-05-01) reconoce que el mapa necesita auditoría.

---

## Pág. 8 — Convenciones del servidor compartido

### Propósito

El servidor `192.168.1.2` aloja múltiples proyectos que comparten CPU, RAM, VRAM y red. Este documento es el contrato de convivencia: define cómo se estructura un proyecto, cómo se despliega, qué recursos comparte y cuáles son las reglas no negociables. Cualquier proyecto que se sume al servidor debe respetarlo. Si un proyecto no puede hacerlo, renegociá las reglas antes de desplegar, no las ignores.

### El modelo en una línea

1 usuario Linux = 1 proyecto. N contenedores por componente. Podman rootless + systemd --user vía Quadlet. GPU vía NVIDIA CDI. Red interna por proyecto. Servicios de plataforma compartidos alojados bajo usuario `infra`.

### Granularidad

- Un proyecto = un usuario Linux. Si el proyecto tiene 8 componentes (API, worker, scheduler, DB propia, etc.), los 8 viven bajo el mismo usuario como contenedores separados. No se crean 8 usuarios.
- Un componente = un contenedor. Cada unidad deployable corre en su propio contenedor con su propio Dockerfile.
- Se separan en usuarios distintos solo cuando: son proyectos con dueños distintos, requieren aislamiento fuerte entre sí, o tienen ciclos de vida totalmente desacoplados.

### Hardware compartido

- **CPU**: Threadripper PRO 7965WX, 24c/48t. El LLM 72B de KZA puede consumir hasta 24 threads cuando activo; el resto debe convivir con eso.
- **RAM**: 128 GB DDR5. Qwen2.5-72B Q6_K reserva ~71 GB cuando cargado. Presupuesto restante: ~50 GB. *(Nota 2026-05-30: el 72B local fue reemplazado por gateway cloud — ver cutover abajo; libera ~63 GB.)*
- **GPUs**: 2x RTX 3070 (8 GB VRAM c/u). cuda:0 reservada a audio de KZA (~6 GB). cuda:1 compartida para LLM + embeddings.
- **Driver NVIDIA**: 580.126.09 / CUDA 13.0. Secure Boot disabled.
- **Red**: puertos TCP son recurso global. Antes de bindear uno nuevo, revisar el mapa de esta página.

### Servicios de plataforma compartidos (bajo usuario `infra`)

- **vLLM**: inferencia LLM compartida vía API OpenAI-compatible en `127.0.0.1:8100`. Modelo: Qwen2.5-7B-AWQ. Cualquier proyecto consume por HTTP; no levantar una segunda instancia. *(Estado 2026-05-30: CAÍDO/disabled — ver cutover.)*
- **nginx reverse proxy** (futuro): expone servicios al LAN vía :80/:443. Cada proyecto que necesita exposición externa registra su hostname interno + puerto.
- **Postgres compartido**: ~~futuro~~ **CANCELADO 2026-04-21** — cada proyecto trae su propio ecosistema completo (Postgres/Redis/colas/cache) bajo su usuario rootless. Centralizar BD crea coupling y blast radius grande. Los únicos servicios compartidos en infra quedan los que tienen sentido por costo real: vLLM (por VRAM) y futuro nginx (por bind privilegiado :80/:443).
- Estos servicios son los únicos que un proyecto puede esperar encontrar al desplegarse. Todo lo demás (DB, caché, cola) lo trae en sus propios contenedores.

### Reglas duras

- **R1 — No duplicar servicios de plataforma.** Si necesitás un LLM vía HTTP, consumí el compartido. Si necesitás un modelo distinto, se discute antes de levantar una segunda instancia.
- **R2 — No matar contenedores o procesos de otro usuario.** Nunca `podman kill/stop` ni `systemctl --user stop` sobre servicios ajenos sin coordinar con su dueño.
- **R3 — Cambios a servicios compartidos** (modelo, parámetros, recursos) requieren coordinación previa y registro en la pág. 8 antes de aplicarlos.
- **R4 — Naming.** Imagen: `localhost/<proyecto>/<componente>:<version>`. Contenedor: `<proyecto>-<componente>`. Quadlet: `~/.config/containers/systemd/<proyecto>-<componente>.container`. Todo grepeable por prefijo del proyecto.
- **R5 — Declarar puertos en la pág. 8 antes del primer deploy.** El mapa es fuente de verdad. Puertos no declarados pueden ser reasignados sin aviso.
- **R6 — Publicación solo a 127.0.0.1.** Mapear con `-p 127.0.0.1:PORT:PORT`. Exposición a LAN solo vía nginx de infra. Comunicación entre contenedores del mismo proyecto por su red interna, no por loopback del host.
- **R7 — Límites declarados en todo contenedor.** `--memory`/`--memory-swap`, `--cpus`/CPUQuota, y `--device nvidia.com/gpu=X` para GPU específica.
- **R8 — Restart policy responsable.** En Quadlet: `Restart=on-failure`, `RestartSec=5s` mínimo, `StartLimitBurst=5`. Nunca `Restart=always` sin RestartSec.
- **R9 — Logs por journald.** Consulta: `journalctl --user -u <proyecto>-<componente> -f`. Prohibido escribir logs en `/var/log` o fuera del home del usuario.
- **R10 — Rootless obligatorio.** Podman rootless. Zero grupo docker. Zero sudo operacional (sudo solo para: crear usuario, instalar paquetes del sistema, generar CDI la primera vez). *Excepciones documentadas abajo.*
- **R11 — Dockerfile versionado en el repo del proyecto.** Build reproducible: FROM con digest o tag inmutable, dependencias pinneadas.
- **R12 — Imágenes con tag inmutable en deploy.** No `:latest` en Quadlet de producción. Versión semántica o SHA del commit.
- **R13 — Prefijo de índices en Elasticsearch compartido (obs).** Patrón `<app>-logs-YYYY.MM.dd` (ej: `kza-logs-*`). Prohibidos los prefijos reservados de ES 8.x: `logs-*`, `metrics-*`, `traces-*`, `synthetics-*`. El template `logs-apps` (priority 600) aplica ILM logs-1d + 1 shard + 0 replicas. Registrar el prefijo reclamado antes del primer deploy. Output hacia `http://host.containers.internal:9200` (contenedor) o `http://127.0.0.1:9200` (nativo).
- **R14 — Servicios con vista privilegiada del host** (node_exporter, cAdvisor): pueden montar `/proc /sys /` read-only — excepción documentada. Siguen rootless y publicados solo a 127.0.0.1.

### Excepciones a R10 (documentadas)

1. **vLLM nativo bajo infra (2026-04-20)** — el overhead de NVIDIA CDI (~0.34 GB sobre 8 GB) obligaba a `gpu-memory-utilization 0.92`; se revirtió a nativo (`/home/infra/vllm-venv/`, systemd --user) con 0.80. Criterio para nativo en servicios de plataforma: (a) servicio estable y pinneado, (b) recursos al límite y el overhead del container lo agrava, (c) el dueño del servicio opera la infra. Forma nativa correcta: venv en `/home/<usuario>/<proyecto>-venv/`, unit en `~/.config/systemd/user/`, Type=simple, ExecStart absoluto al venv, Restart=on-failure.
2. **ha-core rootful + host network (2026-04-21)** — Podman rootless con `--network=host` no ve la red física (slirp4netns/pasta); HA necesita broadcasts mDNS/SSDP para discovery. Quadlet rootful en `/etc/containers/systemd/ha-core.container`, `MemoryMax=3G`, `AddCapability=NET_ADMIN,NET_RAW`. Criterio rootful: (a) requiere visibilidad de red física real, (b) multicast/broadcast esencial, (c) sin alternativa rootless.
3. **mail/Mailcow bajo root (requisito Docker)** — user `mail` (UID 8) existe pero Mailcow corre bajo root directamente.
4. **unbound nativo bajo dns (2026-04-30)** — imagen mvance no es rootless-friendly. AppArmor override en `/etc/apparmor.d/local/usr.sbin.unbound` para leer config bajo `/home/dns/` (template para futuros servicios nativos). *(KZA-voice también corre nativo bajo systemd --user — excepción R10 propia, análoga a la #1 (vLLM nativo): acceso USB ReSpeaker + MA1260 serial + presupuesto <300ms hacen al contenedor inviable.)*
5. **obs-filebeat nativo bajo obs (2026-06-04)** — imagen oficial 8.15.5 (libsystemd 245) no lee formato compact de systemd 255; keep-groups requiere crun no instalado.

### Excepción a R6 — proxy intra-proyecto (2026-04-24)

Cuando un proyecto tiene su propio reverse proxy contenedorizado dentro de su red privada (ej: postpilot-nginx), los upstream NO necesitan publicar puerto al host. El único egress es el cliente de tunnel (cloudflared) vía outbound 443. R6 aplica a servicios sin proxy; con proxy intra-proyecto + tunnel no hace falta PublishPort.

### Mapa de puertos (fuente de verdad: Notion pág. 8; snapshot 2026-06-09)

| Puerto | Servicio | Owner |
|---|---|---|
| :22 | SSH | sistema |
| :53 | DNS (dnsmasq+libvirt; AGH balanced/aggressive en 192.168.30.2/.3) | sistema / dns |
| :80, :3001 | nginx reverse proxy (nativo) | sistema (futuro: infra) |
| :631 | cupsd | sistema |
| :1883 | Mosquitto MQTT (Quadlet) | infra |
| :5053 | unbound recursivo (0.0.0.0 con access-control) | dns |
| :5432 | PostgreSQL (Docker nativo legacy, pendiente cleanup) | kza |
| :5514 | Logstash syslog (UDP+TCP, bind LAN para OPNsense) | obs |
| :5900 | QEMU/KVM VNC (consola OPNsense) — cerrado a LAN vía nftables 2026-05-30, solo localhost/SSH | sistema |
| :8000 | trading-bot API | trading (bajo kza, migración pendiente) |
| :8080 | nginx → Kibana (htpasswd) | sistema |
| :8081 | nginx → Grafana (htpasswd) | sistema |
| :8100 | vLLM compartido — **CAÍDO al 2026-05-30** (disabled) | infra |
| :8101 | llama-server Qwen2.5-7B-Instruct-Q4_K_M (fast-path domótica KZA), full GPU cuda:1, activo | kza |
| :8123 | Home Assistant (rootful, host network) | ha |
| :8200 | LiteLLM gateway → MiniMax-M2.7 cloud (container rootless). Consumers: KZA reasoner + Open-WebUI con virtual key | infra |
| :8300/:8301 | postpilot-embeddings (bge-m3, knowledge/code) — bindean 0.0.0.0, pendiente restringir | (bajo kza) |
| :8953 | unbound remote-control (127.0.0.1) | dns |
| :9090/:9100/:9835 | Prometheus / node_exporter / nvidia_gpu_exporter (loopback) | obs |
| :3000 | Grafana (loopback; LAN vía :8081) | obs |
| :9200 | Elasticsearch (loopback + red obs-internal) | obs |
| :9501 | Kibana (loopback; LAN vía :8080) | obs |
| :9520/:9521 | Open-WebUI (→:8200) / SearXNG | kza |
| :18554 | go2rtc cámaras | video/HA |

**Sub-rangos del rango :9000-9999 para proyectos nuevos** (cada proyecto reclama el suyo al registrarse):

- 9200-9299 → postpilot · 9300-9399 → konsi (9301 prod, 9310-9339 slots feature envs, 9340-9399 reserva) · 9320-9329 → thermouy · 9400-9499 → sockerdata · **9500-9599 → KZA** · 9700-9711 → dns/obs.
- ⚠️ Auditoría 2026-05-01: hay puertos en 9000-9999 ya en uso no declarados (9090, 9100, 9101, 9114, 9198-9199, 9210-9220, 9310-9312, 9400-9402, 9443, 9465, 9500-9501, 9510, 9587, 9617, 9700-9711, 9835, 9993-9995) — el mapa necesita auditoría para que R5 sea efectivo.

### Mapa de GPU y VRAM

- **cuda:0** (bus c1:00.0) — dedicada a audio de KZA: Whisper STT + ECAPA-TDNN SpeakerID + wav2vec2 Emotion + Kokoro TTS + BGE-M3 (~1.5 GB; va acá porque cuda:1 no tiene espacio). Presupuesto: ~7.5 GB. Dueño: kza. Reservada exclusivamente a KZA.
- **cuda:1** (bus c2:00.0) — vLLM compartido bajo infra (nativo, `--gpu-memory-utilization 0.80`, ~7.2 GiB pinneada). *(2026-05-30: vLLM caído; hoy cuda:1 la usa el llama-server :8101 de KZA.)*
- **CPU inference** — Qwen2.5-72B Q6_K bajo kza (slow path, esporádico). Mientras activo, los demás procesos CPU-intensivos reducen prioridad (nice). *(2026-05-30: reemplazado por gateway :8200 → MiniMax cloud.)*
- Proyectos nuevos que necesiten GPU negocian en la pág. 8 antes de `AddDevice`. Declarar VRAM máxima y GPU preferida. `--gpus all` sin declarar está prohibido.
- vLLM no hace scheduling multi-tenant; el contrato "quién usa qué GPU" se enforza por doc + `AddDevice=nvidia.com/gpu=N` en Quadlets.

### Estructura estándar de un proyecto

- `/home/<proyecto>/app/` — código fuente (clone del repo), un Dockerfile por componente.
- `/home/<proyecto>/data/` — volúmenes persistentes por componente (`Volume=%h/data/<componente>:/app/data:Z`).
- `/home/<proyecto>/logs/` — opcional; por defecto todo va al journal del usuario.
- `/home/<proyecto>/secrets/` — chmod 700. `.env` y otros secrets, inyectados con `EnvironmentFile=` en Quadlet, nunca `COPY` en la imagen.
- `/home/<proyecto>/.config/containers/systemd/` — archivos Quadlet (.container, .network, .volume, .pod). El generator de Podman los lee SOLO de acá, no de `.config/systemd/user/`.

### Patrones de red

- **Red interna por proyecto**: Quadlet `.network` (ej: `<proyecto>-internal.network`); los contenedores resuelven entre sí por nombre (DNS interno de Podman).
- **Servicios compartidos del host** desde un contenedor: `http://host.containers.internal:<puerto>`.
- **Publicación al host**: solo componentes edge con `-p 127.0.0.1:PORT:PORT`; los internos no publican.
- **Exposición a LAN**: edge publica a 127.0.0.1, nginx de infra hace proxy. Nunca 0.0.0.0 directo desde un contenedor.

### GPU sharing vía NVIDIA CDI

- Setup (una vez, sudo): instalar nvidia-container-toolkit, `sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml`. Regenerar al cambiar driver o GPUs.
- Uso rootless: `podman run --device nvidia.com/gpu=1 ...` (0/1/all). En Quadlet: `AddDevice=nvidia.com/gpu=1`. Sin grupo docker, sin `--privileged`.

### Catálogo y administración de modelos GPU (2026-04-20; amplía "Cómo consumir el vLLM compartido")

Todos los modelos GPU son administrados por `infra`. Los proyectos consumen por HTTP; no levantan modelos propios (R1).

**4.1 Catálogo** (snapshot; verificar contra Notion):

| Nombre lógico | Tipo | Endpoint | Modelo | GPU | VRAM | Max ctx | Owner | Estado |
|---|---|---|---|---|---|---|---|---|
| qwen2.5-7b-awq | LLM | `http://host.containers.internal:8100/v1` | Qwen2.5-7B-Instruct-AWQ | cuda:1 | ~7.7 GB | 4096 | infra | estable *(caído 05-30)* |
| bge-m3 | embedding | interno kza (no-API) | BAAI/bge-m3 | cuda:0 | ~1.5 GB | 8192 | kza | estable |
| whisper-large-v3 | ASR | interno kza (no-API) | openai/whisper-large-v3 | cuda:0 | ~2 GB | — | kza | estable |
| qwen2.5-72b-q6_k | LLM slow | CPU, sin HTTP | Qwen2.5-72B-Instruct Q6_K | CPU | 71 GB RAM | 32k | kza | esporádico *(retirado 05-30)* |

**4.2 Política de consumo**: health check `GET /v1/models` (timeout <2s, cache 60s); retry con backoff exponencial (250ms→cap 5s) ante 503/timeout; sin fallback local (R1); timeout cliente 30s default; respetar max ctx truncando del lado consumidor.

**4.4 Pedir un modelo nuevo**: entrada en sub-tabla "Pedidos pendientes" de Notion (modelo, VRAM, GPU, caso de uso, owner, urgencia) → infra evalúa → resolución ≤5 días hábiles → si aprueba, infra lo aloja y actualiza 4.1.

**4.5 Ciclo de vida**: add = ventana 24h antes de "estable"; swap = ventana ≥3 días hábiles, API compatible obligatoria; retire = aviso ≥2 semanas con notificación 1:1 a consumidores. Todo cambio va a la sub-tabla "Historial" (2026-04-20: alta qwen2.5-7b-awq :8100; 2026-04-20: bind 127.0.0.1→0.0.0.0 para acceso desde contenedores rootless).

### Usuarios / proyectos registrados (UIDs)

| Usuario | UID | Rol |
|---|---|---|
| kza | 1000 | Asistente de voz KZA (nativo, excepción R10; cuda:0 dedicada; sub-rango 9500-9599) |
| infra | 1001 | Plataforma compartida: vLLM (nativo), Mosquitto :1883 (Quadlet), LiteLLM gateway :8200, futuro nginx. Grupos video/render, linger |
| trading-bot | 1002 | Consumer activo de trading (config viva en `/home/trading-bot/secrets/trading.env`; tras cutover apunta al gateway :8200). Existe además user `trading` con config vieja NO activa |
| postpilot | 1004 | Social media mgmt, 11 contenedores, sub-rango 9200-9299, ~18 GB RAM |
| obs | 1005 | Observabilidad: ELK 8.15.5 + Prometheus/Grafana/exporters (Quadlets, red obs-internal) |
| sockerdata | 1007 | ML predicción deportiva, 7 contenedores, sub-rango 9400-9499, ~17.5 GB RAM, autocontenido |
| cftunnel | 1008 | Daemon cloudflared único para N dominios públicos (blast radius mínimo) |
| mail | 8 | Mailcow bajo root (excepción R10 #3) |
| dns | 1009 | DNS interno per-VLAN: unbound nativo + AGH balanced/aggressive (192.168.30.2/.3:53) |
| thermouy | 1010 | SaaS termografía, 8 contenedores, sub-rango 9320-9329, ~14 GB RAM |
| konsi | 1012 | SaaS B2B consultia.com.uy, 3 contenedores, sub-rango 9300-9399 (9301 edge), tunnel CF dedicado |
| ha (root) | — | ha-core rootful host-network :8123 (excepción R10 #2) |

Regla para nuevos servicios compartidos (2026-04-24): (1) stateful de aplicaciones (cache, broker, cola) → `infra`; (2) expone-servicios-públicos (tunnel, reverse proxy, DNS) → user dedicado con blast radius mínimo; (3) per-app → user del app.

### Proceso para sumar un proyecto / pedir cambios

- Onboarding: checklist de la pág. 9 (abajo). Al finalizar: actualizar "Proyectos registrados" en Notion pág. 8.
- Cambios que afectan a otros (modelo/parámetros de vLLM, reasignar GPU, IP/hostname, servicio en 0.0.0.0, cliente pesado al vLLM): registrar propuesta en pág. 8 con impacto y rollback → OK de los dueños afectados → ejecutar → actualizar secciones.

### Qué NO hacer (resumen)

- Correr servicios como root o como el usuario kza (kza es dueño de su proyecto, no owner universal).
- Instalar dependencias globales con sudo pip/apt — todo vive en la imagen.
- Crear units en `/etc/systemd/system/` — todo va en systemd --user vía Quadlet.
- Bindear 0.0.0.0 desde un contenedor.
- Duplicar vLLM, Postgres o nginx.
- Usar `:latest` en producción.
- Matar procesos/contenedores ajenos sin coordinar.
- Levantar un proyecto sin registrarlo en la pág. 8 (único inventario válido del servidor).

### Actualizaciones operativas relevantes (cronológico)

- **2026-04-23 — Estabilidad del host tras incidente OOM**: needrestart (vía apt-daily-upgrade) escaneó `/proc/*/maps` con el 72B mmapeado (~37 GB) → ~86 GiB RSS → OOM global, host colgado ~10h, OPNsense (VM en el host) cayó con él. Mitigación: `apt-daily{,-upgrade}.timer` disabled+masked; needrestart en list-only (`$nrconf{restart}='l'`). `virsh autostart opnsense`. dnsmasq del host disabled+masked (DHCP roto redundante). NIC eno2np1 (LAN activa) autoconnect=yes. Señales de recurrencia: oom-killer con `task_memcg=/system.slice/apt-*`, journald "Under memory pressure", workqueue hogged CPU.
- **2026-04-30 — proyecto dns + hardening LLM Tier 1**: vLLM :8100 y llama-server :8200 requieren bearer token desde 2026-04-30. Consumers (kza-voice, trading-backend) actualizados con auth. Sysctl persistente `net.ipv4.ip_unprivileged_port_start=53` (puertos bajos rootless).
- **2026-05-20 — feature envs por sesión Claude Code**: patrón konsi de slots efímeros (containers `<proyecto>-slot-N-*`, redes aisladas por slot). Generalizable, ver Notion pág. 17.
- **2026-05-30 — Cutover del razonador a MiniMax + auditoría** (prevalece ante conflicto):
  - gpt-oss-120b local (:8200, kza-llm-ik) reemplazado por **gateway LiteLLM rootless bajo infra → MiniMax-M2.7** (api.minimax.io). Libera ~63 GB RAM y 128 GB disco. kza-llm-ik inhabilitado de raíz (unit `.disabled`).
  - Rompe el 100% local en el razonador (decisión informada; KZA con `cloud.consent=true`). La key real de MiniMax vive solo en `/home/infra/secrets/minimax.env`; KZA y Open-WebUI usan virtual key local (LITELLM_MASTER_KEY). El gateway es el único proceso LLM con egress a internet (resto mantiene IPAddressDeny).
  - Nota LiteLLM: con master_key estática, un consumer con key distinta produce `400 'No connected db'` — es síntoma de **key incorrecta del consumer**, no de DB caída.
  - Convenciones reforzadas: healthchecks con herramientas presentes en la imagen base; `podman image prune` periódico para usuarios con builds frecuentes; binding default 127.0.0.1 (0.0.0.0 solo justificado); prohibido password plano en compose (usar secrets); toda DB de proyecto debe tener dump automatizado con retención (hoy ninguna lo tiene); `vm.swappiness=10` aplicado y persistido.
  - Deudas detectadas: kza-chroma en crash-loop (Quadlet monta `chroma` pero el path real es `chroma_db`); postpilot-frontend-v2 crash-loop (conflicto :9211); dns-sync.service roto (git pull bloqueado por untracked); AGH sin admin; exporters/targets de Prometheus DOWN (HA 401, trading :9302).

---

## Pág. 9 — Onboarding: cómo sumar un proyecto nuevo

### Audiencia y prerrequisitos

Manual de ejecución para sumar un proyecto al servidor `192.168.1.2` con el modelo Podman rootless. Prerrequisito: leer la pág. 8 (reglas R1-R12 que este checklist ejecuta).

### Flujo en 5 pasos

1. Crear el usuario Linux del proyecto (admin, una vez).
2. Preparar el entorno del usuario (Podman, directorios, linger, CDI si usa GPU).
3. Escribir Dockerfile(s) y buildear imágenes.
4. Escribir Quadlets (.network + .container por componente) y arrancar vía systemd --user.
5. Registrar el proyecto en pág. 8 y monitorear 24 h.

### Checklist detallado (P1-P16)

- **P1 — Planificar**: nombre corto (= usuario Linux y prefijo), listar componentes, estimar VRAM/RAM/CPU/puertos, elegir sub-rango libre en :9000-9999.
- **P2 — Crear usuario** (admin): `sudo useradd -m -s /bin/bash -c "Proyecto <nombre>" <nombre>`. Sin password.
- **P3 — Linger** (admin): `sudo loginctl enable-linger <nombre>`.
- **P4 — Grupos GPU** (admin, si usa GPU): `sudo usermod -aG video,render <nombre>` (higiene; CDI no lo exige).
- **P5 — Entrar como el usuario**: `sudo -u <nombre> -i`. Todo lo siguiente sin sudo.
- **P6 — Estructura**: `mkdir -p ~/app ~/data ~/logs ~/secrets ~/.config/systemd/user ~/.config/containers/systemd; chmod 700 ~/secrets`.
- **P7 — Verificar Podman y GPU**: `podman --version`; GPU: `podman run --rm --device nvidia.com/gpu=1 docker.io/nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi`. Si falla: regenerar CDI.
- **P8 — Clonar repo** en `~/app` (un Dockerfile por componente).
- **P9 — Secrets**: `~/secrets/.env` chmod 600. Nunca commitear.
- **P10 — Build**: `podman build -t localhost/<proyecto>/<componente>:v0.1 .` por componente.
- **P11 — Red interna**: `~/.config/containers/systemd/<proyecto>-internal.network`.
- **P12 — Quadlet por componente**: `<proyecto>-<componente>.container` (imagen, red, puertos solo edge, volúmenes, límites).
- **P13 — Arrancar**: `systemctl --user daemon-reload && systemctl --user start <svc>`; `enable` para boot.
- **P14 — Verificar**: `podman ps`, `systemctl --user status`, `journalctl --user -u <svc> -f`, `ss -tlnp | grep 127.0.0.1:<PUERTO>`.
- **P15 — Registrar en pág. 8** (nombre, dueño, componentes, puertos, VRAM/RAM, servicios compartidos consumidos, estado).
- **P16 — Monitorear 24 h**: nvidia-smi, free -g, podman stats, journalctl sin errores recurrentes.

### Templates

**Dockerfile (Python genérico):**

```dockerfile
FROM python:3.13-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["python", "-m", "src.main"]
```

**Quadlet `.network`** (`~/.config/containers/systemd/<proyecto>-internal.network`):

```ini
[Network]
NetworkName=<proyecto>-internal
Internal=false
DNS=10.89.0.1

[Install]
WantedBy=default.target
```

**Quadlet `.container`** (`~/.config/containers/systemd/<proyecto>-<componente>.container`):

```ini
[Unit]
Description=<proyecto> <componente>
After=network-online.target
Wants=network-online.target

[Container]
Image=localhost/<proyecto>/<componente>:v0.1
ContainerName=<proyecto>-<componente>
Network=<proyecto>-internal.network
EnvironmentFile=%h/secrets/.env
Volume=%h/data/<componente>:/app/data:Z
# Publicar SOLO si es componente edge (los internos omiten PublishPort)
PublishPort=127.0.0.1:9100:8000

[Service]
Restart=on-failure
RestartSec=5
StartLimitBurst=5
MemoryMax=2G

[Install]
WantedBy=default.target
```

**Componente con GPU** — agregar al `[Container]`:

```ini
AddDevice=nvidia.com/gpu=1   # 0 o 1 según GPU asignada; consultar pág. 8
Environment=NVIDIA_VISIBLE_DEVICES=all
Environment=NVIDIA_DRIVER_CAPABILITIES=compute,utility
```

Imagen base con CUDA runtime compatible con driver 580.126.09 / CUDA 13.0 (ej: `docker.io/nvidia/cuda:12.4.1-base-ubuntu22.04`).

**Cliente del vLLM compartido (Python):**

```python
from openai import OpenAI
# Desde dentro de un contenedor, usar host.containers.internal
client = OpenAI(base_url='http://host.containers.internal:8100/v1', api_key='not-used', timeout=30)
resp = client.chat.completions.create(model='qwen2.5-7b-awq', messages=[{'role': 'user', 'content': '...'}])
# Health check previo: requests.get('http://host.containers.internal:8100/v1/models', timeout=2)
```

*(Nota 2026-04-30: los endpoints LLM ahora requieren bearer token — la `api_key` ya no es decorativa; tokens en `/home/kza/secrets/*.env` para KZA.)*

### Deploy pull-based (sin sudo, sin SSH root)

- El servidor jala los cambios del repo (no push). `~/bin/deploy.sh` bajo el usuario: `git fetch --tags && git checkout <tag> && podman build -t localhost/<proyecto>/<componente>:<tag> && systemctl --user restart <svc>`.
- Automático: `<proyecto>-deploy.service` + `.timer` (`OnUnitActiveSec=5min`) en `.config/systemd/user/` (los .service plain van ahí; solo Quadlets van en `.config/containers/systemd/`).
- Con CI (GitHub Actions + registry): deploy.sh hace `podman pull` en vez de build.
- Alternativa push: webhook HTTP (puerto ≥1024). Prohibido SSH a root o sudo NOPASSWD amplio.

### Troubleshooting común

- `Failed to connect to bus`: falta `XDG_RUNTIME_DIR` (`export XDG_RUNTIME_DIR=/run/user/$(id -u)`).
- Contenedor muere al cerrar sesión: falta linger.
- `Error configuring CDI devices`: regenerar `/etc/cdi/nvidia.yaml`.
- nvidia-smi no ve GPU en contenedor: imagen base sin libs CUDA o falta `NVIDIA_VISIBLE_DEVICES`.
- `host.containers.internal` no resuelve: probar `10.0.2.2` (slirp4netns) o `--network=pasta`.
- `permission denied` con `:Z`: el host no usa SELinux (Ubuntu default) → quitar `:Z`.
- `port already in use`: conflicto con otro proyecto; revisar mapa pág. 8.
- OOM: excedió `MemoryMax` o presión del host; revisar `podman stats`.
- vLLM 503: retry con backoff; NO segunda instancia (R1).
- Quadlet no genera .service: debe estar en `~/.config/containers/systemd/`; `systemctl --user daemon-reload`; debug con `/usr/libexec/podman/quadlet -dryrun -user`.
- `ValueError: No available memory for the cache blocks` (modelo en contenedor CDI): subir `gpu-memory-utilization` — el budget del contenedor es ~0.3 GiB menor que el host.

### Errata 2026-04-20 (aprendida en la migración de vLLM)

- Quadlets en `~/.config/containers/systemd/`, NO en `~/.config/systemd/user/` (el generator de Podman 4.9 solo lee del primero).
- El límite de memoria NO va como `Memory=` en `[Container]` (Podman 4.9 lo rechaza); va como `MemoryMax=` en `[Service]`.
- AWQ 7B en RTX 3070 dentro de contenedor: `gpu-memory-utilization >= 0.92` (CDI reporta 7.66 GiB efectivos vs 8.00).

### Lecciones transferibles (2026-04-21, pendiente integrar al checklist)

- Podman 4.9.3 NO soporta `NetworkAlias=` en Quadlet → usar `PodmanArgs=--network-alias <short>` cuando los ContainerName llevan prefijo pero el .env usa hostnames cortos. (5.0+: `NetworkAlias=`.)
- Todo `Volume=%h/data/<name>` requiere `mkdir` explícito en el host (rootless no auto-crea; falla con "statfs ... no such file").
- Bind mount de source + deps en imagen → volumen named independiente para node_modules/site-packages.
- `MemoryMax`/`CPUQuota` se enforzan por systemd/cgroups, pero `podman stats` muestra el total del host; verificar con `systemctl --user show <svc> --property=MemoryMax --value`.

### Exponer un app al público (2026-04-24)

NO instalar cloudflared propio ni port forward. Usar la infra bajo user `cftunnel`: (1) tu app expone puerto en localhost; (2) pedir al admin de infra: dominio en la cuenta CF, route DNS al tunnel, ingress rule en config.yml → `http://localhost:<puerto>`; (3) restart del service cloudflared. Runbook completo: Notion pág. 13. Mail propio por dominio: stack Mailcow existente, Notion pág. 12 (outbound vía Resend, inbound vía CF Email Routing).

### Sub-páginas 9.x (entornos privados por proyecto — solo en Notion, no espejadas acá)

9.1 PostPilot · 9.2 trading · 9.2 thermouy · 9.3 sockerdata · 9.4 ha-core (Home Assistant) · 9.5 mosquitto (infra, broker MQTT compartido) · 9.6 zigbee2mqtt (Rpi PostPilot, migrado al server — ver repo `homelab-domo`).

---

> Credenciales iniciales que figuran en Notion (ej. htpasswd de Kibana/Grafana) fueron **redactadas** de este espejo a propósito; consultarlas en Notion / rotarlas antes de uso real.
