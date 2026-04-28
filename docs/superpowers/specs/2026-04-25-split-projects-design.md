# Split del proyecto KZA en tres contextos de Claude Code

**Fecha**: 2026-04-25
**Estado**: aprobado, pendiente implementación
**Autor**: Gabriel + Claude

## Problema

El proyecto `~/Documents/kza/` se desvirtuó. Empezó como asistente de voz local para domótica (~38K líneas Python en `src/`) pero la memoria persistente de Claude (`~/.claude/projects/-Users-yo-Documents-kza/memory/`) acumuló 33 archivos cubriendo dominios que no tienen nada que ver con el pipeline de voz: red (OPNsense, MikroTik, Cloudflare Tunnels, DDNS), observabilidad (ELK, obs-analyzer), mail (Mailcow), Home Assistant migrado al server, hardware del server, usuarios del sistema, incidentes de red.

Consecuencia: cuando se trabaja en `src/pipeline/` se carga contexto irrelevante de mail/firewall, y al revés. Las decisiones se sesgan por información que no aplica al dominio actual.

## Decisión

Separar en **tres proyectos de Claude Code**, cada uno con su propio `CLAUDE.md` y memoria, con **Notion como source of truth cross-project** (no duplicar archivos entre memorias locales).

```
~/Documents/
├── kza/                      ← solo pipeline de voz (ya existe)
├── homelab-infra/            ← nuevo: server base + red + identidad
└── homelab-services/         ← nuevo: apps que corren sobre la infra
```

### Justificación del corte

- **kza** y **homelab-infra** tienen ciclos de trabajo distintos. Tocar `src/llm/reasoner.py` no debería cargar context de firewall rules.
- **homelab-infra** (plomería: server, red, users) y **homelab-services** (apps: obs, mail, HA) tienen frecuencia de cambio diferente. La infra cambia poco; las apps cambian seguido.
- Un solo `homelab/` monolítico tendría ~25 memorias mezclando dominios muy distintos. Tres proyectos con ~10 cada uno es manejable.
- Los proyectos preexistentes vacíos `~/Documents/infra/` e `~/Documents/infra_kza/` se dejan como están (no se reutilizan, no son relevantes). `~/Documents/insfrastructura/` era para dashboard de admin de apps, fuera de scope ahora.

### Por qué Notion como source of truth (no shared files locales)

Tres opciones consideradas:

- **A)** Single source en `homelab-infra/`, otros referencian por path → no carga automáticamente.
- **B)** Duplicar con header de sync → drift garantizado.
- **C)** Resumen mínimo en cada uno + detalle en infra → defasajes inevitables.
- **D (elegida)**: cada proyecto dueño de su slice, **Notion como bridge cross-project** vía MCP `mcp__notion__*`.

El workspace Notion ya tiene la estructura: página root **KZA** con 13 subpáginas que mapean naturalmente al corte. La info compartida ya vive ahí; las memorias locales solo deben tener lo que **no** está en Notion (sesiones WIP, feedback de patterns local, debugging context).

## Mapping completo

### Notion (source of truth cross-project, MCP `mcp__notion__*`)

Workspace KZA, página root → 13 subpáginas existentes:

| Pág | Título | Dominio |
|---|---|---|
| 1 | Qué es KZA | kza |
| 2 | Arquitectura | kza |
| 3 | Hardware | kza + homelab-infra |
| 4 | Deployment | kza |
| 5 | Configuración | kza |
| 6 | Desarrollo | kza |
| 7 | Estado y roadmap | kza |
| 8 | Convenciones del servidor compartido | homelab-infra (contrato R10 #3, sub-rangos puertos) |
| 9 | Onboarding sumar proyecto nuevo | homelab-infra |
| 10 | Domótica HA + MQTT + Zigbee | homelab-services |
| 11 | Red y seguridad perimetral (con DDNS) | homelab-infra |
| 12 | Mail self-hosted | homelab-services |
| 13 | Cloudflare Tunnels | homelab-infra |
| **14** | **Observabilidad** *(a crear, stub)* | **homelab-services** |

### Reparto de las 33 memorias locales

#### `kza/` (voz)

Archivos que **se quedan** (delta sobre Notion o info que no aplica a Notion):
- `architecture.md` — revisar; si replica pág 2, achicar a "ver Notion pág 2 + delta local"
- `decisions.md` — revisar idem
- `patterns.md` — patrones código Python específicos, no replica Notion
- `wake_word_roadmap.md`
- `project_wake_tv_filter_pipeline_regression.md` — sesión WIP abierta
- `feedback_dense_retrieval_antonyms.md` — feedback BGE-M3, específico voz
- `feedback_cuda_import_order.md` — torch/llama-cpp, específico voz
- `gpu-ecosystem.md` — **partir**: cuda:0/cuda:1 que usa KZA queda; el contrato del vLLM compartido va a homelab-infra y a Notion pág 8

#### `homelab-infra/` (server + red + identidad)

Archivos que **se mueven desde kza**:

Server base:
- `server_hardware.md` (achicar; detalle largo a Notion pág 3)
- `server_network_cleanup_2026-04-23.md`
- `incident_oom_needrestart_2026-04-23.md`
- `shared_server_environment.md` (achicar; espejo de Notion pág 8)
- `infra_user.md`
- `deployment_march15.md`
- `kza_firmware_incident.md` (X710)
- (parte vLLM de) `gpu-ecosystem.md`

Red:
- `network_setup.md`
- `opnsense_access.md`
- `opnsense_firewall_rules.md`
- `device_inventory.md`
- `mikrotik_switch.md`
- `ddns_duckdns.md`
- `cloudflare_tunnels.md`
- `finding_hue_bridge_false_positive.md`

#### `homelab-services/` (apps sobre la infra)

Archivos que **se mueven desde kza**:

Observabilidad:
- `project_obs_stack.md`
- `project_obs_analyzer.md`
- `cf_analytics_exporter.md`
- `feedback_es_data_streams.md`
- `feedback_logstash_pipeline_gotchas.md`

Mail:
- `project_mail_mvp.md`
- `playbook_add_mail_domain.md`

Domótica HA:
- `project_ha.md`
- `project_mqtt_zigbee.md`
- `ha_light_naming_convention.md`

## Plantilla de `CLAUDE.md` cross-project

Cada uno de los tres proyectos arranca con un bloque idéntico que establece la regla de Notion:

```markdown
## Source of truth cross-project

Información que afecta a otros proyectos vive en Notion (workspace KZA,
página root id `345ab24f-c493-80b2-b6f4-ef917e865f26`). Consultala vía MCP
`mcp__notion__*` antes de asumir o de buscar en otra memoria local.

Subpáginas relevantes para este proyecto:
- (lista específica del proyecto)

Reglas:
- No duplicar info entre memorias de proyectos distintos.
- No replicar literalmente lo que ya está en Notion: achicar a "ver Notion
  pág X" más el delta local si lo hay.
- Cuando este proyecto necesita info de otro dominio → Notion vía MCP, no
  leer la memoria del otro proyecto.
```

Y luego sigue con su contenido específico (arquitectura, comandos, convenciones del dominio).

## Orden de implementación

1. Escribir este spec (acá) y commitear en kza.
2. Crear `~/Documents/homelab-infra/` y `~/Documents/homelab-services/` vacíos.
3. Crear los CLAUDE.md y MEMORY.md de los proyectos nuevos con la plantilla cross-project + contenido específico.
4. Mover las memorias según la tabla. Las paths reales de memoria de Claude son `~/.claude/projects/-Users-yo-Documents-homelab-infra/memory/` y `~/.claude/projects/-Users-yo-Documents-homelab-services/memory/`.
5. Achicar las memorias que tienen overlap con Notion (architecture, decisions, server_hardware, shared_server_environment, gpu-ecosystem split).
6. Crear página Notion **14. Observabilidad** como sub-página de KZA root, stub mínimo (puede poblarse después con contenido de las memorias de obs).
7. Limpiar `kza/CLAUDE.md` y `kza/MEMORY.md` quitando referencias a infra/red/obs/mail/HA. Dejar solo voz.
8. Verificación: abrir Claude Code en cada uno de los tres directorios y confirmar que el contexto cargado es el correcto y que las memorias migradas figuran en su MEMORY.md respectivo.

## Out of scope

- Mover código real entre proyectos (los proyectos nuevos no tienen código, solo son contextos de Claude).
- Renombrar/borrar `~/Documents/infra/`, `~/Documents/infra_kza/`, `~/Documents/insfrastructura/` (preexistentes vacíos, no tocar).
- Cambiar la estructura de Notion más allá de crear la pág 14.
- Reorganizar el repo `kza` en sí (sigue intacto).
- Migrar memoria de OTROS proyectos del usuario (agrotrace, postpilot, etc.) — fuera del alcance.

## Riesgos

- **Olvido de memoria al moverla**: usar `git mv`/copy explícito por archivo, verificar contra la tabla de mapping antes de borrar el original.
- **Pérdida de contexto en sesiones WIP**: `project_wake_tv_filter_pipeline_regression.md` queda en kza (es la sesión abierta). Verificar que el flag de "sesión abierta" se preserve en MEMORY.md.
- **MCP Notion offline**: si Notion MCP falla, los proyectos siguen funcionando con su memoria local pero pierden el cross-context. No es crítico para sesiones aisladas en un dominio.
- **Achicar memorias mata contexto útil**: antes de borrar contenido por overlap con Notion, verificar que Notion realmente tenga el dato. Si Notion está desactualizado, primero actualizar Notion, después achicar local.
