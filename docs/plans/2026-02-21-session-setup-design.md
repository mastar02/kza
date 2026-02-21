# KZA Session Setup Design

**Fecha**: 2026-02-21
**Objetivo**: Optimizar sesiones de Claude Code para máxima productividad

## Problema

Las sesiones anteriores sufrían de:
1. **Contexto perdido**: Claude no recordaba decisiones previas
2. **Código inconsistente**: No seguía convenciones del proyecto
3. **Lentitud explorando**: Gastaba tiempo re-descubriendo el codebase

## Solución: Enfoque "CLAUDE.md Quirúrgico"

### Entregable 1: CLAUDE.md Reescrito
- De ~250 líneas de spec sheet de hardware → ~150 líneas de guía de desarrollo
- Secciones: Reglas imperativas, convenciones, arquitectura, mapa de archivos, comandos
- Hardware detallado movido a `docs/HARDWARE.md`

### Entregable 2: Memoria Persistente
Archivos en `.claude/projects/.../memory/`:
- `MEMORY.md` — Índice cargado automáticamente en cada sesión
- `architecture.md` — Mapa detallado de módulos y data flow
- `decisions.md` — Registro de decisiones de diseño con justificación
- `patterns.md` — Patrones de código y soluciones a problemas comunes

### Entregable 3: Skills Instaladas
- `python-testing-patterns` (4.2K installs) — pytest patterns
- `voice-ai-development` (279 installs) — Voice AI patterns
- `fastapi-async-patterns` (200 installs) — async/await patterns
- `docker-expert` (3.1K installs) — Docker compose
- `claude-md-architect` (43 installs) — CLAUDE.md structure

### Entregable 4: Hardware specs en docs/
- `docs/HARDWARE.md` — Todo el detalle de CPU, GPU, RAM, PSU, cooling

## Decisiones Clave

1. **CLAUDE.md es la single source of truth** para reglas de código
2. **Memory files** persisten conocimiento entre sesiones
3. **Hardware NO va en CLAUDE.md** porque no ayuda a escribir código correcto
4. **Reglas son imperativas** ("SIEMPRE", "NUNCA") no descriptivas

## Resultado Esperado

Cada nueva sesión de Claude Code arranca con:
- CLAUDE.md cargado automáticamente → sabe las reglas
- MEMORY.md cargado automáticamente → recuerda decisiones previas
- Skills instaladas → mejores patrones de código
- Puede empezar a ser productivo en <30 segundos
