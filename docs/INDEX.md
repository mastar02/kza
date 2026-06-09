# Índice de documentación — KZA

> Convención: `architecture/` = docs vivos del sistema (módulos, hardware, flujo de trabajo);
> `research/` = investigaciones, análisis y registros de sesión (con fecha, históricos);
> `plans/` = planes de implementación; `runbooks/` = procedimientos operativos paso a paso;
> `superpowers/` = planes/specs generados con el plugin superpowers; `examples/` = ejemplos de configuración.

## Raíz

- [SERVER_CONVENTIONS.md](SERVER_CONVENTIONS.md) — Espejo local de Notion pág. 8 y 9: contrato del servidor compartido (usuarios/UID, puertos, Podman rootless + Quadlets, GPU por CDI, onboarding).

## architecture/ — docs vivos del sistema

- [ARCHITECTURE_REVIEW.md](architecture/ARCHITECTURE_REVIEW.md) — Review de arquitectura del sistema de voz: fortalezas, deudas y recomendaciones.
- [DEPLOYMENT.md](architecture/DEPLOYMENT.md) — **Fuente de verdad de deploy**: producción = kza-voice nativo systemd --user (excepción R10); docker/ es experimental, compose es legacy.
- [ECOSISTEMA_LOCAL_SERVER.md](architecture/ECOSISTEMA_LOCAL_SERVER.md) — Flujo de trabajo Local ↔ Server ↔ GitHub: el server deploya in-place pero no pushea; la laptop es el puente (`kza-push`/`kza-sync`).
- [EMOTION_DETECTOR.md](architecture/EMOTION_DETECTOR.md) — Módulo de detección de emociones (wav2vec2): API, integración y configuración.
- [ESTADO_PROYECTO_KZA.md](architecture/ESTADO_PROYECTO_KZA.md) — Reporte de estado del proyecto: qué funciona, métricas y pendientes.
- [HARDWARE.md](architecture/HARDWARE.md) — Especificaciones de hardware: Threadripper PRO, 128GB RAM, GPUs, XVF3800, MA1260, BLE.
- [KZA_ANALISIS_Y_ROADMAP.md](architecture/KZA_ANALISIS_Y_ROADMAP.md) — Análisis completo del sistema y roadmap de mejoras por área.
- [OPTIMIZACION_128GB_8x16.md](architecture/OPTIMIZACION_128GB_8x16.md) — Análisis de la config de RAM 8x16GB (8 canales) para inferencia LLM en CPU.
- [ORCHESTRATOR.md](architecture/ORCHESTRATOR.md) — Sistema de orquestación multi-usuario: colas de prioridad, contexto por usuario, cancelación.
- [SPOTIFY.md](architecture/SPOTIFY.md) — Integración Spotify multi-room: dispatcher, mood mapping, grupos de bocinas.
- [WAKE_WORD_TRAINING.md](architecture/WAKE_WORD_TRAINING.md) — Guía de entrenamiento del wake word personalizado.

## research/ — investigaciones, análisis y sesiones

- [REPORTE_KZA_FEBRERO_2026.md](research/REPORTE_KZA_FEBRERO_2026.md) — Reporte general del asistente (febrero 2026): visión, capacidades y estado.
- [MEJORAS_IMPLEMENTADAS.md](research/MEJORAS_IMPLEMENTADAS.md) — Registro histórico de mejoras implementadas por área.
- [OPENCLAW_INTEGRATION_ANALYSIS.md](research/OPENCLAW_INTEGRATION_ANALYSIS.md) — Análisis de patrones de OpenClaw transferibles a KZA (failover, compaction, hooks).
- [RECOMENDACION_DISCOS_URUGUAY.md](research/RECOMENDACION_DISCOS_URUGUAY.md) — Recomendaciones de almacenamiento (mercado uruguayo).
- [notion_page8_kza_update.md](research/notion_page8_kza_update.md) — Borrador del update para Notion pág 8 ("Proyectos registrados" → entrada KZA).
- [SESSION_2026-04-28_LLM_STACK_OVERHAUL.md](research/SESSION_2026-04-28_LLM_STACK_OVERHAUL.md) — Sesión: overhaul del stack LLM y consolidación de branches.
- [2026-05-31_DIAGNOSTICO_ALUCINACIONES_Y_ROADMAP_ALEXA.md](research/2026-05-31_DIAGNOSTICO_ALUCINACIONES_Y_ROADMAP_ALEXA.md) — Diagnóstico de alucinaciones de Whisper + roadmap "Alexa a tope" (v1).
- [2026-05-31_BUG_RAIZ_WAKE_NEXA_Y_ROADMAP_v2.md](research/2026-05-31_BUG_RAIZ_WAKE_NEXA_Y_ROADMAP_v2.md) — Bug raíz del wake "Nexa" + roadmap corregido (v2).
- [2026-05-31_SESION_CONSOLIDADA_wake_alucinaciones.md](research/2026-05-31_SESION_CONSOLIDADA_wake_alucinaciones.md) — Cierre consolidado de la sesión wake/alucinaciones/XVF3800: lo probado vs lo descartado.
- [SESSION_2026-05-30_XVF3800_WAKE_NLU_FIXES.md](research/SESSION_2026-05-30_XVF3800_WAKE_NLU_FIXES.md) — Sesión: swap del XVF3800 + cadena wake→NLU→HA arreglada.
- [2026-06-03_XVF3800_MAX_PROVECHO_VS_FLEX_Y_ALTERNATIVAS.md](research/2026-06-03_XVF3800_MAX_PROVECHO_VS_FLEX_Y_ALTERNATIVAS.md) — Investigación XVF3800: cómo exprimirlo vs ReSpeaker Flex y alternativas (veredicto: no migrar).
- [2026-06-06_HERMES4_RAG_TOTAL_ANALISIS.md](research/2026-06-06_HERMES4_RAG_TOTAL_ANALISIS.md) — Análisis Hermes 4 + RAG total (veredicto: no adoptar Hermes; el potencial es RAG sobre el stack actual).
- [2026-06-07_SOTA_ASR_ESPANOL_INVESTIGACION.md](research/2026-06-07_SOTA_ASR_ESPANOL_INVESTIGACION.md) — Investigación SOTA ASR español local; motivó el swap del ambient path a Parakeet-TDT.

## plans/ — planes de implementación

- [2026-02-21-holistic-improvements-design.md](plans/2026-02-21-holistic-improvements-design.md) — Plan holístico de mejoras "foundation first".
- [2026-02-21-multi-interface-integration-design.md](plans/2026-02-21-multi-interface-integration-design.md) — Diseño de integración multi-interfaz (mic + BT por habitación).
- [2026-02-21-multi-interface-implementation.md](plans/2026-02-21-multi-interface-implementation.md) — Plan de implementación de la integración multi-interfaz.
- [2026-02-21-q1-architecture-refactor.md](plans/2026-02-21-q1-architecture-refactor.md) — Q1: refactor de arquitectura.
- [2026-02-21-q2-robustness-quality.md](plans/2026-02-21-q2-robustness-quality.md) — Q2: robustez y calidad de código.
- [2026-02-21-session-setup-design.md](plans/2026-02-21-session-setup-design.md) — Diseño del setup de sesión de desarrollo.
- [2026-03-05-lists-reminders-design.md](plans/2026-03-05-lists-reminders-design.md) — Diseño de listas y recordatorios.
- [2026-03-05-lists-reminders-plan.md](plans/2026-03-05-lists-reminders-plan.md) — Plan de implementación de listas y recordatorios.
- [2026-03-06-production-backlog.md](plans/2026-03-06-production-backlog.md) — Backlog de productionización (BL-XXX).
- [2026-03-09-bl001-consolidate-runtime.md](plans/2026-03-09-bl001-consolidate-runtime.md) — BL-001: consolidar runtime canónico.
- [2026-03-09-bl002-fix-user-contract.md](plans/2026-03-09-bl002-fix-user-contract.md) — BL-002: arreglar contrato command/user.
- [2026-06-06_PLAN_MINIMAX_HIGHSPEED_HERMES_AGENT.md](plans/2026-06-06_PLAN_MINIMAX_HIGHSPEED_HERMES_AGENT.md) — Plan MiniMax M2.7-highspeed + hermes-agent sandbox: etapas y prompts por proyecto.
- [2026-06-06_TAREA_KZA_MIGRACION_HIGHSPEED.md](plans/2026-06-06_TAREA_KZA_MIGRACION_HIGHSPEED.md) — Tarea KZA: migración del reasoner a MiniMax-M2.7-highspeed (Etapa A deployada).

## runbooks/ — procedimientos operativos

- [2026-06-06-xvf3800-flasheo-6ch.md](runbooks/2026-06-06-xvf3800-flasheo-6ch.md) — Runbook: flasheo del XVF3800 a firmware USB 6 canales.

## Otros

- `superpowers/plans/` y `superpowers/specs/` — planes y specs generados con el plugin superpowers (workflow propio, no se reorganizan).
- `examples/` — ejemplos de configuración (alertas).
