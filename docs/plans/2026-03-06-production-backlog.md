# KZA Productionization Backlog

**Date:** 2026-03-06  
**Status:** Proposed backlog  
**Primary goal:** Convert KZA from an ambitious local-first prototype into a reliable local voice server for home use.  
**Recommended product shape:** Modular monolith as the source of truth, with optional local worker services only after interfaces and tests are stable.

**Related docs:**
- `docs/ARCHITECTURE_REVIEW.md`
- `docs/ESTADO_PROYECTO_KZA.md`
- `docs/plans/2026-02-21-q1-architecture-refactor.md`
- `docs/plans/2026-02-21-q2-robustness-quality.md`

---

## Planning assumptions

- Canonical runtime should be the Python application assembled in `src/main.py`.
- `src/kza_server.py` has been removed. `docker/services/pipeline_service.py` is marked EXPERIMENTAL and scoped under BL-005.
- Stabilization target should be Python 3.13, which is the intended project runtime and satisfies the `vLLM` constraint of `<3.14`.
- The earlier local test failure under Python 3.9 should be treated as an environment mismatch, not as evidence that the project should target an older interpreter.
- Fast path latency for home automation remains the primary UX target.
- Slow path quality matters, but correctness, cancellation, and observability matter more than raw capability right now.
- No production data exists yet. Persisted data (`./data/`) is dev/test only and can be wiped during contract changes without migration.

---

## Release gates

### Gate A: Stable Developer Baseline
- One supported Python version documented and enforced.
- Test suite runs on the supported version.
- Canonical startup path is unambiguous.

### Gate B: Correct Voice Core
- Audio input, STT, routing, HA action, and TTS work end-to-end.
- User identity, permissions, and room context use one consistent contract.
- Multi-room and cancellation flows are covered by tests.

### Gate C: Operable Home Server
- Health, latency, queue state, and model state are observable.
- Config and secret handling are production-safe.
- Deployment flow is repeatable on the target machine.

---

## Priority overview

| ID | Priority | Theme | Status |
|---|---|---|---|
| BL-001 | P0 | Consolidate canonical runtime | Done |
| BL-002 | P0 | Fix command/user contract breakage | Done |
| BL-003 | P0 | Establish supported Python and test baseline | Done |
| BL-004 | P0 | Restore trustworthy automated verification | Proposed |
| BL-005 | P2 | Align Docker and secondary runtimes | Proposed |
| BL-006 | P1 | Refresh documentation and operational claims | Proposed |
| BL-007 | P1 | Harden HA integration and failure handling | Proposed |
| BL-008 | P1 | Multi-room and multi-user hardening | Proposed |
| BL-009 | P1 | Integrate observability into runtime (backend API) | Proposed |
| BL-010 | P1 | Security and configuration hardening | Proposed |
| BL-011 | P2 | Operator dashboard frontend | Proposed |
| BL-012 | P2 | Packaging and deployment profiles | Proposed |
| BL-013 | P3 | Optional worker extraction for heavy models | Proposed |

---

## Backlog items

### BL-001 — Consolidate canonical runtime
**Priority:** P0  
**Outcome:** One obvious way to run KZA locally.

**Files likely involved:**
- `src/main.py`
- `src/kza_server.py`
- `docker/services/pipeline_service.py`
- `scripts/start.sh`
- `docker-compose.yml`
- `README.md`

**Tasks:**
- Choose `src/main.py` as the source of truth for runtime assembly.
- Mark `src/kza_server.py` as experimental or remove overlapping responsibilities.
- Decide whether `docker/services/pipeline_service.py` is a future adapter or deprecated prototype.
- Make startup scripts point to the canonical runtime.
- Update README startup instructions to match the chosen path.

**Definition of done:**
- A new contributor can answer “how do I run KZA?” with one command and one document.
- No secondary runtime claims functionality it does not actually implement.

### BL-002 — Fix command, user, and room-context contracts
**Priority:** P0  
**Outcome:** The voice core passes around one consistent command result shape.

**Files likely involved:**
- `src/pipeline/command_processor.py`
- `src/pipeline/request_router.py`
- `src/pipeline/command_event.py`
- `src/rooms/room_context.py`
- `tests/unit/pipeline/test_request_router.py`
- `tests/unit/pipeline/test_request_router_room.py`
- `tests/unit/pipeline/test_voice_pipeline_room.py`

**Known bug (discovered 2026-03-09):**
`CommandProcessor._identify_speaker()` returns `{"user": User, "confidence": float, "timing_ms": float}`,
but `RequestRouter` (lines 221-224, 335-343) accesses `cmd_result["user"]` as if it were a `User` object
(`.user_id`, `.name`, `.permission_level`). This is a latent `AttributeError` that only doesn't crash because
no speakers are enrolled yet. Both `_process_command_orchestrated()` and `_process_command_legacy()` are affected.

**Design decision pending:** Whether `process_command()` should return a typed `@dataclass ProcessedCommand`
or remain a dict with a fixed shape. Dataclass recommended for compile-time safety.

**Tasks:**
- Standardize what `CommandProcessor.process_command()` returns for `user`.
- Remove the mismatch between dict-based speaker results and object-style access in `RequestRouter`.
- Define a canonical typed result for processed commands.
- Ensure room metadata travels through the same path for single-room and multi-room flows.
- Ensure the new dispatcher fast paths (`FAST_LIST`, `FAST_REMINDER`) consume the same canonical command result shape.
- Add regression tests for identified user, unknown user, room-resolved command, and no-room command.

**Definition of done:**
- `RequestRouter` no longer depends on ambiguous result shapes.
- User ID, permission level, and room context behave the same in orchestrated and legacy flows.
- List and reminder dispatcher paths use the standardized contract.

### BL-003 — Establish supported Python and dependency baseline
**Priority:** P0  
**Outcome:** The repository has one real compatibility target instead of an assumed one.

**Files likely involved:**
- `requirements.txt`
- `README.md`
- `pytest.ini`
- `systemd/kza-voice.service`
- `docker/Dockerfile.*`

**Tasks:**
- Set the stabilization target to Python 3.13.
- Verify every required package on the target interpreter.
- Install and document `pytest` on the supported runtime.
- Reconcile `requirements.txt` with the runtime that actually works.
- Capture the target Python version in docs and deployment assets.

**Definition of done:**
- Running tests on the supported Python version is a documented, repeatable operation.
- There is no contradiction between docs, dependency metadata, and code syntax.

### BL-004 — Restore trustworthy automated verification
**Priority:** P0  
**Outcome:** The test suite becomes a signal, not a historical claim.

**Files likely involved:**
- `tests/`
- `pytest.ini`
- `TESTING.md`
- `TESTS_REFERENCE.md`

**Tasks:**
- Run the suite on the supported Python version and record the real baseline.
- Fix collection-level breakages first.
- Split smoke tests from hardware-dependent or model-dependent tests.
- Add a minimal “voice core smoke test” path that does not require GPUs.
- Update testing docs to reflect the true state of passing, skipped, and hardware-gated tests.

**Definition of done:**
- The repository can report an honest test status from a single supported environment.
- “All tests pass” is either true again or removed from docs.

### BL-005 — Align Docker and secondary runtimes with the canonical runtime
**Priority:** P2
**Outcome:** Secondary execution modes stop diverging from the real product.

**Files likely involved:**
- `docker/services/pipeline_service.py`
- `docker/services/stt_service.py`
- `docker/services/router_service.py`
- `docker/services/reasoner_service.py`
- `docker-compose.yml`

**Tasks:**
- Audit each service against the canonical monolith behavior.
- Remove placeholder behavior in the Docker pipeline for HA actions.
- Decide whether service mode is “full parity”, “inference workers only”, or “experimental”.
- Document the decision in compose and README.

**Definition of done:**
- Docker mode is either product-valid or explicitly scoped as partial.
- No service advertises full home-control behavior if it only simulates it.

### BL-006 — Refresh documentation and operational claims
**Priority:** P1  
**Outcome:** Docs describe the current codebase, not old snapshots.

**Files likely involved:**
- `README.md`
- `docs/ESTADO_PROYECTO_KZA.md`
- `docs/ARCHITECTURE_REVIEW.md`
- `docs/ORCHESTRATOR.md`
- `docs/HARDWARE.md`

**Tasks:**
- Replace stale line counts, test counts, and architecture descriptions.
- Update the architecture review to reflect the extracted pipeline components.
- Document the lists and reminders modules (`src/lists/`, `src/reminders/`) in architecture descriptions.
- Split aspirational claims from implemented features.
- Add one “current architecture” document and one “future roadmap” document if needed.

**Definition of done:**
- The architecture description matches the code a reader will actually open.
- Quantitative claims are reproducible from the current repo.

### BL-007 — Harden Home Assistant integration and failure handling
**Priority:** P1  
**Outcome:** HA failures degrade gracefully without breaking the server.

**Files likely involved:**
- `src/home_assistant/ha_client.py`
- `src/home_assistant/circuit_breaker.py`
- `src/pipeline/request_router.py`
- `src/pipeline/feature_manager.py`
- `tests/unit/test_ha_client.py`

**Tasks:**
- Use WebSocket fast path where it materially improves latency.
- Add explicit fallback policy when HA is unavailable.
- Surface HA health in runtime status and dashboard.
- Ensure service-call timeouts are predictable and logged.
- Add regression tests for HA unavailable, slow HA, and auth failure.

**Definition of done:**
- A temporary HA outage does not crash or hang the voice server.
- Operators can see HA degradation quickly.

### BL-008 — Multi-room and multi-user hardening
**Priority:** P1  
**Outcome:** Concurrent household use works predictably.

**Files likely involved:**
- `src/pipeline/multi_room_audio_loop.py`
- `src/orchestrator/dispatcher.py`
- `src/orchestrator/priority_queue.py`
- `src/orchestrator/context_manager.py`
- `tests/integration/test_multi_room_concurrent.py`

**Tasks:**
- Verify wake-word deduplication across microphones.
- Stress test concurrent commands from different rooms.
- Verify cancellation when the same user interrupts a slow request.
- Verify that urgent commands do not wait behind slow conversational ones.
- Verify shared list concurrency (two users adding items simultaneously).
- Wire `ReminderScheduler.missed_reminder_on_arrival` to `PresenceDetector` events (currently a TODO in code).
- Define limits for supported simultaneous commands in a home environment.

**Definition of done:**
- Core concurrency scenarios are covered by integration tests.
- The system behavior under simultaneous room input is documented and predictable.
- Shared lists handle concurrent modifications safely.

### BL-009 — Integrate observability into runtime (backend only)
**Priority:** P1
**Outcome:** Latency, queue, subsystem state, and failures are queryable via JSON API endpoints.

**Scope boundary:** This item covers backend metrics and API endpoints only. Frontend visualization belongs to BL-011.

**Files likely involved:**
- `src/monitoring/latency_monitor.py`
- `src/dashboard/api.py`
- `src/main.py`
- `scripts/latency_dashboard.py`

**Tasks:**
- Wire `LatencyMonitor` through the canonical runtime for every processed command.
- Expose p50, p95, p99, queue depth, active zone, and subsystem health via JSON endpoints.
- Add API endpoints for model state, HA state, and recent failures.
- Include `ReminderScheduler` state in subsystem health (pending count, next trigger, delivery failures).
- Separate operator metrics from end-user API responses.

**Definition of done:**
- The operator can answer “what is slow or broken?” with `curl` against the API.
- All subsystem health data is available as structured JSON before any frontend work begins.

### BL-010 — Security and configuration hardening
**Priority:** P1  
**Outcome:** The local server is safe to run inside a home network.

**Files likely involved:**
- `config/settings.yaml`
- `README.md`
- `src/dashboard/api.py`
- `src/core/logging.py`
- `tests/safety/`

**Tasks:**
- Remove hard-coded host examples where environment variables should be used.
- Ensure tokens and secrets are never exposed in logs or status payloads.
- Restrict permissive CORS defaults outside development mode.
- Review local API exposure and unauthenticated control surfaces.
- Expand safety tests for permission boundaries and command execution rules.

**Definition of done:**
- Running KZA on a LAN does not expose avoidable secrets or unsafe control defaults.

### BL-011 — Operator dashboard frontend
**Priority:** P2
**Outcome:** The dashboard becomes a visual control plane, consuming the API endpoints from BL-009.

**Scope boundary:** This item covers frontend UI only. Backend endpoints must already exist from BL-009.

**Files likely involved:**
- `src/dashboard/frontend/`

**Tasks:**
- Build subsystem overview UI for audio, models, HA, queue, users, and rooms.
- Add recent command timeline with latency breakdown visualization.
- Add alerting and failure summary views.
- Add configuration validation views for microphones, zones, and HA connectivity.

**Definition of done:**
- The dashboard is useful for operation, not only for routines CRUD.
- All dashboard views consume BL-009 API endpoints — no backend logic in the frontend layer.

### BL-012 — Packaging and deployment profiles
**Priority:** P2  
**Outcome:** KZA can be installed and restarted predictably on the target machine.

**Files likely involved:**
- `systemd/kza-voice.service`
- `scripts/setup_ubuntu.sh`
- `scripts/start.sh`
- `docker-compose.yml`

**Tasks:**
- Provide one recommended deployment profile for the home server.
- Add environment validation for audio devices, GPUs, and HA connectivity.
- Define restart policy, log locations, and persistent data locations.
- Add a post-install smoke test script.

**Definition of done:**
- A fresh machine can be brought to a working KZA installation from documented steps.

### BL-013 — Optional worker extraction for heavy models
**Priority:** P3  
**Outcome:** Heavy inference services can be separated without changing product behavior.

**Files likely involved:**
- `docker/services/*.py`
- `src/pipeline/model_manager.py`
- `src/llm/reasoner.py`
- `src/stt/whisper_fast.py`
- `src/tts/piper_tts.py`

**Tasks:**
- Define service boundaries only after the monolith contracts are stable.
- Extract STT, router, LLM, or TTS behind explicit provider interfaces.
- Keep home-control orchestration local even if inference workers are split.
- Add parity tests between in-process and worker-backed modes.

**Definition of done:**
- Service extraction reduces operational risk instead of increasing coordination cost.

---

## Recommended execution order

### Wave 1 — P0, sequential (each depends on the previous)
1. BL-001 — Consolidate canonical runtime
2. BL-002 — Fix command/user contracts
3. BL-003 — Python and dependency baseline
4. BL-004 — Restore test suite

### Wave 2 — P1, parallelizable
5. BL-006 — Refresh documentation
6. BL-007 — Harden HA integration
7. BL-008 — Multi-room and multi-user hardening

### Wave 3 — P1, parallelizable
8. BL-009 — Observability
9. BL-010 — Security hardening

### Wave 4 — P2, parallelizable
10. BL-005 — Align Docker runtimes
11. BL-011 — Operator dashboard
12. BL-012 — Packaging and deployment

### Wave 5 — P3
13. BL-013 — Optional worker extraction

---

## Suggested first implementation slice

If execution starts immediately, the first slice should be:

1. Consolidate `src/main.py` as the canonical runtime.
2. Fix the `CommandProcessor` and `RequestRouter` user contract.
3. Lock Python 3.13 as the supported stabilization target.
4. Re-run and repair the test suite on that runtime.
5. Rewrite the top-level docs to match the actual state of the repo.

This slice creates a stable foundation for every later feature and prevents the project from drifting further between implementation, tests, and documentation.
