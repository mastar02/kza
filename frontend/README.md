# `frontend/` — Unified Vite + TypeScript SPA (Phase 5)

Consolidates NEXO (`/`) and the direct-URL technical panel (`/obs`, preserving Decision D1 with no link from NEXO) into one minified static bundle served by `nexa-web` (Rust Axum).

## Goals
- Eliminate 4.23 MiB of vendored React 18 development builds + Babel Standalone in-browser JSX transpilation.
- Load `/api/home/model` (`casa.glb`) and `/api/home/plan` strictly over authenticated `home:read` endpoints (`Cache-Control: no-store`).
- Verify end-to-end flows with Playwright in CI against `nexa-web --test-mode`.
