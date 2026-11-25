# KZA frontend

NEXO is served at `/`. The technical panel is available at `/obs` by direct URL. The Vite build produces `dist/`, which `nexa-web` serves with the same origin as the API and WebSocket endpoints.

## Build and checks

From `frontend/`, with Node 22.12+ and npm 10:

```sh
npm ci
npm run check
npm run test
npm run build
```

The Vitest suite mounts the SPA with synthetic responses and verifies route isolation, permission gating, private asset requests, action receipts and v3 WebSocket retirement. Sessions use `GET /api/admin/auth/whoami`, `POST /api/admin/auth/login`, and `POST /api/admin/auth/logout`. The three WebSocket routes use session-bound REST responses and `live_hello` / `live_reset_ack` / `live_ack`.

Casa lazily loads the Three.js renderer when the 3D view is opened. The model and plan are authenticated requests to `/api/home/model` and `/api/home/plan`. Rooms remain usable when the private model cannot be rendered.
