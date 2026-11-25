# `workers/nexa-ml/` — Stateless GPU Inference Worker (Phase 3)

Thin, stateless Python 3.13 worker speaking `nexa.v1.MlService` over `$XDG_RUNTIME_DIR/nexa/ml.sock`.

## Hard Boundaries
- **Zero network access** (`IPAddressDeny=any` in `nexa-ml.service`).
- **Zero SQLite / state** (all persistence lives in `nexa-memoryd` and `nexa-brain`).
- **Zero audio hardware** (ALSA/XVF3800/MA1260 live exclusively in Rust `nexa-audiod`).
- **Zero PyTorch in PR unit CI** (`dev` extra tests wire contracts against lightweight stubs; GPU benchmarks run separately on the server).
