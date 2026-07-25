# OKF Bundle Log

## 2026-07-24 — Initial bundle generation

Generated the initial OKF v0.1 bundle for the `kza` repository (base commit
`f4e1edc` on `main`, work done on branch `add-okf-bundle`).

- Recon sources: `README.md`, `CLAUDE.md`, `docs/architecture/DEPLOYMENT.md`,
  `docs/architecture/HARDWARE.md`, `docker/README.md`,
  `docs/runbooks/CODE_INDEX.md`, `config/settings.yaml`, and source files
  under `src/` (wakeword, ambient, pipeline, stt, nlu, llm, tts, vectordb,
  home_assistant, spotify, orchestrator, users, code_index, main.py),
  `systemd/kza-voice.service`, `scripts/kza-llm-ik.service`,
  `scripts/kza-code-index.service`, `deploy/udev/99-xvf3800.rules`,
  `config/homeassistant/README.md`, `docker-compose.yml`.
- 20 concept files written across `system/`, `pipeline/`, `datastores/`,
  `integrations/`, `orchestration/`, `services/`, `deployment/`, `config/`.
- Noise excluded: `venv/`, `.venv/`, `models/`, `data/`, `__pycache__/`,
  `.pytest_cache/`, `.worktrees/`. Not bundled as dedicated concepts:
  `tests/`, `benchmarks/`, `examples/`, `docs/plans/`, `docs/research/`,
  `docs/superpowers/`.
- Key judgment calls: treated `CLAUDE.md` and `docs/architecture/DEPLOYMENT.md`
  as current/authoritative over `README.md` and
  `docs/architecture/HARDWARE.md` (stale: 4-GPU description, openwakeword
  `hey_jarvis`, local 70B reasoner); flagged `systemd/kza-voice.service`
  (system-level unit) as contradicting `docs/architecture/DEPLOYMENT.md`'s
  documented production reality (systemd `--user` unit); flagged
  `docker/`/`docker-compose.yml` as explicitly experimental/non-production
  per three independent in-repo sources; flagged `kza-llm-fast.service`
  (port `:8101`) as referenced but not found tracked in-repo; left precise
  per-component GPU (`cuda:N`) assignment unasserted at the detail level
  where sources disagreed.
