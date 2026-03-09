# BL-001: Consolidate Canonical Runtime — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make `src/main.py` the single obvious way to run KZA by removing the competing `kza_server.py` and marking Docker services as experimental.

**Architecture:** No functional changes. Delete dead code, add banners, update docs. `src/main.py` remains untouched.

**Tech Stack:** Shell, Python, YAML, Markdown

---

### Task 1: Delete `src/kza_server.py`

**Files:**
- Delete: `src/kza_server.py`

**Step 1: Verify no runtime imports depend on kza_server**

Run: `grep -r "from src.kza_server\|import kza_server" src/`
Expected: No output (zero imports)

**Step 2: Delete the file**

```bash
rm src/kza_server.py
```

**Step 3: Verify deletion doesn't break imports**

Run: `python -c "from src.main import main; print('OK')"`
Expected: `OK`

**Step 4: Commit**

```bash
git add -u src/kza_server.py
git commit -m "refactor(BL-001): remove kza_server.py — src/main.py is the canonical runtime"
```

---

### Task 2: Add EXPERIMENTAL banner to Docker services

**Files:**
- Modify: `docker-compose.yml` (top of file)
- Modify: `docker/services/pipeline_service.py` (docstring)

**Step 1: Add banner to docker-compose.yml**

Add this comment block at the very top of `docker-compose.yml`, before line 1:

```yaml
# ⚠️ EXPERIMENTAL — Docker services do NOT have full parity with the canonical runtime.
# For production use: python -m src.main (or ./scripts/start.sh)
# Docker services are provided for future worker extraction (see BL-005, BL-013).
# They lack: speaker ID, multi-room, lists/reminders, emotion detection, memory, training.

```

**Step 2: Update pipeline_service.py docstring**

Replace the docstring at the top of `docker/services/pipeline_service.py` (lines 1-4):

Old:
```python
"""
Voice Pipeline Service - Main Orchestrator
Connects to all other services via HTTP
"""
```

New:
```python
"""
Voice Pipeline Service - Main Orchestrator (EXPERIMENTAL)

⚠️ This service does NOT have full parity with the canonical runtime (src/main.py).
Missing: speaker ID, multi-room, lists/reminders, emotion detection, memory, training.
Use src/main.py for production. See BL-005 for Docker alignment scope.
"""
```

**Step 3: Commit**

```bash
git add docker-compose.yml docker/services/pipeline_service.py
git commit -m "docs(BL-001): mark Docker services as EXPERIMENTAL"
```

---

### Task 3: Update README startup instructions

**Files:**
- Modify: `README.md` (lines 93-97, startup section)

**Step 1: Fix the startup command**

Replace the current "4. Ejecutar" section (lines 93-97):

Old:
```markdown
### 4. Ejecutar

```bash
python src/main.py
```
```

New:
```markdown
### 4. Ejecutar

```bash
# Recommended (includes environment checks, GPU verification, health check):
./scripts/start.sh

# Or directly:
python -m src.main
```
```

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs(BL-001): update README startup to use canonical entry point"
```

---

### Task 4: Update backlog plan status

**Files:**
- Modify: `docs/plans/2026-03-06-production-backlog.md` (BL-001 status)

**Step 1: Update BL-001 status in priority table**

Change BL-001 row from `Proposed` to `Done`:

Old: `| BL-001 | P0 | Consolidate canonical runtime | Proposed |`
New: `| BL-001 | P0 | Consolidate canonical runtime | Done |`

**Step 2: Update planning assumptions**

Replace the `kza_server.py` assumption line:

Old: `- \`src/kza_server.py\` and \`docker/services/pipeline_service.py\` should be treated as secondary until they either match the canonical runtime or are explicitly deprecated.`

New: `- \`src/kza_server.py\` has been removed. \`docker/services/pipeline_service.py\` is marked EXPERIMENTAL and scoped under BL-005.`

**Step 3: Commit**

```bash
git add docs/plans/2026-03-06-production-backlog.md
git commit -m "docs(BL-001): mark BL-001 as done in backlog"
```

---

### Task 5: Final verification

**Step 1: Verify kza_server references are gone from source**

Run: `grep -r "kza_server" src/`
Expected: No output

**Step 2: Verify start.sh still works**

Run: `head -n 5 scripts/start.sh && tail -n 3 scripts/start.sh`
Expected: Last line is `exec python3 -m src.main`

**Step 3: Verify Docker banner is present**

Run: `head -n 4 docker-compose.yml`
Expected: EXPERIMENTAL banner visible

**Step 4: Verify README shows correct startup**

Run: `grep -A 4 "4. Ejecutar" README.md`
Expected: Shows `./scripts/start.sh` and `python -m src.main`
