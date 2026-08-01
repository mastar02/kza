# Runbook de deploy — `feat/remediacion-fallos-silenciosos`

**Destino:** `kza@192.168.1.2:/home/kza/app` · **Desde:** `f3c94f1` → **hacia:** `510c2a93a3e6f31310791a5ea23cef976c3da758` (18 commits, 20 archivos, +2870/−23) · **Fast-forward puro verificado.**

**Leyenda:** ✅ verificado en el server · 🔶 inferido, verificar en el momento · 🛑 punto de no retorno · ⏸ parada obligatoria antes de seguir.

---

## Resumen ejecutivo de lo que cambia el runbook respecto de lo esperado

Tres correcciones que salieron del reconocimiento y que **modifican el plan original**:

1. **El sync de Chroma queda FUERA de este deploy.** No es "caro", es **destructivo en ambas variantes**. Detalle en §7. El cambio de código es inerte hasta que alguien lo ejecute: deployarlo no obliga a correrlo.
2. **`kza-voice` ocupa las DOS GPUs** ✅ (960 MiB en cuda:0 + **2176 MiB en cuda:1**, medido por PID). Y cuda:1 tiene **423 MiB libres**. El restart libera y re-pide esos 2176 MiB: es el punto más frágil de todo el procedimiento.
3. **`source` no exporta.** `/home/kza/secrets/*.env` son `EnvironmentFile=` de systemd (`VAR=valor`, **sin `export`**) ✅. Un `source archivo` define variable de *shell*, que Python **no hereda**. Todo source en este runbook usa `set -a; . archivo; set +a`. Esta es la causa raíz mecánica del incidente del 27/7.

---

## §0 · PRE-VUELO

### 0.1 Ventana de deploy

**Recomendada: 07:00–08:00 de un día hábil.** ✅ Sobre `events.db` (1113 comandos): 0 eventos en 14 días, 3 en 30 días, 1,5% histórico. El operador está despierto y puede validar por voz — que pesa más de lo que parece: con un peor caso de arranque de mic de 135s, *silencio ≠ roto*, y hace falta alguien que distinga "calentando" de "sordo".

**Prohibido:** ✅
- **16:00–19:00 cualquier día** — 59% del uso de los últimos 7 días (18h solo = 29%).
- **Domingo 03:00–05:00** — `retrain_cron` de `/home/kza/trading` con `CUDA_VISIBLE_DEVICES=0,1` (toma las dos GPUs) + `e2scrub_all` 03:10 + `podman-prune-all` 04:30.
- **03:00–03:30 cualquier día** — entrenamiento LoRA nocturno *in-process* dentro de `kza-voice`.

Colisión menor tolerable: `homelab-backup.timer` 03:37 (restic, `IOSchedulingClass=idle`, `TimeoutStartSec=6h`). Chequear que no esté corriendo: `ssh kza 'systemctl status homelab-backup.service --no-pager | head -3'`.

### 0.2 Avisarle a la casa (obligatorio antes de empezar)

Dos cambios audibles inmediatos:

- **5 de las 8 luces indexadas están `unavailable` desde el 2026-07-27 02:22:59** ✅ (`light.grupo_balcon / bano / cuarto / escalera / pasillo`). A partir del restart, pedirlas por voz **produce earcon de error** donde antes había silencio absoluto. **Eso es el fix funcionando**, no un bug: antes HA aceptaba la llamada, la filtraba en silencio y devolvía `success=true`.
- **La ventana de restart deja la casa sin voz varios minutos**, no segundos.

### 0.3 Laptop — estado y suite

```bash
cd /Users/yo/Documents/kza

git rev-parse --abbrev-ref HEAD   # -> feat/remediacion-fallos-silenciosos
git rev-parse HEAD                # -> 510c2a93a3e6f31310791a5ea23cef976c3da758
echo "porcelain:"; git status --porcelain --untracked-files=all   # -> vacío

git merge-base --is-ancestor feat/comandos-silenciosos HEAD && echo "OK: FF puro"
git rev-list --left-right --count feat/comandos-silenciosos...HEAD   # -> 0  18
```
✅ Los tres salieron correctos al redactar este runbook.

```bash
# Suite completa (~100s)
.venv/bin/python -m pytest tests/ -q --no-header -p no:cacheprovider 2>&1 | tail -20
```
**Esperado: `13 failed, 2739 passed, 1 xfailed` o `12 failed, 2740 passed, 1 xfailed`.** Los dos son verdes ✅.

Los **12 preexistentes** (verificados fallando idénticos en un HEAD limpio de `f3c94f1` extraído con `git archive`, sin tocar el working tree):
```
tests/safety/test_no_hardcoded_secrets.py::TestNoHardcodedSecrets::test_no_hardcoded_ips_in_source
tests/safety/test_no_hardcoded_secrets.py::TestNoHardcodedSecrets::test_no_hardcoded_ips_in_config
tests/unit/dashboard/test_system_monitor.py::test_services_snapshot_http_probe_marks_active
tests/unit/orchestrator/test_dispatcher.py::TestDispatchFastPath::test_dispatch_domotics_success
tests/unit/orchestrator/test_dispatcher.py::TestDispatchFastPath::test_dispatch_router_simple
tests/unit/orchestrator/test_dispatcher.py::TestDispatchBatch::test_dispatch_batch
tests/unit/pipeline/test_endpointing.py::TestVoiceProb::test_voice_prob_uses_vad_when_available
tests/unit/pipeline/test_request_router_gate.py::test_high_confidence_llm_command_dispatched
tests/unit/test_optimizations.py::TestVADStreaming::test_transcribe_streaming_yields_partial_results
tests/unit/test_optimizations.py::TestPrefixCaching::test_get_cache_stats
tests/unit/test_reasoner.py::TestFastRouterLoRA::test_fast_router_lora_init
tests/unit/test_reasoner.py::TestFastRouterLoRA::test_fast_router_generate_with_lora
```
El 13º es la flaky conocida `tests/unit/wakeword/test_wake_clip_writer.py::TestRotation::test_max_files_enforced_oldest_deleted` — **medida: falla 2 de 5 corridas (~40%)** ✅.

> ⚠️ Las 3 de `test_dispatcher.py` importan porque esta rama toca `dispatcher.py`. Están confirmadas como previas: **ya fallan contra el código que corre hoy en producción.**

**Si aparece un fallo fuera de esa lista, PARAR.** Para dirimir sin tocar el working tree ni worktrees:
```bash
B=/tmp/kza-baseline-f3c94f1; rm -rf $B; mkdir -p $B
git archive f3c94f1 | tar -x -C $B
(cd $B && /Users/yo/Documents/kza/.venv/bin/python -m pytest <ruta::del::test> -q --no-header -p no:cacheprovider)
```

```bash
# El cambio #1 (colección desbloqueada) — la evidencia dura
.venv/bin/python -m pytest tests/ -q --collect-only -p no:cacheprovider 2>&1 | tail -2
```
**Esperado: `2753 tests collected`, 0 errors.** ✅ Baseline en `f3c94f1`: `2668 collected, 4 errors → Interrupted` (`ValueError: torch.__spec__ is not set`). +85 tests y desaparecen los 4 errores que **hoy interrumpen la colección en producción**.

### 0.4 Server — estado y foto "ANTES"

⚠️ **El journal retiene ~22 h** ✅ (`[oww-dbg]` mete ~78k líneas/día; el arranque del 30/7 ya se cayó). **Todo lo que quieras conservar hay que volcarlo a archivo.**

```bash
D=/tmp/kza-deploy-$(date +%Y%m%d); mkdir -p $D

ssh kza 'echo "### FECHA"; date
echo "### SERVICIO"
systemctl --user show kza-voice -p ActiveState -p SubState -p NRestarts \
  -p ExecMainStartTimestamp -p MemoryCurrent -p WorkingDirectory
echo "### GPU"
nvidia-smi --query-gpu=index,gpu_uuid,memory.used,memory.free --format=csv
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv
echo "### DISCO"; df -h / | tail -1
echo "### GIT"; cd /home/kza/app && git rev-parse HEAD && git rev-parse --abbrev-ref HEAD
echo "porcelain:"; git status --porcelain
git worktree list; ls -la .git/index.lock 2>/dev/null || echo "sin index.lock: OK"
echo "### UNITS"; systemctl --user list-units "kza*" --all --no-pager
echo "### TIMERS"; systemctl --user list-timers --all --no-pager
echo "### CODE-INDEX"; curl -fsS http://127.0.0.1:9515/health' 2>&1 | tee $D/antes.txt
```

**Criterios de continuación (todos verificados al redactar):**

| Chequeo | Esperado ✅ | Si no |
|---|---|---|
| `git rev-parse HEAD` | `f3c94f1d6e7930754e57ba41ba56aa03c32c256d` | Parar; el server no está donde se cree |
| `git status --porcelain` | vacío | Parar; hay WIP sin commitear |
| `NRestarts` | `0` | Investigar por qué reinició |
| `WorkingDirectory` | `/home/kza/app` | **Crítico**: si cambió, el `health_path` del poller no coincide (§3) |
| `index.lock` | ausente | ⚠️ `/home/kza/bench-ambient` comparte `.git`. `pgrep -u kza git` antes de tocar nada. **No borrar el lock a ciegas** |
| disco `/` | ≥5G libres (había 327G) | — |
| `indexed_sha` de :9515 | `f3c94f1d6e79...` | Anotarlo; cambia en §2.5 |
| timers | ninguno de KZA | Si ya hay `kza-audio-*`, resolver antes de §3 |

**VRAM — el chequeo que decide si se puede seguir.** Medición al redactar:
```
GPU0  used 1919  free 5913    ← kza-voice 960 + Ollama(pid 2276091) 878   ⚠️ el Ollama NO es KZA
GPU1  used 7418  free  423    ← kza-llm-fast 5214 + kza-voice 2176
```
🔶 En el restart, `kza-voice` suelta 2176 MiB de cuda:1 y los re-pide. Es autoconsistente **solo si `kza-llm-fast` no creció** (KV cache) en el ínterin. Con 423 MiB de headroom no hay margen. **Si `free` en GPU1 bajó de ~400 MiB, no deployar**: reiniciar primero `kza-llm-fast` para que suelte el KV cache, o esperar.

### 0.5 Foto "ANTES" de HA, Chroma, smoke y latencia

```bash
# Entidades HA (read-only; el token nunca se imprime)
ssh kza 'set -a; . /home/kza/secrets/.env >/dev/null 2>&1; set +a
curl -s -m 20 -H "Authorization: Bearer ${HOME_ASSISTANT_TOKEN}" "${HOME_ASSISTANT_URL}/api/states" -o /tmp/.ha.json
/home/kza/app/.venv/bin/python - <<PY
import json, collections
d=json.load(open("/tmp/.ha.json")); st=collections.Counter(e["state"] for e in d)
print("total:", len(d), "| unavailable:", st["unavailable"], "| unknown:", st["unknown"])
for e in sorted([x for x in d if x["entity_id"].startswith(("light.grupo_","scene."))], key=lambda x:x["entity_id"]):
    print("  %-26s %-12s attrs=%d" % (e["entity_id"], e["state"], len(e["attributes"])))
PY
rm -f /tmp/.ha.json' 2>&1 | tee -a $D/antes.txt
```
**Esperado ✅ (medido):** 712 entidades, 439 `unavailable`, 30 `unknown`. De los 8 `light.grupo_*`: **3 vivas** (cocina/escritorio/living, 15 atributos c/u) y **5 muertas** (balcón/baño/cuarto/escalera/pasillo, **4 atributos** — la amputación de atributos que justifica el guard). `scene.fria` y `scene.relax` en `unknown`, que **no** es problema.

> 👉 **Anotar la lista exacta de entidades `unavailable` en este momento.** Es el criterio de aceptación del smoke test post-deploy (§4.4), no el conteo.

```bash
# Índice de Chroma (read-only, immutable, no toma locks)
ssh kza '/home/kza/app/.venv/bin/python -' <<'PY' 2>&1 | tee -a $D/antes.txt
import sqlite3, collections
c=sqlite3.connect("file:/home/kza/app/data/chroma_db/chroma.sqlite3?mode=ro&immutable=1", uri=True)
cnt=collections.Counter()
for k,v in c.execute("select key,string_value from embedding_metadata where key='entity_id'"): cnt[v]+=1
print("TOTAL docs:", sum(cnt.values()))
for k,v in sorted(cnt.items()): print("  %-30s %4d" % (k,v))
PY
```
**Esperado ✅: 337 docs.** Reparto: cocina/escritorio/living **92 c/u**, las 5 caídas **8 c/u**, escenas 21. Ese `8 vs 92` es exactamente el índice empobrecido que el guard existe para prevenir.

```bash
# Smoke test ANTES (con el servicio vivo — es la práctica establecida)
ssh kza 'cd /home/kza/app && set -a && . /home/kza/secrets/.env && set +a && .venv/bin/python tools/smoke_test.py; echo "exit=$?"' 2>&1 | tee $D/smoke-ANTES.txt
```
**Esperado ✅: 2 problemas** (`light.grupo_pasillo` vía hall, `light.grupo_bano`), exit 1. Cobertura vieja = 5 `default_light`.

```bash
# Latencia baseline
ssh kza '/home/kza/app/.venv/bin/python -' <<'PY' 2>&1 | tee -a $D/antes.txt
import sqlite3, datetime, statistics, time
c=sqlite3.connect("file:/home/kza/app/data/latency.db?mode=ro", uri=True)
cut=time.time()-7*86400
rows=[(t,ms,it) for t,ms,mt,it in c.execute("select timestamp,total_ms,met_target,intent from latency_records") if t>=cut]
dom=sorted(ms for _,ms,it in rows if it=="domotics")
print("7d: n=%d | domotics n=%d p50=%.0f p90=%.0f" % (len(rows), len(dom),
      statistics.median(dom) if dom else 0, dom[int(.9*(len(dom)-1))] if dom else 0))
for t,ms,it in rows[-8:]:
    print("  %s %8.0fms %s" % (datetime.datetime.fromtimestamp(t).strftime("%m-%d %H:%M:%S"), ms, it))
PY
```
> ⚠️ Los timestamps son **epoch float**: `strftime('%H', timestamp)` devuelve NULL en silencio. Convertir con `datetime.fromtimestamp()`.
>
> 🔶 Baseline actual: `domotics n=6 p50=880 p90=1447` con `met_target=0` en los 8 registros desde el último restart; `llm_router=824ms` en el último. Se está cayendo del fast path seguido. **Esto es preexistente** (`early_dispatch:false` + destrozo del STT) y **este deploy no lo cambia**. Es la vara, no un criterio de éxito.

### 0.6 Publicar la rama 🛑

⚠️ **La rama NO existe en `origin`** ✅ y no tiene upstream. Sin esto el server no puede hacer checkout.

```bash
cd /Users/yo/Documents/kza
git push -u origin feat/remediacion-fallos-silenciosos
git ls-remote --heads origin feat/remediacion-fallos-silenciosos   # -> 510c2a9... (ya no vacío)
```

> ⚠️ **NO usar `scripts/kza-push`.** Ese script hace `git fetch kza:/home/kza/app` — va **del server hacia GitHub**, para WIP nacido en el server. Esta rama nació en la laptop. `scripts/kza-sync` sí sirve: es un reporte read-only puro.
>
> 🔶 El server llega a origin por **HTTPS** (`git ls-remote` y `git fetch --dry-run` dieron exit 0 sin pedir credenciales ✅). No inspeccioné `credential.helper`. Si el fetch de §2.1 fallara por auth, el plan B desde la laptop es `git push kza:/home/kza/app feat/remediacion-fallos-silenciosos` (funciona porque el árbol del server está limpio y no está parado en esa rama).

⏸ **PARADA.** Antes de seguir: suite verde contra el baseline, server limpio en `f3c94f1`, GPU1 con headroom, casa avisada, foto "antes" guardada en `$D`.

---

## §1 · BACKUP 🛑

⚠️ **No tarear el repo entero: `.git` pesa 72G** ✅. Lo irreemplazable es `data/` (206M) y `config/` (444K); el código se recupera con `git switch` porque ya está en origin.

⚠️ **El orden importa.** `data/` tiene 5 SQLite abiertos por el proceso (`chroma.sqlite3`, `events.db`, `latency.db`, `ambient.db`, `audit.db`). Tarear con el servicio vivo da un backup posiblemente inconsistente — inútil justo cuando hace falta. Cuesta ~20s más de downtime y vale la pena.

```bash
# 1.1 Punto de retorno por escrito
ssh kza 'cd /home/kza/app && git rev-parse HEAD | tee /home/kza/ROLLBACK_SHA.txt'
#   -> f3c94f1d6e7930754e57ba41ba56aa03c32c256d
```

🛑 **A partir de acá arranca el downtime.**

```bash
# 1.2 Parar
ssh kza 'systemctl --user stop kza-voice; systemctl --user is-active kza-voice'   # -> inactive

# 1.3 Backup consistente (~206M, segundos; 327G libres)
ssh kza 'mkdir -p /home/kza/backups && cd /home/kza/app && \
  tar czf /home/kza/backups/kza-pre-deploy-$(date +%Y%m%d-%H%M).tar.gz data config && \
  ls -lh /home/kza/backups/ | tail -3'

# 1.4 Verificar que el tar se puede LEER antes de seguir
ssh kza 'T=$(ls -t /home/kza/backups/kza-pre-deploy-*.tar.gz | head -1); echo "$T"; \
         tar tzf "$T" | wc -l; tar tzf "$T" | grep -c "^data/chroma_db/"'
```

> ⚠️ **Nunca `git stash -u` en el server** — se lleva puesto `models/` (15G). Los 4 stashes viejos que ya están ahí (`pre-merge-deploy-2026-04-28`, 3× `pre-pull-2026-04-23`): no tocarlos.

---

## §2 · DEPLOY DEL CÓDIGO 🛑

```bash
ssh kza
cd /home/kza/app

# 2.1 Traer refs (no toca el working tree)
git fetch origin feat/remediacion-fallos-silenciosos
git rev-parse origin/feat/remediacion-fallos-silenciosos     # -> 510c2a9...

# 2.2 Último chequeo
git status --porcelain && echo "(limpio)"
git merge-base --is-ancestor HEAD origin/feat/remediacion-fallos-silenciosos \
  && echo "OK: FF puro desde el HEAD actual" || echo "ABORTAR"

# 2.3 Cambiar de rama — el working tree ES el deploy
git switch -c feat/remediacion-fallos-silenciosos origin/feat/remediacion-fallos-silenciosos
git rev-parse HEAD          # -> 510c2a93a3e6f31310791a5ea23cef976c3da758
git status --porcelain      # -> vacío
```

> **Por qué `switch` y no `merge --ff-only`:** mergear estando parado en `feat/comandos-silenciosos` avanzaría esa rama a `510c2a9` y la haría divergir de `origin/feat/comandos-silenciosos` — que es el punto de rollback. `switch` deja la rama vieja intacta apuntando a `f3c94f1`.
>
> ⚠️ **`git switch` NO dispara el hook `post-merge`** ✅ (el único hook del server, que reindexa el code-index). El reindex va a mano en §2.5. Omitirlo deja `code_search` devolviendo resultados del código viejo — otro proxy mentiroso.

```bash
# 2.4 Arrancar, capturando el arranque a archivo (el journal rota en 22h)
date +"%Y-%m-%d %H:%M:%S" > /home/kza/backups/restart-mark.txt
systemctl --user start kza-voice
systemctl --user is-active kza-voice     # -> active
```

En otra terminal, desde la laptop:
```bash
ssh kza 'journalctl --user -u kza-voice -f --no-pager' | tee /tmp/kza-deploy-$(date +%Y%m%d)/arranque.log
```

Buscar en ese log:
- `MultiRoomAudioLoop created (2 rooms: cocina, escritorio)` — **debe decir exactamente 2** (son las únicas con `mic_usb_port` ✅: cocina `5-5.3`, escritorio `3-1.4`; living y hall lo tienen en `null`).
- 🔶 Warning por `Wants=kza-llm-ik.service` — **ese unit ya no existe** ✅ (`not-found`). Es `Wants=` (el `Requires=` fue removido a propósito), **no bloquea**. Ruido esperado.
- Un OOM en cuda:1 aquí = rollback inmediato (§6.1).

```bash
# 2.5 Reindex del code-index (lo que haría el hook post-merge)
curl -fsS -X POST -m 5 http://127.0.0.1:9515/reindex -H 'Content-Type: application/json' -d '{"mode":"incremental"}'
sleep 20
curl -fsS http://127.0.0.1:9515/health
#   -> "indexed_sha": "510c2a93a3e6f31310791a5ea23cef976c3da758", "reindex_running": false
```

### ⏸ 2.6 GATE — el snapshot de audio

**Este es el criterio de continuación de todo el frente de observabilidad. Si no pasa, NO instalar el timer (§3): pasar a diagnóstico o rollback.**

⚠️ **NO CONCLUIR NADA ANTES DE 4 MINUTOS.** Arranques del XVF3800 medidos: 1,5–2s típico, **135s el peor caso**. Encima hay que cargar Whisper v3-turbo + Kokoro + BGE-M3 + ECAPA + wav2vec2. `first_frame_grace_s=180` está puesto por eso. Este es *exactamente* el error documentado en la memoria del proyecto ("medí a los 20-40s y concluí mal").

```bash
for i in $(seq 1 24); do date +%H:%M:%S; \
  cat /home/kza/app/data/audio_health.json 2>/dev/null || echo "(sin snapshot todavia)"; \
  echo; sleep 15; done
```

**Éxito:**
```json
{"wall": 1754...., "rooms": {"cocina": {"age_s": 0.3, "ever": true},
                             "escritorio": {"age_s": 0.5, "ever": true}}}
```
- ✅ **Exactamente 2 rooms**, ambas con `"ever": true`. Ni una más ni una menos. (El archivo **no existía antes del deploy** ✅ — estado inicial limpio.)
- El snapshot se reescribe **cada ~2s** (`rooms.stream_watchdog.check_interval_s: 2.0` ✅). Verificar que avanza: `stat -c %y ...; sleep 5; stat -c %y ...`.
- Permisos: `ls -l` debe dar **0644 kza:kza** (el `os.fchmod(fd, 0o644)` del writer).
- `ever=false` con `age_s` creciendo durante los primeros 180s es **normal**.

**Anotar el reloj cuando cada room pasa a `"ever": true`.** 🔶 Es el primer dato duro de arranque que va a tener el proyecto (el arranque anterior se cayó del journal) y calibra si `first_frame_grace_s=180` está bien puesto.

**Fallo (→ §6.1):** el archivo no aparece en 4 min, o alguna room queda en `ever:false`, o aparecen ≠2 rooms.

```bash
# Chequeo puro de la lógica (SOLO LEE, no notifica a HA)
cd /home/kza/app && .venv/bin/python -c "
import json,time,sys; sys.path.insert(0,'/home/kza/app')
from src.monitoring.audio_health import evaluate_health
s=json.load(open('data/audio_health.json'))
print('antiguedad snapshot:', round(time.time()-s['wall'],1),'s')
for r,i in sorted(s['rooms'].items()): print(f\"  {r}: age_s={i['age_s']:.1f} ever={i['ever']}\")
print('SORDAS:', evaluate_health(s, time.time(), 300.0))
viejo={'wall':time.time(),'rooms':{'X':{'age_s':9999,'ever':True}}}
print('prueba negativa (debe listar X):', evaluate_health(viejo, time.time(), 300.0))"
```
**Esperado: `SORDAS: []` y `prueba negativa: ['X']`.**

> Antigüedad del snapshot >120s (`snapshot_stale_after_s`) ⇒ el watchdog dejó de publicar ⇒ el poller marcará **todas** las rooms como sordas. Es por diseño: el silencio del vigilante no es "todo bien".

---

## §3 · INSTALACIÓN DEL TIMER DEL POLLER

**Solo después de que §2.6 pase.** Instalarlo antes crea una notificación falsa en HA que hay que borrar a mano — y la notificación **nunca se auto-borra** (ver riesgo abajo).

**Corre como `kza`** — no por el traverse (`/home/kza` es **0751** ✅, `drwxr-x--x`: otros usuarios sí pueden atravesarlo; el "0700" del review era falso). **El bloqueo real es `/home/kza/secrets` = 0700** ✅: el `HOME_ASSISTANT_TOKEN` solo lo lee `kza`. Sigue siendo un proceso **externo** a `kza-voice`, que es lo que exige el diseño.

Nombre elegido: **`kza-audio-watchdog`** (los frentes propusieron dos nombres distintos; este es el que queda).

### 3.1 Crear las units

```bash
cat > /home/kza/.config/systemd/user/kza-audio-watchdog.service <<'EOF'
[Unit]
Description=KZA audio watchdog — avisa por HA si un micrófono quedó sordo
Documentation=file:///home/kza/app/tools/audio_watchdog_alert.py
# A propósito SIN Requires=/After=kza-voice.service: este poller tiene que correr
# aunque kza-voice esté caído o trabado — "no hay snapshot" es justamente una de
# las anomalías que debe reportar, no una razón para no correr.
# A propósito SIN ConditionPathExists= sobre audio_health.json, por lo mismo: esa
# condición convertiría la ausencia del snapshot en un skip SILENCIOSO.
# StartLimitIntervalSec=0: un watchdog no puede quedar silenciado por el rate
# limiter de systemd.
StartLimitIntervalSec=0

[Service]
Type=oneshot
WorkingDirectory=/home/kza/app
# Aporta HOME_ASSISTANT_URL (http://localhost:8123) y HOME_ASSISTANT_TOKEN.
# Es el mismo archivo que ya consume kza-voice.service.
EnvironmentFile=/home/kza/secrets/.env
Environment=PYTHONUNBUFFERED=1
# --health-path explícito aunque coincide con el default: deja VISIBLE que esto
# depende de que kza-voice conserve WorkingDirectory=/home/kza/app (su YAML
# declara el path relativo "./data/audio_health.json").
ExecStart=/home/kza/app/.venv/bin/python /home/kza/app/tools/audio_watchdog_alert.py --once --health-path /home/kza/app/data/audio_health.json --deaf-after-s 300
# Exit codes: 0 = sano | 1 = sordera detectada (o error del ciclo) | 2 = falta token.
# NO se pone SuccessExitStatus=1 a propósito: así la unit queda en `failed`
# mientras hay sordera, y `systemctl --user --failed` es un segundo canal de
# alarma que NO depende de que HA esté vivo.
TimeoutStartSec=60s
EOF

cat > /home/kza/.config/systemd/user/kza-audio-watchdog.timer <<'EOF'
[Unit]
Description=KZA audio watchdog — chequeo periódico de sordera de micrófonos
Documentation=file:///home/kza/app/tools/audio_watchdog_alert.py

[Timer]
Unit=kza-audio-watchdog.service
# 15 min post-boot antes del PRIMER chequeo: kza-voice tiene que cargar 5 modelos
# en cuda:0 antes de que _stream_watchdog publique. Chequear antes daría
# "(sin snapshot de audio)" — un falso positivo que deja una notificación PEGADA.
OnBootSec=15min
# OnUnitInactiveSec y NO OnUnitActiveSec: un Type=oneshot que sale ≠0 pasa de
# `activating` a `failed` SIN llegar a `active`, y OnUnitActiveSec podría no
# reprogramar nunca — congelando el watchdog justo cuando detectó algo.
OnUnitInactiveSec=5min
AccuracySec=1min

[Install]
WantedBy=timers.target
EOF

chmod 644 /home/kza/.config/systemd/user/kza-audio-watchdog.*
```

**Justificación del intervalo de 5 min:** la señal misma tiene resolución de 300s (`--deaf-after-s`). Pollear más rápido no adelanta la detección, solo suma arranques de Python. Peor caso: 300s (umbral) + 300s (hueco de poll) ≈ **10 minutos**, contra los incidentes de 27h y 7h. Sin riesgo de spam: `notify_ha` manda siempre `notification_id: "kza_audio_deaf"`, así que HA **reemplaza** en vez de acumular.

**Por qué `--once` + timer y no el bucle `--interval-s`:** el bucle vive en un proceso propio y, si se traba, se traba en silencio — el mismo modo de falla que vino a detectar. Con el timer, el scheduler es systemd y cada corrida es un proceso nuevo y desechable.

**Lingering:** ✅ `Linger=yes` para `kza` (`loginctl show-user kza -p Linger`). No hace falta habilitar nada.

**Costo:** ✅ la cadena de imports del poller es **stdlib pura** (`src/monitoring/__init__.py` → `latency_monitor` + `health_aggregator`; `audio_health.py` → `contextlib/json/logging/os/tempfile`). **Cero torch, cero CUDA.** ~0,3–0,5s por corrida.

### 3.2 Validar sin arrancar el timer

```bash
systemctl --user daemon-reload
systemd-analyze --user verify /home/kza/.config/systemd/user/kza-audio-watchdog.service \
                              /home/kza/.config/systemd/user/kza-audio-watchdog.timer
# Salida vacía = OK. 🔶 No pude verificarlo (regla read-only): systemd 255 en el server,
# macOS no tiene systemd-analyze. Revisado a mano: StartLimitIntervalSec en [Unit] (no
# en [Service], el error clásico), service sin [Install], timer con WantedBy=timers.target.
systemctl --user cat kza-audio-watchdog.service kza-audio-watchdog.timer
```

### 3.3 Corrida manual con audio SANO

```bash
systemctl --user start kza-audio-watchdog.service
systemctl --user show kza-audio-watchdog.service -p Result -p ExecMainStatus
#   -> Result=success  ExecMainStatus=0
journalctl --user -u kza-audio-watchdog -n 20 --no-pager
```
- ✅ El único ruido esperado es `RequestsDependencyWarning: Unable to find acceptable character detection dependency` — `requests 2.32.5` está instalado y el warning es **inocuo para este script** (`notify_ha` solo lee `resp.status_code`, nunca `.text`/`apparent_encoding`). Aparecerá en cada corrida. **No confundirlo con un fallo.**
- `ExecMainStatus=1` con `Sin audio de:` → hay una room realmente sorda, o el `--health-path` apunta mal.
- `ExecMainStatus=2` → `HOME_ASSISTANT_TOKEN` no llegó del `EnvironmentFile`.

### 3.4 Prueba negativa end-to-end (⚠️ CREA una notificación real en HA)

Sin desenchufar nada:
```bash
python3 -c "
import json, time, sys
json.dump({'wall': time.time(), 'rooms': {'__test__': {'age_s': 9999.0, 'ever': True}}}, sys.stdout)" \
  > /tmp/kza_health_fake.json

cd /home/kza/app && set -a && . /home/kza/secrets/.env && set +a && \
  .venv/bin/python tools/audio_watchdog_alert.py --once --health-path /tmp/kza_health_fake.json
echo "exit=$?"
#   Esperado: "Sin audio de: __test__"  exit=1  + notificación real en HA
```
🔶 Este es el **único** chequeo que valida el POST real contra HA. Todo lo demás sobre el camino de alarma es lógica verificada offline.

**Borrar la notificación después (obligatorio — nunca se auto-borra):**
```bash
set -a; . /home/kza/secrets/.env; set +a
curl -s -X POST -H "Authorization: Bearer $HOME_ASSISTANT_TOKEN" -H "Content-Type: application/json" \
  -d '{"notification_id":"kza_audio_deaf"}' \
  "$HOME_ASSISTANT_URL/api/services/persistent_notification/dismiss"
rm -f /tmp/kza_health_fake.json
```

> ⚠️ **Gap de diseño a saber:** `audio_watchdog_alert.py` **solo llama `create`, nunca `dismiss`** ✅. Cuando el mic se recupera, la notificación **queda pegada hasta que un humano la descarte**. Cualquier falso positivo transitorio deja una alarma visible indefinidamente — y así es como el humano aprende a ignorar el watchdog. El arreglo de fondo (fuera de este deploy) es llamar a `dismiss` cuando `deaf` vuelve a estar vacío.

### 3.5 Armar el timer

```bash
systemctl --user enable --now kza-audio-watchdog.timer
systemctl --user list-timers kza-audio-watchdog.timer --all --no-pager
#   -> una fila con NEXT ≈ ahora+5min, ACTIVATES=kza-audio-watchdog.service
systemctl --user is-enabled kza-audio-watchdog.timer   # -> enabled
```

### 3.6 🔶 Verificación crítica: ¿el timer reprograma tras un fallo?

Es el modo de falla peor de este frente (un watchdog congelado justo cuando detecta algo) y **no pude confirmarlo read-only**.

```bash
systemctl --user start kza-audio-watchdog.service --no-block 2>/dev/null
# Forzar un fallo apuntando a un path inexistente, sin tocar el snapshot real:
cd /home/kza/app && set -a && . /home/kza/secrets/.env && set +a && \
  .venv/bin/python tools/audio_watchdog_alert.py --once --health-path /tmp/no-existe.json; echo "exit=$?"
# Después de la próxima corrida programada que falle:
systemctl --user list-timers kza-audio-watchdog.timer --no-pager
```
**El campo `NEXT` tiene que mostrar una próxima corrida (~5 min), NO `n/a`.** Si sale `n/a`: agregar `SuccessExitStatus=1` al `[Service]` + `daemon-reload` — con eso la unit siempre termina `success` y cualquier trigger funciona, al precio de perder el canal `systemctl --user --failed`.

---

## §4 · VERIFICACIÓN

### 4.1 Servicio estable

```bash
ssh kza 'systemctl --user show kza-voice -p ActiveState -p SubState -p NRestarts -p ExecMainStartTimestamp
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv'
```
`NRestarts=0`. ⚠️ **`StartLimitBurst=3 / StartLimitIntervalSec=300`** ✅: tres restarts fallidos en 5 min dejan la unit `failed` y **systemd no la vuelve a levantar sola** — la casa queda sin voz hasta intervención manual. Si pasa: leer el journal **primero**, después `systemctl --user reset-failed kza-voice`, y recién entonces reintentar.

### 4.2 El precheck de disponibilidad

```bash
# ⚠️ El grep SIN `grep -v oww-dbg` es inútil: [oww-dbg] mete ~2 líneas/s ✅
ssh kza 'journalctl --user -u kza-voice --since "30 min ago" --no-pager | grep -v oww-dbg | grep -F "[HA-UNAVAILABLE]"'
```

**Prueba por voz positiva** (debe seguir igual que siempre): *"nexa, prendé la luz del living"* → la luz prende, sin `[HA-UNAVAILABLE]`, sin earcon.

**Prueba por voz negativa** (el cambio nuevo): *"nexa, prendé la luz del cuarto"* → **suena el earcon** + línea `[HA-UNAVAILABLE]`. **Esto es ÉXITO.** Antes: silencio absoluto.

🔶 El earcon: el asset existe ✅ (`/home/kza/app/data/earcons/not_understood.wav`, 9644 B) y la firma `play_earcon(zone_id=...)` encaja con la llamada del dispatcher. **Pero `play_earcon` es no-op silencioso si `self._earcon_audio is None`** y la carga del asset + ruteo por MA1260 no son observables read-only. Si aparece la línea pero **no** se escucha nada, buscar `No pude reproducir earcon`; si tampoco está, el asset no cargó o la zona resolvió mal.

**REGRESIÓN si aparece `[HA-UNAVAILABLE]` sobre `light.grupo_living|cocina|escritorio` o sobre `scene.*`** ⇒ el precheck está reteniendo algo vivo (cache stale) ⇒ **rollback parcial (§6.2)**.

```bash
# Auditoría — el comando retenido debe quedar registrado, no evaporarse
ssh kza 'cd /home/kza/app && sqlite3 -readonly data/audit.db ".tables"'
ssh kza 'cd /home/kza/app && sqlite3 -readonly data/audit.db \
  "select ts, rule_name, reason from events where rule_name=\"entity_unavailable\" order by ts desc limit 5;"'
```
🔶 El esquema de `audit.db` no lo inspeccioné (no quise abrir la DB de producción): correr `.tables`/`.schema` primero. **Y el evento `ha_action_blocked` solo se emite si `self.hooks is not None`** — en memoria consta que OpenClaw quedó con `hooks.enabled=false`. Si están apagados, **el precheck funciona igual pero no deja rastro en `audit.db`**, y eso no es un fallo del deploy. Verificar con `grep -n -A3 "^hooks:" config/settings.yaml`.

### 4.3 El poller

```bash
ssh kza 'cd /home/kza/app && set -a && . /home/kza/secrets/.env && set +a && \
  .venv/bin/python tools/audio_watchdog_alert.py --once; echo "exit=$?"'
#   ÉXITO: sin salida + exit=0
```
⚠️ Este comando **escribe en HA** si detecta sordera. Para inspeccionar sin notificar, usar el chequeo puro de §2.6.

### 4.4 Smoke test — **va a reportar MÁS problemas que antes, y eso es ÉXITO**

```bash
ssh kza 'cd /home/kza/app && set -a && . /home/kza/secrets/.env && set +a && \
  .venv/bin/python tools/smoke_test.py; echo "exit=$?"' 2>&1 | tee $D/smoke-DESPUES.txt
diff $D/smoke-ANTES.txt $D/smoke-DESPUES.txt
```

Antes cubría **5 entidades** (los `default_light`) ⇒ 2 problemas. Ahora cubre **13** (índice de Chroma ∪ `default_light` = 8 grupos + 5 escenas) ⇒ **5 problemas con el estado de HA medido, exit 1.**

> ⚠️ **Discrepancia entre frentes, resuelta:** el review original predijo **8 problemas** (5 luces + climate + 2 TV); dos frentes independientes midieron **5** contra el HA de hoy. No es contradicción de fondo — el número es función de cuántas entidades estén `unavailable` en ese instante (hoy 439 de 712, un número que se mueve).
>
> **El criterio de aceptación NO es el conteo, es la lista de nombres.** Fallos ACEPTABLES: exactamente las entidades que `GET /api/states` (§0.5) reportó `unavailable`. Los 3 nuevos respecto de antes (balcón, cuarto, escalera) **nunca fueron chequeados** porque no son `default_light` de ninguna room — es literalmente el punto del commit `cc16b56`.

**REGRESIÓN si:** aparece en rojo `light.grupo_living|cocina|escritorio`; falla alguna `scene.*` (las que están en `unknown` deben pasar **en verde** — `entity_problem` solo marca `unavailable`, alineado con `is_entity_available`; es el fix del commit `2780f44`); falla alguna de las 5 frases canónicas; o sale `no pude leer el índice de Chroma`.

Costo: ~30–60s. ✅ Carga BGE-M3 **en CPU** (`embeddings.device: "cpu"`), **no compite por VRAM** — importante con cuda:1 en 423 MiB libres.

### 4.5 Validación final por voz

Decir en **escritorio** y en **cocina**: *"nexa, prendé la luz del living"* / *"nexa, apagá la luz de la cocina"*. Confirmar **visualmente**.

```bash
ssh kza 'tail -3 /home/kza/logs/kza-metrics.jsonl | cut -c1-400'
ssh kza 'journalctl --user -u kza-voice --since "-10 min" --no-pager | grep -v oww-dbg | \
  grep -iE "HA-CALL|HA-UNAVAILABLE|earcon|\[OK\]|\[SLOW\]" | tail -25'
```
🔑 Verificar por **`HA-CALL`**, no por `source=manual`.

---

## §5 · PRIMERAS 24 HORAS

⚠️ **El journal se borra solo en ~22h.** Volcar a archivo lo que se quiera conservar.

```bash
# D0 — snapshot horario (cron manual o a mano un par de veces)
ssh kza 'journalctl --user -u kza-voice --since "-1h" --no-pager | grep -v oww-dbg' \
  >> ~/kza-postdeploy-$(date +%F).log
```

| # | Qué | Comando | Sano | Alarma |
|---|---|---|---|---|
| D1 | Heartbeat del snapshot | `ssh kza 'stat -c "%y" /home/kza/app/data/audio_health.json'` | <10s de antigüedad, **siempre** | cualquier otra cosa |
| D2 | El publicador falla | `... \| grep -F "No pude publicar audio_health"` | **vacío** | cualquier línea = el vigilante no puede publicar |
| D3 | Earcon mudo | `... \| grep -F "No pude reproducir earcon"` | **vacío** | el precheck retiene pero el usuario no se entera |
| D4 | Volumen del precheck | ver abajo | solo las 5 luces caídas, volumen bajo | living/cocina/escritorio en la lista, o conteo en decenas ⇒ **§6.2** |
| D5 | Fast path vivo | ver abajo | `[HA-CALL]` con volumen comparable al de antes | **cero `[HA-CALL]` + varios `[HA-UNAVAILABLE]` = el precheck se está comiendo todo** |
| D6 | Reinicios | `ssh kza 'systemctl --user show kza-voice -p NRestarts -p ActiveEnterTimestamp'` | `NRestarts` no crece | ver §4.1 |
| D7 | El poller | `ssh kza 'journalctl --user -u kza-audio-watchdog --since "-24h" --no-pager \| tail -30'` | corridas silenciosas (+ el warning de chardet) | `Sin audio de:`, `snapshot ilegible`, `no pude avisar a HA` |
| D8 | Excepciones nuevas | `... \| grep -E "Traceback\|ERROR" \| tail -20` | nada nuevo vs la foto ANTES | — |
| D9 | Canal independiente de HA | `ssh kza 'systemctl --user --failed --no-pager'` | `kza-audio-watchdog` ausente | presente = ir al journal (D7) |

```bash
# D4
ssh kza 'journalctl --user -u kza-voice --since "24 hours ago" --no-pager | grep -v oww-dbg \
  | grep -F "[HA-UNAVAILABLE]" | grep -oE "@[a-z_]+\.[a-z0-9_]+" | sort | uniq -c | sort -rn'

# D5
ssh kza 'journalctl --user -u kza-voice --since "24 hours ago" --no-pager | grep -v oww-dbg | grep -cE "\[HA-CALL\]"'
```

> ⚠️ **Lo que NO se validó y solo se confirma en el primer incidente real:** el escenario de sordera **no se reproduce sin desenchufar físicamente un XVF3800**. La prueba de §3.4 valida la lógica de detección + el camino a HA con un snapshot sintético; **no** valida que `_stream_watchdog` publique correctamente ante una falla real de isócronos (endpoint isócrono muerto → ceros silenciosos, sin error en dmesg). **No declarar "validado" lo que solo está "instalado".**

---

## §6 · ROLLBACK

### 6.1 Total — volver a `f3c94f1`

**Disparar si:** `NRestarts` sube · OOM en cuda:1 · `audio_health.json` no aparece o ninguna room llega a `"ever": true` en 4 minutos · regresión en el smoke test (nombres, no conteo) · `[HA-UNAVAILABLE]` masivo sobre entidades vivas.

**El orden importa: el timer se para ANTES del checkout.** Si se revierte el código con el timer vivo, `audio_health.json` queda congelado, su `wall` supera `snapshot_stale_after_s=120` y el poller reporta **todas** las rooms como sordas para siempre — alarma permanente.

```bash
ssh kza
# 1) Parar el poller PRIMERO (su código desaparece en el paso 3)
systemctl --user disable --now kza-audio-watchdog.timer
systemctl --user stop kza-audio-watchdog.service 2>/dev/null
systemctl --user reset-failed kza-audio-watchdog.service 2>/dev/null

# 2) ⚠️ Si se usó el rollback parcial (§6.2), config/settings.yaml está sucio y el switch FALLA
cd /home/kza/app
git status --porcelain config/settings.yaml
#   si hay salida:  git checkout -- config/settings.yaml    (descarta el flag editado)

# 3) Volver
systemctl --user stop kza-voice
git switch feat/comandos-silenciosos
git rev-parse HEAD          # DEBE dar f3c94f1d6e7930754e57ba41ba56aa03c32c256d
git status --porcelain      # vacío

# 4) Arrancar
systemctl --user start kza-voice
sleep 20 && systemctl --user is-active kza-voice

# 5) Reindex del code-index (switch no dispara post-merge)
curl -fsS -X POST -m 5 http://127.0.0.1:9515/reindex -H 'Content-Type: application/json' -d '{"mode":"incremental"}'

# 6) Limpiar artefactos que el código viejo no escribe ni lee
rm -f /home/kza/app/data/audio_health.json
rm -f /home/kza/.config/systemd/user/kza-audio-watchdog.{service,timer}
systemctl --user daemon-reload
systemctl --user list-unit-files "kza-audio*"   # -> 0 unit files listed
```

**Verificar la vuelta:**
```bash
journalctl --user -u kza-voice --since "5 min ago" --no-pager | grep -v oww-dbg | grep -c "HA-UNAVAILABLE"   # -> 0
cd /home/kza/app && set -a && . /home/kza/secrets/.env && set +a && .venv/bin/python tools/smoke_test.py | tail -3   # -> vuelve a 2 problemas
```

**Restore de `data/` — solo si quedó corrupto** (no debería: el deploy no escribe ahí; `data/` está gitignored ✅ y no puede bloquear un checkout):
```bash
ssh kza 'systemctl --user stop kza-voice && cd /home/kza/app && \
  tar xzf /home/kza/backups/kza-pre-deploy-<TIMESTAMP>.tar.gz && \
  systemctl --user start kza-voice'
```

**Chroma:** si se siguió la recomendación de §7 (**no correr el sync**), **no hay nada que revertir** — el guard solo aborta, no escribe, y el cambio de cache key es inerte. Verificar con el comando de §0.5: debe seguir dando **337 docs**.

### 6.2 Parcial — apagar solo el precheck, sin revertir código

**Cuándo:** el precheck retiene comandos contra dispositivos **que en realidad ya volvieron**. Típicamente tras un reinicio de Z2M/HA con el WS de `state_changed` muerto en silencio — el cache no sana hasta el snapshot REST (`full_refresh_interval_s: 300`), o sea hasta 5 min de comandos rechazados con earcon. **No hay hoy cota de frescura sobre el cache: este flag es la única mitigación sin tocar código.**

```bash
ssh kza
cd /home/kza/app
grep -n "unavailable_precheck_enabled" config/settings.yaml
# editar:  unavailable_precheck_enabled: true   ->   false

# ⚠️ REINICIO OBLIGATORIO. El flag se lee UNA sola vez, en main.py al arranque
#    (config.get("home_assistant", {}).get("unavailable_precheck_enabled", True)),
#    se pasa por constructor al MultiUserOrchestrator y de ahí al RequestDispatcher.
#    Editar el YAML sin reiniciar NO cambia nada. ✅
systemctl --user restart kza-voice
sleep 20 && systemctl --user is-active kza-voice

journalctl --user -u kza-voice --since "5 min ago" --no-pager | grep -v oww-dbg | grep -c "HA-UNAVAILABLE"   # -> 0
```
Pedir por voz una luz caída debe volver a ser **silencio total** (el bug viejo, aceptado a cambio de destrabar los comandos contra dispositivos vivos).

**Se conserva:** cambios 1, 3, 4 y 5 (colección de pytest, alerta de sordera, guard del sync, cobertura del smoke). Solo se apaga el cambio 2.

⚠️ **Costo:** un segundo restart, o sea otra ventana de sordera. Y **deja el working tree sucio** — es la causa más probable de que el rollback total falle (paso 2 de §6.1).

Para volver a encenderlo: revertir el YAML a `true` + restart.

---

## §7 · SYNC DE CHROMA — **NO EN ESTA VENTANA** 🛑

**Cambio respecto del plan original.** El review lo trataba como "caro" (~100-120 llamadas al LLM). Es peor: **es destructivo en las dos variantes.**

`scripts/sync_ha_to_chroma.py:679` → `doc_id = f"{item['key']}_{j}"` — **el id del documento deriva de la cache key**, y la key cambió para *todas* las entidades (pasó de `f"{eid}|{fname}|{area}|{cap}|{value}"` a `...|{vitality}"`). Y `collection.add()` **no borra los viejos**. Entonces, con 5 de 8 `light.grupo_*` caídos desde hace 5 días:

- **con `--wipe`** → borra los 337 docs y reindexa solo las 3 rooms vivas + escenas (~297). **Balcón, baño, cuarto, escalera y pasillo pasan de 8 docs a CERO**: pierden el control por voz por completo. Hoy funcionan degradadas; después no funcionarían.
- **sin `--wipe`, con `--allow-unavailable`** → agrega ~276 docs de las rooms vivas + 21 de escenas **encima** de los 337 → **índice duplicado (~613)**, con la búsqueda vectorial devolviendo pares redundantes.
- **sin flags** → exit 2 (el guard hace su trabajo).

**Secuencia correcta:** (1) deployar · (2) resolver los 5 grupos Zigbee caídos · (3) recién ahí `--wipe` + resync completo, que es la única forma de deduplicar **y** devolverles las 92 frases.

Si aun así hay que inspeccionar qué haría el guard — **read-only, no escribe en Chroma**:
```bash
ssh kza 'cd /home/kza/app && set -a && . /home/kza/secrets/.env && . /home/kza/secrets/llama-api-key.env && set +a && \
  python3 -c "import os; print(\"LLAMA_API_KEY presente:\", bool(os.environ.get(\"LLAMA_API_KEY\")))" && \
  .venv/bin/python scripts/sync_ha_to_chroma.py --dry-run \
    --vllm-url http://127.0.0.1:8101/v1 \
    --vllm-model /home/kza/kza/models/Qwen2.5-7B-Instruct-Q4_K_M/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
    --embedder-device cuda:0; echo "exit=$?"'
```
**Esperado hoy: exit 2** listando los 5 `light.grupo_*` caídos. **Eso es el guard funcionando, no un error.**

Cuatro trampas ✅, todas verificadas:
1. **`set -a` es obligatorio** — los `.env` no tienen `export`; sin él la key no cruza al proceso Python, `--wipe` (línea 576) corre **antes** de construir el `VLLMClient` (~625), y borra igual. **Ese es el mecanismo exacto del incidente del 27/7.** La receta que quedó en memoria (`source llama-api-key.env`) es **insuficiente por sí sola**.
2. **Los defaults apuntan a un servicio caído** — `--vllm-url` default es `:8100/v1` y **:8100 está muerto** (no aparece en `ss -ltnp`). El vivo es `:8101`, y el model id debe ser el **path GGUF completo** (verificado contra `:8101/v1/models`).
3. **Nunca `--embedder-device cuda:1`** — 423 MiB libres. `cuda:0` tiene 5913.
4. El **guard nuevo** (`select_syncable` → `sys.exit(2)`, ~línea 550) corre **antes** del wipe (576) ✅. Protege la colección.

---

## Decisiones que quedan al usuario

1. **Merge de `feat/comandos-silenciosos` (y esta rama) a `main`.** `main == origin/main == 2450e29` ✅, y `feat/remediacion-fallos-silenciosos` está **25 commits por delante en fast-forward puro**. Los 7 de `comandos-silenciosos` llevan validados en producción desde el 30/7. El server no está parado en `main`, así que esto **no afecta el deploy** — es higiene de repo. Recomendación: mergear después de que las 24h de observación cierren limpias, en un solo fast-forward desde la laptop. (Nota: el `main` local del server está en `0d0e483`, 2 atrás de origin — irrelevante, pero explica ruido en `kza-sync`.)

2. **Intervalo del poller: 5 min (propuesto) vs 2 min.** 5 min está justificado porque la señal tiene resolución de 300s (`--deaf-after-s`) y el peor caso de detección queda en ~10 min, contra incidentes de 27h y 7h. 2 min no adelanta la detección, solo suma arranques. Cambiar el `OnUnitInactiveSec=` del timer + `daemon-reload`.

3. **`SuccessExitStatus=0 1` sí o no.** Sin él (lo propuesto), la unit queda en `failed` mientras haya sordera y `systemctl --user --failed` es un **segundo canal de alarma independiente de que HA esté vivo** — justo el punto del proyecto. El costo: exit 1 es ambiguo (sordera *o* error del ciclo *o* traceback), así que `failed` significa "andá a mirar el journal", no "hay un mic sordo". Si nadie mira `--failed` en la práctica, es ruido puro y conviene agregarlo. **Depende de cómo monitoreás.**

4. **Ventana de deploy: 07:00–08:00 hábil (recomendada) vs 04:30–05:30 no-domingo.** La segunda tiene exposición mínima (0 eventos en 30 días) pero nadie valida hasta la mañana: si el deploy dejó el sistema sordo, son 4h de sordera sin testigo — exactamente el modo de falla que esta rama viene a eliminar. Solo elegirla si el operador se queda despierto validando.

5. **Commitear los unit files a `scripts/`.** La convención del repo es versionarlos ahí (`scripts/kza-code-index.service`, `scripts/kza-llm-ik.service`, con header de instalación). La rama **no los incluye** hoy. Este runbook los crea con heredoc para no cambiar el SHA `510c2a9` que todo el reconocimiento verificó. Deuda: commitearlos en un commit posterior, o quedan invisibles para el próximo `kza-sync`.

6. **Los 5 `light.grupo_*` caídos desde el 2026-07-27 02:22:59.** El timestamp idéntico en los 5 apunta a un evento único (coincide con la ventana del re-sync Z2M de ese día). **Es un problema preexistente de la casa que este deploy solo vuelve visible.** Si son dispositivos que se sacaron a propósito, el arreglo no es esperarlos sino sacarlos también de HA — y eso desbloquea el sync de §7.

7. **`Wants=kza-llm-ik.service` en `kza-voice.service` apunta a un unit que ya no existe** (deprecado el 28/4). No bloquea, pero deja un warning en cada restart que enmascara problemas reales. Limpieza aparte.

8. **El spam de `[oww-dbg]`** (~78k líneas/día) reduce la retención del journal a ~22h. Bajar el nivel del logger `src.wakeword.detector` es el arreglo de fondo; fuera de alcance de este deploy.