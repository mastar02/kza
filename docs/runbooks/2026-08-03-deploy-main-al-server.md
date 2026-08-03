# Runbook — deploy de `main` al server (53 commits, 5 merges)

> **Estado**: escrito 2026-08-03, **sin ejecutar**. Todo lo marcado ✅ VERIFICADO
> se comprobó por comando contra el server real al escribirlo; lo demás es
> procedimiento a ejecutar.
>
> Complementa —no reemplaza— a
> [`docs/superpowers/2026-08-01-runbook-deploy-remediacion.md`](../superpowers/2026-08-01-runbook-deploy-remediacion.md),
> que **nunca se ejecutó** (el server sigue en su punto de retorno `f3c94f1`).
> Aquel cubre en detalle el poller de `audio_health` (§3) y las fotos ANTES de
> HA/Chroma (§0.5); esas partes siguen valiendo tal cual y no se repiten acá.

---

## Resumen

| | |
|---|---|
| Server ahora | rama `feat/comandos-silenciosos` @ `f3c94f1`, working tree **limpio** ✅ |
| Destino | `main` @ `e6ae878` |
| Entra | **53 commits**, 5 merges (PRs #10, #11, #13, #12) |
| Servicios a reiniciar | `kza-voice` **y `kza-code-index`** (ver §3 — no es obvio) |
| Disco libre | 320 G ✅ |

### 🔑 El hallazgo que hace este deploy seguro

El server corre una rama que **nunca se pusheó a GitHub**, con 7 commits
propios — incluidos los fixes validados por voz (mic ausente, watchdog,
early dispatch, ack de 308 ms). La pregunta obvia era si pasar a `main`
los pierde.

**No los pierde: los 7 ya están en `main` por patch-id**, reintegrados vía
PR #10/#11 con otros hashes. Verificado ✅ — devuelve vacío:

```bash
git fetch kza:/home/kza/app feat/comandos-silenciosos:refs/remotes/server/comandos-silenciosos
git log --oneline --no-merges --cherry-pick --right-only main...server/comandos-silenciosos
#   (vacío = todo lo del server está en main)
git rev-list --count --no-merges --cherry-pick --left-only main...server/comandos-silenciosos
#   -> 48   (lo que main tiene y el server no)
```

Confirmado además por contenido: `first_frame_grace_s`, `early_dispatch`,
`call_service_ws`, `VALID_SERVICE_DATA_SLOTS` y `smoke_check` están todos en
`main`. **Re-correr esos dos comandos es el paso 0 del pre-vuelo**: si el de
la derecha deja de dar vacío, alguien commiteó en el server desde el 03-08 y
este runbook queda invalidado hasta reconciliarlo.

### Qué cambia en `config/settings.yaml`

Cinco deltas. Los dos de `${LLM_GATEWAY_URL}` son el riesgo real:

| Delta | Nota |
|---|---|
| `reasoner.http_base_url` → `${LLM_GATEWAY_URL}` | precondición ✅ **ya resuelta** (var seteada en el `.env` el 2026-08-02) |
| `code_index.cards.base_url` → `${LLM_GATEWAY_URL}` | ⚠️ obliga a reiniciar `kza-code-index` — ver §3 |
| `+ unavailable_precheck_enabled: true` | PR #10 |
| `+ audio.watchdog.health_path: ./data/audio_health.json` | el archivo **no existe hoy** ✅; aparece con este deploy |
| `− reasoner.cloud.strip_home_state` | key muerta, cero lectores |

---

## §0 · PRE-VUELO

### 0.1 Ventana
No deployar con la casa usando el sistema. **Avisar antes** (el pipeline queda
mudo durante §2–§3). Reservar ≥ 40 min: el gate de audio de §4.2 **exige
minutos**, no segundos.

### 0.2 Laptop
```bash
cd ~/Documents/kza && git checkout main && git pull --ff-only
git log --oneline -1                      # -> e6ae878 (o posterior)
.venv/bin/python -m pytest tests/ -q      # -> 2762 passed, 1 xfailed
echo "EXIT=$?"                            # -> 0   (NO usar `| tail`: se traga el exit code)
```

### 0.3 Server — foto ANTES
```bash
ssh kza
cd /home/kza/app
git rev-parse --abbrev-ref HEAD && git rev-parse --short HEAD   # -> feat/comandos-silenciosos f3c94f1
git status --porcelain | wc -l                                  # -> 0  (si NO es 0: PARAR, hay WIP)
systemctl --user is-active kza-voice kza-code-index             # -> active active
df -h /home/kza | tail -1                                       # -> ≥ 5 G libres
```

**El proceso en ejecución corresponde al código en disco** ✅ (verificado
2026-08-03): PID arrancado el 2026-08-01 19:20, cero archivos de `src/`,
`config/` o `tools/` con mtime posterior, y el último `.git/HEAD` es del 07-29
—anterior al arranque—. Re-chequearlo si pasan días antes de deployar:

```bash
PID=$(pgrep -u kza -f '.venv/bin/python -m src.main' | head -1)
cd /home/kza/app
find src/ config/ tools/ -type f -newer /proc/$PID          # -> vacío
find src/ -name '*.py' -newermt '2020-01-01' | wc -l        # control: >0, el find funciona
```

⚠️ Comparar contra `/proc/$PID`, **no** contra `ps -o lstart=` pasado por
`date -d`: en locale español devuelve `sáb ago  1` y `date` no lo parsea, así
que el `find` falla y —si además se traga stderr— "0 archivos" se ve idéntico a
"todo en orden". Pasó al escribir este chequeo. El control de la segunda línea
existe justamente para que un `find` roto no se disfrace de resultado limpio.

Fotos ANTES de HA / Chroma / smoke / latencia: usar §0.5 del runbook del
2026-08-01 tal cual.

---

## §1 · BACKUP 🛑

```bash
cd /home/kza/app
git tag pre-deploy-main-20260803          # punto de retorno explícito
git rev-parse pre-deploy-main-20260803    # ANOTARLO fuera de la terminal

systemctl --user stop kza-voice.service

# OJO con las rutas (ver recuadro): /home/kza/app es un SYMLINK a /home/kza/kza,
# y /home/kza/data NO es /home/kza/app/data.
tar czf ~/kza-data-20260803.tgz -C /home/kza/kza data      # ~206 M ✅
tar tzf ~/kza-data-20260803.tgz > /dev/null && echo "backup LEGIBLE"
```

El `tar tzf` no es ceremonia: un backup que no se puede leer no es un backup.
El tag es barato y es lo que convierte el rollback de §5 en un comando.

> ⚠️ **Por qué el tar apunta ahí y no a `app/`** — tres trampas verificadas ✅,
> las tres detectadas revisando este runbook, no ejecutándolo:
> 1. **`/home/kza/app` es un symlink** a `/home/kza/kza`. `tar` **no** sigue
>    symlinks pasados como argumento: `tar ... -C /home/kza app` archiva el
>    symlink, no el código. Un backup vacío que parece exitoso.
> 2. **`/home/kza/data` existe y NO es `/home/kza/app/data`.** Son directorios
>    distintos; el estado del pipeline (`chroma_db`, `events.db`, `users.json`,
>    `contexts/`, `audio_health.json`) vive en el segundo.
> 3. **`/home/kza/kza` pesa 99 G** (15 G solo `models/`). Un tar del árbol
>    entero es absurdo y no aporta: el código lo cubre git (tag + rama), y los
>    modelos no los toca este deploy.
>
> Lo único que git **no** puede restaurar es `data/` (**206 M** ✅), y por eso
> es lo único que se respalda. `secrets/` no lo toca este deploy, y copiarlo
> solo multiplicaría tokens en disco.

---

## §2 · DEPLOY DEL CÓDIGO 🛑

```bash
cd /home/kza/app
git fetch origin                                   # solo refs; no toca el working tree

# El working tree ES el deploy: este checkout es el deploy.
# Verificado ✅ hoy (main local del server = 0d0e483, ancestro de origin/main).
# Si esto imprime "PARAR", el main del server divergió y --ff-only va a fallar:
# no forzar, reconciliar primero.
git merge-base --is-ancestor main origin/main && echo "ff limpio" || echo "PARAR: main divergió"
git checkout main
git merge --ff-only origin/main
git rev-parse --short HEAD                         # -> e6ae878 (o posterior)
git status --porcelain | wc -l                     # -> 0
```

El hook `post-merge` ✅ existe y dispara el reindex del code-index.

**No hace falta tocar el venv** ✅: `requirements.txt` no cambia entre el server
y `main` (lo único nuevo es `requirements-test.txt`, que solo usa el CI). Si
algún deploy futuro sí cambia dependencias de runtime, ese `pip install` va
acá, entre el checkout y el start — nunca con el servicio arriba.

---

## §3 · REINICIO — los DOS servicios

```bash
# La captura PRIMERO: el journal rota en ~22 h y el arranque es lo que más
# importa. Arrancar el tee después del start se pierde justo esos segundos.
journalctl --user -u kza-voice -f > ~/arranque-20260803.log &
TEE_PID=$!

systemctl --user start kza-voice.service
systemctl --user restart kza-code-index.service

systemctl --user is-active kza-voice kza-code-index    # -> active active
# ...dejar corriendo durante §4.2; al terminar: kill $TEE_PID
```

⚠️ **`kza-code-index` no es opcional y es fácil de olvidar.** Razón verificada ✅:
carga `EnvironmentFile=/home/kza/secrets/.env`, el mismo archivo — pero su
proceso arrancó **antes** de que se agregara `LLM_GATEWAY_URL`, así que su
entorno actual no la tiene. Tras el deploy su `settings.yaml` pasa a
`${LLM_GATEWAY_URL}`; sin reiniciar, resuelve al default local y degrada en
silencio. Exactamente la clase de fallo que este repo viene persiguiendo.

---

## §4 · GATES DE VERIFICACIÓN ⏸

### 4.1 El gate de privacidad (nuevo, PR #12)
Con la config viva (`consent: true`) el compactor debe heredar el gateway;
con `consent: false` debe degradar a `:8101` **sin** la key cloud:

> 🔑 **`set -a; . archivo; set +a`, nunca `source` a secas.** Los
> `/home/kza/secrets/*.env` son `EnvironmentFile=` de systemd (`VAR=valor`, sin
> `export`): un `source` define variable de *shell* y **Python no la hereda**.
> Sin esto, `load_config` aborta en `check_unresolved_env_vars` por
> `home_assistant.*` y el chequeo falla por el motivo equivocado. Es la causa
> raíz mecánica del incidente del 27/7 y está documentada en el §preámbulo del
> runbook del 2026-08-01 — **la omití al escribir este** y la reintroduje en
> los dos comandos de abajo.

```bash
cd /home/kza/app && set -a && . /home/kza/secrets/.env && set +a && .venv/bin/python - <<'EOF'
from src.core.settings_schema import DEFAULT_LOCAL_LLM_GATEWAY
from src.llm.cloud_consent import resolve_reasoner_gate, resolve_compaction_endpoint, is_cloud_endpoint
from src.main import load_config
for consent in (True, False):
    c = load_config("config/settings.yaml"); r = c["reasoner"]
    r.setdefault("cloud", {})["consent"] = consent
    g, _ = resolve_reasoner_gate(r, r.get("mode", "http"), DEFAULT_LOCAL_LLM_GATEWAY)
    e = resolve_compaction_endpoint(c["orchestrator"]["context"]["compaction"], r, g)
    print(consent, g, e.base_url, e.api_key_env, is_cloud_endpoint(e.base_url))
EOF
```
Esperado — **y esto además prueba que `LLM_GATEWAY_URL` se resolvió**: si la
primera línea dice `127.0.0.1:8200` en vez de `192.168.1.2:8200`, la var no
llegó al proceso.
```
True  True  http://192.168.1.2:8200/v1 MINIMAX_API_KEY True
False False http://127.0.0.1:8101/v1   None            False
```

### 4.2 Audio — ⏸ **esperar MINUTOS**
```bash
ls -l /home/kza/app/data/audio_health.json     # debe APARECER (hoy no existe ✅)
```
El criterio es **muestras no-cero**, no un umbral de RMS (el piso de ruido real
es 0.0104). El primer frame puede tardar hasta **135 s** medidos; por eso
`first_frame_grace_s=180`. **Medir a los 20–40 s lleva a concluir mal** — ya
pasó, está documentado. Esperar ≥ 3 min antes de declarar nada.

⚠️ **NUNCA hacer polling USB externo al XVF3800 con `kza-voice` vivo**: congela
la captura.

### 4.3 Smoke test
```bash
cd /home/kza/app && set -a && . /home/kza/secrets/.env && set +a && \
  .venv/bin/python tools/smoke_test.py; echo "exit=$?"
```
Dry-run frase → vector search → entidad viva → payload, **sin ejecutar**. No
cubre el path del LLM ni nada de audio: verde acá no dice nada sobre §4.2.

### 4.4 Por voz
Un comando real por habitación. `[HA-CALL]` en el journal es la evidencia dura
de que la acción salió — **no** `source=manual`, que mide otra cosa.

---

## §5 · ROLLBACK

Disparadores: audio sin frames no-cero tras §4.2, `[HA-CALL]` ausente en §4.4,
o el gate de §4.1 devolviendo cloud con `consent=false`.

```bash
systemctl --user stop kza-voice.service
cd /home/kza/app
git checkout feat/comandos-silenciosos     # vuelve a f3c94f1, el estado de hoy
git status --porcelain | wc -l             # -> 0
systemctl --user start kza-voice.service
systemctl --user restart kza-code-index.service
```

La rama de vuelta sigue existiendo en el server (no se borra en §2) y el tag
`pre-deploy-main-20260803` apunta al mismo commit. `data/` está gitignoreado ✅
(`.gitignore:26`), así que el checkout no lo toca y el `wc -l` da 0 aunque el
código nuevo ya haya escrito `audio_health.json`.

El `.env` **no se toca** en el rollback: `LLM_GATEWAY_URL` es inofensiva para el
código viejo, que tiene la URL hardcodeada y ni la lee.

### Si volver el código no alcanza

Restaurar `data/` es **escalación, no el paso por defecto**: el estado suele ser
compatible hacia adelante, y volverlo tira lo que la casa generó desde el
backup (contextos, eventos, preferencias). Hacerlo solo si el rollback de código
deja el sistema roto — p. ej. Chroma o `events.db` ilegibles para el código
viejo:

```bash
systemctl --user stop kza-voice.service
mv /home/kza/kza/data /home/kza/kza/data.roto-20260803    # NO borrar: es la evidencia
tar xzf ~/kza-data-20260803.tgz -C /home/kza/kza
systemctl --user start kza-voice.service
```

---

## Apéndice · qué se verificó al escribir esto (2026-08-03)

| Afirmación | Cómo |
|---|---|
| Los 7 commits del server ya están en `main` | `git log --cherry-pick --right-only` → vacío + 5 firmas por contenido |
| Working tree del server limpio | `git status --porcelain \| wc -l` → 0 |
| `LLM_GATEWAY_URL` seteada en el `.env` | `grep` → `http://192.168.1.2:8200/v1` |
| `kza-code-index` usa el mismo `.env` | `systemctl --user cat` → `EnvironmentFile=/home/kza/secrets/.env` |
| `audio_health.json` no existe todavía | `ls` → no existe |
| Hook `post-merge` presente | `ls -l .git/hooks/post-merge` |
| 320 G libres | `df -h /home/kza` |
| La rama del server no está en GitHub | `git ls-remote --heads origin` |
| Suite en `main` verde | `pytest tests/ -q` → 2762 passed, EXIT=0 |
| CI en `main` verde | los 6 jobs `success` |

### Defectos que encontró el review de este runbook (antes de ejecutarlo)

Cuatro, ninguno detectable leyendo el texto sin ir al server:

1. **El backup era inservible**: `/home/kza/app` es symlink (tar no lo sigue),
   `/home/kza/data` ≠ `/home/kza/app/data`, y el árbol pesa 99 G.
2. **Los comandos de verificación no tenían entorno** — `set -a; . .env; set +a`.
   La lección estaba escrita en el runbook del 2026-08-01 y la omití: el error
   que ese runbook documenta como causa raíz del incidente del 27/7.
3. **La captura del journal empezaba después del `start`**, perdiendo el arranque.
4. **El chequeo de ancestro fallaba en silencio** (`&&` sin `||`).

| El proceso en ejecución corresponde al código en disco | 0 archivos de `src/`/`config/`/`tools/` con mtime posterior al arranque del PID, con control de que el `find` funciona (§0.3) |

**No verificado / a mirar durante el deploy**
- El poller de `audio_health` del runbook del 01-08 **no está instalado** ✅
  (`systemctl --user list-timers` no lo lista). Este deploy solo hace que el
  snapshot se escriba; que alguien lo mire sigue pendiente.
- Comportamiento del code-index tras el cambio de `base_url` a placeholder,
  más allá de que arranque.
