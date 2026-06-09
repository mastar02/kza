# Ecosistema de trabajo Local ↔ Server ↔ GitHub

> Diseñado el 2026-06-01 tras un análisis profundo del drift (ver al final).
> Opción adoptada: **A (puente formalizado) + D (higiene + worktrees)**, con **B** como evolución a mediano plazo.

## 1. Topología real (verificado, no asumido)

```
   LAPTOP  ──(git push/pull, SSH)──►  GitHub (origin: mastar02/kza)
     ▲                                      ▲
     │ git fetch kza:/home/kza/app          │  (la laptop hace de PUENTE)
     │                                      │
   SERVER (kza@192.168.1.2:/home/kza/app)   │
     └── ES repo git, deploya IN-PLACE,  ────┘  ✗ NO puede pushear directo
         pero su SSH key NO está autorizada en GitHub
```

- **El server SÍ es git.** `/home/kza/app` → symlink a `/home/kza/kza`, branch de trabajo.
- **El server deploya in-place**: el código que corre es el working tree. `git commit` **no cambia los archivos en disco**, así que commitear es seguro respecto al servicio `kza-voice` (no hace falta reiniciar).
- **El server NO pushea a GitHub** (SSH key no autorizada). La **laptop es el puente**: `git fetch` del server por SSH + `git push origin`.
- **El server tampoco puede pullear si tiene WIP sin commitear** sobre los archivos que cambiarían → de ahí el drift.

## 2. La causa raíz del drift (no es git, es el WIP)

El análisis del 2026-06-01 mostró que la historia commiteada **nunca diverge** (local suele ser ancestro fast-forwardeable de `origin/main` = server). **Todo el drift vive en los árboles de trabajo sin commitear** + el ruido de `.bak` que esconde el trabajo real. Los dos males:

1. **WIP sin commitear** que solo existe en un working tree (se pierde con `stash -u` / `clean -fdx` / `reset --hard` — ver `feedback_git_stash_u_models`).
2. **Ruido `.bak`/benchmarks/log** que infla `git status` y disfraza el trabajo real.

## 3. Invariante del ecosistema

> **`origin/main` == el `main` del server == baseline estable.**
> El trabajo en curso vive en ramas `feat/*` (o `wip/*`), nunca suelto en el working tree por más de una sesión.

## 4. Flujo diario (Opción A — puente formalizado)

**Al cerrar cada sesión de dev en el server:**

```bash
# En el server (vía SSH): commitear el WIP en una rama feat/*, NUNCA dejarlo suelto.
ssh kza
cd /home/kza/app
git switch -c feat/<lo-que-sea>          # o git switch a una rama feat existente
git add <archivos reales>                 # el .gitignore ya filtra .bak/bench/log
git commit -m "feat(...): ..."
exit
```

**Desde la laptop, publicar a GitHub (el server no puede):**

```bash
scripts/kza-push feat/<lo-que-sea>        # fetch del server + push a origin
```

**Detectar drift / WIP olvidado en cualquier momento (read-only):**

```bash
scripts/kza-sync                          # estado local ↔ server ↔ origin + WIP real vs ruido
```

## 5. Higiene + aislamiento (Opción D)

- **`.gitignore`** ya ignora `*.bak`, `*.bak.*`, `*.orig`, `/llama.log`, `/benchmarks/router/results_*.json`. Así `git status` muestra **solo trabajo real**.
- **Worktrees por feature** cuando dos líneas tocan el mismo archivo (p.ej. wake y grupos/escenas ambos editaban `config/settings.yaml` y colisionaban):
  ```bash
  cd /home/kza/app
  git worktree add ../wt-wake feat/wake-openwakeword-nexa
  git worktree add ../wt-grupos feat/escritorio-grupos-escenas
  ```
  Cada feature en su carpeta, sin mezclar WIP en el mismo `settings.yaml`.
- **Alerta de WIP > 24h** (propuesto, aún no instalado): un timer systemd `--user` que corra `git status --porcelain | grep -vE '<ruido>'` y avise (mail/Notion) si hay trabajo sin commitear. Convierte la disciplina en automatismo.

## 6. Evolución a mediano plazo (Opción B)

Cuando se acepte meter una credencial con push en el server de producción y haya ventana de mantenimiento: crear una **deploy key / PAT** scoped al repo en `/home/kza/secrets/`, configurar el `origin` del server con ella → el server pushea directo y la laptop solo hace `git pull`. Elimina el puente manual y el SPOF de la laptop. (Opción C — server como remoto git canónico con bare repo + mirror — se descartó por sobre-ingeniería para el volumen actual.)

## 7. Receta de reconciliación "no perder nada" (ejecutada el 2026-06-01)

Cuando local y server acumulan WIP divergente, el orden seguro es:

1. **Backup tar** de todo lo uncommitted en ambos lados (red de seguridad de riesgo cero).
2. **Local**: commitear lo at-risk en una rama; `git restore` lo superado; borrar untracked idénticos al remoto (bloquean el pull); `git merge --ff-only` para ponerse al día.
3. **Server**: commitear el WIP real en una rama `wip/snapshot-<fecha>` (excluyendo ruido); `kza-push` esa rama a GitHub.
4. **Split fino en la laptop** (sin riesgo de prod): separar el snapshot en ramas `feat/*` limpias, reconciliar con lo que ya esté en GitHub, y recién ahí mergear a `main` + redeployar.

Estado tras la sesión 2026-06-01:
- `origin/main` = server `main` = `224912d` (baseline).
- `docs/wake-sesion-2026-05-31` — 5 docs + fix MA1260 (era WIP local).
- `wip/server-snapshot-2026-06-01` — WIP de wake + grupos/escenas + tools (era WIP server, deployado).
- **Pendiente (próxima sesión, deliberado)**: split del snapshot en `feat/wake-openwakeword-nexa` + reconciliar grupos con `feat/escritorio-grupos-escenas` (10 commits ya en GitHub, no mergeados), decidir el scaffold porcupine (descartado), y mergear a `main` + redeploy.
