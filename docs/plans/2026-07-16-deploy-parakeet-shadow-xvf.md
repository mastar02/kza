# Runbook — Ventana de deploy 2026-07-16: main + shadow Parakeet + palancas XVF

**Objetivo:** un solo restart de `kza-voice` que lleva el server a `main` (entra
en vivo el **textual wake**, se consolida el **watchdog** ya corriendo hace 11
días, y queda disponible el código Parakeet), activa el **shadow A/B**
Whisper vs Parakeet y las **2 palancas anti-TV del XVF3800**.

Contexto y evidencia: plan Parakeet fast path (sesiones 2026-07-14/16).
Snapshot del chip pre-cambio: `/tmp/xvf_fase3/snapshot_20260716_0943.txt`.

## Qué cambia con este deploy

| Cambio | Mecanismo | Kill-switch / rollback |
|---|---|---|
| Textual wake "nexa" EN VIVO (1ª vez) | código main 61ffb9d + `ambient.textual_wake.enabled: true` (ya estaba) | `ambient.textual_wake.enabled: false` + restart |
| Watchdog consolidado en main | merge PR #8 (12a003c) — ya corría desde 07-05 | no aplica (sin cambio de comportamiento) |
| Shadow A/B STT | `stt.shadow_engine: "parakeet"` | `shadow_engine: null` + restart |
| XVF: `PP_ATTNS_MODE 0→1`, `PP_MIN_NN 0.51→0.4` (+ ancla `ATTNS_SLOPE 1.0`) | `rooms.xvf_tuning.params` (aplica al boot, RAM) | re-comentar + restart (o re-enchufar mic restaura preset) |
| STT primario | **SIN CAMBIO** (`stt.engine: whisper`) | — |

## Pre-checks (laptop, antes de pedir la ventana)

- [ ] `scripts/kza-sync` → sin WIP real en el server, divergencia entendida.
- [ ] `main` local == `origin/main` == este runbook + config de activación.
- [ ] Suite local corrida; fallas = solo baseline conocido.
- [ ] `ssh kza nvidia-smi` → cuda:1 con margen (~400MB libres con el servicio
      corriendo es lo normal; el restart libera y re-toma lo suyo).
- [ ] Hogar avisado: ~1 min sin voz.

## Ventana (server) — downtime ≈ checkout + boot (~60s)

```bash
ssh kza
cd ~/app
git fetch origin
git status --short           # debe estar limpio
systemctl --user stop kza-voice.service
git checkout main && git pull --ff-only origin main
systemctl --user start kza-voice.service
```

## Verificación post-boot (journal)

```bash
journalctl --user -u kza-voice.service --since "5 minutes ago" --no-pager \
  | grep -E "XVF-tuning|audio-watchdog|STT shadow|Shadow STT|TextualWake|Parakeet|Whisper cargado|Traceback|ERROR"
```

Esperado:
- [ ] `[XVF-tuning] PP_AGCMAXGAIN … → [16.0]`, `PP_ATTNS_MODE (0,) → [1]`,
      `PP_ATTNS_SLOPE … → [1.0]`, `PP_MIN_NN (0.51…) → [0.4]` (before→after).
- [ ] `[audio-watchdog] ACTIVO (timeout=8.0s…)`.
- [ ] `STT shadow habilitado: engine=parakeet` (main.py) y
      `Shadow STT loaded` / `Parakeet cargado` (CommandProcessor).
- [ ] Sin `Traceback`/`ERROR` (el timeout de subscribe HA al boot con
      reconexión es conocido y benigno).
- [ ] `oww-dbg` emitiendo (pipeline vivo).

## Validación por voz (usuario presente)

1. **Comando normal near-field** — "nexa, prendé/apagá la <luz>": ejecuta, y el
   journal muestra `[STT-shadow] primary=… shadow=…` (el A/B captura el
   comando). Anotar si el texto de Parakeet es español correcto.
2. **Textual wake fix 2 (no doble ejecución)** — comando a volumen normal: el
   wake acústico dispara y NO hay segunda ejecución del textual (dedup ts).
3. **Textual wake camino propio** — comando en voz BAJA (que el acústico no
   dispare): `[TextualWake] DISPARO` en journal + ejecución en ~1.5–3s.
4. **Textual wake fix 1 (audio mono / speaker-ID)** — en el journal del disparo
   textual: "Using wake-detector text as pretranscribed" y ningún error de
   speaker-ID/ECAPA.
5. **Anti-TV** — con la TV sonando unos minutos: no hay acciones fantasma;
   comparar tasa de `[CommandGate]` accept=True vs los ~90/día previos.
6. **Cascada acústica intacta** — repetir (1) desde la distancia habitual: la
   captura no se corta a mitad de frase (si se corta: sospechar PP_MIN_NN/ATTNS
   moviendo el piso del endpointing → rollback XVF).

## Rollback

- **No arranca / Traceback en boot:** `systemctl --user stop kza-voice &&
  git checkout 2363979 && systemctl --user start kza-voice` (estado previo
  exacto del server) y debuggear en frío.
- **Comandos dejan de transcribir / captura cortada:** re-comentar las 3
  líneas XVF en `rooms.xvf_tuning.params` (commit en el server) + restart.
- **Textual wake dispara fantasmas:** `ambient.textual_wake.enabled: false`
  + restart. (⚠ conocida: `tv_azimuth` null → defensa TV del SourceClassifier
  inerte; auditar `[TextualWake]` los primeros días.)
- **Shadow molesta (CPU/logs):** `stt.shadow_engine: null` + restart.

## Post-ventana

- Borrar la rama `feat/audio-watchdog-usb-recovery` (server y origin) y
  `feat/parakeet-fastpath` local cuando `git log <rama> ^main` quede vacío.
- Días siguientes: cosechar `[STT-shadow]` (`journalctl … | grep STT-shadow`)
  → comparar texto/latencia sobre comandos reales → decidir el flip
  `stt.engine: parakeet` (sesión aparte).
- Auditar `[TextualWake]` (tasa de disparos, falsos por TV).
- Ventana futura corta: medir azimut usuario/TV (`xvf_tune --read
  AEC_AZIMUTH_VALUES` con servicio parado, bundle `/tmp/xvf_fase3`) →
  activar fixed beams + `tv_azimuth`.
