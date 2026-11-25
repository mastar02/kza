# Despliegue Nativo de Nexa v2 (Usuario Dedicado `nexa`, Sin `sudo`, Sin Pods)

Este directorio define el despliegue nativo de KZA v2 bajo el usuario Linux dedicado **`nexa`** (sin privilegios `sudo`, sin Docker/Podman/CDI), gestionado íntegramente por `systemd --user` + permisos POSIX + Unix Domain Sockets (`$XDG_RUNTIME_DIR/nexa/*.sock`).

## 1. Aprovisionamiento único en el servidor (requiere `sudo` una sola vez)

```bash
sudo bash deploy/nexa/provision-nexa-user.sh
```

Verifica automáticamente que `nexa` **no** pertenezca a `sudo`, `adm`, `wheel` ni `docker`, asigna solo `audio,dialout,video,render`, habilita `loginctl enable-linger nexa`, instala la regla `udev` `99-xvf3800-nexa.rules` (`GROUP="nexa", MODE="0660"`), crea `/opt/nexa`, `/etc/nexa`, `/var/lib/nexa` y configura `TimeoutStopSec=30s` únicamente para el UID de `nexa`.

Antes de iniciar los servicios, instalar `llama-server` en `/opt/nexa/bin/` y la
biblioteca de ONNX Runtime en `/opt/nexa/lib/libonnxruntime.so`. La unidad
`nexa-audiod` carga esa biblioteca mediante `ORT_DYLIB_PATH`. Guardar la clave
de la API en `/etc/nexa/secrets/llama-api-key` (propietario `nexa:nexa`, modo
`0600`). La unidad `nexa-llm-fast` la lee mediante `--api-key-file`.

Antes del primer arranque de ML, sembrar la caché persistente
`/var/lib/nexa/.cache` (directorio `0700`, propietario `nexa:nexa`).
Copiar el modelo ECAPA completo, incluidos `hyperparams.yaml` y
`embedding_model.ckpt`, a
`/var/lib/nexa/.cache/nexa-ml/speechbrain/speechbrain--spkrec-ecapa-voxceleb/`.
Copiar la caché de Hugging Face de los modelos y voces configurados a
`/var/lib/nexa/.cache/huggingface/`, conservando `hub/`, los snapshots,
los blobs y sus enlaces. Verificar que `nexa` pueda leer los archivos y
actualizar la caché. El worker arranca sin red y reutiliza estos datos
entre reinicios.

## 2. Despliegue atómico y rollback (< 60 s)

En el checkout del commit a distribuir, preparar los cuatro binarios nativos en
`target/release/` y la SPA en `frontend/dist/`. Empaquetar en una carpeta nueva
o vacía:

```bash
python3 deploy/nexa/package-release.py v2.0.0 /tmp/nexa-release-v2.0.0
```

El manifiesto identifica el commit del checkout y la plataforma y arquitectura
del host de empaquetado. El árbol debe estar limpio. Para pruebas locales,
`--allow-dirty` permite cambios y el manifiesto registra `git_dirty`. El instalador de producción rechaza
las releases que declaran `git_dirty: true`, incluso en rollback.
Fijar `SOURCE_DATE_EPOCH` (segundos Unix) para reproducir la fecha UTC. Usar binarios compilados para el host de destino.
`--allow-missing-frontend` permite un paquete parcial para pruebas.
El instalador usa Python 3 con `stage-release.py` y `release_policy.py` en el mismo
directorio. Verifica inventario, rutas prohibidas, hashes, binarios y units antes
de activar una versión, tanto en instalación como en rollback. En producción
exige una release con manifiesto y al menos un archivo en `frontend/`.

```bash
sudo bash deploy/nexa/install-release.sh install /tmp/nexa-release-v2.0.0 v2.0.0
# Rollback inmediato a la release anterior:
sudo bash deploy/nexa/install-release.sh rollback v1.9.9
```

Para ejecutar L5, usar `tests-v2/smoke/run_l5_smoke.sh` con pytest disponible
en `python3` (o seleccionar el intérprete con `PYTHON`). El smoke compila
`nexa-memoryd` con Cargo y protoc o usa el binario indicado por `NEXA_MEMORYD_BIN`.
En macOS verifica el socket privado y el apagado por SIGTERM. En Linux también
exige los datagramas `READY=1` y `STOPPING=1` del proceso.

Los cuatro daemons Rust y el worker ML usan `Type=notify`. ML carga el código
desde la release con el intérprete del entorno compartido en `/opt/nexa/ml-venv`.
NGINX y llama-server son ejecutables externos y conservan `Type=simple`.

En `/etc/nexa/nginx-proxy.conf`, configurar `pid` con la ruta absoluta a
`$XDG_RUNTIME_DIR/nexa-proxy/nginx.pid` del usuario nexa (sustituir
`$XDG_RUNTIME_DIR` por `/run/user/<UID>`). Dirigir `error_log` a `stderr` y
`access_log` a `off` o a un archivo en `/var/lib/nexa/proxy/`. Dentro de
`http`, asignar subdirectorios distintos de `/var/lib/nexa/proxy/` a
`client_body_temp_path`, `proxy_temp_path`, `fastcgi_temp_path`,
`uwsgi_temp_path` y `scgi_temp_path` para los módulos habilitados.

## 3. Registro propuesto para `docs/SERVER_CONVENTIONS.md` (y Notion pág. 8)

| Campo | Valor |
|---|---|
| Usuario Linux | `nexa` (UID asignado `>= 1015`, sin `sudo`, linger habilitado) |
| Modelo | Nativo sin pods (Excepción R10 #4: USB XVF3800 + MA1260 + presupuesto `<300 ms` + sin overhead CDI) |
| Unidades | `nexa-audiod`, `nexa-ml`, `nexa-memoryd`, `nexa-brain`, `nexa-web`, `nexa-llm-fast`, `nexa-dashboard-proxy` (`nexa.target`) |
| IPC interno | Unix Domain Sockets `$XDG_RUNTIME_DIR/nexa/{audio,ml,memory,brain}.sock` (`0600`) — **0 puertos TCP internos nuevos** |
| Puertos TCP | Sub-rango reservado KZA `9500-9599`: `127.0.0.1:9500` (`nexa-web`), `192.168.1.2:9540` (NGINX TLS), `127.0.0.1:8101` (`nexa-llm-fast`) |
| GPUs | 2 × RTX 3070 de 8 GiB cada una |
| `cuda:1` | `llama-server 7B :8101`, `whisper-v3-turbo` de comandos y `Kokoro-82M` en streaming (~6,7 GiB) |
| `cuda:0` | 2 × `whisper-large-v3` ambientales y 1 × ECAPA-TDNN (~4,1 GiB) |
| CPU | Embeddings `BAAI/bge-m3` y modelos ONNX Parakeet, Silero y OpenWakeWord |
| Logs | `journalctl --user -u 'nexa-*' -f` (cumple R9) |
