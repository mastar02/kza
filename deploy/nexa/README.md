# Despliegue Nativo de Nexa v2 (Usuario Dedicado `nexa`, Sin `sudo`, Sin Pods)

Este directorio define el despliegue nativo de KZA v2 bajo el usuario Linux dedicado **`nexa`** (sin privilegios `sudo`, sin Docker/Podman/CDI), gestionado íntegramente por `systemd --user` + permisos POSIX + Unix Domain Sockets (`$XDG_RUNTIME_DIR/nexa/*.sock`).

## 1. Aprovisionamiento único en el servidor (requiere `sudo` una sola vez)

```bash
sudo bash deploy/nexa/provision-nexa-user.sh
```

Verifica automáticamente que `nexa` **no** pertenezca a `sudo`, `adm`, `wheel` ni `docker`, asigna solo `audio,dialout,video,render`, habilita `loginctl enable-linger nexa`, instala la regla `udev` `99-xvf3800-nexa.rules` (`GROUP="nexa", MODE="0660"`), crea `/opt/nexa`, `/etc/nexa`, `/var/lib/nexa` y configura `TimeoutStopSec=30s` únicamente para el UID de `nexa`.

## 2. Despliegue atómico y rollback (< 60 s)

En el checkout del commit a distribuir, preparar los cuatro binarios nativos en
`target/release/` y la SPA en `frontend/dist/`. Empaquetar en una carpeta nueva
o vacía:

```bash
python3 deploy/nexa/package-release.py v2.0.0 /tmp/nexa-release-v2.0.0
```

El manifiesto identifica el commit del checkout y la plataforma y arquitectura
del host de empaquetado. Usar binarios compilados para el host de destino.
`--allow-missing-frontend` permite un paquete parcial para pruebas.
El instalador usa Python 3 y su auxiliar `stage-release.py` en el mismo directorio.
Verifica el inventario y los hashes en staging antes de promover la versión.

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

## 3. Registro propuesto para `docs/SERVER_CONVENTIONS.md` (y Notion pág. 8)

| Campo | Valor |
|---|---|
| Usuario Linux | `nexa` (UID asignado `>= 1015`, sin `sudo`, linger habilitado) |
| Modelo | Nativo sin pods (Excepción R10 #4: USB XVF3800 + MA1260 + presupuesto `<300 ms` + sin overhead CDI) |
| Unidades | `nexa-audiod`, `nexa-ml`, `nexa-memoryd`, `nexa-brain`, `nexa-web`, `nexa-llm-fast`, `nexa-dashboard-proxy` (`nexa.target`) |
| IPC interno | Unix Domain Sockets `$XDG_RUNTIME_DIR/nexa/{audio,ml,memory,brain}.sock` (`0600`) — **0 puertos TCP internos nuevos** |
| Puertos TCP | Sub-rango reservado KZA `9500-9599`: `127.0.0.1:9500` (`nexa-web`), `192.168.1.2:9540` (NGINX TLS), `127.0.0.1:8101` (`nexa-llm-fast`) |
| GPUs | `cuda:1` (`whisper-v3-turbo` + `Kokoro` + `llama-server 7B :8101`); `cuda:0` (`2x whisper large-v3` + `1x ECAPA`) |
| Logs | `journalctl --user -u 'nexa-*' -f` (cumple R9) |
