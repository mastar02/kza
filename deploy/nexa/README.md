# Despliegue Nativo de Nexa v2 (Usuario Dedicado `nexa`, Sin `sudo`, Sin Pods)

Este directorio define el despliegue nativo de KZA v2 bajo el usuario Linux dedicado **`nexa`** (sin privilegios `sudo`, sin Docker/Podman/CDI), gestionado íntegramente por `systemd --user` + permisos POSIX + Unix Domain Sockets (`$XDG_RUNTIME_DIR/nexa/*.sock`).

## 1. Aprovisionamiento único en el servidor (requiere `sudo` una sola vez)

```bash
sudo bash deploy/nexa/provision-nexa-user.sh
```

Verifica automáticamente que `nexa` **no** pertenezca a `sudo`, `adm`, `wheel` ni `docker`, asigna solo `audio,dialout,video,render`, habilita `loginctl enable-linger nexa`, instala la regla `udev` `99-xvf3800-nexa.rules` (`GROUP="nexa", MODE="0660"`), crea `/opt/nexa`, `/etc/nexa`, `/var/lib/nexa` y configura `TimeoutStopSec=30s` únicamente para el UID de `nexa`.

## 2. Despliegue atómico y rollback (< 60 s)

```bash
sudo bash deploy/nexa/install-release.sh install /tmp/nexa-release-v2.0.0 v2.0.0
# Rollback inmediato a la release anterior:
sudo bash deploy/nexa/install-release.sh rollback v1.9.9
```

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
