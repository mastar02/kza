# Despliegue nativo de Nexa v2

La unidad de entrega es una release versionada para Linux con cuatro binarios Rust, el código del worker Python 3.13, la SPA compilada y las units. Desarrollar y probar en Mac o como `yo` en Linux. Solo una release con GO explícito habilita promoción, instalación o reinicios en producción. El monolito bajo `kza` y su rollback se conservan durante la transición.

## Cuenta, configuración y estado

El usuario dedicado `nexa` ejecuta `systemd --user` con linger (`loginctl enable-linger nexa`), grupos suplementarios `audio,dialout,video,render` y sin membresía en `sudo`, `adm`, `wheel` o `docker`. La regla `99-xvf3800-nexa.rules` concede acceso al XVF3800 con modo `0660`. Las units aplican `ProtectSystem=strict` y declaran las rutas de escritura necesarias. El aprovisionamiento queda fuera del flujo de desarrollo. [Definición de la cuenta](../../deploy/nexa/provision-nexa-user.sh) y [regla udev](../../deploy/nexa/udev/99-xvf3800-nexa.rules).

En la ventana de aprovisionamiento autorizada, el administrador ejecuta una vez `deploy/nexa/provision-nexa-user.sh`. El script asigna a `nexa:nexa` los directorios `/opt/nexa` y `/opt/nexa/releases` con modo `0750`. Las siete unidades conservan `ProtectSystem=strict`. En sesiones sin variables de systemd, el instalador deriva `XDG_RUNTIME_DIR` del UID de la cuenta (por ejemplo, `/run/user/501`) y usa el bus de ese directorio. El operador puede promover y revertir releases desde una sesión de `nexa` sin sudo.

La configuración tipada vive en `/etc/nexa/{nexa,rooms,ha-controls}.toml`. Partir de los [modelos de configuración](../../config/nexa/) y materializar las habitaciones y controles propios fuera de Git. Los archivos secretos tienen modo `0600` en `/etc/nexa/secrets/`. El runtime usa `/opt/nexa` para programas y modelos, `/var/lib/nexa` para estado privado y `$XDG_RUNTIME_DIR/nexa/{audio,ml,memory,brain}.sock` para IPC con modo `0600`. Las rutas de Nexa son independientes de la cuenta legacy.

| Servicio | Contrato de ejecución |
| --- | --- |
| nexa-audiod | Compilar en Linux con `--features hardware`. ALSA por habitación, control USB XVF3800 vía rusb y MA1260 en simulación. Silero VAD, OpenWakeWord y veto en sombra Parakeet usan ONNX CPU con `ORT_DYLIB_PATH=/opt/nexa/lib/libonnxruntime.so`. El sello persistente de captura vive en `/var/lib/nexa/audiod`. |
| nexa-ml | Worker GPU stateless en `/opt/nexa/ml-venv`, con `PYTHONPATH=/opt/nexa/current/ml/src` y `RestrictAddressFamilies=AF_UNIX`. Precargar modelos y voces en la caché persistente `/var/lib/nexa/.cache` (`0700`), con `HF_HOME=/var/lib/nexa/.cache/huggingface`. |
| nexa-memoryd | Sin red. SQLite WAL y sqlite-vec 0.1.7 en `/var/lib/nexa/memory/`, con `secure_delete=ON`. El borrado forense usa `write_both`, tombstones y deletion ledger. La voz no confiere autoridad. |
| nexa-brain | Compone nexa-nlu, nexa-ha, nexa-household (SQLite v3 en `/var/lib/nexa/household/`, rutinas programadas y actualización CAS de recordatorios) y nexa-llm (failover, Bearer auth y CloudConsentGate). |
| nexa-web | Axum y `/var/lib/nexa/auth/auth.db`. RBAC OPERATOR/HOME/EVALUATOR, WebSocket v3 con revalidación anti-TOCTOU. Feature `test-mode` apagada en release. Sirve Vite + TypeScript + Svelte 5, carga Three.js bajo demanda para el recurso autenticado `casa.glb`. NEXO en `/` y observabilidad en `/obs`, sin enlace entre vistas (D1). |
| nexa-llm-fast | llama-server Qwen2.5-7B-Instruct-Q4_K_M en `127.0.0.1:8101`, con `LD_LIBRARY_PATH=/opt/nexa/lib` y `--api-key-file /etc/nexa/secrets/llama-api-key`. |
| nexa-dashboard-proxy | NGINX termina TLS en `192.168.30.2:9540` y reenvía a nexa-web en `127.0.0.1:9500`. El certificado, la clave y el GLB privado se provisionan fuera del paquete. |

## Construcción y validación

1. Desde un checkout limpio del SHA aprobado, compilar los cuatro binarios en modo release para el host destino y construir `frontend/dist/`. Compilar audiod con hardware y web con features de release. Ejecutar la [CI v2](../../.github/workflows/v2-ci.yml).
2. Fijar `SOURCE_DATE_EPOCH` y ejecutar `python3 deploy/nexa/package-release.py VERSION DESTINO` en un directorio nuevo o vacío. Conservar manifiesto, SHA, inventario y hashes. El instalador rechaza `git_dirty: true`, también durante rollback. [Empaquetador](../../deploy/nexa/package-release.py).
3. Ensayar la [migración offline](../../tools/nexa-migrate/README.md) con escritores detenidos sobre un origen consistente. El migrador lee SQLite/JSON en modo de solo lectura, verifica WAL y ledger y promueve conjuntamente el destino y su informe. Validar `memory/memory.db`, `memory/ambient.db`, `household/household.db` v3 y `auth/auth.db`. Comprobar que ningún tombstone, persona retirada o consentimiento revocado resucite. Revisar los conteos de rutinas deshabilitadas y los hechos pendientes de reindexación. Guardar el informe fuera de Git.
4. Con GO, ejecutar como `nexa` el comando `deploy/nexa/install-release.sh --user install PAQUETE VERSION`. El instalador serializa operaciones con `.install-lock`, verifica `SHA256SUMS`, quita los permisos de escritura a la release y conmuta `/opt/nexa/current` atómicamente. Reinicia el target con `systemctl --user` y comprueba las siete unidades con un plazo de 130 s. Verificar units, sockets, permisos, autenticación, modelo offline y captura por habitación. Consultar logs como `nexa` con `journalctl --user -u 'nexa-*'`.
5. Ensayar y cronometrar `deploy/nexa/install-release.sh --user rollback VERSION_ANTERIOR` con una release validada. El objetivo es menos de 60 s. El switch cambia la release de código. Conservar por separado el estado privado compatible y acordar su recuperación antes de habilitar escrituras. [Instalador](../../deploy/nexa/install-release.sh).

## Verificación de cutover

Antes de promover, contrastar paquete, unidades efectivas, configuración NGINX y procesos con el [inventario físico](HARDWARE.md). Registrar el resultado en el gate de cutover.
