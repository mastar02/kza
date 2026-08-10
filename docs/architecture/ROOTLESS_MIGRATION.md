# Migración de KZA a rootless real — análisis y plan

> Análisis point-in-time (2026-08-10), disparado por el incidente del instalador de Hermes
> Agent: un script sin revisar corrió `sudo apt install` (Xvfb, drivers mesa, libs de browser)
> sin pedir contraseña ni confirmación. Investigado en vivo contra el server `kza`
> (`192.168.1.2`), todo de solo lectura. Ver `docs/SERVER_CONVENTIONS.md` §R10 para la política
> que este documento intenta cerrar.

## TL;DR

**El pipeline de voz en sí ya corre rootless.** `kza-voice`, `kza-code-index` y el
`llama-server :8101` corren como usuario `kza` (UID 1000) vía `systemctl --user`, no como root
— verificado contra los procesos reales del server, no contra documentación. El problema real
**no es que algo corra como root hoy**, sino que **la cuenta `kza` puede convertirse en root en
cualquier momento, sin fricción**:

```
$ sudo -l
User kza may run the following commands on kza:
    (ALL : ALL) ALL
    (ALL) NOPASSWD: ALL
```

Sumado a membresía en los grupos `docker`, `lxd`, `kvm`, `libvirt` (todos root-equivalentes por
diseño), la cuenta `kza` es hoy funcionalmente una cuenta de administrador del server completo,
no la cuenta de un proyecto aislado. El incidente de Hermes fue la primera vez que ese poder se
ejerció por accidente — pero la capacidad estaba ahí desde antes, disponible para cualquier
script, dependencia comprometida, o error de cualquier sesión (incluida esta).

## Qué se verificó y cómo

Todo verificado en vivo por SSH contra el server, solo lectura, sin cambios:

| Chequeo | Comando | Resultado |
|---|---|---|
| Procesos que corren como root relacionados a kza | `ps -eo user,pid,cmd` filtrado | **Ninguno.** Todos los root PIDs pertenecen a mailcow, ha-core, o servicios de sistema (nginx, haproxy, libvirtd, cups) — proyectos distintos |
| Units systemd a nivel sistema para kza | `find /etc/systemd/system -iname '*kza*'` | **Vacío.** Solo existen las `systemctl --user` |
| Crontab de root con referencias a kza | `sudo crontab -l -u root \| grep kza` | **Vacío** |
| Sudo del usuario kza | `sudo -l` | `(ALL) NOPASSWD: ALL` — sin restricción |
| Origen de esa regla | `sudo grep -rn kza /etc/sudoers.d/` | `/etc/sudoers.d/kza`, fechado **16 de marzo** — antes de que existiera R10 (abril-junio) |
| Comparación con otros proyectos | `ls /etc/sudoers.d/` + leer cada uno | `kza` es el **único** archivo de proyecto; ningún otro usuario (`infra`, `postpilot`, `konsi`, `sockerdata`, `obs`, `dns`, `thermouy`, `cftunnel`, `trading-bot`) tiene sudoers propio |
| Grupos de kza | `id kza` | `sudo,audio,dialout,docker,lxd,kvm,libvirt,adm,cdrom,dip,plugdev,secrets-llm` |
| Qué corre vía el grupo `docker` | `docker ps -a` | Los **17 contenedores de mailcow** (proyecto de correo, no relacionado a KZA) — `kza` tiene acceso root-equivalente a infraestructura ajena |
| Linger (necesario para systemd --user) | `loginctl show-user kza --property=Linger` | `Linger=yes` — correcto, no tocar |
| Crontab propio de kza | `crontab -l -u kza` | Un job de **reentrenamiento de "trading"** corriendo desde `/home/kza/trading` — proyecto distinto (existe usuario `trading-bot` UID 1002 dedicado), viviendo bajo la cuenta de kza por herencia histórica |
| Scripts del repo con sudo embebido | `grep -rl sudo scripts/` | `scripts/setup_ubuntu.sh` — pensado para correr como root una vez, pero sus instrucciones finales (`sudo systemctl enable kza-voice`) están **desactualizadas**: el deploy real usa `systemctl --user`, no una unit de sistema |

## Por qué esto pasó (contexto, no excusa)

`kza` es, por la fecha del sudoers file, casi con certeza el **primer proyecto** que existió en
este server — de antes de que se formalizara el modelo multi-tenant rootless (Podman + systemd
--user + CDI) descrito en `SERVER_CONVENTIONS.md`. Cuando esa política se escribió para los
proyectos *nuevos* (`postpilot`, `konsi`, `sockerdata`, etc.), la cuenta `kza` ya existía con
privilegios de una era anterior y nunca se recortó. La propia excepción documentada de KZA a R10
(#4: "KZA-voice también corre nativo... acceso USB ReSpeaker + MA1260 serial + presupuesto
<300ms") es legítima y sigue siendo válida — el pipeline de voz necesita correr nativo, no en
contenedor. Pero "correr nativo como usuario sin privilegios" y "tener sudo irrestricto +
membresía en `docker`/`lxd`/`kvm`" son cosas completamente distintas, y solo la primera está
justificada por esa excepción.

## Lo que SÍ necesita el pipeline de voz (y ya lo tiene, sin root)

- **Mic ReSpeaker (USB)** → grupo `audio` — ✅ ya lo tiene, es lo que usa hoy.
- **Amplificador MA1260 (serial RS-232)** → grupo `dialout` — ✅ ya lo tiene.
- **GPU (STT/TTS/embeddings/llama-server)** → funciona hoy sin estar en `video`/`render` (los
  `/dev/nvidia*` deben tener permisos permisivos por defecto en esta instalación) — no requiere
  sudo ni root en ningún punto del arranque observado.
- **Puertos** (`:8101`, rango `9500-9599`, más `:8300`/`:8301` de postpilot-embeddings alojado
  acá) — todos >1024, sin necesidad de `CAP_NET_BIND_SERVICE` ni root.
- **systemd --user + linger** — ya configurado correctamente, sobrevive logout/reboot sin root.

Ningún componente del pipeline de voz que se observó corriendo (16 h+ de uptime antes del deploy
de hoy) tocó sudo, docker, ni ningún grupo root-equivalente. La necesidad de esos privilegios es,
por lo que se pudo verificar, **puramente histórica/heredada**, no funcional.

## Cambios requeridos

1. **Reemplazar `/etc/sudoers.d/kza`.** Borrar la línea `NOPASSWD: ALL`. Default recomendado:
   **sin sudo en absoluto** (igual que todo otro proyecto del server). Si aparece una necesidad
   real y puntual (instalar un paquete de sistema nuevo, regenerar CDI la primera vez — los dos
   casos que R10 ya contempla como excepción), se ejecuta desde una sesión de admin explícita,
   no desde una regla permanente en la cuenta del proyecto.
2. **Sacar a `kza` de los grupos `docker`, `lxd`, `kvm`, `libvirt`, `sudo`.** Ninguno tiene un
   consumidor legítimo verificado en el pipeline de voz. `docker` en particular es la vía de
   acceso a la infraestructura de mailcow — sacarla no afecta a KZA y cierra ese cruce entre
   proyectos.
3. **Corregir `scripts/setup_ubuntu.sh`** para que documente el modelo real de deploy
   (`systemctl --user`, `loginctl enable-linger`) en vez del modelo de unit de sistema que ya no
   se usa. Sin este fix, un re-provisioning futuro reintroduce el problema desde cero.
4. **Reubicar (o dar de baja) el cron de "trading"** que vive en `crontab -u kza` /
   `/home/kza/trading` — pertenece al proyecto `trading-bot` (UID 1002), no a KZA. No es un
   problema de root, pero es la misma clase de mezcla de responsabilidades que causó que KZA
   terminara con privilegios que no le corresponden; conviene resolverlo en la misma pasada.
5. **Confirmar el cierre de la Postgres legacy vía Docker** (`:5432`, marcada "pendiente cleanup"
   en `SERVER_CONVENTIONS.md`, snapshot 2026-06-09) — no aparece en `docker ps -a` hoy, así que
   probablemente ya se limpió; falta solo actualizar esa línea del mapa de puertos para que el
   doc no mienta.
6. **Política hacia adelante para componentes nuevos** (ej. el propio Hermes CLI, pendiente de
   reinstalar): si necesita dependencias de sistema pesadas (como el stack Xvfb/browser que
   instaló el otro día), evaluar contenerizarlo en un Podman rootless dedicado en vez de
   instalarlo nativo en la cuenta compartida — así su huella queda contenida a su propio
   contenedor y límites de recursos, y no vuelve a depender de que alguien apruebe `sudo apt
   install` a mano cada vez.

## Riesgos y cosas para confirmar ANTES de tocar nada

Estos no son bloqueantes técnicos — son preguntas de "¿alguien usa esto hoy?" que solo vos podés
responder, porque no hay forma de verificarlo de forma segura desde afuera sin arriesgar cortar
algo en uso:

- **¿Se usa la sesión de `kza` para administrar mailcow** (via `docker exec`/`docker logs` sobre
  los contenedores `mailcowdockerized-*`) **como atajo operativo?** Si sí, sacar el grupo
  `docker` corta ese atajo — la alternativa correcta es entrar como el usuario dueño de mailcow
  (o como admin) para esas tareas, no mantener el acceso cruzado desde `kza`.
- **¿Se usa `kza` para gestionar la VM de OPNsense** (`virsh`, mencionado en el historial de
  incidentes de abril) **u otras VMs vía libvirt/kvm?** Si sí, mismo razonamiento: esa gestión
  debería vivir en la cuenta de administración del host, no en la de un proyecto.
- **¿Hay algún script o hábito manual que dependa de poder hacer `sudo <lo que sea>` sin pensar**
  desde la sesión de `kza`, que se rompería al sacar el `NOPASSWD: ALL`? Vale la pena un `grep -r
  sudo` más amplio sobre `/home/kza` (no solo el repo de KZA) antes de aplicar el cambio, por si
  hay algo fuera de este proyecto que también vive en esa cuenta.

## Pasos de la transición (orden de menor a mayor riesgo)

Cada paso es independiente y reversible hasta el punto en que se ejecuta; no hace falta hacerlos
todos de una sentada.

1. ✅ **Corregir `scripts/setup_ubuntu.sh`** — hecho 2026-08-10 (`082d3cd`): el script instala
   user unit + linger, layout real (`~/kza` + symlink `~/app`, `~/secrets`), y
   `systemd/kza-voice.service` en el repo es ahora espejo del unit real del server.
2. ✅ **Actualizar el mapa de puertos de `SERVER_CONVENTIONS.md`** — hecho 2026-08-10
   (`e04ca67`): la Postgres legacy `:5432` ya no existe (verificado en vivo); línea cerrada.
   Pendiente reflejarlo en Notion pág 8 desde una sesión de homelab-infra.
3. ✅ **Reubicar el cron de trading** — resuelto 2026-08-10 dándolo de baja: apuntaba a
   `/home/kza/trading`, que **ya no existe** (el proyecto migró a `/home/trading-bot` en junio),
   así que el cron llevaba semanas fallando en el `cd` sin loguear nada — no había nada que
   reubicar. Backup de la línea en `~kza/backups/crontab-kza-2026-08-10.bak`; `crontab -u kza`
   quedó vacío. Nota: `trading-bot` no tiene hoy ningún retrain semanal propio (solo el timer
   `trading-carry-monitor`) — si el retrain se quiere de vuelta, es una decisión del proyecto
   trading, no de KZA.
4. **Responder las preguntas de la sección anterior** (mailcow/libvirt vía `kza`) — esto
   determina si los pasos 5-6 son seguros tal cual o necesitan un intermedio.
5. **Sacar los grupos** `docker`, `lxd`, `kvm`, `libvirt`, `sudo` de `kza`
   (`sudo gpasswd -d kza <grupo>` por cada uno) — reiniciar la sesión SSH después para que el
   cambio de grupos tome efecto, y volver a correr el smoke test + verificar que `kza-voice`
   sigue sano (no debería tocarlo en absoluto, pero es la verificación barata).
6. **Borrar `/etc/sudoers.d/kza`** (o reemplazarlo por una regla acotada si el paso 4 reveló una
   necesidad real y puntual) — último paso, el de mayor blast radius si algo se pasó por alto en
   los pasos anteriores.

Verificación final post-migración: repetir exactamente los chequeos de la tabla de arriba
(`sudo -l`, `id kza`, `ps` filtrado por root) y confirmar que el pipeline de voz sigue sano
(`systemctl --user status kza-voice`, journal sin errores nuevos, smoke test verde).

## Lo que este documento NO cubre

- `ha-core` (Home Assistant) sigue corriendo rootful por diseño — es una excepción R10 propia
  (#2), de un proyecto distinto (`ha`), no de KZA. No es parte de esta migración.
- mailcow bajo root — excepción R10 #3, proyecto `mail`, tampoco es KZA.
- Durante la investigación original (2026-08-10 AM) no se tocó nada del server. Los pasos 1-3
  de la transición se ejecutaron ese mismo día (ver checkmarks arriba); el único cambio de
  estado en el server fue la baja del cron muerto de trading (con backup). Los pasos 4-6
  siguen pendientes.
