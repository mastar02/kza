# Runbook: mic XVF3800 que enumera y no entrega audio (isócronos mudos)

**Fecha:** 2026-07-25 · **Contexto:** alta del segundo XVF3800 (cocina) por extensor Cat5e.

## Síntoma

El mic aparece en `lsusb` y `arecord -l`, `arecord` **termina con exit 0** y escribe un WAV
del tamaño exacto esperado… y todas las muestras son **cero absoluto**. Sin un solo error
en `dmesg`, sin xruns, sin mute en el mixer.

Es el modo de falla más difícil de diagnosticar que tiene este hardware, porque cada
chequeo individual da verde.

## Causa

Un puerto de hub marginal (acá el puerto 1 del hub del extensor, `5-5.1`).

Los **control transfers** de USB se reintentan a nivel protocolo: por eso la enumeración
pasa, se leen producto/serial/descriptores, y hasta se leen todos los parámetros del DSP
por la vendor interface. Los **transfers isócronos** del audio **no tienen reintentos, por
diseño**. Un puerto marginal los pierde, `snd-usb-audio` rellena el buffer de ALSA con
ceros, entrega el byte-count correcto y no reporta nada.

Resultado: el mic "existe", responde todo lo que le preguntes, y no entrega audio.

## Procedimiento de diagnóstico

La medición decisiva es **co-observar las dos capas en la misma ventana temporal**: lo que
el DSP oye (adentro del chip) contra lo que llega por USB.

```bash
# 1) ¿El array de micrófonos oye? — SPENERGY, adentro del chip.
#    Apuntar SIEMPRE a un device concreto: con >1 mic, usb.core.find() agarra
#    el primero que enumere.
cd /home/kza/app && ./.venv/bin/python -c "
import sys; sys.path.insert(0,'/home/kza/app')
import usb.core
from src.audio.xvf_controller import XvfController
dev = usb.core.find(idVendor=0x2886, idProduct=0x001a,
                    custom_match=lambda d: d.bus == 5)   # ajustar bus
c = XvfController(device=dev)
print('SPENERGY', c.read_param('AEC_SPENERGY_VALUES'))
print('AZIMUTH ', c.read_param('AEC_AZIMUTH_VALUES'))
print('AGCGAIN ', c.read_param('PP_AGCGAIN'))"

# 2) ¿Llega audio por USB? — contar muestras NO-CERO, no confiar en el exit code.
arecord -D hw:<CARD>,0 -f S16_LE -r 16000 -c 2 -d 3 /tmp/t.wav
python3 -c "
import wave,array
w=wave.open('/tmp/t.wav'); d=array.array('h'); d.frombytes(w.readframes(w.getnframes()))
print('no_cero=%d de %d' % (sum(1 for x in d if x), len(d)))"
```

Interpretación:

| SPENERGY con ruido | Muestras no-cero | Conclusión |
|---|---|---|
| alto (1e6–1e8) | ~0% | **Isócronos caídos** → puerto/cable. Probar otro puerto del hub. |
| alto | >99% | Sano. |
| 0 con ruido real | ~0% | Array de micrófonos o unidad fallada. |
| 0 en silencio | — | Normal: SPENERGY es un VAD de hardware, lee 0 sin voz. **No concluir nada.** |

`PP_AGCGAIN` con un valor no-redondo (ej `7.5779447`) es prueba de que el pipeline del DSP
está corriendo y adaptándose: el AGC lo recalcula en runtime.

## Trampas encontradas en esta sesión

- **`arecord` exit 0 no significa audio.** Siempre contar muestras no-cero. Un soak que
  solo mira el exit code reporta "OK" para siempre sobre silencio digital.
- **`grep "error -71"` no matchea** los errores de este driver, que escribe `err -71`
  (`cannot get freq (v2/v3): err -71`). Grepear `err -` y `error -`.
- **El ring buffer de `dmesg` rota rápido** en este server (churn de veth/podman): un
  conteo histórico en cero no prueba que nunca hubo errores.
- **Desenchufar el mic sin que el kernel loguee `USB disconnect`** = el enlace a ese mic ya
  estaba caído y el hub no lo reportaba.
- **Sacar el cable del lado del server sí loguea disconnect del hijo**, pero eso viene del
  modelo interno del kernel al destruir el subárbol — no prueba que el mic estuviera
  eléctricamente presente.
- El salto grande en el número de device (`30` → `100` en minutos) indica decenas de
  intentos de enumeración que no completaron.

## Los resets por software NO alcanzan — hay que cortar VBUS

El mismo día, el mic del **escritorio** cayó en un estado parecido pero peor: `arecord`
devolvía `pcm_read: read error: Input/output error` en la primera lectura (ni siquiera ceros).
El DSP seguía vivo (`VERSION 2.0.6`, `PP_AGCGAIN 2.139` calculado en runtime) y los control
transfers funcionaban — solo el endpoint isócrono estaba muerto.

Se probaron, en orden, y **ninguno lo revivió**:

```bash
# 1) re-enumerar el device
echo 0 | sudo tee /sys/bus/usb/devices/3-1.4/authorized; echo 1 | sudo tee ...
# 2) disable/enable del puerto del hub
echo 1 | sudo tee /sys/bus/usb/devices/3-1:1.0/3-1-port4/disable; echo 0 | sudo tee ...
# 3) reset del hub entero (por el "clear tt ... error -71" que lo señalaba)
echo 0 | sudo tee /sys/bus/usb/devices/3-1/authorized; echo 1 | sudo tee ...
```

**Lo que lo arregló fue desenchufarlo a mano.** Los hubs Terminus `1a40:0101` del setup no
tienen conmutación de alimentación por puerto: `authorized` y `port/disable` re-enumeran pero
**no cortan VBUS**, así que el XVF3800 nunca pierde alimentación y su DSP queda atascado en
el mismo estado. Tras el replug físico: `ch0 rms 130.6 / ch1 rms 352.1`, 99.6-99.8% no-cero.

**Regla:** para un XVF3800 con el isócrono muerto, el único reset efectivo es el físico.
No perder tiempo con sysfs. Si el mic es inaccesible, hace falta un hub con per-port power
switching (PPPS, `uhubctl`) para poder hacerlo remotamente.

**Disparador probable:** perturbación USB al manipular el rack. A las 13:15 aparecieron
`usb 3-1: clear tt 1 (9032) error -71` y `usb 1-2: clear tt ...` mientras se conectaba el
extensor de la cocina; a las 13:17:55 el mic del escritorio dejó de entregar frames y el
audio-watchdog entró en loop cada ~2m46s (recupera → 8s sin frames → cae) durante 2 horas.

## Fix aplicado

Mover el mic al puerto 4 del mismo hub (`5-5.4`). Mismo cable, mismo extensor, misma
unidad. El puerto 1 queda documentado como quemado en `config/settings.yaml` (room
`cocina`).

## Bug de software que destapó

`XvfController.open()` hacía `usb.core.find(VID, PID)` sin filtro → el primer XVF3800 que
enumere. Con `xvf_tuning.apply_on_start: true`, un restart de `kza-voice` podía escribir el
tuning sobre el mic equivocado y dejar al otro en preset de fábrica (`AGCMAXGAIN=64`) en
silencio — la cascada de sordera del 2026-07-05.

Arreglado con binding por puerto USB y un controller por room. Ver
`src/audio/xvf_controller.py` (`parse_usb_port`) y `MultiRoomAudioLoop.xvf_controllers`.
Las CLIs `tools/xvf_tune.py` y `tools/acoustic_calibration.py` tienen ahora `--port`.
