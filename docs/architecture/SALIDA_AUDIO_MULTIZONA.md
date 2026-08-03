# Salida de audio multizona — KZA → MA1260 → parlantes

**Estado:** hardware comprado, sin instalar. Sesión de decisión: 2026-07-28 / 2026-08-01.
**Decisión:** Behringer U-Phoria **UMC1820**, USD 399 (Uruguay).

Este documento cubre **cómo el audio sale del server hacia los parlantes**. No cubre el
diseño del software de ruteo, que queda pendiente para cuando el hardware esté instalado.

---

## 1. El constraint que define todo: la referencia del AEC

XMOS, sobre el XVF3800:

> *"A far-end AEC reference signal must be provided on the left (0) channel of the I²S
> or USB input signal"* — y la parte *"supports a monophonic audio output and uses a
> single channel to provide the reference signal for the acoustic echo canceller (AEC)."*
>
> — https://www.xmos.com/documentation/XM-014888-PC/html/modules/fwk_xvf/doc/datasheet/03_audio_pipeline.html

**El cancelador de eco por hardware se alimenta del stream de playback que el host le
manda al chip.** Si el audio sale por otra placa, el XVF3800 no tiene con qué cancelar:
el micrófono escucha los parlantes sin AEC → falsos wakes con la propia voz de KZA y
barge-in degradado.

### Consecuencia operativa

A cada XVF3800 hay que escribirle, por su endpoint USB de playback, **el mix mono de su
zona como referencia-only** — sin conectar nada a su salida de 3,5mm.

Y una decisión que se deriva de esto: **el volumen se aplica antes del split** (arriba,
en el bus de zona), no después. XMOS advierte que `AEC_FAR_EXTGAIN` tiene que seguir al
volumen (−6 dB de atenuación → setear −6). Si el volumen se aplica arriba del split, la
referencia ve exactamente lo mismo que sale por el parlante y `FAR_EXTGAIN` queda fijo.
Si se aplicara después, habría que actualizarlo por RPC en cada cambio de volumen.

### Arquitectura alternativa que se evaluó y se descartó

El ReSpeaker XVF3800 **tiene jack de 3,5mm + conector de parlante dedicado**, y enumera
como sound card USB en ambas direcciones. Eso permitía una topología sin ninguna interfaz:
cada cuarto usa su XVF3800 como DAC, salida al LINE IN del canal correspondiente del
MA1260, AEC perfecto y costo cero en hardware.

**Descartada porque el XVF3800 saca mono** y living y escritorio son zonas estéreo.

---

## 2. El amplificador no tiene control digital

Confirmado contra el manual (2026-05-18) y re-verificado contra specs de Dayton en esta
sesión. El **Dayton Audio MA1260**:

- 12 canales, Class-D, 60W/ch @ 4Ω
- **12 entradas RCA dedicadas** (una por canal) + 2 entradas BUS compartidas
- Trim de nivel **físico** por canal
- Selección de fuente por canal (BUS1 / BUS2 / LINE IN): **switch físico del panel trasero**
- Único control eléctrico: trigger 12V para encendido
- **Sin RS-232, sin red, sin IR, sin USB**

Todo el ruteo, volumen y mezcla tiene que pasar en el dominio digital, río arriba. El amp
es una etapa de potencia tonta. Por eso hace falta una interfaz multicanal: es el único
lugar donde puede vivir la lógica de zonas.

### Deuda de código que esto expone

`src/audio/ma1260_controller.py` implementa un protocolo serial `<STX><Zone><Command>`
9600 8N1 **que este amplificador no tiene**. Hoy no rompe nada porque está en
`connection_type: "simulation"`, pero:

- `src/main.py:562` y `src/main.py:807` lo instancian (dos veces)
- `src/audio/zone_manager.py:271-273` llama `select_zone()` + `set_volume()` en el camino
  caliente del TTS — son **no-ops silenciosos**
- `play_to_zone()` termina en `sd.play()` sobre `audio_output_device: null` → **device
  default de ALSA, mono**

**El enrutado por zona hoy es decorativo.** Cuando alguien ponga `connection_type: "serial"`
va a escribir bytes al vacío y creer que ruteó. Hay que reducir `MA1260Controller` a lo
que el amp soporta (trigger 12V) o borrarlo.

Esto también explica por qué `echo_suppressor.py` implementa ducking por software y por
qué el umbral de barge-in está calibrado *"por encima del eco residual típico"*: el AEC
por hardware nunca estuvo recibiendo referencia.

---

## 3. Reparto de zonas y mapa de canales

| Zona | Configuración | Canales fuente |
|---|---|---|
| Living | estéreo | 2 |
| Escritorio | estéreo | 2 |
| Hall | mono | 1 |
| Cocina | mono | 1 |
| Baño | mono | 1 |
| **Total** | | **7** |

Zona 6 del MA1260 libre, sin destino asignado.

```
UMC1820 out 1 → MA1260 ch1 LINE IN → Living L
UMC1820 out 2 → MA1260 ch2 LINE IN → Living R
UMC1820 out 3 → MA1260 ch3 LINE IN → Escritorio L
UMC1820 out 4 → MA1260 ch4 LINE IN → Escritorio R
UMC1820 out 5 → MA1260 ch5 LINE IN → Hall
UMC1820 out 6 → MA1260 ch6 LINE IN → Cocina
UMC1820 out 7 → MA1260 ch7 LINE IN → Baño
UMC1820 out 8-10 → libres
```

Los 7 switches traseros del MA1260 van a **LINE IN**. Los trims de canal se calibran una
vez en la instalación a un nivel de referencia parejo y no se tocan más: de ahí en
adelante el volumen lo maneja el software.

---

## 4. Por qué el UMC1820

10 salidas analógicas: MAIN OUT 1-2 + LINE OUT 3-10 (8× TRS balanceadas). Class compliant
(USB ID `1397:0503`), soportado por `snd-usb-audio` sin drivers.

- **10 salidas contra 7 necesarias** — la zona 6 entra y sobra una.
- **Disponibilidad y garantía local.** Pesa: es un sistema que la casa usa a diario.
- **Expansión por ADAT** si algún día hacen falta más canales (ADA8200 = +8 analógicas).
- **El riesgo del MAIN OUT está acotado.** Las salidas 1-2 tienen un switch que las pone
  en "mezcla de todas las salidas" o discretas. Si no se pueden aislar, quedan las 8
  LINE OUT 3-10, que **igual cubren los 7 canales**. El peor caso cuesta la zona 6, no el
  sistema. (Con 5 zonas estéreo habría sido fatal; con este reparto mono/estéreo no lo es.)

**Lo que se paga de más:** 8 preamps Midas y phantom power que no se van a usar nunca, y
2U de rack. Es una interfaz de grabación usada como DAC multicanal.

### Alternativas descartadas

| Modelo | Por qué no |
|---|---|
| **ESI Gigaport eX** | **Mejor ajuste técnico**: 8 salidas RCA que enchufan directo al MA1260, playback puro sin entradas muertas, ESI confirma ALSA en su base de conocimiento. Descartada por disponibilidad: sin stock ni distribuidor local, listings de ML son revendedores por importación desde EE.UU. |
| **Focusrite Scarlett 18i20** | 2× a 3,4× el precio. Su driver de kernel (FCP, mainline 6.14) resuelve el **control** de la interfaz — pad, air, phantom, mixer interno — no el playback. Para playback puro, class-compliance alcanza. La ventaja es irrelevante para este caso. |
| **MOTU 24Ao / UltraLite mk5** | MOTU mismo describe su soporte Linux como *"very spotty"*. Reportes concretos: canales que no aparecen (24Ao), estática en canales sin usar (mk5 sobre Pi4/Ubuntu). |
| **Placas 7.1 genéricas (CM6206)** | 4 jacks estéreo, independencia por canal sin verificar bajo ALSA. Solo servía para prototipar. |
| **N× DAC USB estéreo** | Suma N dispositivos USB más a un bus que ya tuvo problemas de re-enumeración (ver watchdog). Sin sample-sync entre zonas. |
| **Snapcast (red)** | Sync sub-ms *entre clientes*, pero el buffering necesario para lograrlo mata el presupuesto de 300ms del TTS. Y pide un cliente + amp por cuarto. |

---

## 5. Qué verificar al instalar

1. **Nivel de salida.** Las LINE OUT del UMC1820 son balanceadas a nivel pro (+4dBu); el
   LINE IN del MA1260 es RCA consumer. Los trims por canal del amp deberían absorber la
   diferencia, pero **probar con un canal antes de cablear los siete** — si entra muy
   caliente se trabaja en el extremo del recorrido del trim.

2. **MAIN OUT 1-2 discretas o no.** Determina si hay 10 salidas útiles u 8.

3. **⚠️ Índices de ALSA.** Agregar la placa **corre los índices de card**. Toda la lógica
   `mic_usb_port` → `mic_device_index` existe porque eso ya mordió antes. La interfaz
   debe bindearse por nombre o puerto USB, nunca por índice.

4. **⚠️ Convivencia con la captura ALSA cruda.** Si se usa PipeWire para el ruteo, va a
   enumerar los XVF3800 y necesita su lado de *playback* para la referencia del AEC —
   pero KZA captura de esos mismos dispositivos por ALSA directo, y de eso dependen el
   binding por puerto, el watchdog y el presupuesto de latencia. Abrir playback y capture
   por separado sobre la misma placa debería funcionar, pero **hay que verificarlo antes
   de tocar producción, no asumirlo.**

5. **Delay de la referencia del AEC.** La referencia llega antes que el eco (orden
   correcto), pero la diferencia tiene que caer dentro de la ventana del AEC. Se mide.

6. **Referencia mono en zonas estéreo.** El chip acepta un solo canal: living y
   escritorio cancelan contra un downmix. Peor que las mono — y son justo los cuartos
   donde están los micrófonos.

### Validación barata, sin cablear nada

Se puede probar el punto más riesgoso con el hardware que ya hay: mandar el mismo audio
al XVF3800 del escritorio como referencia-only y la salida real a otro device, y medir si
el wake deja de dispararse con la voz de KZA.

**Esta prueba no pone en duda la interfaz** — el UMC1820 hace falta en los dos escenarios,
porque living y escritorio son estéreo y el XVF3800 saca mono. Lo que decide es *cuáles
cuartos pasan por él*:

- **AEC converge** → los 7 canales por el UMC1820 (el plan de arriba).
- **AEC no converge** → hall, cocina y baño caen de vuelta al XVF3800 como DAC de su
  cuarto (son mono, no pierden nada, y conservan AEC por hardware perfecto); living y
  escritorio siguen necesitando 4 canales del UMC1820.

⚠️ En el escenario B los tres cuartos mono suenan desde dispositivos USB distintos, así
que **no quedan sample-synced** con living/escritorio. Irrelevante para contenido
independiente por cuarto; molesta si se quiere la misma música sincronizada en toda la casa.

### Cables

**7× TRS 6,35mm → RCA.** Es costo real que no está en el precio de la caja. (Con el
Gigaport hubieran sido 7 RCA→RCA, más baratos y sin adaptación.)

---

## 6. Consecuencia: la música se muda

Decisión tomada en esta sesión: **la música pasa a los parlantes cableados**, los
Echo/Sonos salen. Eso implica un player local (librespot / Music Assistant) apuntando a
las zonas, y hace preferible que el ruteo lo haga **PipeWire con un sink por zona** en
vez de un router propio — así cualquier app puede targetear una zona sin que KZA sea
dueño del stream de música.

**`src/spotify/` son 4.568 líneas que hoy targetean por `spotify_device_name`**
(`"Echo Studio Sala"`, etc., ver `config/settings.yaml`). Migrar eso es un proyecto
aparte, no parte de la instalación del hardware.
