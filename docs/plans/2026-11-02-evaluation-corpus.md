# Contrato del corpus de evaluación doméstica

Preparación de T004 del [roadmap](2026-11-02-autonomous-roadmap.md).
Este contrato define admisión, separación y retirada. No inicia grabaciones,
concede consentimientos ni declara completo el borrado del sistema.

## Finalidad y datos privados

El corpus nuevo sirve exclusivamente para calibración y evaluación consentidas
del perfil `home-2026-11-v1`. No admite capturas ordinarias, material legacy,
entrenamiento LoRA ni clonación. La retención ordinaria no se convierte en 30
días por copiar un archivo o cambiarle su finalidad. Consentimientos, personas,
WAV, referencias, embeddings y rutas concretas quedan fuera de Git y RAG.

El modelo validable está en `src/evaluation/corpus.py`. Sus dataclasses son
inmutables; los manifiestos privados deberán serializar datetimes con zona
horaria, enums por valor y tuplas como arrays. El esquema 1 conserva el inventario
previo, con procedencia técnica desconocida. El esquema 2 permite declarar
procedencia tipada y comparaciones explícitas; una versión desconocida se
rechaza. No hay promoción automática de los registros anteriores. El modelo no
agrega un adaptador de archivos ni un productor de grabaciones.

## Esquema y procedencia

| Registro | Campos y regla |
|---|---|
| Corpus | Finalidad fija, revisión del protocolo, instante de revisión, sesiones, capturas, artefactos y cierre opcional. El protocolo identifica la distribución, exclusiones y evaluador congelados antes de abrir reserva. |
| Sesión completa | ID estable, participante seudónimo, referencia al consentimiento, instante del consentimiento, comienzo/fin y partición `calibration` o `reserved`. La revocación es un evento irreversible. |
| Captura original | ID, sesión, captura, vencimiento original, SHA-256 del archivo original, duración, canal físico/lógico, revisión DSP, habitación, distancia, condición y revisión del candidato. Origen obligatorio: captura nueva consentida de evaluación. |
| Artefacto | ID, captura, padre opcional, tipo, ruta relativa privada, SHA-256 e intervalo en milisegundos de la captura original. Incluye original, copias, recortes, aumentos, referencias y embeddings. |
| Referencia humana | Artefacto `human_reference` con intervalo exacto y referencia a su revisión humana. El texto vive en el archivo privado; una transcripción de otro STT no lo sustituye. |

El tiempo de captura y los vencimientos no son parámetros del derivado. El
registro rechaza IDs duplicados, la reimportación del mismo original, padres
ausentes/cíclicos, cambios de fuente e intervalos fuera del padre o la grabación.
Un preview de seis segundos sólo se compara con su referencia de seis segundos.
La mera existencia de una referencia no demuestra su revisión o calidad.

### Procedencia y alineación por frames

`CaptureProvenance` identifica el origen, una referencia opaca de entrada, la
rama de procesamiento, las generaciones separadas de apertura y segmento,
el número de canales, el canal seleccionado, la tasa y los frames de la PCM
poseída. No se serializa `CaptureObservation` ni se infiere información técnica
del nombre de habitación. `DSPObservation` distingue estado desconocido,
configuración declarada y revisión observada. Una intención de configuración
no acredita una lectura del DSP; la comparación exige una revisión observada.
Las etiquetas históricas `channel` y `dsp_revision` se conservan como descripción;
no sustituyen la procedencia tipada ni habilitan una comparación.

Los artefactos del esquema 2 pueden declarar `FrameInterval`: intervalo
semiabierto relativo al frame cero de esa PCM. Sus campos en milisegundos son
sólo la envolvente inferior/superior de compatibilidad. Dos intervalos que
redondean igual siguen siendo diferentes. Un hijo con frames requiere que cada
padre tenga frames y contenga exactamente su intervalo; no se completa un
padre desconocido a partir del hijo.

`CorpusComparison` vincula un audio `recording` o `clip`, una referencia
`human_reference` revisada y una hipótesis `stt_hypothesis`. Los tres pertenecen
a la misma captura y al mismo intervalo exacto; referencia e hipótesis declaran
ese audio como padre y fuente explícita. Un texto del segmento completo no se
recorta por proporción para compararlo con un preview. Los registros sin
procedencia suficiente pueden seguir inventariados, pero no son comparables.

`require_comparison` comprueba esa coherencia y el vencimiento del snapshot.
No abre archivos, calcula métricas, prueba el canal físico o reemplaza la
autoridad vigente de consentimiento, revocaciones y tombstones. El adaptador
de lectura deberá revalidar esa autoridad y la identidad real de los archivos.
Las garantías generales de borrado y restauración se completan con T022/T024.

En las capturas nativas, `CaptureOpening` viaja con cada bloque desde el holder
del stream. Pre-roll, segmentos y snapshots conservan las aportaciones
desconocidas; un bloque posterior conocido no vuelve a etiquetar el audio
anterior. Un cambio conocido de apertura o canal retira el buffer y el estado
recurrente antes de mezclar entradas. La ventana completa que recibió STT y el
audio recortado del comando tienen referencias independientes. El canal de STT
se fija antes de la inferencia y no se consulta otra vez al reutilizar el texto.
Sin lectura real del DSP, su estado permanece desconocido.

Las revisiones de voz conservan documentos nativos cerrados de **versión 1**,
distintos del esquema 2 del manifiesto del corpus:

- `audio_metadata` contiene `schema_version`, `provenance` y `span`. La ventana
  del fragmento corresponde a los frames reales del WAV mono de 16 kHz, con
  hasta seis segundos de una entrada de entre 0,8 y 30 segundos.
- `transcript_metadata` añade `text_truncated`, booleano o desconocido. Su
  procedencia, tasa, canal y ventana pertenecen a la entrada original de STT;
  no se les aplica el límite de 30 segundos del audio de revisión. El recorte
  local del texto se registra antes de limitarlo a 8192 caracteres.

Un documento `null` conserva el desconocido histórico. El texto puede conocer
su recorte aunque su procedencia y ventana sean desconocidas. Un documento
no nulo malformado vuelve la muestra no disponible; no se sustituye por datos
de la configuración actual. El operador recibe ambos documentos y el evaluador
recibe únicamente el de la modalidad concedida. Los lectores limitan tipos y
bytes en SQLite antes de obtener las columnas; cada modalidad valida sólo su
payload. El navegador valida y copia el mismo contrato y rechaza contadores
que no pueda representar exactamente. Retirar acceso también retira los
metadatos, el DOM y el recurso de audio del consumidor original.

La interfaz distingue el fragmento, la entrada de STT y el recorte conocido o
desconocido del texto. Estos campos no prueban alineación de palabras ni
autentican personas, conceden permisos o sustituyen consentimiento vigente.

## Separación y apertura de la reserva

Se asigna una sesión completa a una sola partición antes de procesarla. Dos
registros del mismo ID con particiones distintas son inválidos. Copias, recortes
y aumentos heredan sesión, persona, partición y vencimiento de la captura.
El escritor persistente deberá conservar los IDs y asignaciones entre revisiones;
no puede borrar y volver a admitir el mismo original con otro hash o sesión.
Los hashes son una defensa adicional, no prueba de consentimiento ni detector
universal de material derivado.

Antes de abrir reserva se registra la revisión del candidato, evaluador,
normalización de referencia, exclusiones, familias y distribución por condición,
persona, habitación y distancia. Siguen vigentes todos los mínimos C12–C18 y la
estratificación del roadmap; este esquema no los reduce. Las sesiones completas
de calibración y reserva se realizan en días distintos. No se consulta la reserva
para ajustar parámetros. Un cambio del candidato exige nueva validación y, si
se utilizó la reserva para ajustarlo, nueva reserva independiente acordada.

## Vencimiento y purga

Para cada captura y todos sus artefactos:

`vencimiento efectivo = min(vencimiento original, captura + 30 días, cierre, revocación aplicable)`

El acceso se deniega al alcanzar ese instante aunque la eliminación física siga
pendiente. Cierre y revocaciones se aplican a las copias y derivados registrados.
El orden temporal y los 30 días se calculan en UTC, también cuando un horario
local se repite; la comparación no ignora el offset ni el indicador fold.
Cambiar una revisión/configuración, crear un derivado al día 29 o restaurar un
manifiesto antiguo no renueva la autorización. Una revocación de participante se
aplica a todas sus sesiones; nunca sólo a la muestra visible en la interfaz.

`CorpusManifest.retired_artifacts()` produce el plan completo de retirada del
snapshot; no borra archivos ni acredita eliminación. El adaptador persistente
deberá mantener revocaciones/cierre y tombstones fuera de backups restaurables,
verificar esa autoridad vigente antes de leer/escribir y rechazar una restauración
si no puede reconciliarla. Los archivos tardíos de trabajos en curso se registran
y eliminan sin exponerse. No se permiten copias sin registrar.

La purga sólo abre/elimina rutas dentro de la raíz privada registrada, sin seguir
symlinks; una ruta léxicamente relativa no basta. Cada fallo de unlink o copia
desconocida queda pendiente y visible. No se retira la entrada de limpieza hasta
confirmar la ausencia del archivo. El recibo posterior conserva sólo identificador
opaco, instante, versión de política y resultado necesarios para impedir
resurrección; no texto, audio, identidad, huellas ni embeddings.

La integración del registro con productores, SDK, copias y el coordinador de
borrado se verifica en T014/T022/T024. Las pruebas puras de P0 no demuestran
concurrencia, borrado físico ni restauración segura del conjunto completo.
