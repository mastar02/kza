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
horaria, enums por valor y tuplas como arrays. Se usa `schema_version: 1` en el
contenedor persistido; una versión desconocida se rechaza, sin inferir defaults.
No se agrega una configuración runtime ni un productor de grabaciones en P0.

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
