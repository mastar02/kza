"""Controlador del XVF3800 (vendor interface): lee SPENERGY = VAD por hardware
(pre-gate antes de Whisper) y lee/escribe parámetros de tuning del DSP en RAM.

El DSP del XVF3800 expone ``AEC_SPENERGY_VALUES`` (4 floats; índice 3 = beam
auto-select): >0 indica voz, 0 = ruido de banda ancha (secador) o silencio. Es
**pre-AGC y 0-GPU**, así que separa la voz del ruido del secador con un margen
enorme (medido 2026-05-31: secador=0, voz=101–2.1M). Correr este gate antes de
Whisper evita las alucinaciones ("Gracias" sobre silencio/secador).

Protocolo (verificado 2026-06-04 contra ``python_control/xvf_host.py`` oficial
de ``respeaker/reSpeaker_XVF3800_USB_4MIC_ARRAY@master``): control transfer
vendor sobre Interface 3 (NO toca el audio streaming).
- READ  = ctrl_transfer(0xC0, 0, wValue=0x80|cmdid, wIndex=resid, length)
- WRITE = ctrl_transfer(0x40, 0, wValue=cmdid,      wIndex=resid, payload)
Payload tipado little-endian: float/radians = f32, int32/uint32 = 4 bytes,
uint8 = 1 byte por valor.

⚠️ SOLO RAM: los writes son reversibles desenchufando el device. Los comandos
que persisten o rebootean (SAVE_CONFIGURATION, CLEAR_CONFIGURATION, REBOOT,
TEST_*) están DELIBERADAMENTE fuera de ``PARAMETERS`` — el issue #8 del repo
Seeed documenta que ``save_configuration`` puede dejar el device sin enumerar.

Diseño FAIL-OPEN operativo: cualquier problema de hardware/USB (sin device,
sin permisos, timeout) degrada a None/False sin excepción, de modo que NUNCA
se bloquea un comando de voz por un fallo del USB. Los errores de PROGRAMACIÓN
(parámetro desconocido, read-only, count incorrecto) sí levantan ValueError.
"""
from __future__ import annotations

import logging
import struct
import threading
import time
from collections import deque

logger = logging.getLogger(__name__)

# XVF3800 USB IDs (Seeed reSpeaker 4-Mic Array)
XVF_VID = 0x2886
XVF_PID = 0x001A

# Subconjunto curado del dict PARAMETERS del xvf_host.py oficial (verificado
# 2026-06-04 contra master). name: (resid, cmdid, count, rw, type).
# Solo tuning en RAM — ver advertencia del docstring del módulo.
PARAMETERS: dict[str, tuple[int, int, int, str, str]] = {
    # Firmware info
    "VERSION": (48, 0, 3, "ro", "uint8"),
    # VAD por hardware / dirección
    "AEC_SPENERGY_VALUES": (33, 80, 4, "ro", "float"),
    "AEC_AZIMUTH_VALUES": (33, 75, 4, "ro", "radians"),
    # AGC del post-processor (L-2: preset Seeed MAXGAIN=64 infla el piso)
    "PP_AGCONOFF": (17, 10, 1, "rw", "int32"),
    "PP_AGCMAXGAIN": (17, 11, 1, "rw", "float"),
    "PP_AGCDESIREDLEVEL": (17, 12, 1, "rw", "float"),
    "PP_AGCGAIN": (17, 13, 1, "rw", "float"),
    # Salida ASR del beamformer (tap pre-post-processor, gain fijo)
    "AEC_ASROUTONOFF": (33, 35, 1, "rw", "int32"),
    "AEC_ASROUTGAIN": (33, 36, 1, "rw", "float"),
    # Beams fijos (L-4: fijar dirección de escucha si el DoA se estabiliza)
    "AEC_FIXEDBEAMSONOFF": (33, 37, 1, "rw", "int32"),
    "AEC_FIXEDBEAMSAZIMUTH_VALUES": (33, 81, 2, "rw", "radians"),
    "AEC_FIXEDBEAMSELEVATION_VALUES": (33, 82, 2, "rw", "radians"),
    "AEC_FIXEDBEAMSGATING": (33, 83, 1, "rw", "uint8"),
    # Routing de canales de salida USB: <category>, <source> por canal (L-3)
    "AUDIO_MGR_OP_L": (35, 15, 2, "rw", "uint8"),
    "AUDIO_MGR_OP_R": (35, 19, 2, "rw", "uint8"),
}

_AUTO_BEAM_IDX = 3  # índice 3 = beam auto-select (el más robusto)

# bmRequestType vendor: READ = CTRL_IN(0x80)|VENDOR(0x40)|DEVICE(0x00);
# WRITE = CTRL_OUT(0x00)|VENDOR(0x40)|DEVICE(0x00).
_BM_REQUEST_READ = 0xC0
_BM_REQUEST_WRITE = 0x40

# Códigos de status del primer byte de la respuesta (del firmware XMOS).
_CONTROL_SUCCESS = 0
_SERVICER_COMMAND_RETRY = 64

# Bytes por valor según tipo declarado en PARAMETERS.
_TYPE_SIZE = {"uint8": 1, "int32": 4, "uint32": 4, "float": 4, "radians": 4}
_TYPE_STRUCT = {"int32": "i", "uint32": "I", "float": "f", "radians": "f"}


class XvfController:
    """Lee SPENERGY del XVF3800 vía pyusb y mantiene un pico móvil para gatear.

    El device se puede inyectar (tests) o se descubre con pyusb (producción).
    Todas las operaciones son fail-open.
    """

    def __init__(
        self,
        vid: int = XVF_VID,
        pid: int = XVF_PID,
        poll_interval_s: float = 0.04,
        window_s: float = 3.0,
        read_timeout_ms: int = 200,
        max_retries: int = 20,
        device=None,
    ):
        self.vid = vid
        self.pid = pid
        self.poll_interval_s = poll_interval_s
        self.window_s = window_s
        self.read_timeout_ms = read_timeout_ms
        self.max_retries = max_retries
        self._dev = device  # inyectable para tests
        self._samples: deque[tuple[float, float]] = deque()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._running = False

    # ---- USB device ----
    def open(self) -> bool:
        """Descubre y abre el XVF3800. True si hay device usable. Fail-open."""
        if self._dev is not None:
            return True
        try:
            import usb.core
        except Exception as e:  # pyusb no instalado
            logger.warning(f"pyusb no disponible — SPENERGY gate OFF: {e}")
            return False
        try:
            dev = usb.core.find(idVendor=self.vid, idProduct=self.pid)
        except Exception as e:  # sin permisos / error de backend
            logger.warning(f"XVF3800 no accesible (¿udev rule?) — SPENERGY gate OFF: {e}")
            return False
        if dev is None:
            logger.warning("XVF3800 no encontrado — SPENERGY gate OFF")
            return False
        self._dev = dev
        return True

    @staticmethod
    def _spec(name: str) -> tuple[int, int, int, str, str]:
        spec = PARAMETERS.get(name)
        if spec is None:
            raise ValueError(f"parámetro desconocido: {name!r}")
        return spec

    @staticmethod
    def _decode(dtype: str, count: int, raw: bytes) -> tuple:
        if dtype == "uint8":
            return tuple(int(b) for b in raw[:count])
        fmt = f"<{count}{_TYPE_STRUCT[dtype]}"
        return struct.unpack(fmt, raw[: count * 4])

    @staticmethod
    def _encode(dtype: str, values: list) -> bytes:
        if dtype == "uint8":
            return b"".join(int(v).to_bytes(1, "little") for v in values)
        fmt = f"<{len(values)}{_TYPE_STRUCT[dtype]}"
        if _TYPE_STRUCT[dtype] in ("i", "I"):
            return struct.pack(fmt, *(int(v) for v in values))
        return struct.pack(fmt, *(float(v) for v in values))

    def read_param(self, name: str) -> tuple | None:
        """Lee un parámetro por nombre (ver ``PARAMETERS``).

        Returns:
            Tupla de valores decodificados según el tipo del parámetro, o
            None ante cualquier fallo de hardware/USB (fail-open).

        Raises:
            ValueError: Si el nombre no está en ``PARAMETERS`` (error de
                programación, no de hardware).
        """
        resid, cmdid, count, _rw, dtype = self._spec(name)
        read_len = count * _TYPE_SIZE[dtype] + 1  # + 1 byte de status
        dev = self._dev
        if dev is None:
            if not self.open():
                return None
            dev = self._dev
        try:
            attempts = 0
            while attempts < self.max_retries:
                resp = dev.ctrl_transfer(
                    _BM_REQUEST_READ, 0, 0x80 | cmdid, resid, read_len, self.read_timeout_ms
                )
                status = resp[0]
                if status == _CONTROL_SUCCESS:
                    return self._decode(dtype, count, bytes(resp[1:]))
                if status != _SERVICER_COMMAND_RETRY:
                    logger.debug(f"read_param({name}) status desconocido: {status}")
                    return None
                attempts += 1
                time.sleep(0.005)
            return None
        except Exception as e:
            logger.debug(f"read_param({name}) fail-open: {e}")
            return None

    def write_param(self, name: str, values) -> bool:
        """Escribe un parámetro ``rw`` por nombre, EN RAM del chip.

        Reversible: desenchufar el device restaura la config persistida.
        Nunca toca flash (SAVE_CONFIGURATION no está expuesto a propósito).

        Args:
            name: Clave de ``PARAMETERS`` con rw="rw".
            values: Secuencia con exactamente ``count`` valores del tipo
                del parámetro.

        Returns:
            True si el control transfer se emitió; False ante fallo de
            hardware/USB (fail-open operativo — el chip queda como estaba).

        Raises:
            ValueError: Parámetro desconocido, read-only o count incorrecto
                (errores de programación, no de hardware).
        """
        resid, cmdid, count, rw, dtype = self._spec(name)
        if rw == "ro":
            raise ValueError(f"{name} es read-only")
        values = list(values)
        if len(values) != count:
            raise ValueError(f"{name} espera {count} valores, recibió {len(values)}")
        payload = self._encode(dtype, values)  # errores de tipo explotan acá
        dev = self._dev
        if dev is None:
            if not self.open():
                return False
            dev = self._dev
        try:
            dev.ctrl_transfer(
                _BM_REQUEST_WRITE, 0, cmdid, resid, payload, self.read_timeout_ms
            )
            logger.info(
                f"XvfController: write {name}={values} (RAM — reversible al re-enchufar)"
            )
            return True
        except Exception as e:
            logger.warning(f"write_param({name}) fail-open: {e}")
            return False

    def read_spenergy(self) -> tuple[float, ...] | None:
        """Lee los 4 floats de AEC_SPENERGY_VALUES. None ante cualquier fallo."""
        return self.read_param("AEC_SPENERGY_VALUES")

    # ---- poller ----
    def start(self) -> bool:
        """Arranca el thread de polling. True si quedó corriendo. Fail-open."""
        if self._thread is not None:
            return True
        if self._dev is None and not self.open():
            return False
        self._running = True
        self._thread = threading.Thread(
            target=self._poll_loop, daemon=True, name="xvf-spenergy"
        )
        self._thread.start()
        logger.info("XvfController: poller SPENERGY iniciado")
        return True

    def stop(self) -> None:
        self._running = False
        t = self._thread
        if t is not None:
            t.join(timeout=1.0)
        self._thread = None

    def _poll_loop(self) -> None:
        while self._running:
            vals = self.read_spenergy()
            if vals is not None:
                self._record(vals[_AUTO_BEAM_IDX], time.time())
            time.sleep(self.poll_interval_s)

    def _record(self, value: float, now: float) -> None:
        with self._lock:
            self._samples.append((now, value))
            cutoff = now - self.window_s
            while self._samples and self._samples[0][0] < cutoff:
                self._samples.popleft()

    def peak_since(self, since_ts: float) -> float | None:
        """Máximo SPENERGY[3] con timestamp >= since_ts.

        None si no hay muestras en la ventana (gate fail-open: el caller debe
        procesar el comando cuando recibe None).
        """
        with self._lock:
            vals = [v for (ts, v) in self._samples if ts >= since_ts]
        return max(vals) if vals else None
