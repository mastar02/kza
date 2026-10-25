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
from dataclasses import dataclass
from enum import Enum

try:
    from usb.core import USBTimeoutError
except ImportError:
    _USB_TIMEOUT_ERRORS: tuple[type[Exception], ...] = ()
else:
    _USB_TIMEOUT_ERRORS = (USBTimeoutError,)

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
    "DOA_VALUE": (20, 18, 2, "ro", "uint16"),
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
    # --- Fase 3 tuning anti-TV (2026-07-16, IDs verificados contra
    # python_control/xvf_host.py oficial @ master) ---
    # Ganancia analógica del array (pre-SHF) — la palanca correcta para "voz
    # bajo el umbral de endpointing" ANTES de forzar el AGC del post-processor.
    # ⚠️ Sin rango oficial declarado → sin validación; 0.0 silencia el mic.
    "AUDIO_MGR_MIC_GAIN": (35, 0, 1, "rw", "float"),
    # Gain-floors de supresión de ruido del post-processor (MENOR = más
    # supresión). MIN_NN (no-estacionario) es la perilla contra habla/música
    # de TV; MIN_NS (estacionario) contra ruido constante.
    "PP_MIN_NS": (17, 21, 1, "rw", "float"),
    "PP_MIN_NN": (17, 22, 1, "rw", "float"),
    # Bajo SPEINDEX Hz se suprime más en double-talk (parámetro de dispositivo,
    # no de gusto — default Seeed 1300.0 vs guía XMOS 593.75: leer antes).
    "PP_FMIN_SPEINDEX": (17, 30, 1, "rw", "float"),
    # ATTNS: reducción EXTRA del gain del AGC durante no-voz. Doma el efecto
    # "AGC amplifica la TV en los silencios" sin apagar el AGC.
    "PP_ATTNS_MODE": (17, 32, 1, "rw", "int32"),
    "PP_ATTNS_NOMINAL": (17, 33, 1, "rw", "float"),
    "PP_ATTNS_SLOPE": (17, 34, 1, "rw", "float"),
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
_TYPE_SIZE = {"uint8": 1, "uint16": 2, "int32": 4, "uint32": 4, "float": 4, "radians": 4}
_TYPE_STRUCT = {"uint16": "H", "int32": "i", "uint32": "I", "float": "f", "radians": "f"}


class XvfReadOutcome(str, Enum):
    """Classify a transport read without inferring firmware support."""

    OK = "ok"
    BUSY = "busy"
    NO_DEVICE = "no_device"
    TIMEOUT = "timeout"
    USB_ERROR = "usb_error"
    DEVICE_STATUS = "device_status"
    INVALID_PAYLOAD = "invalid_payload"
    RETRIES_EXHAUSTED = "retries_exhausted"


@dataclass(frozen=True)
class XvfReadResult:
    """Carry decoded values or a transport failure and the received status."""

    outcome: XvfReadOutcome
    values: tuple[int | float, ...] | None = None
    device_status: int | None = None

# Rangos válidos por parámetro, tomados de las descripciones del xvf_host.py
# oficial ("Valid range: ..."). Solo se valida lo que el oficial declara;
# los uint8 además acotan 0..255 implícito del tipo. Un fat-finger (AGC=0,
# negativo) NO debe llegar al DSP de producción (review Fase 1 2026-06-04).
_VALID_RANGES: dict[str, tuple[float, float]] = {
    "PP_AGCONOFF": (0, 1),
    "PP_AGCMAXGAIN": (1.0, 1000.0),
    "PP_AGCDESIREDLEVEL": (1e-8, 1.0),
    "PP_AGCGAIN": (1.0, 1000.0),
    "AEC_ASROUTONOFF": (0, 1),
    "AEC_ASROUTGAIN": (0.0, 1000.0),
    "AEC_FIXEDBEAMSONOFF": (0, 1),
    "AEC_FIXEDBEAMSGATING": (0, 1),
    # Fase 3 (rangos "Valid range" del xvf_host.py oficial; AUDIO_MGR_MIC_GAIN
    # no declara rango → sin entrada, por la regla de este dict)
    "PP_MIN_NS": (0.0, 1.0),
    "PP_MIN_NN": (0.0, 1.0),
    "PP_FMIN_SPEINDEX": (0.0, 7999.0),
    "PP_ATTNS_MODE": (0, 1),
    "PP_ATTNS_NOMINAL": (0.0, 1.0),
    "PP_ATTNS_SLOPE": (0.0, 5.0),
}


def validate_values(name: str, values: list) -> None:
    """Valida cantidad y rango de valores para un parámetro escribible.

    Reutilizable fuera del controller (CLI valida ANTES de abrir el device).

    Raises:
        ValueError: Parámetro desconocido, cantidad incorrecta o valor fuera
            del rango oficial.
    """
    spec = PARAMETERS.get(name)
    if spec is None:
        raise ValueError(f"parámetro desconocido: {name!r}")
    _resid, _cmdid, count, _rw, dtype = spec
    if len(values) != count:
        raise ValueError(f"{name} espera {count} valores, recibió {len(values)}")
    lo, hi = _VALID_RANGES.get(name, (None, None))
    if dtype == "uint8" and lo is None:
        lo, hi = 0, 255  # acotado por el tipo
    if lo is not None:
        for v in values:
            if not (lo <= v <= hi):
                raise ValueError(
                    f"{name}: valor {v} fuera de rango oficial [{lo} .. {hi}]"
                )


def parse_usb_port(usb_port: str | None) -> tuple[int, tuple[int, ...]] | None:
    """Parsear un puerto USB de sysfs a (bus, cadena de puertos).

    Mismo formato que ``mic_usb_port`` de ``RoomConfig``: ``"5-5.4"`` es el bus
    5, hub en el puerto 5 del root, device en el puerto 4 de ese hub.

    Args:
        usb_port: Puerto en notación sysfs (``"3-1"``, ``"5-5.4"``).

    Returns:
        ``(bus, (puerto, ...))``, o None si no se puede parsear.
    """
    if not usb_port:
        return None
    try:
        bus_str, ports_str = usb_port.split("-", 1)
        return int(bus_str), tuple(int(p) for p in ports_str.split("."))
    except (AttributeError, ValueError):
        return None


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
        usb_port: str | None = None,
    ):
        self.vid = vid
        self.pid = pid
        self.usb_port = usb_port
        self.poll_interval_s = poll_interval_s
        self.window_s = window_s
        self.read_timeout_ms = read_timeout_ms
        self.max_retries = max_retries
        self._dev = device  # inyectable para tests
        self._samples: deque[tuple[float, float]] = deque()
        self._lock = threading.Lock()
        self._device_generation = 0
        # Lock DEDICADO para serializar ctrl_transfer sobre el device handle
        # compartido (poller thread vs write_param desde otro thread). libusb
        # no garantiza transfers síncronos concurrentes sobre un mismo handle
        # (review Fase 1 2026-06-04). NO reusar _lock (es del deque).
        self._usb_lock = threading.Lock()
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
        target = None
        if self.usb_port:
            target = parse_usb_port(self.usb_port)
            if target is None:
                logger.warning(
                    f"XVF3800 usb_port {self.usb_port!r} no parseable — gate OFF"
                )
                return False
        try:
            if target is None:
                dev = usb.core.find(idVendor=self.vid, idProduct=self.pid)
            else:
                bus, ports = target
                dev = usb.core.find(
                    idVendor=self.vid,
                    idProduct=self.pid,
                    custom_match=lambda d: d.bus == bus
                    and tuple(getattr(d, "port_numbers", None) or ()) == ports,
                )
        except Exception as e:  # sin permisos / error de backend
            logger.warning(f"XVF3800 no accesible (¿udev rule?) — SPENERGY gate OFF: {e}")
            return False
        if dev is None:
            # Con usb_port declarado NO caemos al primer XVF3800 disponible:
            # gatear una room con el SPENERGY de OTRA es peor que no gatear.
            where = f" en {self.usb_port}" if self.usb_port else ""
            logger.warning(f"XVF3800 no encontrado{where} — SPENERGY gate OFF")
            return False
        self._dev = dev
        return True

    def invalidate_device(self) -> None:
        """Retire a disconnected handle and cached energy without USB transfers.

        The lifecycle owner drains direction reads before calling this method.
        The next read discovers the current device on this controller's bound port.
        """
        with self._usb_lock:
            self._dev = None
            with self._lock:
                self._device_generation += 1
                self._samples.clear()

    @staticmethod
    def _spec(name: str) -> tuple[int, int, int, str, str]:
        spec = PARAMETERS.get(name)
        if spec is None:
            raise ValueError(f"parámetro desconocido: {name!r}")
        return spec

    @staticmethod
    def _decode(dtype: str, count: int, raw: bytes) -> tuple:
        if len(raw) != count * _TYPE_SIZE[dtype]:
            raise ValueError("invalid payload length")
        if dtype == "uint8":
            return tuple(raw)
        return struct.unpack(f"<{count}{_TYPE_STRUCT[dtype]}", raw)

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
        result = self.read_param_result(
            name, timeout_ms=self.read_timeout_ms, max_attempts=self.max_retries,
        )
        return result.values if result.outcome is XvfReadOutcome.OK else None

    def read_param_result(
        self,
        name: str,
        *,
        timeout_ms: int,
        max_attempts: int,
        skip_if_busy: bool = False,
    ) -> XvfReadResult:
        """Read a parameter with an explicit outcome and bounded retry budget.

        Args:
            name: Key in ``PARAMETERS``.
            timeout_ms: Timeout for each USB transfer in milliseconds.
            max_attempts: Maximum transfers, retrying only device status 64.
            skip_if_busy: Return BUSY instead of waiting for the USB lock.

        Returns:
            Decoded values or a transport outcome, preserving device status.

        Raises:
            ValueError: If the parameter name is unknown.
        """
        resid, cmdid, count, _rw, dtype = self._spec(name)
        read_len = count * _TYPE_SIZE[dtype] + 1
        dev = self._dev
        if dev is None:
            if not self._usb_lock.acquire(blocking=not skip_if_busy):
                return XvfReadResult(XvfReadOutcome.BUSY)
            try:
                if not self.open():
                    return XvfReadResult(XvfReadOutcome.NO_DEVICE)
                dev = self._dev
            finally:
                self._usb_lock.release()
        try:
            status = None
            for _ in range(max_attempts):
                if not self._usb_lock.acquire(blocking=not skip_if_busy):
                    return XvfReadResult(XvfReadOutcome.BUSY)
                try:
                    if dev is not self._dev:
                        return XvfReadResult(XvfReadOutcome.NO_DEVICE)
                    resp = dev.ctrl_transfer(
                        _BM_REQUEST_READ, 0, 0x80 | cmdid, resid, read_len, timeout_ms
                    )
                finally:
                    self._usb_lock.release()
                if not len(resp):
                    return XvfReadResult(XvfReadOutcome.INVALID_PAYLOAD)
                status = resp[0]
                if status == _CONTROL_SUCCESS:
                    if len(resp) != read_len:
                        return XvfReadResult(XvfReadOutcome.INVALID_PAYLOAD, device_status=status)
                    return XvfReadResult(
                        XvfReadOutcome.OK, self._decode(dtype, count, bytes(resp[1:])), status,
                    )
                if status != _SERVICER_COMMAND_RETRY:
                    return XvfReadResult(XvfReadOutcome.DEVICE_STATUS, device_status=status)
                time.sleep(0.005)
            return XvfReadResult(XvfReadOutcome.RETRIES_EXHAUSTED, device_status=status)
        except _USB_TIMEOUT_ERRORS as e:
            logger.debug(f"read_param({name}) timeout: {e}")
            return XvfReadResult(XvfReadOutcome.TIMEOUT)
        except Exception as e:
            logger.debug(f"read_param({name}) fail-open: {e}")
            return XvfReadResult(XvfReadOutcome.USB_ERROR)

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
        validate_values(name, values)  # count + rango oficial
        payload = self._encode(dtype, values)  # errores de tipo explotan acá
        dev = self._dev
        if dev is None:
            if not self.open():
                return False
            dev = self._dev
        try:
            with self._usb_lock:  # serializar vs reads del poller
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
            with self._lock:
                generation = self._device_generation
            vals = self.read_spenergy()
            if vals is not None:
                self._record(vals[_AUTO_BEAM_IDX], time.time(), generation)
            time.sleep(self.poll_interval_s)

    def _record(self, value: float, now: float, generation: int | None = None) -> None:
        with self._lock:
            if generation is not None and generation != self._device_generation:
                return
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
