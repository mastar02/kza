"""Controlador del XVF3800 (vendor interface) para leer SPENERGY = VAD por
hardware, y usarlo como pre-gate antes de Whisper.

El DSP del XVF3800 expone ``AEC_SPENERGY_VALUES`` (4 floats; índice 3 = beam
auto-select): >0 indica voz, 0 = ruido de banda ancha (secador) o silencio. Es
**pre-AGC y 0-GPU**, así que separa la voz del ruido del secador con un margen
enorme (medido 2026-05-31: secador=0, voz=101–2.1M). Correr este gate antes de
Whisper evita las alucinaciones ("Gracias" sobre silencio/secador).

Protocolo (portado del xvf_host.py oficial de Seeed,
``respeaker/reSpeaker_XVF3800_USB_4MIC_ARRAY``): control transfer vendor sobre
Interface 3 (NO toca el audio streaming). AEC_SPENERGY_VALUES = resid 33,
cmd 80, 4 floats. Lectura = ctrl_transfer(IN|VENDOR|DEVICE, wValue=0x80|cmd).

Diseño FAIL-OPEN: cualquier problema (sin device, sin permisos USB, timeout,
error de parseo) hace que el gate quede desactivado (``peak_since`` → None), de
modo que NUNCA se bloquea un comando de voz por un fallo del hardware/USB.
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

# AEC_SPENERGY_VALUES: (resource_id, command_id, count). Ver PARAMETERS en xvf_host.py.
_SPENERGY_RESID = 33
_SPENERGY_CMDID = 80
_SPENERGY_COUNT = 4
_AUTO_BEAM_IDX = 3  # índice 3 = beam auto-select (el más robusto)

# bmRequestType para lectura vendor: CTRL_IN(0x80)|CTRL_TYPE_VENDOR(0x40)|RECIPIENT_DEVICE(0x00)
_BM_REQUEST_READ = 0xC0
# wValue lleva el cmd id con el bit 0x80 (read).
_WVALUE = 0x80 | _SPENERGY_CMDID
_READ_LEN = _SPENERGY_COUNT * 4 + 1  # 4 floats + 1 byte de status

# Códigos de status del primer byte de la respuesta (del firmware XMOS).
_CONTROL_SUCCESS = 0
_SERVICER_COMMAND_RETRY = 64


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

    def read_spenergy(self) -> tuple[float, ...] | None:
        """Lee los 4 floats de AEC_SPENERGY_VALUES. None ante cualquier fallo."""
        dev = self._dev
        if dev is None:
            if not self.open():
                return None
            dev = self._dev
        try:
            attempts = 0
            while attempts < self.max_retries:
                resp = dev.ctrl_transfer(
                    _BM_REQUEST_READ, 0, _WVALUE, _SPENERGY_RESID, _READ_LEN, self.read_timeout_ms
                )
                status = resp[0]
                if status == _CONTROL_SUCCESS:
                    raw = bytes(resp[1:1 + _SPENERGY_COUNT * 4])
                    return struct.unpack("<ffff", raw)
                if status != _SERVICER_COMMAND_RETRY:
                    logger.debug(f"SPENERGY status desconocido: {status}")
                    return None
                attempts += 1
                time.sleep(0.005)
            return None
        except Exception as e:
            logger.debug(f"read_spenergy fail-open: {e}")
            return None

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
