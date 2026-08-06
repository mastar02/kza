"""AudioArchiver — persistencia FLAC del audio de cada segmento del ambient.

Existe SOLO para la campaña de medición de fidelidad (spec 2026-08-05): sin el
audio no hay referencia posible y no se puede calcular WER ni re-transcribir.
Apagado por default.

Best-effort por diseño: cualquier fallo devuelve None. Los fallos de escritura
se loguean siempre, el rechazo por piso de disco a lo sumo una vez por hora, y
enabled=False retorna None en silencio. NUNCA propaga — el pipeline de voz no
se cae porque no se pudo guardar un FLAC.
"""
from __future__ import annotations

import asyncio
import logging
import shutil
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_DISK_WARN_INTERVAL_S = 3600.0


class AudioArchiver:
    """Escribe el audio de un segmento a FLAC mono, best-effort."""

    def __init__(
        self,
        base_dir: str,
        enabled: bool = False,
        sample_rate: int = 16000,
        min_free_bytes: int = 1_000_000_000,
    ):
        self.base_dir = Path(base_dir)
        self.enabled = enabled
        self.sample_rate = sample_rate
        self.min_free_bytes = min_free_bytes
        self._last_disk_warn = 0.0
        self.stats = {"written": 0, "skipped_disk": 0, "failed": 0}

    async def write(
        self, room_id: str, utt_id: int, audio: np.ndarray
    ) -> str | None:
        """Guardar el audio de una utterance.

        Args:
            room_id: Habitación de origen (define el subdirectorio).
            utt_id: rowid de la utterance; da el nombre del archivo.
            audio: (n_samples, n_channels) float32, o (n_samples,) mono.

        Returns:
            La ruta del archivo escrito, o None si está deshabilitado, no
            hay disco por encima del piso, o falló la escritura. Los tres
            casos se distinguen en ``self.stats`` (skipped_disk / failed;
            deshabilitado no cuenta) y los dos últimos se loguean — el
            rechazo por piso de disco como máximo una vez por hora.
        """
        if not self.enabled:
            return None
        try:
            mono = audio[:, 0] if audio.ndim == 2 else audio
            if mono.size == 0:
                raise ValueError("audio vacío")
            path = self.base_dir / room_id / f"{utt_id}.flac"
            # El chequeo de disco (stat/statvfs) va DENTRO del thread: en el
            # loop bloqueaba el voice path una vez por segmento (review
            # PR #14, 2026-08-06).
            written = await asyncio.to_thread(self._write_sync, path, mono)
            if not written:
                self.stats["skipped_disk"] += 1
                return None
            self.stats["written"] += 1
            return str(path)
        except Exception as e:
            self.stats["failed"] += 1
            logger.warning(
                "AudioArchiver: no se pudo guardar %s/%d (%s)", room_id, utt_id, e
            )
            return None

    def _write_sync(self, path: Path, mono: np.ndarray) -> bool:
        """Escritura bloqueante — corre en un hilo aparte.

        Returns:
            False si el disco está debajo del piso (no se escribió nada),
            True si el archivo quedó en disco.
        """
        import soundfile as sf

        if not self._has_room():
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(path), mono, self.sample_rate, format="FLAC")
        return True

    def _has_room(self) -> bool:
        """¿Queda espacio por encima del piso configurado?"""
        try:
            probe = self.base_dir if self.base_dir.exists() else self.base_dir.parent
            while not probe.exists() and probe != probe.parent:
                probe = probe.parent
            free = shutil.disk_usage(probe).free
        except OSError:
            return True  # si no se puede medir, no bloquear la escritura
        if free >= self.min_free_bytes:
            return True
        now = time.time()
        if now - self._last_disk_warn > _DISK_WARN_INTERVAL_S:
            self._last_disk_warn = now
            logger.warning(
                "AudioArchiver: %d MB libres < piso %d MB — audio no archivado",
                free // 1_000_000, self.min_free_bytes // 1_000_000,
            )
        return False
