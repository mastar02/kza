"""AudioArchiver — persistencia FLAC del audio de cada segmento del ambient.

Existe SOLO para la campaña de medición de fidelidad (spec 2026-08-05): sin el
audio no hay referencia posible y no se puede calcular WER ni re-transcribir.
Apagado por default.

Best-effort por diseño: cualquier fallo devuelve None. Los fallos de escritura
se loguean siempre, el rechazo por piso de disco ~una vez por hora
(best-effort entre threads: `_last_disk_warn` se muta desde threads del pool
sin lock, así que la ventana puede duplicar bajo carrera — no es estricto), y
enabled=False retorna None en silencio. NUNCA propaga — el pipeline de voz no
se cae porque no se pudo guardar un FLAC.
"""
from __future__ import annotations

import asyncio
import logging
import os
import shutil
import time
from collections.abc import Awaitable, Callable
from dataclasses import replace
from pathlib import Path

import numpy as np

from src.memory.fact_ledger import MemorySource

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
        discard_audio: Callable[[str], Awaitable[None]] | None = None,
        privacy=None,
        store=None,
        capacity: int = 8,
    ):
        self.base_dir = Path(base_dir)
        self.enabled = enabled
        self.sample_rate = sample_rate
        self.min_free_bytes = min_free_bytes
        self._discard_audio = discard_audio
        if type(capacity) is not int or capacity < 1:
            raise ValueError('invalid_archive_capacity')
        self._privacy, self._store, self._capacity = privacy, store, capacity
        self._writes: dict[asyncio.Task, tuple[MemorySource | None, str]] = {}
        self._calls: dict[asyncio.Future, MemorySource | None] = {}
        self._closed = False
        self._closing: asyncio.Task | None = None
        self._last_disk_warn = 0.0
        self.stats = {"written": 0, "skipped_disk": 0, "failed": 0}

    async def write(
        self, room_id: str, utt_id: int, audio: np.ndarray, *, source: MemorySource | None = None,
    ) -> str | None:
        """Bound and retain reservation, encoding and cancellation cleanup together."""
        if not self.enabled or self._closed:
            return None
        if len(self._calls) >= self._capacity:
            self.stats['failed'] += 1
            return None
        completion = asyncio.get_running_loop().create_future()
        self._calls[completion] = source
        try:
            return await self._write(room_id, utt_id, audio, source=source)
        finally:
            self._calls.pop(completion, None)
            completion.set_result(None)

    async def _write(
        self, room_id: str, utt_id: int, audio: np.ndarray, *, source: MemorySource | None = None,
    ) -> str | None:
        """Guardar el audio de una utterance.

        Args:
            room_id: Habitación de origen (define el subdirectorio).
            utt_id: rowid de la utterance; da el nombre del archivo.
            audio: (n_samples, n_channels) float32, o (n_samples,) mono.

        Returns:
            La ruta del archivo escrito, o None si está deshabilitado, no
            hay disco por encima del piso, o falló la escritura.
            ``skipped_disk`` y ``failed`` se distinguen en ``self.stats`` y
            ambos se loguean (el rechazo por piso de disco como máximo una
            vez por hora); deshabilitado retorna None sin contar ni loguear.
        """
        if not self.enabled or self._closed:
            return None
        if self._privacy is not None and (source is None or not self._privacy.sources_available((source,))):
            return None
        try:
            mono = audio[:, 0] if audio.ndim == 2 else audio
            if mono.size == 0:
                raise ValueError("audio vacío")
            path = self.base_dir / room_id / f"{utt_id}.flac"
            if self._privacy is not None:
                if self._store is None:
                    raise RuntimeError('archive_requires_capture_store')
                path = self.base_dir / 'policy_v1' / f'{source.capture_id}.flac'
                reserved = await self._store.reserve_audio_path(utt_id, str(path), source)
                if reserved is None:
                    return None
                path = Path(reserved.path)
                source = replace(source, expires_at=reserved.expires_at)
                # No await between revalidation, admission and native ownership.
                if self._closed or not self._privacy.sources_available((source,)):
                    return None
            if len(self._writes) >= self._capacity:
                raise RuntimeError('archive_capacity')
            # El chequeo de disco (stat/statvfs) va DENTRO del thread: en el
            # loop bloqueaba el voice path una vez por segmento (review
            # PR #14, 2026-08-06).
            def encode():
                if self._privacy is not None and not self._privacy.sources_available((source,)):
                    return None
                return self._write_sync(path, mono)
            writing = asyncio.create_task(asyncio.to_thread(encode))
            self._writes[writing] = (source, str(path))
            def completed(task):
                self._writes.pop(task, None)
                if not task.cancelled():
                    task.exception()
            writing.add_done_callback(completed)
            cancelled = False
            while not writing.done():
                try:
                    await asyncio.shield(writing)
                except asyncio.CancelledError:
                    cancelled = True
                except Exception:
                    break  # Retrieve the actual write outcome below.
            error = writing.exception()
            revoked = self._closed or self._privacy is not None and not self._privacy.sources_available((source,))
            if cancelled or error is not None or revoked:
                cleanup = asyncio.create_task(self._discard(str(path)))
                while not cleanup.done():
                    try:
                        await asyncio.shield(cleanup)
                    except asyncio.CancelledError:
                        cancelled = True
                    except Exception:
                        break
                if cancelled:
                    if not cleanup.cancelled() and cleanup.exception() is not None:
                        logger.warning("AudioArchiver cancelled-write cleanup failed: %s", type(cleanup.exception()).__name__)
                    raise asyncio.CancelledError
                cleanup.result()
                if error is not None:
                    raise error
                return None
            written = writing.result()
            if not written:
                self.stats["skipped_disk"] += 1
                return None
            self.stats["written"] += 1
            return str(path)
        except Exception as e:
            self.stats["failed"] += 1
            logger.warning(
                "AudioArchiver write failed: %s", type(e).__name__
            )
            return None

    def is_writing(self, path: str) -> bool:
        """Report actual encoders, without filesystem I/O or clip content."""
        return any(saved_path == path and not task.done() for task, (_, saved_path) in self._writes.items())

    async def drain_conversation(self, conversation_id: str) -> None:
        """Wait actual pre-revocation encoders; observation cancellation is shielded."""
        tasks = [task for task, source in self._calls.items()
                 if source is not None and source.conversation_id == conversation_id]
        await asyncio.gather(*(asyncio.shield(task) for task in tasks), return_exceptions=True)

    def begin_close(self) -> asyncio.Task:
        """Close native admission and retain its real drain for the privacy lease."""
        self._closed = True
        if self._closing is None:
            async def finish():
                await asyncio.gather(*(asyncio.shield(task) for task in list(self._calls)), return_exceptions=True)
            self._closing = asyncio.create_task(finish())
        return self._closing

    async def _discard(self, path: str) -> None:
        if self._discard_audio is not None:
            await self._discard_audio(path)
        else:
            await asyncio.to_thread(Path(path).unlink, missing_ok=True)

    def _write_sync(self, path: Path, mono: np.ndarray) -> bool:
        """Escritura bloqueante — corre en un hilo aparte.

        Returns:
            False si el disco está debajo del piso (no se escribió nada),
            True si el archivo quedó en disco.
        """
        import soundfile as sf

        if not self._has_room():
            return False
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        descriptor = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, 'wb', closefd=False) as output:
                sf.write(output, mono, self.sample_rate, format="FLAC")
        finally:
            os.close(descriptor)
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
