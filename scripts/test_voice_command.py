"""
End-to-end test: mic (ReSpeaker) -> Whisper STT -> Chroma search -> HA action.

Uso:
  python scripts/test_voice_command.py --dashboard-url "$KZA_DASHBOARD_URL" --no-execute
  python scripts/test_voice_command.py --dashboard-url "$KZA_DASHBOARD_URL" --duration 4 --device 9 --once

Configurar KZA_DASHBOARD_URL y KZA_DASHBOARD_TOKEN (admin:manage).
Para ejecutar acciones, configurar además KZA_DASHBOARD_ACTION_TOKEN
(home:control). Captura y acción usan credenciales independientes del Dashboard.

Press-to-talk (sin wake word) para validación inicial del pipeline.
"""

import argparse
import logging
import os
import re
import sys
import time
import uuid
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("voice_test")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Credentials are selected explicitly by main; importing never reads .env.


# Regex de intent: verbos en español para turn_on/turn_off.
# Incluye tú (apaga/prende), vos (apagá/prendé), infinitivo (apagar/prender).
_RE_TURN_OFF = re.compile(r"\b(apag[aá]r?|cort[aá]r?|corte[a-z]*)\b", re.IGNORECASE)
_RE_TURN_ON = re.compile(r"\b(prend[eé]r?|encend[eé]r?|enciendo|ilumin[aá]r?|activ[aá]r?)\b", re.IGNORECASE)


def classify_intent(text: str) -> str | None:
    """Clasifica intent por léxico (antes del vector search). Retorna 'turn_on', 'turn_off', o None."""
    if _RE_TURN_OFF.search(text):
        return "turn_off"
    if _RE_TURN_ON.search(text):
        return "turn_on"
    return None


def call_ha_service(
    domain: str, service: str, entity_id: str, *,
    action_client=None, request_id: str | None = None,
):
    """Submit one typed intent under a separate original home authority."""
    from src.audio.manual_command_action import ManualVoiceActionClient

    if not isinstance(action_client, ManualVoiceActionClient):
        raise RuntimeError("manual_voice_action_authority_required")
    identity = request_id if request_id is not None else uuid.uuid4().hex
    return action_client.execute(domain, service, entity_id, request_id=identity)


def record(
    duration: float, device: int, fs: int = 16000, *, authority_factory=None,
):
    """Record one authorized 6ch input; transfer its unchanged mono view."""
    from src.audio.manual_command_capture import ManualCommandCapture

    capture = ManualCommandCapture(authority_factory)
    mono = primary = cleanup = None
    try:
        capture.start()
        with capture.operation("main"):
            mono = _record(capture, duration, device, fs)
    except BaseException as error:
        primary = _fresh_error(error)
    finally:
        try:
            capture.finish()
        except BaseException as error:
            cleanup = _fresh_error(error)
    if primary is not None and cleanup is not None:
        raise BaseExceptionGroup("manual_command_failed", [primary, cleanup]) from None
    if primary is not None:
        raise primary from None
    if cleanup is not None:
        raise cleanup from None
    return mono


def _fresh_error(error):
    """Copy only safe failure/control information, never an SDK traceback."""
    from src.audio.manual_command_capture import ManualCommandStopped
    from src.audio.zone_capture import ZoneCaptureStopped

    if isinstance(error, BaseExceptionGroup):
        return BaseExceptionGroup("manual_command_failed", [
            _fresh_error(child) for child in error.exceptions
        ])
    if isinstance(error, KeyboardInterrupt):
        return KeyboardInterrupt()
    if isinstance(error, SystemExit):
        code = error.code
        return SystemExit(code if code is None or type(code) is int else 1)
    if isinstance(error, ZoneCaptureStopped):
        return ManualCommandStopped("manual_command_stopped")
    if (type(error) is RuntimeError and len(error.args) == 1
            and type(error.args[0]) is str and error.args[0] in {
                "manual_command_authority_required",
                "manual_voice_action_authority_required",
            }):
        return RuntimeError(error.args[0])
    if isinstance(error, Exception):
        return RuntimeError("manual_command_failed")
    return BaseException("manual_command_failed")


def _record(capture, duration: float, device: int, fs: int = 16000):
    import numpy as np
    import sounddevice as sd

    audio_6ch = mono = stream = None
    primary = cleanup = None
    key = None
    with capture.operation("record"):
        try:
            capture.publish(logger.info, "Grabando %ss en device %s...", duration, device)
            key, stream = capture.open_record_stream(lambda: sd.InputStream(
                samplerate=fs, channels=6, device=device, dtype="float32",
            ))
            try:
                capture.require_current()
                audio_6ch, _overflowed = stream.read(int(duration * fs))
            except BaseException as error:
                primary = _fresh_error(error)
            finally:
                stream = None
                close_key, key = key, None
                try:
                    capture.close_record_stream(close_key)
                except BaseException as error:
                    cleanup = _fresh_error(error)
            if primary is not None or cleanup is not None:
                # No raw SDK exception or PCM alias enters the fresh traceback.
                audio_6ch = mono = stream = None
                _overflowed = None
                if primary is not None and cleanup is not None:
                    raise BaseExceptionGroup(
                        "manual_command_record_failed", [primary, cleanup],
                    ) from None
                raise (primary if primary is not None else cleanup) from None
            capture.require_current()
            mono = audio_6ch[:, 0]
            rms = float(np.sqrt(np.mean(mono**2)))
            capture.publish(
                logger.info, "Grabación: %s samples, RMS=%.4f", len(mono), rms,
            )
            if rms < 0.001:
                capture.publish(
                    logger.warning, "¡Audio casi en silencio! Verificá ganancia del mic.",
                )
            return mono
        finally:
            audio_6ch = mono = stream = None
            _overflowed = None
            if key is not None:
                capture.close_record_stream(key)


def _authority_factory(args):
    from src.audio.native_capture_authority import ManualVoiceCommandAuthorityClient

    if not args.capture_url or not args.capture_token_env:
        raise RuntimeError("manual_command_authority_required")
    token = os.environ.get(args.capture_token_env)
    if not token:
        raise RuntimeError("manual_command_authority_required")
    return lambda: ManualVoiceCommandAuthorityClient(args.capture_url, token)


def _action_factory(args):
    from src.audio.manual_command_action import ManualVoiceActionClient

    if not args.capture_url or not args.action_token_env:
        raise RuntimeError("manual_voice_action_authority_required")
    token = os.environ.get(args.action_token_env)
    if not token:
        raise RuntimeError("manual_voice_action_authority_required")
    return lambda: ManualVoiceActionClient(args.capture_url, token)


def _action_confirmed(result) -> bool:
    return (
        result.status == "confirmed" and result.dispatch == "accepted"
        and result.effect == "observed" and result.observation_scope == "ha_state"
    )


def _run_turn(capture, args, stt, embedder, col, action_client):
    audio = segments = _info = text = q_emb = where = res = None
    top_meta = meta = doc = entity_id = domain = service = result = None
    try:
        t_total = time.time()
        audio = _record(capture, args.duration, args.device)
        capture.require_current()
        t0 = time.time()
        segments, _info = capture.transcribe(
            stt, audio, language="es", vad_filter=True, beam_size=1,
        )
        text = " ".join(s.text for s in segments).strip()
        t_stt = (time.time() - t0) * 1000
        capture.publish(logger.info, f"STT [{t_stt:.0f}ms]: {text!r}")

        if not text:
            capture.publish(logger.warning, "Transcripción vacía. Reintentá.")
            return 1 if args.once else None

        intent = classify_intent(text)
        if intent:
            capture.publish(logger.info, f"Intent léxico: {intent}")

        capture.require_current()
        t0 = time.time()
        q_emb = embedder.encode(text).tolist()
        where = {"service": intent} if intent else None
        capture.require_current()
        res = col.query(
            query_embeddings=[q_emb],
            n_results=3,
            where=where,
            include=["metadatas", "distances", "documents"],
        )
        t_search = (time.time() - t0) * 1000

        if not res["ids"][0]:
            capture.publish(logger.warning, f"Sin resultados en Chroma [{t_search:.0f}ms]")
            return 1 if args.once else None

        capture.publish(logger.info, f"Vector search [{t_search:.0f}ms] — top 3:")
        for i in range(len(res["ids"][0])):
            dist = res["distances"][0][i]
            sim = 1 - (dist / 2)
            meta = res["metadatas"][0][i]
            doc = res["documents"][0][i]
            capture.publish(
                logger.info,
                f"  [{i+1}] sim={sim:.3f} | {meta['entity_id']} {meta['service']} | {doc!r}",
            )

        top_dist = res["distances"][0][0]
        top_sim = 1 - (top_dist / 2)
        top_meta = res["metadatas"][0][0]

        if top_sim < args.threshold:
            capture.publish(
                logger.warning,
                f"Similitud {top_sim:.3f} < threshold {args.threshold}. No ejecutando.",
            )
            return 1 if args.once else None

        entity_id = top_meta["entity_id"]
        domain = top_meta["domain"]
        service = top_meta["service"]
        capture.publish(logger.info, f"MATCH: {domain}.{service}({entity_id})")

        if args.no_execute:
            capture.publish(logger.info, "(--no-execute activo, no llamo HA)")
        else:
            capture.require_current()
            t0 = time.time()
            result = call_ha_service(
                domain, service, entity_id, action_client=action_client,
                request_id=uuid.uuid4().hex,
            )
            t_ha = (time.time() - t0) * 1000
            if not _action_confirmed(result):
                capture.publish(logger.error, "HA no confirmado: %s", result.status)
                return 1
            capture.publish(logger.info, f"HA OK [{t_ha:.0f}ms]: {service} sobre {entity_id}")

        t_end = (time.time() - t_total) * 1000
        capture.publish(logger.info, f"Total end-to-end: {t_end:.0f}ms")
        return 0 if args.once else None
    finally:
        audio = segments = _info = text = q_emb = where = res = None
        top_meta = meta = doc = entity_id = domain = service = result = None
        stt = embedder = col = action_client = None
        capture.close_segments()


def main(argv=None):
    from src.audio.manual_command_capture import ManualCommandCapture

    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=5.0)
    ap.add_argument("--device", type=int, default=9)
    ap.add_argument("--whisper-model", default=str(ROOT / "models/whisper-v3-turbo"))
    ap.add_argument("--chroma-path", default=str(ROOT / "data/chroma_db"))
    ap.add_argument("--threshold", type=float, default=0.55, help="Similitud mínima (0-1)")
    ap.add_argument("--no-execute", action="store_true", help="No llamar HA, solo transcribir+buscar")
    ap.add_argument("--once", action="store_true", help="Una sola iteración (no loop)")
    ap.add_argument("--embedder-device", default="cuda:0",
                    help="Device para BGE-M3. Default cuda:0 (cuda:1 está ocupado por vLLM)")
    ap.add_argument(
        "--dashboard-url", "--capture-url", dest="capture_url", required=True,
        help="Dashboard explícito para captura y acciones tipadas",
    )
    ap.add_argument(
        "--capture-token-env", "--token-env", default="KZA_DASHBOARD_TOKEN",
        help="Variable con credencial original admin:manage",
    )
    ap.add_argument(
        "--action-token-env", default="KZA_DASHBOARD_ACTION_TOKEN",
        help="Variable explícita con autoridad home:control",
    )
    args = ap.parse_args(argv)

    capture = ManualCommandCapture(None)
    stt = embedder = client = col = action_client = None
    authority_factory = action_factory = None
    primary = cleanup = None
    exit_code = 1
    try:
        authority_factory = _authority_factory(args)
        capture._authority_factory = authority_factory
        authority_factory = None
        capture.start()
        with capture.operation("main"):
            try:
                if not args.no_execute:
                    action_factory = _action_factory(args)
                    action_client = capture.construct("action", action_factory, close=True)
                    action_factory = None

                capture.publish(
                    logger.info, f"Cargando Whisper desde {args.whisper_model} (cuda:0)...",
                )
                from faster_whisper import WhisperModel
                stt = capture.construct("whisper", lambda: WhisperModel(
                    args.whisper_model, device="cuda", device_index=0, compute_type="float16",
                ))
                capture.publish(logger.info, "Whisper cargado")

                capture.publish(logger.info, f"Cargando BGE-M3 ({args.embedder_device})...")
                from sentence_transformers import SentenceTransformer
                embedder = capture.construct("embedder", lambda: SentenceTransformer(
                    "BAAI/bge-m3", device=args.embedder_device,
                ))

                import chromadb
                from chromadb.config import Settings
                client = capture.construct("chroma", lambda: chromadb.PersistentClient(
                    path=args.chroma_path, settings=Settings(anonymized_telemetry=False),
                ), close=True)
                capture.require_current()
                # U3b-0d2 R5: the collection the runtime reads (pointer-resolved).
                from src.vectordb.command_index import open_active_commands
                col, active = open_active_commands(client, create_base=False)
                if active.degraded:
                    capture.publish(logger.error, f"Índice de comandos degradado ({active.degraded})")
                capture.require_current()
                count = col.count()
                capture.publish(logger.info, f"Chroma: {count} frases indexadas")
                if count == 0:
                    capture.publish(
                        logger.error,
                        "Chroma vacío. Corré primero: scripts/sync_ha_to_chroma.py --entity light.escritorio",
                    )
                else:
                    while True:
                        try:
                            entered = capture.wait_for_enter(
                                "\n>>> Enter para grabar (Ctrl-C para salir)...",
                            )
                        except KeyboardInterrupt:
                            entered = False
                        if not entered:
                            exit_code = 0
                            break
                        turn = _run_turn(capture, args, stt, embedder, col, action_client)
                        if turn is not None:
                            exit_code = turn
                            break
            except BaseException as error:
                primary = _fresh_error(error)
            finally:
                stt = embedder = client = col = action_client = None
                authority_factory = action_factory = None
                try:
                    capture.close_segments()
                finally:
                    capture.release_models()
    except BaseException as error:
        primary = _fresh_error(error)
    finally:
        stt = embedder = client = col = action_client = None
        authority_factory = action_factory = None
        try:
            capture.finish()
        except BaseException as error:
            cleanup = _fresh_error(error)
    if primary is not None and cleanup is not None:
        error = BaseExceptionGroup("manual_command_failed", [primary, cleanup])
    else:
        error = primary if primary is not None else cleanup
    if error is not None:
        if isinstance(error, Exception):
            logger.error("Diagnóstico no completado: %s", type(error).__name__)
            return 1
        raise error from None
    return exit_code


if __name__ == "__main__":
    sys.exit(main() or 0)
