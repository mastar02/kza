"""Bounded listening preview owned by the voice registry, not a permanent recording."""
from dataclasses import dataclass, field
from io import BytesIO
import wave

import numpy as np

from src.evaluation.provenance import CaptureProvenance, FrameInterval
from src.users.voice_capture_metadata import encode_audio_metadata, encode_transcript_metadata


def _preview_frame_count(audio_wav: bytes) -> int | None:
    """Read a canonical mono PCM16/16 kHz WAV byte count, without float inference."""
    try:
        if type(audio_wav) is not bytes or not 25644 <= len(audio_wav) <= 192044:
            return None
        if ((len(audio_wav) - 44) % 2 or audio_wav[:4] != b'RIFF'
                or int.from_bytes(audio_wav[4:8], 'little') != len(audio_wav) - 8
                or audio_wav[8:16] != b'WAVEfmt '
                or int.from_bytes(audio_wav[16:20], 'little') != 16
                or int.from_bytes(audio_wav[20:22], 'little') != 1
                or int.from_bytes(audio_wav[22:24], 'little') != 1
                or int.from_bytes(audio_wav[24:28], 'little') != 16000
                or int.from_bytes(audio_wav[28:32], 'little') != 32000
                or int.from_bytes(audio_wav[32:34], 'little') != 2
                or int.from_bytes(audio_wav[34:36], 'little') != 16
                or audio_wav[36:40] != b'data'
                or int.from_bytes(audio_wav[40:44], 'little') != len(audio_wav) - 44):
            return None
        return (len(audio_wav) - 44) // 2
    finally:
        audio_wav = None


def validate_voice_preview(audio_wav: bytes, clip_duration_s: float) -> None:
    """Validate one canonical preview without reading text, room or source metadata."""
    frames = _preview_frame_count(audio_wav)
    audio_wav = None
    if (frames is None or type(clip_duration_s) not in (int, float)
            or clip_duration_s != frames / 16000):
        clip_duration_s = None
        raise ValueError('invalid_voice_review_capture')


@dataclass(frozen=True, repr=False)
class VoiceCaptureReview:
    """Own a preview and text with independent, optionally unknown PCM descriptors."""

    audio_wav: bytes = field(repr=False)
    transcript: str = field(repr=False)
    room_id: str | None
    duration_s: float
    clip_duration_s: float
    audio_provenance: CaptureProvenance | None = None
    audio_span: FrameInterval | None = None
    transcript_provenance: CaptureProvenance | None = None
    transcript_span: FrameInterval | None = None
    transcript_truncated: bool | None = None

    def __post_init__(self) -> None:
        invalid = False
        try:
            if (type(self.transcript) is not str or len(self.transcript) > 8192
                    or self.room_id is not None and (type(self.room_id) is not str or not 1 <= len(self.room_id) <= 128)
                    or type(self.duration_s) not in (int, float) or not .8 <= self.duration_s <= 30
                    or type(self.clip_duration_s) not in (int, float)
                    or not .8 <= self.clip_duration_s <= min(6., self.duration_s)):
                raise ValueError('invalid_voice_review_capture')
            validate_voice_preview(self.audio_wav, self.clip_duration_s)
            encode_audio_metadata(self.audio_provenance, self.audio_span)
            encode_transcript_metadata(self.transcript_provenance, self.transcript_span, self.transcript_truncated)
            if self.audio_provenance is not None:
                provenance, span = self.audio_provenance, self.audio_span
                if (provenance.sample_rate_hz != 16000
                        or self.duration_s != provenance.frame_count / 16000
                        or span.start_frame != 0 or span.end_frame != min(provenance.frame_count, 96000)
                        or _preview_frame_count(self.audio_wav) != span.end_frame - span.start_frame
                        or self.clip_duration_s != (span.end_frame - span.start_frame) / 16000):
                    raise ValueError('invalid_voice_review_capture')
            if self.transcript_provenance is not None:
                if (self.transcript_span.start_frame != 0
                        or self.transcript_span.end_frame != self.transcript_provenance.frame_count):
                    raise ValueError('invalid_voice_review_capture')
        except (ValueError, TypeError, AttributeError, OverflowError):
            invalid = True
        finally:
            self = provenance = span = None
        if invalid:
            raise ValueError('invalid_voice_review_capture')


def build_review_capture(audio: np.ndarray, transcript: str, room_id: str | None, *,
                         audio_provenance: CaptureProvenance | None = None,
                         transcript_provenance: CaptureProvenance | None = None,
                         transcript_span: FrameInterval | None = None,
                         transcript_truncated: bool | None = None) -> VoiceCaptureReview:
    """Encode the actual first six seconds; never infer STT lineage from preview PCM."""
    pcm = output = audio_span = None
    invalid = False
    try:
        if (type(audio) is not np.ndarray or audio.dtype != np.float32 or audio.ndim != 1
                or not 12800 <= audio.size <= 480000 or not np.isfinite(audio).all()):
            raise ValueError('invalid_voice_review_capture')
        if audio_provenance is not None:
            if type(audio_provenance) is not CaptureProvenance:
                raise ValueError('invalid_voice_review_capture')
            CaptureProvenance.__post_init__(audio_provenance)
            if audio_provenance.sample_rate_hz != 16000 or audio_provenance.frame_count != audio.size:
                raise ValueError('invalid_voice_review_capture')
            audio_span = FrameInterval(0, min(audio.size, 96000))
            encode_audio_metadata(audio_provenance, audio_span)
        encode_transcript_metadata(transcript_provenance, transcript_span, transcript_truncated)
        if transcript_provenance is not None and (transcript_span.start_frame != 0
                or transcript_span.end_frame != transcript_provenance.frame_count):
            raise ValueError('invalid_voice_review_capture')
        # PCM16 is the listening preview, independently of the model's float32 input.
        pcm = np.rint(np.clip(audio[:96000], -1., 1.) * 32767).astype('<i2').tobytes()
        output = BytesIO()
        with wave.open(output, 'wb') as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(16000)
            writer.writeframes(pcm)
        return VoiceCaptureReview(output.getvalue(), transcript, room_id, audio.size / 16000, len(pcm) / 32000,
                                  audio_provenance, audio_span, transcript_provenance,
                                  transcript_span, transcript_truncated)
    except (ValueError, TypeError, AttributeError, OverflowError):
        invalid = True
    finally:
        audio = transcript = pcm = output = audio_span = None
        audio_provenance = transcript_provenance = transcript_span = transcript_truncated = None
    if invalid:
        raise ValueError('invalid_voice_review_capture')
