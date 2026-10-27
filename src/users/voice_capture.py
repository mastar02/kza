"""Bounded local listening samples that never outlive their source capture."""

from dataclasses import dataclass,field
from io import BytesIO
import wave

import numpy as np


@dataclass(frozen=True,repr=False)
class VoiceCaptureReview:
    """Own one six-second maximum WAV and bounded source-level review text."""

    audio_wav: bytes = field(repr=False)
    transcript: str = field(repr=False)
    room_id: str | None
    duration_s: float
    clip_duration_s: float

    def __post_init__(self):
        if (type(self.audio_wav) is not bytes or not 44<len(self.audio_wav)<=192044
                or type(self.transcript) is not str or len(self.transcript)>8192
                or self.room_id is not None and (type(self.room_id) is not str or not 1<=len(self.room_id)<=128)
                or type(self.duration_s) not in (int,float) or not .8<=self.duration_s<=30
                or type(self.clip_duration_s) not in (int,float) or not .8<=self.clip_duration_s<=min(6.,self.duration_s)):
            raise ValueError('invalid_voice_review_capture')
        try:
            with wave.open(BytesIO(self.audio_wav),'rb') as reader:
                if (reader.getnchannels()!=1 or reader.getsampwidth()!=2 or reader.getframerate()!=16000
                        or reader.getcomptype()!='NONE' or reader.getnframes()!=round(self.clip_duration_s*16000)
                        or len(reader.readframes(reader.getnframes()))!=reader.getnframes()*2):
                    raise ValueError('invalid_voice_review_capture')
        except (wave.Error,EOFError):
            raise ValueError('invalid_voice_review_capture') from None


def build_review_capture(audio: np.ndarray, transcript: str, room_id: str | None) -> VoiceCaptureReview:
    """Encode the first six seconds off-loop, preserving the full segment duration."""
    if (type(audio) is not np.ndarray or audio.dtype!=np.float32 or audio.ndim!=1
            or not 12800<=audio.size<=480000 or not np.isfinite(audio).all()):
        raise ValueError('invalid_voice_review_capture')
    pcm = output = None
    try:
        # PCM16 is the listening preview, independently of the model's float32 input.
        pcm = np.rint(np.clip(audio[:96000],-1.,1.)*32767).astype('<i2').tobytes()
        output = BytesIO()
        with wave.open(output,'wb') as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(16000)
            writer.writeframes(pcm)
        return VoiceCaptureReview(output.getvalue(),transcript,room_id,audio.size/16000,len(pcm)/32000)
    finally:
        audio = transcript = pcm = output = None
