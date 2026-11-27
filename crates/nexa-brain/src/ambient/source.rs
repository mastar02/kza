use std::collections::HashMap;

use nexa_proto::v1::{AudioUtterance, TranscribeResponse};

const MIN_NO_SPEECH_PROB: f32 = 0.70;
const MIN_AVG_LOGPROB: f32 = -1.10;
const MAX_COMPRESSION_RATIO: f32 = 2.40;
const MEDIA_HA_SPEECH_RATIO: f32 = 0.85;
const MEDIA_HA_DOA_STABILITY: f32 = 0.90;
const MEDIA_STREAK_SPEECH_RATIO: f32 = 0.88;
const MEDIA_STREAK_DOA_STABILITY: f32 = 0.92;
const SAME_DIRECTION_MAX_RAD: f32 = 0.26;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SourceKind {
    Live,
    MediaTv,
    Noise,
}

#[derive(Debug, Default)]
pub(crate) struct TvTracker {
    streaks: HashMap<String, (f32, u32)>,
}

impl TvTracker {
    pub(crate) fn reset(&mut self, room_id: &str) {
        self.streaks.remove(room_id);
    }

    fn observe_static_monologue(&mut self, room_id: &str, azimuth_rad: f32) -> u32 {
        match self.streaks.get_mut(room_id) {
            Some((prev_az, count))
                if circular_angle_diff(*prev_az, azimuth_rad) <= SAME_DIRECTION_MAX_RAD =>
            {
                *count = count.saturating_add(1);
                *prev_az = azimuth_rad;
                *count
            }
            _ => {
                self.streaks.insert(room_id.to_owned(), (azimuth_rad, 1));
                1
            }
        }
    }
}

pub(crate) struct SourceClassifier;

impl SourceClassifier {
    pub(crate) fn classify(
        utterance: &AudioUtterance,
        stt: &TranscribeResponse,
        room_media_playing: bool,
        tv_tracker: &mut TvTracker,
    ) -> SourceKind {
        if stt.text.trim().is_empty()
            || (stt.no_speech_prob.is_finite() && stt.no_speech_prob > MIN_NO_SPEECH_PROB)
            || (stt.avg_logprob.is_finite() && stt.avg_logprob < MIN_AVG_LOGPROB)
            || (stt.compression_ratio.is_finite() && stt.compression_ratio > MAX_COMPRESSION_RATIO)
        {
            tv_tracker.reset(&utterance.room_id);
            return SourceKind::Noise;
        }

        if room_media_playing
            && (utterance.speech_ratio >= MEDIA_HA_SPEECH_RATIO
                || (utterance.has_doa && utterance.doa_stability >= MEDIA_HA_DOA_STABILITY))
        {
            return SourceKind::MediaTv;
        }

        if utterance.has_doa
            && utterance.doa_stability >= MEDIA_STREAK_DOA_STABILITY
            && utterance.speech_ratio >= MEDIA_STREAK_SPEECH_RATIO
        {
            let streak =
                tv_tracker.observe_static_monologue(&utterance.room_id, utterance.doa_azimuth_rad);
            if streak >= 2 {
                return SourceKind::MediaTv;
            }
        } else {
            tv_tracker.reset(&utterance.room_id);
        }

        SourceKind::Live
    }
}

fn circular_angle_diff(a: f32, b: f32) -> f32 {
    let diff = a - b;
    diff.sin().atan2(diff.cos()).abs()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stt(text: &str) -> TranscribeResponse {
        TranscribeResponse {
            text: text.into(),
            no_speech_prob: 0.05,
            avg_logprob: -0.35,
            compression_ratio: 1.2,
            latency_ms: 50,
            model_id: "whisper-v3-turbo".into(),
        }
    }

    #[test]
    fn filters_noise_and_tv_streams() {
        let mut tracker = TvTracker::default();
        let live_utt = AudioUtterance {
            room_id: "living".into(),
            speech_ratio: 0.55,
            ..Default::default()
        };
        assert_eq!(
            SourceClassifier::classify(&live_utt, &stt("hola como estas"), false, &mut tracker),
            SourceKind::Live
        );
        assert_eq!(
            SourceClassifier::classify(&live_utt, &stt("   "), false, &mut tracker),
            SourceKind::Noise
        );
        let tv_utt = AudioUtterance {
            room_id: "living".into(),
            has_doa: true,
            doa_azimuth_rad: 1.1,
            doa_stability: 0.96,
            speech_ratio: 0.92,
            ..Default::default()
        };
        assert_eq!(
            SourceClassifier::classify(&tv_utt, &stt("monologo de television"), true, &mut tracker),
            SourceKind::MediaTv
        );
        // Without HA state, two consecutive static-DoA high-density segments classify as MediaTv.
        assert_eq!(
            SourceClassifier::classify(&tv_utt, &stt("primera parte"), false, &mut tracker),
            SourceKind::Live
        );
        assert_eq!(
            SourceClassifier::classify(&tv_utt, &stt("segunda parte"), false, &mut tracker),
            SourceKind::MediaTv
        );
    }
}
