use std::time::{Duration, Instant};

use futures_util::StreamExt;
use nexa_proto::v1::{
    self, CancelPlaybackRequest, LiveEventKind, PlayAudioChunk, SynthesizeRequest,
};
use serde_json::json;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::Status;

use crate::{Brain, clients::StreamOf};

use super::{Turn, turns::Lease};

impl Brain {
    pub(super) async fn cancel_playback(&self, zone: &str, lease: &Lease) {
        lease
            .cancel_rpc
            .get_or_init(|| async {
                let started = Instant::now();
                let zone_id = self.output_zone(zone);
                let cancelled = tokio::time::timeout(
                    Duration::from_millis(350),
                    self.audio.cancel(CancelPlaybackRequest {
                        zone_id: zone_id.clone(),
                        barge_in_token: lease.id.clone(),
                    }),
                )
                .await;
                let latency_ms = started.elapsed().as_millis();
                match cancelled {
                    Ok(Ok(())) => tracing::info!(%zone_id, latency_ms, status = "cancelled", "TTS playback cancellation acknowledged"),
                    result => {
                        let code = match result {
                            Ok(Err(error)) => error.code(),
                            _ => tonic::Code::DeadlineExceeded,
                        };
                        tracing::warn!(%zone_id, latency_ms, status = ?code, "TTS playback cancellation was not acknowledged");
                    }
                }
            })
            .await;
    }

    pub(crate) async fn say(
        &self,
        turn: &Turn,
        conversation: &str,
        text: impl Into<String>,
    ) -> Result<v1::PlayAudioReceipt, Status> {
        let text = text.into();
        self.speak(
            turn,
            conversation,
            Box::pin(futures_util::stream::iter([text])),
        )
        .await
    }

    pub(super) async fn speak(
        &self,
        turn: &Turn,
        conversation: &str,
        texts: StreamOf<String>,
    ) -> Result<v1::PlayAudioReceipt, Status> {
        let started = Instant::now();
        let result = self.play_speech(turn, conversation, texts).await;
        let zone_id = self.output_zone(&turn.zone);
        let latency_ms = started.elapsed().as_millis();
        let status = playback_status(&result);
        match &result {
            Ok(_) => {
                tracing::info!(%zone_id, utterance_id = %conversation, latency_ms, status, "TTS playback finished")
            }
            Err(error) => {
                tracing::warn!(%zone_id, utterance_id = %conversation, latency_ms, status,
                code = ?error.code(), "TTS playback failed")
            }
        }
        result
    }

    async fn play_speech(
        &self,
        turn: &Turn,
        conversation: &str,
        mut texts: StreamOf<String>,
    ) -> Result<v1::PlayAudioReceipt, Status> {
        let Some(first) = texts.next().await else {
            return Err(Status::failed_precondition("no speech generated"));
        };
        let (tx, rx) = mpsc::channel(8);
        let sample_rate = self.config.core.audio.playback_sample_rate;
        let zone = &turn.zone;
        let chunk = |pcm, is_final| PlayAudioChunk {
            playback_id: turn.lease.id.clone(),
            zone_id: self.output_zone(zone),
            sample_rate,
            pcm_f32le: pcm,
            priority: 1,
            barge_in_token: turn.lease.id.clone(),
            is_final,
        };
        let produce = async {
            let mut texts = futures_util::stream::iter([first]).chain(texts);
            let mut sent = false;
            while let Some(text) = texts.next().await {
                if text.trim().is_empty() {
                    continue;
                }
                let mut audio = self
                    .ml
                    .synthesize(SynthesizeRequest {
                        text,
                        voice: self.config.core.ml.tts_voice.clone(),
                        lang_code: self.config.core.ml.tts_lang_code.clone(),
                        speed: 1.0,
                    })
                    .await?;
                let mut finished = false;
                while let Some(frame) = audio.next().await {
                    let frame = frame?;
                    if frame.sample_rate != sample_rate || !frame.pcm_f32le.len().is_multiple_of(4)
                    {
                        return Err(Status::data_loss("invalid TTS audio format"));
                    }
                    if !frame.pcm_f32le.is_empty() {
                        tx.send(chunk(frame.pcm_f32le, false))
                            .await
                            .map_err(|_| Status::cancelled("playback stopped"))?;
                        sent = true;
                    }
                    if frame.is_final {
                        finished = true;
                        break;
                    }
                }
                if !finished {
                    return Err(Status::data_loss("incomplete TTS audio"));
                }
            }
            if !sent {
                return Err(Status::data_loss("TTS generated no audio"));
            }
            tx.send(chunk(Vec::new(), true))
                .await
                .map_err(|_| Status::cancelled("playback stopped"))?;
            Ok(())
        };
        self.event(
            LiveEventKind::Tts,
            zone,
            conversation,
            json!({"status": "started"}),
        );
        let play = async {
            let receipt = self.audio.play(ReceiverStream::new(rx)).await?;
            if receipt.playback_id != turn.lease.id {
                return Err(Status::data_loss("playback receipt ID mismatch"));
            }
            if receipt.interrupted_by_barge_in {
                turn.lease.cancel.cancel();
                return Err(Status::cancelled("playback interrupted by barge-in"));
            }
            Ok(receipt)
        };
        let result = tokio::try_join!(produce, play).map(|(_, receipt)| receipt);
        self.event(
            LiveEventKind::Tts,
            zone,
            conversation,
            json!({"status": playback_status(&result)}),
        );
        result
    }

    pub(crate) fn output_busy(&self, zone: &str) -> bool {
        self.active.busy(zone)
    }

    pub(crate) async fn announce(
        &self,
        zone: &str,
        message: &str,
    ) -> nexa_household::intercom::Playback {
        use nexa_household::intercom::Playback;
        if !self.zone(zone) {
            return Playback::NotStarted;
        }
        let Some((turn, _)) = self.active.start(zone, &self.shutdown, false) else {
            return Playback::NotStarted;
        };
        let result = tokio::select! {
            biased;
            _ = turn.lease.cancel.cancelled() => None,
            result = tokio::time::timeout(Duration::from_secs(10), self.say(&turn, "", message)) => result.ok(),
        };
        match result {
            Some(Ok(receipt)) if receipt.completed => Playback::Confirmed,
            _ => {
                self.cancel_playback(zone, &turn.lease).await;
                Playback::Uncertain
            }
        }
    }
}

fn playback_status(result: &Result<v1::PlayAudioReceipt, Status>) -> &'static str {
    match result {
        Ok(receipt) if receipt.completed => "completed",
        Err(error) if error.code() == tonic::Code::Cancelled => "cancelled",
        _ => "unconfirmed",
    }
}
