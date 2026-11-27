//! Injectable daemon interfaces and their Tonic Unix-socket adapters.

use std::{fmt::Debug, path::Path, pin::Pin, time::Duration};

use futures_util::Stream;
use hyper_util::rt::TokioIo;
use nexa_proto::v1::{
    self, audio_service_client::AudioServiceClient, memory_service_client::MemoryServiceClient,
    ml_service_client::MlServiceClient,
};
use prost::Message;
use tokio::net::UnixStream;
use tonic::{
    Request, Status,
    metadata::MetadataValue,
    transport::{Channel, Endpoint},
};
use tower::service_fn;

pub type StreamOf<T> = Pin<Box<dyn Stream<Item = T> + Send + 'static>>;

#[tonic::async_trait]
pub trait AudioClient: Debug + Send + Sync {
    async fn subscribe(
        &self,
        ambient: bool,
    ) -> Result<StreamOf<Result<v1::AudioUtterance, Status>>, Status>;
    async fn play(
        &self,
        chunks: tokio_stream::wrappers::ReceiverStream<v1::PlayAudioChunk>,
    ) -> Result<v1::PlayAudioReceipt, Status>;
    async fn cancel(&self, request: v1::CancelPlaybackRequest) -> Result<(), Status>;
}

#[tonic::async_trait]
pub trait MlClient: Debug + Send + Sync {
    async fn transcribe(
        &self,
        request: v1::TranscribeRequest,
    ) -> Result<v1::TranscribeResponse, Status>;
    async fn synthesize(
        &self,
        request: v1::SynthesizeRequest,
    ) -> Result<StreamOf<Result<v1::TtsAudioChunk, Status>>, Status>;
    async fn embed_speaker(
        &self,
        request: v1::EmbedSpeakerRequest,
    ) -> Result<v1::SpeakerEmbeddingResponse, Status>;
    async fn embed_text(
        &self,
        request: v1::EmbedTextRequest,
    ) -> Result<v1::BatchEmbeddingResponse, Status>;
}

#[tonic::async_trait]
pub trait MemoryClient: Debug + Send + Sync {
    async fn query(
        &self,
        request: v1::QueryMemoryRequest,
    ) -> Result<v1::QueryMemoryResponse, Status>;
    async fn observe_voice(
        &self,
        request: v1::RecordVoiceObservationRequest,
    ) -> Result<v1::VoiceObservationReceipt, Status>;

    async fn erase_conversation(
        &self,
        _request: v1::EraseConversationRequest,
    ) -> Result<v1::ErasureReceipt, Status> {
        Err(Status::unimplemented("memory erasure adapter unavailable"))
    }

    async fn overview(
        &self,
        _request: v1::MemoryOverviewRequest,
    ) -> Result<v1::MemoryOverviewResponse, Status> {
        Err(Status::unimplemented("memory service has no overview RPC"))
    }

    async fn mutate(
        &self,
        _request: v1::MemoryMutationRequest,
    ) -> Result<v1::MemoryMutationResponse, Status> {
        Err(Status::unimplemented(
            "memory service has no fact review RPC",
        ))
    }

    async fn voice_switch(
        &self,
        _request: v1::ActorRequest,
    ) -> Result<v1::VoiceGlobalSwitchStatus, Status> {
        Err(Status::unimplemented(
            "memory service has no global voice switch RPC",
        ))
    }

    async fn set_voice_switch(
        &self,
        _request: v1::SetVoiceGlobalSwitchRequest,
    ) -> Result<v1::VoiceGlobalSwitchStatus, Status> {
        Err(Status::unimplemented(
            "memory service has no global voice switch RPC",
        ))
    }

    async fn review_capture(
        &self,
        _request: v1::VoiceCaptureReviewRequest,
    ) -> Result<v1::VoiceCaptureReviewResponse, Status> {
        Err(Status::unimplemented(
            "memory service has no capture review RPC",
        ))
    }

    async fn query_captures(
        &self,
        _request: v1::VoiceCaptureQueryRequest,
    ) -> Result<v1::VoiceCaptureQueryResponse, Status> {
        Err(Status::unimplemented(
            "memory service has no capture query RPC",
        ))
    }

    async fn evaluation_sample(
        &self,
        _request: v1::EvaluationSampleRequest,
    ) -> Result<v1::EvaluationSampleResponse, Status> {
        Err(Status::unimplemented(
            "memory service has no per-sample evaluation RPC",
        ))
    }

    async fn ambient_control(&self) -> Result<v1::AmbientControlStatus, Status> {
        Err(Status::unimplemented(
            "memory service has no ambient control RPC",
        ))
    }

    async fn store_fact(&self, _request: v1::StoreFactRequest) -> Result<v1::MemoryFact, Status> {
        Err(Status::unimplemented(
            "memory service has no store_fact RPC",
        ))
    }

    async fn identify_voice(
        &self,
        _request: v1::IdentifyVoiceRequest,
    ) -> Result<v1::IdentifyVoiceResponse, Status> {
        Err(Status::unimplemented(
            "memory service has no identify_voice RPC",
        ))
    }

    async fn upsert_voice_personality(
        &self,
        _request: v1::UpsertVoicePersonalityRequest,
    ) -> Result<v1::VoicePersonality, Status> {
        Err(Status::unimplemented(
            "memory service has no upsert_voice_personality RPC",
        ))
    }

    async fn get_voice_personality(
        &self,
        _request: v1::GetVoicePersonalityRequest,
    ) -> Result<v1::VoicePersonality, Status> {
        Err(Status::unimplemented(
            "memory service has no get_voice_personality RPC",
        ))
    }
}

/// Lazily connect so mutually dependent daemons can start in any order.
pub fn channel(path: &Path) -> Channel {
    let path = path.to_path_buf();
    Endpoint::from_static("http://[::]:50051")
        .connect_timeout(Duration::from_secs(2))
        .connect_with_connector_lazy(service_fn(move |_: tonic::transport::Uri| {
            let path = path.clone();
            async move { UnixStream::connect(path).await.map(TokioIo::new) }
        }))
}

#[derive(Debug)]
pub struct UdsAudio(pub AudioServiceClient<Channel>);

#[tonic::async_trait]
impl AudioClient for UdsAudio {
    async fn subscribe(
        &self,
        ambient: bool,
    ) -> Result<StreamOf<Result<v1::AudioUtterance, Status>>, Status> {
        let response = self
            .0
            .clone()
            .subscribe_utterances(v1::SubscribeUtterancesRequest {
                include_ambient: ambient,
            })
            .await?;
        Ok(Box::pin(response.into_inner()))
    }

    async fn play(
        &self,
        chunks: tokio_stream::wrappers::ReceiverStream<v1::PlayAudioChunk>,
    ) -> Result<v1::PlayAudioReceipt, Status> {
        self.0
            .clone()
            .play_audio_stream(chunks)
            .await
            .map(tonic::Response::into_inner)
    }

    async fn cancel(&self, request: v1::CancelPlaybackRequest) -> Result<(), Status> {
        self.0.clone().cancel_playback(request).await?;
        Ok(())
    }
}

#[derive(Debug)]
pub struct UdsMl(pub MlServiceClient<Channel>);

#[tonic::async_trait]
impl MlClient for UdsMl {
    async fn transcribe(
        &self,
        request: v1::TranscribeRequest,
    ) -> Result<v1::TranscribeResponse, Status> {
        self.0
            .clone()
            .transcribe(request)
            .await
            .map(tonic::Response::into_inner)
    }

    async fn synthesize(
        &self,
        request: v1::SynthesizeRequest,
    ) -> Result<StreamOf<Result<v1::TtsAudioChunk, Status>>, Status> {
        let response = self.0.clone().synthesize_stream(request).await?;
        Ok(Box::pin(response.into_inner()))
    }

    async fn embed_speaker(
        &self,
        request: v1::EmbedSpeakerRequest,
    ) -> Result<v1::SpeakerEmbeddingResponse, Status> {
        self.0
            .clone()
            .embed_speaker(request)
            .await
            .map(tonic::Response::into_inner)
    }

    async fn embed_text(
        &self,
        request: v1::EmbedTextRequest,
    ) -> Result<v1::BatchEmbeddingResponse, Status> {
        self.0
            .clone()
            .embed_text(request)
            .await
            .map(tonic::Response::into_inner)
    }
}

#[derive(Debug)]
pub struct UdsMemory(pub MemoryServiceClient<Channel>);

#[tonic::async_trait]
impl MemoryClient for UdsMemory {
    async fn overview(
        &self,
        request: v1::MemoryOverviewRequest,
    ) -> Result<v1::MemoryOverviewResponse, Status> {
        self.0
            .clone()
            .list_memory_overview(request)
            .await
            .map(tonic::Response::into_inner)
    }

    async fn mutate(
        &self,
        request: v1::MemoryMutationRequest,
    ) -> Result<v1::MemoryMutationResponse, Status> {
        self.0
            .clone()
            .mutate_memory_item(request)
            .await
            .map(tonic::Response::into_inner)
    }

    async fn voice_switch(
        &self,
        request: v1::ActorRequest,
    ) -> Result<v1::VoiceGlobalSwitchStatus, Status> {
        self.0
            .clone()
            .get_voice_global_switch(request)
            .await
            .map(tonic::Response::into_inner)
    }

    async fn set_voice_switch(
        &self,
        request: v1::SetVoiceGlobalSwitchRequest,
    ) -> Result<v1::VoiceGlobalSwitchStatus, Status> {
        self.0
            .clone()
            .set_voice_global_switch(request)
            .await
            .map(tonic::Response::into_inner)
    }

    async fn review_capture(
        &self,
        request: v1::VoiceCaptureReviewRequest,
    ) -> Result<v1::VoiceCaptureReviewResponse, Status> {
        self.0
            .clone()
            .get_voice_capture_review(request)
            .await
            .map(tonic::Response::into_inner)
    }

    async fn query_captures(
        &self,
        request: v1::VoiceCaptureQueryRequest,
    ) -> Result<v1::VoiceCaptureQueryResponse, Status> {
        self.0
            .clone()
            .query_voice_captures(request)
            .await
            .map(tonic::Response::into_inner)
    }

    async fn evaluation_sample(
        &self,
        request: v1::EvaluationSampleRequest,
    ) -> Result<v1::EvaluationSampleResponse, Status> {
        self.0
            .clone()
            .get_evaluation_sample(request)
            .await
            .map(tonic::Response::into_inner)
    }

    async fn erase_conversation(
        &self,
        request: v1::EraseConversationRequest,
    ) -> Result<v1::ErasureReceipt, Status> {
        self.0
            .clone()
            .erase_conversation(request)
            .await
            .map(tonic::Response::into_inner)
    }

    async fn query(
        &self,
        request: v1::QueryMemoryRequest,
    ) -> Result<v1::QueryMemoryResponse, Status> {
        self.0
            .clone()
            .query_memory(request)
            .await
            .map(tonic::Response::into_inner)
    }

    async fn observe_voice(
        &self,
        request: v1::RecordVoiceObservationRequest,
    ) -> Result<v1::VoiceObservationReceipt, Status> {
        let actor = v1::ActorContext {
            subject: "nexa-brain".into(),
            role: v1::Role::Operator as i32,
            permissions: vec![v1::Permission::VoicesReview as i32],
            ..Default::default()
        };
        let mut request = Request::new(request);
        request.metadata_mut().insert_bin(
            "actor-context-bin",
            MetadataValue::from_bytes(&actor.encode_to_vec()),
        );
        self.0
            .clone()
            .record_voice_observation(request)
            .await
            .map(tonic::Response::into_inner)
    }

    async fn ambient_control(&self) -> Result<v1::AmbientControlStatus, Status> {
        let actor = v1::ActorContext {
            subject: "nexa-brain".into(),
            role: v1::Role::Operator.into(),
            permissions: vec![v1::Permission::AmbientControl.into()],
            ..Default::default()
        };
        let mut request = Request::new(v1::Empty {});
        request.metadata_mut().insert_bin(
            "actor-context-bin",
            MetadataValue::from_bytes(&actor.encode_to_vec()),
        );
        self.0
            .clone()
            .get_ambient_control(request)
            .await
            .map(tonic::Response::into_inner)
    }

    async fn store_fact(&self, request: v1::StoreFactRequest) -> Result<v1::MemoryFact, Status> {
        self.0
            .clone()
            .store_fact(request)
            .await
            .map(tonic::Response::into_inner)
    }

    async fn identify_voice(
        &self,
        request: v1::IdentifyVoiceRequest,
    ) -> Result<v1::IdentifyVoiceResponse, Status> {
        self.0
            .clone()
            .identify_voice(request)
            .await
            .map(tonic::Response::into_inner)
    }

    async fn upsert_voice_personality(
        &self,
        request: v1::UpsertVoicePersonalityRequest,
    ) -> Result<v1::VoicePersonality, Status> {
        self.0
            .clone()
            .upsert_voice_personality(request)
            .await
            .map(tonic::Response::into_inner)
    }

    async fn get_voice_personality(
        &self,
        request: v1::GetVoicePersonalityRequest,
    ) -> Result<v1::VoicePersonality, Status> {
        self.0
            .clone()
            .get_voice_personality(request)
            .await
            .map(tonic::Response::into_inner)
    }
}
