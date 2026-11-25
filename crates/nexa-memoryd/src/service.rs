use std::fs::{self, File};
use std::future::Future;
use std::io;
use std::os::unix::fs::{FileTypeExt, MetadataExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use nexa_proto::sockets;
use nexa_proto::v1::memory_service_server::{MemoryService, MemoryServiceServer};
use nexa_proto::v1::*;
use prost::Message;
use tokio::net::UnixListener;
use tokio::sync::Mutex;
use tokio_stream::wrappers::UnixListenerStream;
use tonic::{Request, Response, Status, transport::Server};

use crate::db::{Error, Result, Store, ensure_private_dir, id, lock_file};
use crate::voice::Observation;

/// Protobuf-encoded ActorContext for RPCs whose request has no actor field.
/// The private socket is restricted to the local UID of trusted Nexa processes.
pub const ACTOR_METADATA: &str = "actor-context-bin";

#[derive(Clone, Debug)]
pub struct MemoryRpc {
    store: Arc<Mutex<Store>>,
    started: Instant,
    capture_enabled: bool,
}

impl MemoryRpc {
    pub fn new(store: Store, capture_enabled: bool) -> Self {
        Self {
            store: Arc::new(Mutex::new(store)),
            started: Instant::now(),
            capture_enabled,
        }
    }

    // The generated tonic service uses its concrete Status type for RPC failures.
    #[allow(clippy::result_large_err)]
    async fn run<T: Send + 'static>(
        &self,
        operation: impl FnOnce(&mut Store) -> Result<T> + Send + 'static,
    ) -> std::result::Result<Response<T>, Status> {
        // Queue on the async runtime and let blocking workers handle SQLite I/O.
        let mut store = self.store.clone().lock_owned().await;
        tokio::task::spawn_blocking(move || operation(&mut store))
            .await
            .map_err(|_| Status::internal("memory worker stopped"))?
            .map(Response::new)
            .map_err(status)
    }
}

struct Actor<'a> {
    context: &'a ActorContext,
    role: Role,
}

impl<'a> Actor<'a> {
    fn validate(context: Option<&'a ActorContext>) -> Result<Self> {
        let context = context.ok_or(Error::Unauthenticated("actor context required"))?;
        let role = Role::try_from(context.role)
            .ok()
            .filter(|role| *role != Role::Unspecified)
            .ok_or(Error::Denied("invalid actor role"))?;
        if id(&context.subject).is_err()
            || (!context.user_id.is_empty() && (role != Role::Home || id(&context.user_id).is_err()))
            || context.permissions.iter().any(|permission| {
                !matches!(Permission::try_from(*permission), Ok(value) if value != Permission::Unspecified)
            })
        {
            return Err(Error::Denied("invalid actor context"));
        }
        Ok(Self { context, role })
    }

    fn require(&self, permission: Permission) -> Result<()> {
        if !self.context.permissions.contains(&(permission as i32)) {
            return Err(Error::Denied("required permission missing"));
        }
        Ok(())
    }

    fn operator(&self, permission: Permission) -> Result<()> {
        if self.role != Role::Operator {
            return Err(Error::Denied("operator actor required"));
        }
        self.require(permission)
    }

    fn own(&self, permission: Permission, user: &str) -> Result<()> {
        if self.role != Role::Home || user.is_empty() || self.context.user_id != user {
            return Err(Error::Denied("personal actor scope mismatch"));
        }
        self.require(permission)
    }

    fn store_fact(&self, owner: &str) -> Result<()> {
        if self.role == Role::Operator {
            self.require(Permission::MemoryReview)?;
            if !owner.is_empty() {
                self.require(Permission::MemoryAssign)?;
            }
            return Ok(());
        }
        self.own(Permission::MemoryOwnWrite, owner)
    }

    fn query(&self, owner: &str, shared: bool) -> Result<()> {
        if self.role == Role::Operator {
            self.require(Permission::MemoryRead)?;
            if !owner.is_empty() {
                self.require(Permission::MemoryAssign)?;
            }
            return Ok(());
        }
        if shared {
            self.require(Permission::MemoryRead)?;
        }
        if !owner.is_empty() {
            self.own(Permission::MemoryOwnRead, owner)?;
        }
        Ok(())
    }

    fn delete_scope(&self) -> Result<Option<String>> {
        if self.role == Role::Operator {
            self.require(Permission::MemoryReview)?;
            return Ok(None);
        }
        self.own(Permission::MemoryOwnWrite, &self.context.user_id)?;
        Ok(Some(self.context.user_id.clone()))
    }
}

fn transport_actor<T>(request: &Request<T>) -> Result<ActorContext> {
    let mut values = request.metadata().get_all_bin(ACTOR_METADATA).iter();
    let Some(value) = values.next() else {
        return Err(Error::Unauthenticated("actor metadata required"));
    };
    if values.next().is_some() {
        return Err(Error::Unauthenticated("ambiguous actor metadata"));
    }
    let bytes = value
        .to_bytes()
        .map_err(|_| Error::Unauthenticated("invalid actor metadata"))?;
    ActorContext::decode(bytes).map_err(|_| Error::Unauthenticated("invalid actor metadata"))
}

fn status(error: Error) -> Status {
    match error {
        Error::Invalid(message) => Status::invalid_argument(message),
        Error::Denied(message) => Status::permission_denied(message),
        Error::Unauthenticated(message) => Status::unauthenticated(message),
        Error::Precondition(message) => Status::failed_precondition(message),
        Error::Conflict(message) => Status::aborted(message),
        Error::NotFound(message) => Status::not_found(message),
        Error::Unavailable(message) => Status::unavailable(message),
        Error::Sql(error) => {
            // Extended SQL errors can contain values from private rows.
            let code = error.sqlite_error_code();
            tracing::error!(?code, "memory database operation failed");
            if matches!(
                code,
                Some(rusqlite::ErrorCode::DatabaseBusy | rusqlite::ErrorCode::DatabaseLocked)
            ) {
                Status::unavailable("memory database busy")
            } else {
                Status::internal("memory database operation failed")
            }
        }
        Error::Io(_) | Error::Schema(_) => Status::internal("memory storage unavailable"),
    }
}

#[tonic::async_trait]
impl MemoryService for MemoryRpc {
    async fn get_health(
        &self,
        req: Request<Empty>,
    ) -> std::result::Result<Response<HealthResponse>, Status> {
        let context = transport_actor(&req).map_err(status)?;
        Actor::validate(Some(&context))
            .and_then(|actor| actor.require(Permission::DiagnosticsRead))
            .map_err(status)?;
        let started = self.started;
        self.run(move |store| {
            store.check_alive()?;
            Ok(HealthResponse {
                service: "nexa-memoryd".into(),
                version: env!("CARGO_PKG_VERSION").into(),
                state: HealthState::Ok as i32,
                uptime_ms: started.elapsed().as_millis() as u64,
                detail: String::new(),
            })
        })
        .await
    }

    async fn store_fact(
        &self,
        req: Request<StoreFactRequest>,
    ) -> std::result::Result<Response<MemoryFact>, Status> {
        let req = req.into_inner();
        Actor::validate(req.actor.as_ref())
            .and_then(|actor| actor.store_fact(&req.owner_user_id))
            .map_err(status)?;
        self.run(move |store| {
            store.store_fact(
                &req.conversation_id,
                &req.owner_user_id,
                &req.content,
                &req.category,
                &req.embedding,
            )
        })
        .await
    }

    async fn query_memory(
        &self,
        req: Request<QueryMemoryRequest>,
    ) -> std::result::Result<Response<QueryMemoryResponse>, Status> {
        let req = req.into_inner();
        Actor::validate(req.actor.as_ref())
            .and_then(|actor| actor.query(&req.owner_user_id, req.include_shared))
            .map_err(status)?;
        self.run(move |store| {
            Ok(QueryMemoryResponse {
                hits: store.query_memory(
                    &req.query_embedding,
                    req.top_k,
                    &req.owner_user_id,
                    req.include_shared,
                )?,
            })
        })
        .await
    }

    async fn delete_fact(
        &self,
        req: Request<DeleteFactRequest>,
    ) -> std::result::Result<Response<ErasureReceipt>, Status> {
        let req = req.into_inner();
        let owner = Actor::validate(req.actor.as_ref())
            .and_then(|actor| actor.delete_scope())
            .map_err(status)?;
        self.run(move |store| store.delete_fact(&req.fact_id, owner.as_deref()))
            .await
    }

    async fn record_voice_observation(
        &self,
        req: Request<RecordVoiceObservationRequest>,
    ) -> std::result::Result<Response<VoiceObservationReceipt>, Status> {
        let context = transport_actor(&req).map_err(status)?;
        Actor::validate(Some(&context))
            .and_then(|actor| actor.operator(Permission::VoicesReview))
            .map_err(status)?;
        if !self.capture_enabled {
            return Err(Status::failed_precondition("ambient capture disabled"));
        }
        let req = req.into_inner();
        self.run(move |store| {
            store.record_observation(Observation {
                utterance: &req.utterance_id,
                room: &req.room_id,
                model: &req.encoder_model_key,
                preview: &req.transcript_preview,
                captured_at: req.captured_at_unix_ms,
                embedding: &req.embedding,
            })
        })
        .await
    }

    async fn list_voice_profiles(
        &self,
        req: Request<ListVoiceProfilesRequest>,
    ) -> std::result::Result<Response<ListVoiceProfilesResponse>, Status> {
        let req = req.into_inner();
        let actor = Actor::validate(req.actor.as_ref()).map_err(status)?;
        actor.operator(Permission::VoicesRead).map_err(status)?;
        actor.require(Permission::VoicesReview).map_err(status)?;
        if req.include_person {
            actor.require(Permission::AdminManage).map_err(status)?;
        }
        self.run(move |store| {
            Ok(ListVoiceProfilesResponse {
                profiles: store.list_voice_profiles(req.include_person)?,
            })
        })
        .await
    }

    async fn link_voice_profile_to_person(
        &self,
        req: Request<LinkVoiceProfileRequest>,
    ) -> std::result::Result<Response<VoiceProfile>, Status> {
        let req = req.into_inner();
        let actor = Actor::validate(req.actor.as_ref()).map_err(status)?;
        actor.operator(Permission::VoicesReview).map_err(status)?;
        actor.require(Permission::AdminManage).map_err(status)?;
        self.run(move |store| {
            store.link_profile(
                &req.profile_id,
                &req.person_id,
                req.expected_profile_revision,
                req.expected_link_revision,
            )
        })
        .await
    }

    async fn set_voice_consent(
        &self,
        req: Request<SetVoiceConsentRequest>,
    ) -> std::result::Result<Response<VoiceConsentStatus>, Status> {
        let req = req.into_inner();
        Actor::validate(req.actor.as_ref())
            .and_then(|actor| actor.own(Permission::MemoryOwnVoice, &req.user_id))
            .map_err(status)?;
        self.run(move |store| store.set_consent(&req.user_id, req.granted))
            .await
    }

    async fn erase_conversation(
        &self,
        req: Request<EraseConversationRequest>,
    ) -> std::result::Result<Response<ErasureReceipt>, Status> {
        let req = req.into_inner();
        Actor::validate(req.actor.as_ref())
            .and_then(|actor| actor.operator(Permission::MemoryReview))
            .map_err(status)?;
        self.run(move |store| store.erase_conversation(&req.conversation_id))
            .await
    }

    async fn purge_retired_user(
        &self,
        req: Request<PurgeRetiredUserRequest>,
    ) -> std::result::Result<Response<ErasureReceipt>, Status> {
        let req = req.into_inner();
        Actor::validate(req.actor.as_ref())
            .and_then(|actor| actor.operator(Permission::AdminManage))
            .map_err(status)?;
        self.run(move |store| store.purge_retired_user(&req.user_id))
            .await
    }

    async fn get_ambient_control(
        &self,
        req: Request<Empty>,
    ) -> std::result::Result<Response<AmbientControlStatus>, Status> {
        let context = transport_actor(&req).map_err(status)?;
        Actor::validate(Some(&context))
            .and_then(|actor| actor.require(Permission::AmbientControl))
            .map_err(status)?;
        self.run(|store| store.ambient_control()).await
    }

    async fn set_ambient_pause(
        &self,
        req: Request<SetAmbientPauseRequest>,
    ) -> std::result::Result<Response<AmbientControlStatus>, Status> {
        let req = req.into_inner();
        Actor::validate(req.actor.as_ref())
            .and_then(|actor| actor.require(Permission::AmbientControl))
            .map_err(status)?;
        self.run(move |store| store.set_ambient_pause(req.paused, req.expected_revision))
            .await
    }
}

#[derive(Debug)]
pub struct BoundSocket {
    listener: Option<UnixListener>,
    path: PathBuf,
    inode: (u64, u64),
    _lock: File,
}

impl BoundSocket {
    pub fn bind(runtime_dir: &Path) -> Result<Self> {
        ensure_private_dir(runtime_dir)?;
        let lock = lock_file(&runtime_dir.join("memory-socket.lock"))?;
        let path = runtime_dir.join(sockets::MEMORY);
        match fs::symlink_metadata(&path) {
            Ok(meta) if meta.file_type().is_socket() => {
                match std::os::unix::net::UnixStream::connect(&path) {
                    Err(error) if error.kind() == io::ErrorKind::ConnectionRefused => {
                        fs::remove_file(&path)?
                    }
                    _ => return Err(Error::Unavailable("memory socket is in use")),
                }
            }
            Ok(_) => return Err(Error::Invalid("memory socket path must be a Unix socket")),
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
        let listener = UnixListener::bind(&path)?;
        let metadata = fs::symlink_metadata(&path)?;
        let bound = Self {
            listener: Some(listener),
            path,
            inode: (metadata.dev(), metadata.ino()),
            _lock: lock,
        };
        fs::set_permissions(&bound.path, fs::Permissions::from_mode(0o600))?;
        Ok(bound)
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub async fn serve(
        mut self,
        service: MemoryRpc,
        shutdown: impl Future<Output = ()>,
    ) -> std::result::Result<(), tonic::transport::Error> {
        let Some(listener) = self.listener.take() else {
            return Ok(());
        };
        Server::builder()
            .add_service(MemoryServiceServer::new(service))
            .serve_with_incoming_shutdown(UnixListenerStream::new(listener), shutdown)
            .await
    }
}

impl Drop for BoundSocket {
    fn drop(&mut self) {
        if let Ok(meta) = fs::symlink_metadata(&self.path)
            && self.inode == (meta.dev(), meta.ino())
            && meta.file_type().is_socket()
            && fs::remove_file(&self.path).is_err()
        {
            tracing::warn!("could not clean up memory socket");
        }
    }
}
