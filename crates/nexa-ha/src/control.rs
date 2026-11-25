use std::{
    collections::{BTreeSet, HashMap},
    sync::{Arc, Mutex, PoisonError},
};

use reqwest::StatusCode;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::sync::{oneshot, watch};

use crate::{
    breaker::{CircuitBreaker, CircuitBreakerConfig, Health},
    client::{HaClient, HaError},
    operations::{Operation, OperationError},
    policy::{ControlPolicy, EffectIntent, EffectPlan, PlanError},
    state::Observation,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ActionRequest {
    pub request_id: String,
    pub action: EffectIntent,
}

#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum ControlError {
    #[error("invalid request_id")]
    InvalidRequestId,
    #[error("request_id is already bound to a different action body")]
    IdempotencyConflict,
    #[error("action worker ended before recording a receipt")]
    ExecutionInterrupted,
}

impl ControlError {
    pub fn http_status_code(&self) -> u16 {
        match self {
            Self::InvalidRequestId => 400,
            Self::IdempotencyConflict => 409,
            Self::ExecutionInterrupted => 503,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NotSentReason {
    Policy(PlanError),
    Unavailable,
    StateUnverified,
    Capability(OperationError),
    CircuitOpen,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActionStatus {
    NotSent(NotSentReason),
    /// HA accepted the service call. The device's physical effect is unverified.
    Accepted,
    Rejected(u16),
    /// The service may have executed. Replaying this request is unsafe.
    Uncertain,
    /// At least one leaf was accepted before a subsequent leaf failed.
    Partial,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EffectReceipt {
    pub entity_id: String,
    pub operation: &'static str,
    pub status: ActionStatus,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActionReceipt {
    pub request_id: String,
    pub entity_id: String,
    pub operation: &'static str,
    pub status: ActionStatus,
    pub effects: Vec<EffectReceipt>,
}

/// The mutation entry point for trusted callers with established control authority.
/// Clones share policy, FIFO lanes, breaker and process-lifetime receipts.
#[derive(Debug, Clone)]
pub struct ControlService {
    inner: Arc<ServiceInner>,
}

#[derive(Debug)]
struct ServiceInner {
    client: HaClient,
    policy: ControlPolicy,
    breaker: CircuitBreaker,
    ledger: Mutex<Ledger>,
}

type ReceiptReceiver = watch::Receiver<Option<Arc<ActionReceipt>>>;

#[derive(Debug, Default)]
struct Ledger {
    requests: HashMap<String, StoredRequest>,
    tails: HashMap<String, oneshot::Receiver<()>>,
}

#[derive(Debug)]
struct StoredRequest {
    body: EffectIntent,
    receipt: ReceiptReceiver,
}

struct Reservation {
    predecessors: Vec<oneshot::Receiver<()>>,
    _completion: Vec<oneshot::Sender<()>>,
}

impl Ledger {
    fn reserve_lanes(&mut self, plan: &EffectPlan) -> Reservation {
        let entities: BTreeSet<_> = plan
            .references
            .keys()
            .chain(plan.leaves.iter().map(|effect| &effect.entity_id))
            .collect();
        let mut reservation = Reservation {
            predecessors: Vec::new(),
            _completion: Vec::new(),
        };
        // Reserve all lanes atomically, so overlapping plans cannot wait on each other.
        for entity_id in entities {
            let (done, tail) = oneshot::channel();
            if let Some(previous) = self.tails.insert(entity_id.clone(), tail) {
                reservation.predecessors.push(previous);
            }
            reservation._completion.push(done);
        }
        reservation
    }
}

impl ControlService {
    pub fn new(client: HaClient, policy: ControlPolicy) -> Self {
        Self::with_breaker(client, policy, CircuitBreakerConfig::default())
    }

    pub fn with_breaker(
        client: HaClient,
        policy: ControlPolicy,
        breaker_config: CircuitBreakerConfig,
    ) -> Self {
        Self {
            inner: Arc::new(ServiceInner {
                client,
                policy,
                breaker: CircuitBreaker::new(breaker_config),
                ledger: Mutex::new(Ledger::default()),
            }),
        }
    }

    /// A matching retry waits for or reuses the original receipt, even if its caller was cancelled.
    pub async fn execute(
        &self,
        request: ActionRequest,
    ) -> Result<Arc<ActionReceipt>, ControlError> {
        if !valid_request_id(&request.request_id) {
            return Err(ControlError::InvalidRequestId);
        }
        let (mut receipt, worker) = {
            let mut ledger = self
                .inner
                .ledger
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            if let Some(stored) = ledger.requests.get(&request.request_id) {
                if stored.body.entity_id != request.action.entity_id
                    || !stored.body.operation.same_body(&request.action.operation)
                {
                    return Err(ControlError::IdempotencyConflict);
                }
                (stored.receipt.clone(), None)
            } else {
                let (sender, receiver) = watch::channel(None);
                let worker = match self.inner.policy.plan(&request.action) {
                    Ok(plan) => {
                        let reservation = ledger.reserve_lanes(&plan);
                        Some((sender, plan, reservation))
                    }
                    Err(reason) => {
                        let status = ActionStatus::NotSent(NotSentReason::Policy(reason));
                        sender.send_replace(Some(Arc::new(make_receipt(
                            &request,
                            status,
                            Vec::new(),
                        ))));
                        None
                    }
                };
                ledger.requests.insert(
                    request.request_id.clone(),
                    StoredRequest {
                        body: request.action.clone(),
                        receipt: receiver.clone(),
                    },
                );
                (receiver, worker)
            }
        };
        if let Some((sender, plan, reservation)) = worker {
            let service = self.clone();
            tokio::spawn(async move {
                for previous in reservation.predecessors {
                    let _ = previous.await;
                }
                let (status, effects) = service.inner.perform(&plan).await;
                sender.send_replace(Some(Arc::new(make_receipt(&request, status, effects))));
                drop(reservation._completion);
            });
        }
        loop {
            if let Some(receipt) = receipt.borrow_and_update().clone() {
                return Ok(receipt);
            }
            receipt
                .changed()
                .await
                .map_err(|_| ControlError::ExecutionInterrupted)?;
        }
    }

    /// Retrieve a completed receipt for a caller whose authority has already been checked.
    pub fn get_receipt(&self, request_id: &str) -> Option<Arc<ActionReceipt>> {
        let ledger = self
            .inner
            .ledger
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        ledger.requests.get(request_id)?.receipt.borrow().clone()
    }
}

impl ServiceInner {
    async fn perform(&self, plan: &EffectPlan) -> (ActionStatus, Vec<EffectReceipt>) {
        let Some(permit) = self.breaker.admit() else {
            return (
                ActionStatus::NotSent(NotSentReason::CircuitOpen),
                Vec::new(),
            );
        };
        let (status, effects, health) = self.run_plan(plan).await;
        permit.finish(health);
        (status, effects)
    }

    async fn run_plan(&self, plan: &EffectPlan) -> (ActionStatus, Vec<EffectReceipt>, Health) {
        let celsius = if plan
            .leaves
            .iter()
            .any(|effect| matches!(effect.operation, Operation::SetTemperature { .. }))
        {
            match self.client.temperature_is_celsius().await {
                Ok(celsius) => celsius,
                Err(error) => {
                    return (
                        ActionStatus::NotSent(NotSentReason::StateUnverified),
                        Vec::new(),
                        dependency_health(&error),
                    );
                }
            }
        } else {
            false
        };
        if !plan.references.is_empty()
            && let Err((reason, health)) = self.preflight_plan(plan, celsius).await
        {
            return (ActionStatus::NotSent(reason), Vec::new(), health);
        }
        let mut effects = Vec::with_capacity(plan.leaves.len());
        for intent in &plan.leaves {
            let (status, health) = match self.preflight_leaf(intent, celsius).await {
                Ok(()) => {
                    let (service, body) = intent.operation.service(&intent.entity_id);
                    // Planning has verified the entity syntax and closed domain allowlist.
                    let domain = intent
                        .entity_id
                        .split_once('.')
                        .map_or("", |(domain, _)| domain);
                    match self.client.call_service(domain, service, &body).await {
                        Ok(()) => (ActionStatus::Accepted, Health::Healthy),
                        Err(error) => (dispatch_failure(&error), dependency_health(&error)),
                    }
                }
                Err((reason, health)) => (ActionStatus::NotSent(reason), health),
            };
            effects.push(EffectReceipt {
                entity_id: intent.entity_id.clone(),
                operation: intent.operation.name(),
                status,
            });
            if status != ActionStatus::Accepted {
                let overall = if effects.len() > 1 {
                    ActionStatus::Partial
                } else {
                    status
                };
                let health = if matches!(health, Health::Unobserved) && effects.len() > 1 {
                    Health::Healthy
                } else {
                    health
                };
                return (overall, effects, health);
            }
        }
        (ActionStatus::Accepted, effects, Health::Healthy)
    }

    async fn preflight_plan(
        &self,
        plan: &EffectPlan,
        celsius: bool,
    ) -> Result<(), PreflightFailure> {
        for (entity_id, reference) in &plan.references {
            let observation = self.available(entity_id).await?;
            reference
                .validate(&observation.state)
                .map_err(|reason| (NotSentReason::Policy(reason), Health::Unobserved))?;
        }
        for intent in &plan.leaves {
            self.preflight_leaf(intent, celsius).await?;
        }
        Ok(())
    }

    async fn preflight_leaf(
        &self,
        intent: &EffectIntent,
        celsius: bool,
    ) -> Result<(), PreflightFailure> {
        let observation = self.available(&intent.entity_id).await?;
        intent
            .operation
            .validate_leaf(&observation.state, celsius)
            .map_err(|error| (NotSentReason::Capability(error), Health::Unobserved))
    }

    async fn available(&self, entity_id: &str) -> Result<Observation, PreflightFailure> {
        let observation = self
            .client
            .current_entity(entity_id)
            .await
            .map_err(|error| (NotSentReason::StateUnverified, dependency_health(&error)))?;
        if observation.state.state() == "unavailable" {
            return Err((NotSentReason::Unavailable, Health::Unobserved));
        }
        Ok(observation)
    }
}

type PreflightFailure = (NotSentReason, Health);

fn valid_request_id(id: &str) -> bool {
    (1..=128).contains(&id.len())
        && id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"_.:-".contains(&byte))
}

fn make_receipt(
    request: &ActionRequest,
    status: ActionStatus,
    effects: Vec<EffectReceipt>,
) -> ActionReceipt {
    ActionReceipt {
        request_id: request.request_id.clone(),
        entity_id: request.action.entity_id.clone(),
        operation: request.action.operation.name(),
        status,
        effects,
    }
}

fn dependency_health(error: &HaError) -> Health {
    match error {
        HaError::Transport | HaError::InvalidResponse => Health::Failed,
        HaError::Http(status)
            if status.is_server_error()
                || matches!(
                    *status,
                    StatusCode::REQUEST_TIMEOUT
                        | StatusCode::TOO_MANY_REQUESTS
                        | StatusCode::UNAUTHORIZED
                        | StatusCode::FORBIDDEN
                ) =>
        {
            Health::Failed
        }
        HaError::Http(_) => Health::Healthy,
        _ => Health::Unobserved,
    }
}

fn dispatch_failure(error: &HaError) -> ActionStatus {
    match error {
        HaError::Http(status)
            if status.is_client_error() && *status != StatusCode::REQUEST_TIMEOUT =>
        {
            ActionStatus::Rejected(status.as_u16())
        }
        _ => ActionStatus::Uncertain,
    }
}
