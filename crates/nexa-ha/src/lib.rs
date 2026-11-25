//! Home Assistant state projection, read-only views and fail-closed control.

mod breaker;
mod client;
mod control;
mod operations;
mod policy;
pub mod readers;
mod state;
mod sync;

pub use breaker::CircuitBreakerConfig;
pub use client::{HaClient, HaError};
pub use control::{
    ActionReceipt, ActionRequest, ActionStatus, ControlError, ControlService, EffectReceipt,
    NotSentReason,
};
pub use operations::{Brightness, HvacMode, Operation, OperationError};
pub use policy::{
    ControlPolicy, ControlPolicyError, EffectIntent, MembershipSource, PlanError, ReviewedGroup,
    authorize_action,
};
pub use state::{EntityState, StateFrame, StateReadError};
