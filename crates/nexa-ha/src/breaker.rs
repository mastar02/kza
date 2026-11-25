use std::{
    num::{NonZeroU32, NonZeroU64},
    sync::{Mutex, PoisonError},
    time::Duration,
};

use tokio::time::Instant;

#[derive(Debug, Clone, Copy)]
pub struct CircuitBreakerConfig {
    pub failure_threshold: NonZeroU32,
    pub cooldown_s: NonZeroU64,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: NonZeroU32::new(3).unwrap_or(NonZeroU32::MIN),
            cooldown_s: NonZeroU64::new(30).unwrap_or(NonZeroU64::MIN),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum Health {
    Healthy,
    Failed,
    Unobserved,
}

#[derive(Debug)]
pub(crate) struct CircuitBreaker {
    config: CircuitBreakerConfig,
    state: Mutex<Circuit>,
}

#[derive(Debug)]
struct Circuit {
    generation: u64,
    phase: Phase,
}

#[derive(Debug, Clone, Copy)]
enum Phase {
    Closed(u32),
    Open(Instant),
    HalfOpen(Instant),
}

pub(crate) struct Permit<'a> {
    breaker: &'a CircuitBreaker,
    generation: u64,
}

impl CircuitBreaker {
    pub fn new(config: CircuitBreakerConfig) -> Self {
        Self {
            config,
            state: Mutex::new(Circuit {
                generation: 0,
                phase: Phase::Closed(0),
            }),
        }
    }

    pub fn admit(&self) -> Option<Permit<'_>> {
        let mut circuit = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        match circuit.phase {
            Phase::Closed(_) => {}
            Phase::Open(since)
                if since.elapsed() >= Duration::from_secs(self.config.cooldown_s.get()) =>
            {
                circuit.phase = Phase::HalfOpen(since);
            }
            _ => return None,
        }
        Some(Permit {
            breaker: self,
            generation: circuit.generation,
        })
    }

    fn record(&self, generation: u64, health: Health) {
        let mut circuit = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        if circuit.generation != generation {
            return;
        }
        circuit.phase = match (circuit.phase, health) {
            (Phase::Closed(failures), Health::Failed) => {
                let failures = failures + 1;
                if failures >= self.config.failure_threshold.get() {
                    circuit.generation = circuit.generation.wrapping_add(1);
                    Phase::Open(Instant::now())
                } else {
                    Phase::Closed(failures)
                }
            }
            (Phase::Closed(_), Health::Healthy) => Phase::Closed(0),
            (Phase::HalfOpen(_), Health::Healthy) => {
                circuit.generation = circuit.generation.wrapping_add(1);
                Phase::Closed(0)
            }
            (Phase::HalfOpen(_), Health::Failed) => {
                circuit.generation = circuit.generation.wrapping_add(1);
                Phase::Open(Instant::now())
            }
            (Phase::HalfOpen(since), Health::Unobserved) => {
                // An action stopped locally. Another action may still probe HA.
                circuit.generation = circuit.generation.wrapping_add(1);
                Phase::Open(since)
            }
            (phase, _) => phase,
        };
    }
}

impl Permit<'_> {
    pub fn finish(self, health: Health) {
        self.breaker.record(self.generation, health);
    }
}

impl Drop for Permit<'_> {
    fn drop(&mut self) {
        self.breaker.record(self.generation, Health::Unobserved);
    }
}
