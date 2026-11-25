//! `nexa-brain` — Voice orchestrator, NLU, Home Assistant gateway & household daemon (Phase 4 target).

use nexa_config::HaControlPolicyConfig;
use nexa_ha::authorize_action;
use nexa_household::home_actions_within_grace;
use nexa_llm::CloudConsentGate;
use nexa_nlu::gate_utterance;
use nexa_proto::sockets;
use tracing::info;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt().with_target(true).init();
    let consent_gate = CloudConsentGate::new(false);
    let default_policy = HaControlPolicyConfig::default();
    let probe_decision = gate_utterance("", 1.0, 2.2);
    info!(
        service = "nexa-brain",
        socket = sockets::BRAIN,
        cloud_consent = consent_gate.is_allowed(),
        fail_closed_ha = authorize_action(&default_policy, "light.probe", "turn_on").is_err(),
        grace_default_ok = home_actions_within_grace(0, 120),
        gate_probe = ?probe_decision.path,
        "nexa-brain skeleton initialized (Phase 1 foundation)"
    );
}
