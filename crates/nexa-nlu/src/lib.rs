//! Pure deterministic NLU layer for Nexa v2 (Phase 4A implementation target).
//!
//! Zero network/disk I/O: CommandGate (compression ratio, prompt echo, noise),
//! Rioplatense HouseholdIntentClassifier, CommandGrammar, and SlotExtractor.

use nexa_proto::v1::PathType;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RouteDecision {
    pub path: PathType,
    pub normalized_text: String,
    pub reason: Option<String>,
}

/// Normalize Spanish/Rioplatense utterance whitespace and case cleanly.
#[must_use]
pub fn normalize_utterance(raw: &str) -> String {
    raw.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

/// Reject empty or hallucinated high-compression transcriptions before routing.
#[must_use]
pub fn gate_utterance(
    raw: &str,
    compression_ratio: f32,
    max_compression_ratio: f32,
) -> RouteDecision {
    let normalized = normalize_utterance(raw);
    if normalized.is_empty() {
        return RouteDecision {
            path: PathType::Rejected,
            normalized_text: normalized,
            reason: Some("empty".into()),
        };
    }
    if compression_ratio > max_compression_ratio {
        return RouteDecision {
            path: PathType::Rejected,
            normalized_text: normalized,
            reason: Some("high_compression".into()),
        };
    }
    RouteDecision {
        path: PathType::SlowLlm,
        normalized_text: normalized,
        reason: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gate_rejects_empty_and_high_compression_hallucinations() {
        let empty = gate_utterance("   \t\n ", 1.0, 2.2);
        assert_eq!(empty.path, PathType::Rejected);
        assert_eq!(empty.reason.as_deref(), Some("empty"));

        let loop_hallucination = gate_utterance("gracias gracias gracias", 3.1, 2.2);
        assert_eq!(loop_hallucination.path, PathType::Rejected);
        assert_eq!(
            loop_hallucination.reason.as_deref(),
            Some("high_compression")
        );

        let ok = gate_utterance("  Prendé la luz del escritorio ", 0.9, 2.2);
        assert_eq!(ok.path, PathType::SlowLlm);
        assert_eq!(ok.normalized_text, "prendé la luz del escritorio");
    }
}
