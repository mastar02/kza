//! Deterministic, I/O-free NLU for Nexa.

#![deny(unsafe_code)]
#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used))]

// All patterns are constant and are compiled once, at first use.
macro_rules! re {
    ($pattern:expr) => {{
        static PATTERN: std::sync::LazyLock<regex::Regex> =
            std::sync::LazyLock::new(|| match regex::Regex::new($pattern) {
                Ok(regex) => regex,
                Err(error) => panic!("invalid NLU pattern: {error}"),
            });
        &*PATTERN
    }};
}

// Group a module's recognizers so construction can keep compilation off the routing path.
macro_rules! regex_patterns {
    ($($name:ident => $pattern:expr;)*) => {
        $(static $name: std::sync::LazyLock<regex::Regex> = std::sync::LazyLock::new(|| {
            match regex::Regex::new($pattern) {
                Ok(regex) => regex,
                Err(error) => panic!("invalid NLU pattern {}: {error}", stringify!($name)),
            }
        });)*

        pub(crate) fn prepare_patterns() {
            $(std::sync::LazyLock::force(&$name);)*
        }
    };
}

pub mod gate;
pub mod grammar;
pub mod household;
pub mod normalize;
pub mod router_engine;
pub mod world;

pub use normalize::normalize_utterance;
pub use router_engine::{RouteDecision, RouterEngine};

/// Compatibility probe used by the existing brain startup path.
#[must_use]
pub fn gate_utterance(
    raw: &str,
    compression_ratio: f32,
    max_compression_ratio: f32,
) -> RouteDecision {
    use nexa_proto::v1::PathType;

    let normalized_text = normalize_utterance(raw);
    let reason = if normalized_text.is_empty() {
        Some("empty".into())
    } else if compression_ratio > max_compression_ratio {
        Some("high_compression".into())
    } else {
        None
    };
    let path = if reason.is_some() {
        PathType::Rejected
    } else {
        PathType::SlowLlm
    };
    RouteDecision {
        path,
        normalized_text,
        reason,
    }
}
