//! Bounded runtime events, with metadata-only conversation history.

use nexa_proto::v1::{LiveEventKind, LiveRuntimeEvent};
use serde_json::{Value, json};
use std::{collections::VecDeque, sync::Mutex};
use tokio::sync::broadcast;

const CAPACITY: usize = 128;

#[derive(Debug)]
pub struct Events {
    sender: broadcast::Sender<LiveRuntimeEvent>,
    history: Mutex<History>,
}

#[derive(Debug, Default)]
struct History {
    recent_turns: VecDeque<LiveRuntimeEvent>,
    erased_turns: VecDeque<String>,
}

impl Default for Events {
    fn default() -> Self {
        Self {
            sender: broadcast::channel(CAPACITY).0,
            history: Mutex::default(),
        }
    }
}

impl Events {
    pub fn publish(&self, kind: LiveEventKind, room: &str, turn: &str, now: u64, payload: Value) {
        let event = LiveRuntimeEvent {
            event_id: crate::id("evt"),
            kind: kind.into(),
            room_id: room.into(),
            conversation_id: turn.into(),
            payload_json: payload.to_string(),
            emitted_at_unix_ms: now,
        };
        // Serialize fences with events so no already-admitted turn update can overtake a fence.
        let mut history = self
            .history
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if kind == LiveEventKind::ErasureFence && !turn.is_empty() {
            history
                .recent_turns
                .retain(|record| record.conversation_id != turn);
            if !history.erased_turns.iter().any(|id| id == turn) {
                if history.erased_turns.len() == CAPACITY {
                    history.erased_turns.pop_front();
                }
                history.erased_turns.push_back(turn.to_owned());
            }
        } else {
            if history.erased_turns.iter().any(|id| id == turn) {
                return;
            }
            if kind == LiveEventKind::Turn {
                if history.recent_turns.len() == CAPACITY {
                    history.recent_turns.pop_front();
                }
                history.recent_turns.push_back(event.clone());
            }
        }
        let _ = self.sender.send(event);
    }

    pub fn subscribe(&self) -> broadcast::Receiver<LiveRuntimeEvent> {
        self.sender.subscribe()
    }

    pub(crate) fn conversations(&self, room: &str, limit: usize) -> Value {
        let history = self
            .history
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        Value::Array(history.recent_turns.iter().rev()
            .filter(|event| room.is_empty() || event.room_id == room)
            .take(limit)
            .map(|event| json!({
                "conversation_id": event.conversation_id,
                "room_id": event.room_id,
                "observed_at_unix_ms": event.emitted_at_unix_ms,
                "routing": serde_json::from_str::<Value>(&event.payload_json).unwrap_or_default(),
            }))
            .collect())
    }
}
