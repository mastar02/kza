use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Mutex;

use nexa_llm::ChatTurn;
use nexa_proto::v1::{AudioUtterance, TranscribeResponse};

use super::source::{SourceClassifier, SourceKind, TvTracker};

const SESSION_IDLE_GAP_MS: u64 = 45_000;
const SESSION_MAX_TURNS: usize = 24;
const ERASED_FENCE_CAPACITY: usize = 256;

#[derive(Debug, Clone)]
pub(crate) struct AmbientTurn {
    pub text: String,
    pub matched_profile_id: Option<String>,
    pub captured_at_unix_ms: u64,
}

#[derive(Debug, Clone)]
pub(crate) struct ClosedSession {
    pub conversation_id: String,
    pub room_id: String,
    pub turns: Vec<AmbientTurn>,
    pub matched_profiles: HashSet<String>,
}

#[derive(Debug)]
struct ActiveSession {
    conversation_id: String,
    room_id: String,
    turns: Vec<AmbientTurn>,
    last_at_unix_ms: u64,
}

impl ActiveSession {
    fn into_closed(self) -> ClosedSession {
        let matched_profiles = self
            .turns
            .iter()
            .filter_map(|turn| turn.matched_profile_id.clone())
            .collect();
        ClosedSession {
            conversation_id: self.conversation_id,
            room_id: self.room_id,
            turns: self.turns,
            matched_profiles,
        }
    }
}

#[derive(Debug, Default)]
pub(crate) struct ConversationTracker {
    sessions: HashMap<String, ActiveSession>,
}

impl ConversationTracker {
    pub(crate) fn prepare_conversation_id(
        &mut self,
        room_id: &str,
        now_ms: u64,
    ) -> (String, Option<ClosedSession>) {
        if let Some(existing) = self.sessions.get(room_id)
            && now_ms.saturating_sub(existing.last_at_unix_ms) <= SESSION_IDLE_GAP_MS
            && existing.turns.len() < SESSION_MAX_TURNS
        {
            return (existing.conversation_id.clone(), None);
        }
        let closed = self
            .sessions
            .remove(room_id)
            .map(ActiveSession::into_closed);
        let conversation_id = crate::id("conv");
        self.sessions.insert(
            room_id.to_owned(),
            ActiveSession {
                conversation_id: conversation_id.clone(),
                room_id: room_id.to_owned(),
                turns: Vec::new(),
                last_at_unix_ms: now_ms,
            },
        );
        (conversation_id, closed)
    }

    pub(crate) fn push_turn(
        &mut self,
        room_id: &str,
        conversation_id: &str,
        turn: AmbientTurn,
    ) -> Option<ClosedSession> {
        let session = self.sessions.get_mut(room_id)?;
        if session.conversation_id != conversation_id {
            return None;
        }
        session.last_at_unix_ms = session.last_at_unix_ms.max(turn.captured_at_unix_ms);
        session.turns.push(turn);
        if session.turns.len() >= SESSION_MAX_TURNS {
            return self
                .sessions
                .remove(room_id)
                .map(ActiveSession::into_closed);
        }
        None
    }

    pub(crate) fn take_idle(&mut self, room_id: &str, now_ms: u64) -> Option<ClosedSession> {
        let expired = self.sessions.get(room_id).is_some_and(|session| {
            now_ms.saturating_sub(session.last_at_unix_ms) > SESSION_IDLE_GAP_MS
        });
        if expired {
            self.sessions
                .remove(room_id)
                .map(ActiveSession::into_closed)
        } else {
            None
        }
    }

    pub(crate) fn evict_conversation(&mut self, conversation_id: &str) {
        self.sessions
            .retain(|_, session| session.conversation_id != conversation_id);
    }
}

#[derive(Debug, Clone)]
struct HistoryEntry {
    conversation_id: String,
    role: String,
    content: String,
}

#[derive(Debug)]
pub(crate) struct ShortTermHistory {
    max_turns: usize,
    per_room: HashMap<String, VecDeque<HistoryEntry>>,
}

impl ShortTermHistory {
    pub(crate) fn new(max_turns: usize) -> Self {
        Self {
            max_turns: max_turns.max(1),
            per_room: HashMap::new(),
        }
    }

    pub(crate) fn push(&mut self, room_id: &str, conversation_id: &str, role: &str, content: &str) {
        let trimmed = content.trim();
        if trimmed.is_empty() {
            return;
        }
        let queue = self.per_room.entry(room_id.to_owned()).or_default();
        if queue.len() >= self.max_turns {
            queue.pop_front();
        }
        queue.push_back(HistoryEntry {
            conversation_id: conversation_id.to_owned(),
            role: role.to_owned(),
            content: trimmed.to_owned(),
        });
    }

    pub(crate) fn chat_turns(&self, room_id: &str) -> Vec<ChatTurn> {
        self.per_room
            .get(room_id)
            .map(|queue| {
                queue
                    .iter()
                    .map(|entry| ChatTurn {
                        role: entry.role.clone(),
                        content: entry.content.clone(),
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    pub(crate) fn evict_conversation(&mut self, conversation_id: &str) {
        for queue in self.per_room.values_mut() {
            queue.retain(|entry| entry.conversation_id != conversation_id);
        }
    }
}

#[derive(Debug)]
struct AmbientInner {
    tv: TvTracker,
    conversations: ConversationTracker,
    history: ShortTermHistory,
    erased: VecDeque<String>,
}

#[derive(Debug)]
pub(crate) struct AmbientState {
    inner: Mutex<AmbientInner>,
}

impl AmbientState {
    pub(crate) fn new(short_term_turns: usize) -> Self {
        Self {
            inner: Mutex::new(AmbientInner {
                tv: TvTracker::default(),
                conversations: ConversationTracker::default(),
                history: ShortTermHistory::new(short_term_turns),
                erased: VecDeque::with_capacity(ERASED_FENCE_CAPACITY),
            }),
        }
    }

    pub(crate) fn classify(
        &self,
        utterance: &AudioUtterance,
        stt: &TranscribeResponse,
        room_media_playing: bool,
    ) -> SourceKind {
        let mut guard = self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        SourceClassifier::classify(utterance, stt, room_media_playing, &mut guard.tv)
    }

    pub(crate) fn prepare_ambient(
        &self,
        room_id: &str,
        now_ms: u64,
    ) -> (String, Option<ClosedSession>) {
        let mut guard = self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        guard.conversations.prepare_conversation_id(room_id, now_ms)
    }

    pub(crate) fn finish_ambient(
        &self,
        room_id: &str,
        conversation_id: &str,
        turn: AmbientTurn,
    ) -> Option<ClosedSession> {
        let mut guard = self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if guard.erased.iter().any(|id| id == conversation_id) {
            return None;
        }
        guard
            .conversations
            .push_turn(room_id, conversation_id, turn)
    }

    pub(crate) fn take_idle(&self, room_id: &str, now_ms: u64) -> Option<ClosedSession> {
        let mut guard = self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        guard.conversations.take_idle(room_id, now_ms)
    }

    pub(crate) fn record_chat_turn(
        &self,
        room_id: &str,
        conversation_id: &str,
        role: &str,
        content: &str,
    ) {
        let mut guard = self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if guard.erased.iter().any(|id| id == conversation_id) {
            return;
        }
        guard.history.push(room_id, conversation_id, role, content);
    }

    pub(crate) fn chat_turns(&self, room_id: &str) -> Vec<ChatTurn> {
        let guard = self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        guard.history.chat_turns(room_id)
    }

    pub(crate) fn evict_conversation(&self, conversation_id: &str) {
        let mut guard = self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        guard.conversations.evict_conversation(conversation_id);
        guard.history.evict_conversation(conversation_id);
        if !guard.erased.iter().any(|id| id == conversation_id) {
            if guard.erased.len() == ERASED_FENCE_CAPACITY {
                guard.erased.pop_front();
            }
            guard.erased.push_back(conversation_id.to_owned());
        }
    }

    pub(crate) fn is_erased(&self, conversation_id: &str) -> bool {
        let guard = self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        guard.erased.iter().any(|id| id == conversation_id)
    }
}
