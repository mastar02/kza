use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Mutex;

use nexa_llm::ChatTurn;
use nexa_proto::v1::{AudioUtterance, TranscribeResponse};

use super::source::{SourceClassifier, SourceKind, TvTracker};

const SESSION_IDLE_GAP_MS: u64 = 45_000;
const DISTILL_WINDOW_TURNS: usize = 24;
const DISTILL_WINDOW_CHARS: usize = 6_000;
const MAX_STITCHED_TURN_CHARS: usize = 1_600;
const MAX_ROLLING_SUMMARY_CHARS: usize = 600;
const ERASED_FENCE_CAPACITY: usize = 256;

#[derive(Debug, Clone)]
pub(crate) struct AmbientTurn {
    pub text: String,
    pub matched_profile_id: Option<String>,
    pub captured_at_unix_ms: u64,
    pub continues_segment: bool,
}

#[derive(Debug, Clone)]
pub(crate) struct ClosedSession {
    pub conversation_id: String,
    pub room_id: String,
    pub turns: Vec<AmbientTurn>,
    pub matched_profiles: HashSet<String>,
    pub window_index: u32,
    pub prior_context: Option<String>,
    pub is_final: bool,
}

#[derive(Debug)]
struct ActiveSession {
    conversation_id: String,
    room_id: String,
    turns: Vec<AmbientTurn>,
    buffered_chars: usize,
    last_at_unix_ms: u64,
    next_window_index: u32,
    rolling_summary: Option<String>,
}

impl ActiveSession {
    fn take_window(&mut self, is_final: bool) -> Option<ClosedSession> {
        if self.turns.is_empty() {
            return None;
        }
        let turns = std::mem::take(&mut self.turns);
        self.buffered_chars = 0;
        let window_index = self.next_window_index;
        self.next_window_index = self.next_window_index.saturating_add(1);
        let matched_profiles = turns
            .iter()
            .filter_map(|turn| turn.matched_profile_id.clone())
            .collect();
        Some(ClosedSession {
            conversation_id: self.conversation_id.clone(),
            room_id: self.room_id.clone(),
            turns,
            matched_profiles,
            window_index,
            prior_context: self.rolling_summary.clone(),
            is_final,
        })
    }
}

fn normalize_word(word: &str) -> String {
    word.trim_matches(|ch: char| {
        matches!(
            ch,
            '.' | ',' | ';' | ':' | '!' | '?' | '¿' | '¡' | '"' | '\'' | '(' | ')'
        )
    })
    .to_lowercase()
}

/// Join a continued segment (`continues_segment = true`), deduplicating up to 4 overlapping
/// boundary words introduced by the 480 ms segment pre-roll.
pub(crate) fn stitch_transcript(prev: &str, next: &str) -> String {
    let prev = prev.trim();
    let next = next.trim();
    if prev.is_empty() {
        return next.to_owned();
    }
    if next.is_empty() {
        return prev.to_owned();
    }
    let prev_words: Vec<&str> = prev.split_whitespace().collect();
    let next_words: Vec<&str> = next.split_whitespace().collect();
    let max_overlap = prev_words.len().min(next_words.len()).min(4);
    for overlap in (1..=max_overlap).rev() {
        let prev_tail = &prev_words[prev_words.len() - overlap..];
        let next_head = &next_words[..overlap];
        let mut has_non_empty = false;
        let matches = prev_tail.iter().zip(next_head.iter()).all(|(a, b)| {
            let na = normalize_word(a);
            let nb = normalize_word(b);
            if !na.is_empty() {
                has_non_empty = true;
            }
            na == nb
        });
        if has_non_empty && matches {
            let suffix = next_words[overlap..].join(" ");
            return if suffix.is_empty() {
                prev.to_owned()
            } else {
                format!("{prev} {suffix}")
            };
        }
    }
    format!("{prev} {next}")
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
        {
            return (existing.conversation_id.clone(), None);
        }
        let closed = self
            .sessions
            .remove(room_id)
            .and_then(|mut session| session.take_window(true));
        let conversation_id = crate::id("conv");
        self.sessions.insert(
            room_id.to_owned(),
            ActiveSession {
                conversation_id: conversation_id.clone(),
                room_id: room_id.to_owned(),
                turns: Vec::new(),
                buffered_chars: 0,
                last_at_unix_ms: now_ms,
                next_window_index: 1,
                rolling_summary: None,
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
        let mut pushed = false;
        if turn.continues_segment
            && let Some(prev) = session.turns.last_mut()
            && prev.matched_profile_id == turn.matched_profile_id
        {
            let stitched = stitch_transcript(&prev.text, &turn.text);
            let new_len = stitched.chars().count();
            if new_len <= MAX_STITCHED_TURN_CHARS {
                let old_len = prev.text.chars().count();
                session.buffered_chars = session
                    .buffered_chars
                    .saturating_sub(old_len)
                    .saturating_add(new_len);
                prev.text = stitched;
                prev.captured_at_unix_ms = prev.captured_at_unix_ms.max(turn.captured_at_unix_ms);
                pushed = true;
            }
        }
        if !pushed {
            session.buffered_chars = session
                .buffered_chars
                .saturating_add(turn.text.chars().count());
            session.turns.push(turn);
        }
        if session.turns.len() >= DISTILL_WINDOW_TURNS
            || session.buffered_chars >= DISTILL_WINDOW_CHARS
        {
            return session.take_window(false);
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
                .and_then(|mut session| session.take_window(true))
        } else {
            None
        }
    }

    pub(crate) fn record_window_summary(
        &mut self,
        room_id: &str,
        conversation_id: &str,
        facts: &[String],
    ) {
        if facts.is_empty() {
            return;
        }
        let Some(session) = self.sessions.get_mut(room_id) else {
            return;
        };
        if session.conversation_id != conversation_id {
            return;
        }
        let mut lines: VecDeque<String> = session
            .rolling_summary
            .as_deref()
            .map(|summary| summary.lines().map(ToOwned::to_owned).collect())
            .unwrap_or_default();
        for fact in facts {
            let trimmed = fact.trim();
            if !trimmed.is_empty() {
                lines.push_back(format!("- {trimmed}"));
            }
        }
        let mut total_chars = lines
            .iter()
            .map(|line| line.chars().count())
            .sum::<usize>()
            .saturating_add(lines.len().saturating_sub(1));
        while lines.len() > 1 && total_chars > MAX_ROLLING_SUMMARY_CHARS {
            if let Some(removed) = lines.pop_front() {
                total_chars = total_chars.saturating_sub(removed.chars().count() + 1);
            }
        }
        if lines.len() == 1
            && let Some(only) = lines.front_mut()
            && only.chars().count() > MAX_ROLLING_SUMMARY_CHARS
        {
            *only = only.chars().take(MAX_ROLLING_SUMMARY_CHARS).collect();
        }
        session.rolling_summary =
            (!lines.is_empty()).then(|| lines.into_iter().collect::<Vec<_>>().join("\n"));
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

    pub(crate) fn record_window_summary(
        &self,
        room_id: &str,
        conversation_id: &str,
        facts: &[String],
    ) {
        if facts.is_empty() {
            return;
        }
        let mut guard = self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if guard.erased.iter().any(|id| id == conversation_id) {
            return;
        }
        guard
            .conversations
            .record_window_summary(room_id, conversation_id, facts);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stitch_transcript_deduplicates_spanish_overlap_and_joins_cleanly() {
        assert_eq!(
            stitch_transcript(
                "Estábamos hablando de organizar,",
                "¿Organizar mejor la semana?"
            ),
            "Estábamos hablando de organizar, mejor la semana?"
        );
        assert_eq!(
            stitch_transcript(
                "Mañana vamos para organizar",
                "para organizar la compra del súper"
            ),
            "Mañana vamos para organizar la compra del súper"
        );
        assert_eq!(
            stitch_transcript("Primera frase completa.", "Segunda frase nueva."),
            "Primera frase completa. Segunda frase nueva."
        );
    }

    #[test]
    fn tracker_keeps_single_conversation_id_across_rolling_windows_and_idle_closes() {
        let mut tracker = ConversationTracker::default();
        let mut now_ms = 1_800_000_000_000_u64;
        let mut first_conv = None;
        let mut windows = Vec::new();

        for idx in 0..100 {
            let (conv_id, idle_closed) = tracker.prepare_conversation_id("living", now_ms);
            assert!(idle_closed.is_none());
            match &first_conv {
                Some(expected) => assert_eq!(&conv_id, expected),
                None => first_conv = Some(conv_id.clone()),
            }
            if let Some(window) = tracker.push_turn(
                "living",
                &conv_id,
                AmbientTurn {
                    text: format!("Turno continuo número {idx} con contenido suficiente"),
                    matched_profile_id: Some("prof-1".into()),
                    captured_at_unix_ms: now_ms,
                    continues_segment: false,
                },
            ) {
                tracker.record_window_summary(
                    "living",
                    &conv_id,
                    &[format!("Hecho ventana {}", window.window_index)],
                );
                windows.push(window);
            }
            now_ms += 12_000;
        }

        let conv_id = first_conv.expect("conversation id");
        assert_eq!(windows.len(), 4);
        assert_eq!(windows[0].window_index, 1);
        assert!(!windows[0].is_final);
        assert!(windows[0].prior_context.is_none());
        assert_eq!(windows[1].window_index, 2);
        assert_eq!(
            windows[1].prior_context.as_deref(),
            Some("- Hecho ventana 1")
        );

        let final_window = tracker
            .take_idle("living", now_ms + SESSION_IDLE_GAP_MS + 1)
            .expect("final idle window");
        assert_eq!(final_window.conversation_id, conv_id);
        assert_eq!(final_window.window_index, 5);
        assert!(final_window.is_final);
        assert_eq!(final_window.turns.len(), 4);
        assert!(
            final_window
                .prior_context
                .as_deref()
                .is_some_and(|ctx| ctx.contains("- Hecho ventana 4"))
        );
    }
}
