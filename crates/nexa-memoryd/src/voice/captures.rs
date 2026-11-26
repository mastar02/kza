use rusqlite::{Connection, OptionalExtension, Row, params};
use serde_json::{Map, Value, json};

use crate::db::{
    Error, Result, Store, conversation_is_live, id, optional_id, page_limit, person_is_live,
};
use crate::erasure::{Kind, is_erased};

use super::query_string;

const CAPTURE_FIELDS: &str =
    "f.capture_id, f.utterance_id, f.encoder_model_key, f.profile_id, f.captured_at_unix_ms,
     u.room_id, u.conversation_id, COALESCE(c.owner_user_id, ''), COALESCE(p.person_id, ''),
     COALESCE(r.decision, 'pending')";
const CAPTURE_JOINS: &str = "FROM voice_fingerprints f
     JOIN ambient.utterances u ON u.utterance_id=f.utterance_id
     JOIN ambient.capture_conversations c ON c.conversation_id=u.conversation_id
     LEFT JOIN voice_profile_people p ON p.profile_id=f.profile_id
     LEFT JOIN voice_capture_reviews r ON r.capture_id=f.capture_id";

pub(super) struct Capture {
    fingerprint_id: String,
    utterance_id: String,
    model: String,
    pub(super) profile_id: Option<String>,
    captured_at: u64,
    room: String,
    conversation: String,
    owner: String,
    linked_person: String,
    pub(super) review: String,
}

impl Capture {
    fn from_row(row: &Row<'_>) -> rusqlite::Result<Self> {
        Ok(Self {
            fingerprint_id: row.get(0)?,
            utterance_id: row.get(1)?,
            model: row.get(2)?,
            profile_id: row.get(3)?,
            captured_at: row.get(4)?,
            room: row.get(5)?,
            conversation: row.get(6)?,
            owner: row.get(7)?,
            linked_person: row.get(8)?,
            review: row.get(9)?,
        })
    }

    fn is_live(&self, conn: &Connection) -> Result<bool> {
        if !conversation_is_live(conn, &self.conversation)?
            || is_erased(conn, Kind::Utterance, &self.utterance_id)?
            || !person_is_live(conn, &self.owner)?
            || !person_is_live(conn, &self.linked_person)?
        {
            return Ok(false);
        }
        let mut stmt = conn.prepare_cached(
            "SELECT s.conversation_id, COALESCE(c.owner_user_id, '')
             FROM voice_fingerprint_sources s
             LEFT JOIN ambient.capture_conversations c USING (conversation_id)
             WHERE s.capture_id=?1",
        )?;
        let mut rows = stmt.query([&self.fingerprint_id])?;
        while let Some(row) = rows.next()? {
            if !conversation_is_live(conn, &row.get::<_, String>(0)?)?
                || !person_is_live(conn, &row.get::<_, String>(1)?)?
            {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn review_json(&self) -> Value {
        json!({
            "fingerprint_id": self.fingerprint_id,
            "capture_id": self.fingerprint_id,
            "utterance_id": self.utterance_id,
            "encoder_model_key": self.model,
            "profile_id": self.profile_id,
            "captured_at_unix_ms": self.captured_at,
            "room_id": self.room,
            "review_status": match self.review.as_str() {
                "accept" => "accepted",
                "reject" => "rejected",
                _ => "pending",
            },
        })
    }
}

pub(super) fn live_capture(conn: &Connection, fingerprint: &str) -> Result<Capture> {
    id(fingerprint)?;
    let sql = format!("SELECT {CAPTURE_FIELDS} {CAPTURE_JOINS} WHERE f.capture_id=?1");
    let capture = conn
        .query_row(&sql, [fingerprint], Capture::from_row)
        .optional()?;
    match capture {
        Some(capture) if capture.is_live(conn)? => Ok(capture),
        _ => Err(Error::NotFound("voice capture unavailable")),
    }
}

impl Store {
    pub(crate) fn voice_capture_review(
        &self,
        fingerprint: &str,
        include_transcript: bool,
    ) -> Result<Value> {
        let conn = self.read()?;
        let capture = live_capture(conn, fingerprint)?;
        let mut review = capture.review_json();
        if include_transcript {
            let preview: String = conn.query_row(
                "SELECT transcript_preview FROM ambient.utterances WHERE utterance_id=?1",
                [&capture.utterance_id],
                |row| row.get(0),
            )?;
            review["transcript_preview"] = json!(preview);
        }
        Ok(review)
    }

    pub(crate) fn evaluation_sample(&self, fingerprint: &str) -> Result<Value> {
        let sample = live_capture(self.read()?, fingerprint)?;
        // The RPC carries no modality or source grant. Expose sample metadata only.
        Ok(json!({
            "fingerprint_id": sample.fingerprint_id,
            "encoder_model_key": sample.model,
            "captured_at_unix_ms": sample.captured_at,
        }))
    }

    pub(crate) fn query_voice_captures(&self, query: &Map<String, Value>) -> Result<Value> {
        for key in query.keys() {
            if !matches!(
                key.as_str(),
                "room" | "room_id" | "cursor" | "limit" | "review_status"
            ) {
                return Err(Error::Invalid("unsupported capture filter"));
            }
        }
        if query.contains_key("room") && query.contains_key("room_id") {
            return Err(Error::Invalid("ambiguous room filter"));
        }
        let room = match query.get("room").or_else(|| query.get("room_id")) {
            None | Some(Value::Null) => "",
            Some(Value::Object(room))
                if room.len() == 2 && room.get("kind").and_then(Value::as_str) == Some("room") =>
            {
                room.get("id")
                    .and_then(Value::as_str)
                    .ok_or(Error::Invalid("invalid room filter"))?
            }
            Some(value) => value
                .as_str()
                .ok_or(Error::Invalid("invalid room filter"))?,
        };
        optional_id(room)?;
        let cursor = query_string(query, "cursor")?;
        optional_id(cursor)?;
        let review = match query_string(query, "review_status")? {
            "" => "",
            "pending" => "pending",
            "accept" | "accepted" => "accept",
            "reject" | "rejected" => "reject",
            _ => return Err(Error::Invalid("invalid capture review status")),
        };
        let limit = match query.get("limit") {
            None | Some(Value::Null) => 0,
            Some(Value::String(value)) => value
                .parse::<u32>()
                .map_err(|_| Error::Invalid("invalid capture limit"))?,
            Some(value) => value
                .as_u64()
                .and_then(|n| u32::try_from(n).ok())
                .ok_or(Error::Invalid("invalid capture limit"))?,
        };
        let limit = page_limit(limit);
        let conn = self.read()?;
        let sql = format!(
            "SELECT {CAPTURE_FIELDS} {CAPTURE_JOINS}
             WHERE f.capture_id>?1 AND (?2='' OR u.room_id=?2)
             AND (?3='' OR COALESCE(r.decision, 'pending')=?3)
             ORDER BY f.capture_id"
        );
        let mut stmt = conn.prepare(&sql)?;
        let mut rows = stmt.query(params![cursor, room, review])?;
        let mut captures = Vec::with_capacity(limit);
        let mut last_id = None;
        let mut next = None;
        while let Some(row) = rows.next()? {
            let capture = Capture::from_row(row)?;
            if !capture.is_live(conn)? {
                continue;
            }
            if captures.len() == limit {
                next = last_id;
                break;
            }
            captures.push(capture.review_json());
            last_id = Some(capture.fingerprint_id);
        }
        Ok(json!({"captures": captures, "next": next, "limit": limit}))
    }
}
