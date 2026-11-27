use nexa_proto::v1::AmbientControlStatus;
use rusqlite::{Connection, params};

use super::{Error, Result, Store, now_ms};

impl Store {
    pub fn ambient_control(&self) -> Result<AmbientControlStatus> {
        ambient_control(self.read()?)
    }

    pub fn set_ambient_pause(
        &mut self,
        paused: bool,
        expected: u64,
    ) -> Result<AmbientControlStatus> {
        let tx = self.write()?;
        let previous = ambient_control(&tx)?;
        if previous.revision != expected {
            return Err(Error::Conflict("ambient revision conflict"));
        }
        if previous.paused != paused {
            tx.execute(
                "UPDATE ambient.ambient_runtime_control
                 SET paused=?1, revision=revision+1, updated_at_unix_ms=?2 WHERE id=1",
                params![paused, now_ms()?],
            )?;
        }
        let current = ambient_control(&tx)?;
        tx.commit()?;
        Ok(current)
    }

    pub fn prune_expired_ambient(&mut self, retention_days: u32) -> Result<u32> {
        if retention_days == 0 {
            return Ok(0);
        }
        let now = now_ms()?;
        let ttl_ms = u64::from(retention_days).saturating_mul(86_400_000);
        let cutoff = now.saturating_sub(ttl_ms);
        if cutoff == 0 {
            return Ok(0);
        }
        let expired: Vec<String> = {
            let conn = self.read()?;
            let mut stmt = conn.prepare(
                "SELECT c.conversation_id
                 FROM ambient.capture_conversations c
                 WHERE c.started_at_unix_ms < ?1
                 AND NOT EXISTS (
                     SELECT 1 FROM facts f
                     WHERE (f.conversation_id = c.conversation_id
                            OR f.fact_id IN (
                                SELECT fs.fact_id FROM fact_sources fs
                                WHERE fs.conversation_id = c.conversation_id
                            ))
                     AND f.confirmed = 1
                 )",
            )?;
            let rows = stmt.query_map([cutoff], |row| row.get(0))?;
            rows.collect::<std::result::Result<_, _>>()?
        };
        let mut erased = 0_u32;
        for conversation_id in expired {
            let _ = self.erase_conversation(&conversation_id)?;
            erased = erased.saturating_add(1);
        }
        Ok(erased)
    }
}

pub(super) fn ambient_control(conn: &Connection) -> Result<AmbientControlStatus> {
    Ok(conn.query_row(
        "SELECT paused, revision, updated_at_unix_ms FROM ambient.ambient_runtime_control WHERE id=1",
        [],
        |row| Ok(AmbientControlStatus {
            paused: row.get(0)?, revision: row.get(1)?, updated_at_unix_ms: row.get(2)?,
        }),
    )?)
}
