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
