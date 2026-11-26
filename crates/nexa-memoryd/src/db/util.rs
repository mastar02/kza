use std::fs::{self, DirBuilder, File, OpenOptions};
use std::io;
use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt, PermissionsExt};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use rusqlite::Connection;
use serde_json::{Map, Value};

use super::{Error, Result};

pub(crate) fn check_revision(expected: Option<u64>, current: u64) -> Result<()> {
    if expected.is_some_and(|revision| revision != current) {
        return Err(Error::Conflict("revision conflict"));
    }
    Ok(())
}

pub(crate) fn json_object(input: &str) -> Result<Map<String, Value>> {
    if input.is_empty() {
        return Ok(Map::new());
    }
    serde_json::from_str(input).map_err(|_| Error::Invalid("expected JSON object"))
}

pub(crate) fn json_revision(body: &Map<String, Value>, key: &str) -> Result<Option<u64>> {
    body.get(key)
        .map(|value| {
            value
                .as_u64()
                .ok_or(Error::Invalid("invalid expected revision"))
        })
        .transpose()
}

pub(crate) fn page_limit(limit: u32) -> usize {
    if limit == 0 {
        50
    } else {
        limit.min(100) as usize
    }
}

pub(crate) fn id(value: &str) -> Result<()> {
    if value.trim().is_empty()
        || value.trim() != value
        || value.len() > 256
        || value.chars().any(char::is_control)
    {
        return Err(Error::Invalid("invalid identifier"));
    }
    Ok(())
}

pub(crate) fn optional_id(value: &str) -> Result<()> {
    if value.is_empty() { Ok(()) } else { id(value) }
}

pub(crate) fn now_ms() -> Result<u64> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .ok()
        .and_then(|duration| i64::try_from(duration.as_millis()).ok())
        .map(|ms| ms as u64)
        .ok_or(Error::Unavailable("system clock out of range"))
}

pub(crate) fn new_id(conn: &Connection) -> Result<String> {
    Ok(conn.query_row("SELECT lower(hex(randomblob(16)))", [], |row| row.get(0))?)
}

pub(crate) fn path_str(path: &Path) -> Result<&str> {
    path.to_str().ok_or(Error::Invalid("non-UTF-8 path"))
}

pub fn ensure_private_dir(path: &Path) -> Result<()> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.is_dir() && metadata.permissions().mode() & 0o077 == 0 => Ok(()),
        Ok(_) => Err(Error::Invalid(
            "database and socket directories must be private directories",
        )),
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            DirBuilder::new().recursive(true).mode(0o700).create(path)?;
            ensure_private_dir(path)
        }
        Err(error) => Err(error.into()),
    }
}

pub(super) fn private_file(path: &Path) -> Result<File> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.is_file() && metadata.permissions().mode() & 0o077 == 0 => {}
        Ok(_) => {
            return Err(Error::Invalid(
                "database and lock files must be private regular files",
            ));
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.into()),
    }
    Ok(OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .mode(0o600)
        .open(path)?)
}

pub(crate) fn lock_file(path: &Path) -> Result<File> {
    let file = private_file(path)?;
    file.try_lock()
        .map_err(|_| Error::Unavailable("exclusive memoryd lock unavailable"))?;
    Ok(file)
}
