use std::sync::OnceLock;

use rusqlite::{Connection, params};

use super::{Error, Result};

pub const FACT_MODEL: &str = "BAAI/bge-m3";
pub const VOICE_MODEL: &str = "speechbrain/spkrec-ecapa-voxceleb";
pub const FACT_DIMENSIONS: usize = 1024;
pub const VOICE_DIMENSIONS: usize = 192;

#[derive(Clone, Copy, Debug)]
pub(crate) enum VectorKind {
    Fact,
    Voice,
}

impl VectorKind {
    pub(super) fn spec(self) -> (&'static str, &'static str, usize) {
        match self {
            Self::Fact => ("fact", FACT_MODEL, FACT_DIMENSIONS),
            Self::Voice => ("voice", VOICE_MODEL, VOICE_DIMENSIONS),
        }
    }
}

pub(crate) fn vector(
    conn: &Connection,
    kind: VectorKind,
    model: &str,
    coordinates: &[f32],
) -> Result<Vec<u8>> {
    let (kind, model_key, expected_dimensions) = kind.spec();
    if model != model_key {
        return Err(Error::Invalid("unregistered vector model"));
    }
    let dimensions: usize = conn.query_row(
        "SELECT dimensions FROM schema_meta WHERE model_key=?1 AND vector_kind=?2",
        params![model, kind],
        |row| row.get(0),
    )?;
    if dimensions != expected_dimensions {
        return Err(Error::Schema("registered vector dimensions changed"));
    }
    if coordinates.len() != dimensions
        || coordinates.iter().any(|coordinate| !coordinate.is_finite())
    {
        return Err(Error::Invalid("invalid vector dimensions or coordinates"));
    }
    // Normalizing in double precision prevents float32 overflow in cosine distance.
    let norm = coordinates
        .iter()
        .map(|value| f64::from(*value).powi(2))
        .sum::<f64>()
        .sqrt();
    if norm == 0.0 {
        return Err(Error::Invalid("zero vector"));
    }
    Ok(coordinates
        .iter()
        .flat_map(|value| ((f64::from(*value) / norm) as f32).to_ne_bytes())
        .collect())
}

/// Register the statically linked extension before opening any SQLite connection.
#[allow(unsafe_code)]
pub(super) fn register_vec() -> Result<()> {
    static STATUS: OnceLock<i32> = OnceLock::new();
    let status = *STATUS.get_or_init(|| {
        // SAFETY: sqlite-vec exposes the SQLite extension entry point with the exact C
        // ABI required by sqlite3_auto_extension. Its linked symbol is process-lived.
        unsafe {
            rusqlite::ffi::sqlite3_auto_extension(Some(std::mem::transmute::<
                *const (),
                unsafe extern "C" fn(
                    *mut rusqlite::ffi::sqlite3,
                    *mut *mut std::ffi::c_char,
                    *const rusqlite::ffi::sqlite3_api_routines,
                ) -> std::ffi::c_int,
            >(
                sqlite_vec::sqlite3_vec_init as *const ()
            )))
        }
    });
    if status != rusqlite::ffi::SQLITE_OK {
        return Err(Error::Unavailable(
            "sqlite-vec extension registration failed",
        ));
    }
    Ok(())
}
