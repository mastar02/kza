//! Atomic on-disk capture policy and single-daemon ownership.

use std::fs::{DirBuilder, File, OpenOptions};
use std::io::{self, Read, Write};
use std::os::unix::fs::{DirBuilderExt, MetadataExt, OpenOptionsExt};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use rustix::fs::{FlockOperation, flock};
use tempfile::NamedTempFile;

use crate::capture_seal::{CaptureControl, CaptureError, CaptureState};

#[derive(Debug)]
pub struct CaptureStore {
    directory: PathBuf,
    _lock: File,
}

impl CaptureStore {
    /// Own the capture state for this data directory until the daemon stops.
    pub fn open(data_dir: &Path) -> io::Result<Self> {
        let directory = data_dir.join("audiod");
        private_directory(&directory)?;
        let lock = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .mode(0o600)
            .custom_flags(rustix::fs::OFlags::NOFOLLOW.bits() as i32)
            .open(directory.join("capture_seal.lock"))?;
        flock(&lock, FlockOperation::NonBlockingLockExclusive)?;
        Ok(Self {
            directory,
            _lock: lock,
        })
    }

    /// Invalid or unreadable state aborts startup before any capture can open.
    pub fn restore(self) -> Result<CaptureControl, CaptureError> {
        let path = self.directory.join("capture_seal.json");
        let state = match OpenOptions::new()
            .read(true)
            .custom_flags(
                (rustix::fs::OFlags::NOFOLLOW | rustix::fs::OFlags::NONBLOCK).bits() as i32,
            )
            .open(path)
        {
            Ok(file) => {
                let metadata = file.metadata().map_err(|_| CaptureError::Persistence)?;
                if !metadata.is_file()
                    || metadata.uid() != rustix::process::geteuid().as_raw()
                    || metadata.mode() & 0o777 != 0o600
                {
                    return Err(CaptureError::InvalidState);
                }
                let mut bytes = Vec::new();
                file.take(1024 * 1024 + 1)
                    .read_to_end(&mut bytes)
                    .map_err(|_| CaptureError::Persistence)?;
                if bytes.len() > 1024 * 1024 {
                    return Err(CaptureError::InvalidState);
                }
                serde_json::from_slice(&bytes).map_err(|_| CaptureError::InvalidState)?
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => CaptureState::default(),
            Err(error) => {
                tracing::error!(%error, "could not read capture seal");
                return Err(CaptureError::Persistence);
            }
        };
        let mut control = CaptureControl::restore(state, true)?;
        control.attach_store(Arc::new(self));
        Ok(control)
    }

    pub(crate) fn save(&self, state: &CaptureState) -> io::Result<()> {
        let mut temporary = NamedTempFile::new_in(&self.directory)?;
        serde_json::to_writer(temporary.as_file_mut(), state)?;
        temporary.write_all(b"\n")?;
        temporary.as_file().sync_all()?;
        temporary.persist(self.directory.join("capture_seal.json"))?;
        File::open(&self.directory)?.sync_all()
    }
}

pub(crate) fn private_directory(path: &Path) -> io::Result<()> {
    DirBuilder::new().recursive(true).mode(0o700).create(path)?;
    let metadata = path.symlink_metadata()?;
    if !metadata.is_dir()
        || metadata.uid() != rustix::process::geteuid().as_raw()
        || metadata.mode() & 0o777 != 0o700
    {
        return Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            "directory must be owned by the daemon with mode 0700",
        ));
    }
    Ok(())
}
