use super::*;

/// A bound socket whose inode is removed when this owner exits.
#[derive(Debug)]
pub struct BoundControl {
    listener: UnixListener,
    path: PathBuf,
    inode: u64,
    device: u64,
}

impl BoundControl {
    pub fn bind(path: &Path) -> io::Result<Self> {
        let dir = path
            .parent()
            .ok_or_else(|| io::Error::other("brain socket requires a parent directory"))?;
        let mut builder = fs::DirBuilder::new();
        builder.recursive(true).mode(0o700).create(dir)?;
        let uid = rustix::process::geteuid().as_raw();
        let metadata = fs::symlink_metadata(dir)?;
        if !metadata.is_dir() || metadata.uid() != uid {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "runtime directory is not owned by daemon UID",
            ));
        }
        if metadata.mode() & 0o7777 != 0o700 {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "runtime directory mode must be 0700",
            ));
        }
        let listener = UnixListener::bind(path)?;
        let metadata = match fs::symlink_metadata(path) {
            Ok(metadata) => metadata,
            Err(error) => {
                let _ = fs::remove_file(path);
                return Err(error);
            }
        };
        let bound = Self {
            listener,
            path: path.to_owned(),
            inode: metadata.ino(),
            device: metadata.dev(),
        };
        fs::set_permissions(path, fs::Permissions::from_mode(0o600))?;
        Ok(bound)
    }

    pub async fn serve(self, brain: Arc<Brain>) -> Result<(), BrainError> {
        let listener = &self.listener;
        let incoming = async_stream::stream! {
            let uid = rustix::process::geteuid().as_raw();
            loop {
                match listener.accept().await {
                    Ok((stream, _)) if stream.peer_cred().is_ok_and(|peer| peer.uid() == uid) => yield Ok(stream),
                    Ok(_) => tracing::warn!("rejected brain socket peer"),
                    Err(error) => { yield Err(error); break; }
                }
            }
        };
        let stop = brain.shutdown_token();
        Server::builder()
            .add_service(BrainControlServiceServer::new(Control(brain)))
            .serve_with_incoming_shutdown(incoming, stop.cancelled_owned())
            .await?;
        Ok(())
    }
}

impl Drop for BoundControl {
    fn drop(&mut self) {
        if fs::symlink_metadata(&self.path).is_ok_and(|metadata| {
            metadata.file_type().is_socket()
                && metadata.ino() == self.inode
                && metadata.dev() == self.device
        }) {
            let _ = fs::remove_file(&self.path);
        }
    }
}
