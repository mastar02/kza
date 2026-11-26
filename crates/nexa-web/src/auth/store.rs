use super::*;

struct Inner {
    db: Mutex<Connection>,
    _owner: File,
    changed: watch::Sender<u64>,
}

#[derive(Clone)]
pub struct AuthStore(Arc<Inner>);

impl fmt::Debug for AuthStore {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("AuthStore").finish_non_exhaustive()
    }
}

fn private_file(path: &Path) -> std::io::Result<File> {
    let file = File::from(open(
        path,
        OFlags::CREATE | OFlags::RDWR | OFlags::NOFOLLOW | OFlags::NONBLOCK | OFlags::CLOEXEC,
        Mode::RUSR | Mode::WUSR,
    )?);
    let metadata = file.metadata()?;
    if !metadata.is_file()
        || metadata.nlink() != 1
        || metadata.mode() & 0o077 != 0
        || metadata.uid() != rustix::process::geteuid().as_raw()
    {
        return Err(std::io::ErrorKind::PermissionDenied.into());
    }
    Ok(file)
}

impl AuthStore {
    /// Open the migrated dashboard authority under the configured data directory.
    pub fn open_in(data_dir: &Path) -> Result<Self, AuthError> {
        use std::os::unix::fs::DirBuilderExt;
        let directory = data_dir.join("auth");
        std::fs::DirBuilder::new()
            .recursive(true)
            .mode(0o700)
            .create(&directory)
            .map_err(|_| AuthError::Unavailable)?;
        Self::open(&directory.join("auth.db"))
    }

    /// Open the single authority writer. The first Rust startup invalidates Python-era cookies.
    pub fn open(path: &Path) -> Result<Self, AuthError> {
        let filename = path
            .file_name()
            .ok_or(AuthError::Unavailable)?
            .to_string_lossy();
        let owner = private_file(&path.with_file_name(format!("{filename}.lease")))
            .map_err(|_| AuthError::Unavailable)?;
        owner.try_lock().map_err(|_| AuthError::Unavailable)?;
        private_file(path).map_err(|_| AuthError::Unavailable)?;
        let mut db = Connection::open(path)?;
        db.busy_timeout(std::time::Duration::from_millis(500))?;
        db.pragma_update(None, "journal_mode", "WAL")?;
        db.pragma_update(None, "secure_delete", "ON")?;
        db.pragma_update(None, "foreign_keys", "ON")?;
        let tx = db.transaction_with_behavior(TransactionBehavior::Immediate)?;
        tx.execute_batch(SCHEMA)?;
        if tx.execute(
            "INSERT OR IGNORE INTO auth_meta VALUES ('nexa_session_cutover','1')",
            [],
        )? == 1
        {
            tx.execute("DELETE FROM sessions", [])?;
        }
        touch_clock(&tx, clock_floor(&tx)?)?;
        tx.commit()?;
        Ok(Self(Arc::new(Inner {
            db: Mutex::new(db),
            _owner: owner,
            changed: watch::channel(0).0,
        })))
    }

    pub(super) async fn access<T: Send + 'static>(
        &self,
        work: impl FnOnce(&mut Connection) -> Result<T, AuthError> + Send + 'static,
    ) -> Result<T, AuthError> {
        let inner = self.0.clone();
        tokio::task::spawn_blocking(move || {
            let mut db = inner.db.lock().map_err(|_| AuthError::Unavailable)?;
            work(&mut db)
        })
        .await
        .map_err(|_| AuthError::Unavailable)?
    }

    pub(crate) fn changes(&self) -> watch::Receiver<u64> {
        self.0.changed.subscribe()
    }

    pub(super) async fn write<T: Send + 'static>(
        &self,
        work: impl FnOnce(&Connection, f64) -> Result<T, AuthError> + Send + 'static,
    ) -> Result<T, AuthError> {
        let inner = self.0.clone();
        self.access(move |db| {
            let tx = db.transaction_with_behavior(TransactionBehavior::Immediate)?;
            let time = clock_floor(&tx)?;
            let result = work(&tx, time)?;
            touch_clock(&tx, time)?;
            tx.commit()?;
            inner
                .changed
                .send_modify(|counter| *counter = counter.wrapping_add(1));
            Ok(result)
        })
        .await
    }
}
impl AuthStore {
    pub(crate) async fn users(&self) -> Result<Value, AuthError> {
        self.access(|db| {
            let mut query =
                db.prepare("SELECT id,name,permission_level FROM nexa_accounts ORDER BY id")?;
            let rows = query
                .query_map([], |row| {
                    Ok(json!({
                        "id": row.get::<_,String>(0)?, "name": row.get::<_,String>(1)?,
                        "permission_level": row.get::<_,String>(2)?, "is_active": true,
                    }))
                })?
                .collect::<Result<Vec<_>, _>>()?;
            Ok(json!(rows))
        })
        .await
    }

    pub(crate) async fn write_user(
        &self,
        proof: Presented,
        id: Option<String>,
        body: Value,
    ) -> Result<Value, AuthError> {
        let level = body
            .get("permission_level")
            .and_then(Value::as_str)
            .unwrap_or("adult")
            .to_ascii_lowercase();
        if !["guest", "child", "teen", "adult", "admin"].contains(&level.as_str()) {
            return Err(AuthError::Invalid);
        }
        self.write(move |db, _| {
            lookup(db, &proof)?.require(&[Permission::AdminManage])?;
            if let Some(id) = id {
                if db.execute(
                    "UPDATE nexa_accounts SET permission_level=?1 WHERE id=?2",
                    params![level, id],
                )? != 1
                {
                    return Err(AuthError::NotFound);
                }
                Ok(json!({"ok": true, "permission_level":level.to_ascii_uppercase()}))
            } else {
                let name = body
                    .get("name")
                    .and_then(Value::as_str)
                    .filter(|s| valid_id(s) && s.chars().count() <= 64)
                    .ok_or(AuthError::Invalid)?;
                let id = Uuid::new_v4().to_string();
                if db.execute(
                    "INSERT OR IGNORE INTO nexa_accounts VALUES (?,?,?)",
                    params![id, name, level],
                )? != 1
                {
                    return Err(AuthError::Conflict);
                }
                Ok(json!({"ok": true, "user_id":id, "name":name, "message":"Usuario creado"}))
            }
        })
        .await
    }

    pub(crate) async fn retire(
        &self,
        proof: Presented,
        user_id: String,
    ) -> Result<bool, AuthError> {
        if !valid_id(&user_id) {
            return Err(AuthError::Invalid);
        }
        self.write(move |db, time| {
            let lease = lookup(db, &proof)?;
            lease.require(&[Permission::AdminManage, Permission::AuthManage])?;
            let retired = db.query_row("SELECT EXISTS(SELECT 1 FROM retired_users WHERE user_id=?1)", [&user_id],
                |row| row.get::<_,bool>(0))?;
            if !retired {
                let exists = db.query_row(
                    "SELECT EXISTS(SELECT 1 FROM nexa_accounts WHERE id=?1)
                     OR EXISTS(SELECT 1 FROM credentials WHERE user_id=?1)",
                    [&user_id], |row| row.get::<_,bool>(0),
                )?;
                if !exists {
                    return Err(AuthError::NotFound);
                }
                db.execute("INSERT INTO retired_users VALUES (?,?)", params![user_id,time])?;
                db.execute("DELETE FROM sessions WHERE credential IN (SELECT id FROM credentials WHERE user_id=?1)", [&user_id])?;
                db.execute("DELETE FROM credentials WHERE user_id=?1", [&user_id])?;
                db.execute("DELETE FROM nexa_accounts WHERE id=?1", [&user_id])?;
                db.execute("INSERT INTO nexa_purge_pending VALUES (?1)", [&user_id])?;
            }
            let pending = db.query_row("SELECT EXISTS(SELECT 1 FROM nexa_purge_pending WHERE user_id=?1)", [&user_id],
                |row| row.get::<_,bool>(0))?;
            Ok(pending)
        }).await
    }

    pub(crate) async fn finish_purge(&self, user_id: String) -> Result<(), AuthError> {
        self.write(move |db, _| {
            db.execute("DELETE FROM nexa_purge_pending WHERE user_id=?1", [user_id])?;
            Ok(())
        })
        .await
    }
}
