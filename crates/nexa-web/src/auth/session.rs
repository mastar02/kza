use super::*;

#[derive(Clone)]
pub(crate) struct Presented {
    digest: String,
    cookie: bool,
    binding: Option<(String, u64)>,
}

impl Presented {
    fn new(raw: &str, cookie: bool) -> Option<Self> {
        if raw.is_empty() || raw.len() > 4096 || (cookie && !raw.starts_with("kzas_")) {
            return None;
        }
        Some(Self {
            digest: sha256(raw),
            cookie,
            binding: None,
        })
    }

    pub(crate) fn bind(mut self, lease: &Lease) -> Self {
        self.binding = Some((lease.lease_id.clone(), lease.revision));
        self
    }
}

pub(crate) fn cookie(headers: &HeaderMap) -> Option<Presented> {
    let mut tokens = headers
        .get_all(header::COOKIE)
        .iter()
        .filter_map(|value| value.to_str().ok())
        .flat_map(|value| value.split(';'))
        .filter_map(|pair| pair.trim().split_once('='))
        .filter(|(name, _)| *name == COOKIE_NAME);
    let (_, token) = tokens.next()?;
    if tokens.next().is_some() {
        return None;
    }
    Presented::new(token, true)
}

pub(crate) fn presented(headers: &HeaderMap) -> Option<Presented> {
    let mut values = headers.get_all(header::AUTHORIZATION).iter();
    if let Some(value) = values.next() {
        if values.next().is_some() {
            return None;
        }
        let raw = value.to_str().ok()?;
        let token = raw
            .split_once(' ')
            .filter(|(scheme, _)| scheme.eq_ignore_ascii_case("bearer"))
            .map_or(raw, |(_, value)| value.trim());
        return Presented::new(token, false);
    }
    cookie(headers)
}

pub(crate) fn session_cookie(
    token: &str,
    max_age: u64,
    insecure: bool,
) -> Result<HeaderValue, AuthError> {
    let secure = if insecure { "" } else { "; Secure" };
    let value = format!(
        "{COOKIE_NAME}={token}; Max-Age={max_age}; HttpOnly; SameSite=Strict; Path=/{secure}"
    );
    HeaderValue::from_str(&value).map_err(|_| AuthError::Unavailable)
}

#[derive(Deserialize)]
pub(super) struct Sample {
    pub(super) fingerprint_id: String,
    source_digests: Vec<String>,
    pub(super) modalities: Vec<String>,
    pub(super) expires_at: f64,
}

pub(crate) struct Lease {
    pub(super) id: String,
    subject: String,
    role: Role,
    user_id: Option<String>,
    expires_at: f64,
    revision: u64,
    lease_id: String,
    samples: Vec<Sample>,
    session: bool,
}

impl Lease {
    pub(crate) fn actor(&self) -> ActorContext {
        ActorContext {
            subject: self.subject.clone(),
            role: self.role as i32,
            permissions: permissions(self.role, self.user_id.is_some())
                .into_iter()
                .map(|p| p as i32)
                .collect(),
            user_id: self.user_id.clone().unwrap_or_default(),
            lease_id: self.lease_id.clone(),
            lease_epoch: self.revision,
        }
    }

    pub(crate) fn has(&self, permission: Permission) -> bool {
        permissions(self.role, self.user_id.is_some()).contains(&permission)
    }

    pub(crate) fn require(&self, required: &[Permission]) -> Result<(), AuthError> {
        if required.iter().all(|permission| self.has(*permission)) {
            Ok(())
        } else {
            Err(AuthError::Forbidden)
        }
    }

    pub(crate) fn expires_at(&self) -> f64 {
        self.expires_at
    }

    pub(crate) fn whoami(&self) -> Value {
        let role = role_name(self.role);
        let binding = self.user_id.as_ref().map(|user| {
            let material = ["kza-dashboard-account-binding-v1", &self.id, role, user].join("\0");
            sha256(&material)[..32].to_owned()
        });
        let mut names: Vec<_> = permissions(self.role, self.user_id.is_some())
            .into_iter()
            .map(permission_name)
            .collect();
        names.sort_unstable();
        json!({
            "ok": true, "auth_configured": true, "subject": self.subject, "role": role,
            "permissions": names, "expires_at": self.expires_at, "personal_scope": self.user_id.is_some(),
            "account_binding": binding, "routine_actor": format!("\u{001f}routine:{}", sha256(&self.id))
        })
    }

    pub(crate) fn evaluation_grants(&self) -> Value {
        let grants: Vec<_> = self
            .samples
            .iter()
            .filter(|sample| sample.expires_at > now())
            .map(|sample| {
                json!({"fingerprint_id": sample.fingerprint_id, "modalities": sample.modalities,
                                 "expires_at": sample.expires_at.min(self.expires_at)})
            })
            .collect();
        json!({"grants": grants})
    }

    pub(crate) fn evaluation_sources(
        &self,
        id: &str,
        modality: &str,
    ) -> Result<&[String], AuthError> {
        self.samples
            .iter()
            .find(|sample| {
                sample.fingerprint_id == id
                    && sample.expires_at > now()
                    && sample.modalities.iter().any(|value| value == modality)
            })
            .map(|sample| sample.source_digests.as_slice())
            .ok_or(AuthError::Forbidden)
    }
}

struct Row {
    id: String,
    digest: String,
    subject: String,
    role: String,
    user: Option<String>,
    expiry: f64,
    revision: u64,
    samples: String,
}

pub(super) fn lookup(db: &Connection, proof: &Presented) -> Result<Lease, AuthError> {
    let credential =
        "SELECT c.id,c.digest,c.subject,c.role,c.user_id,c.expires_at,c.revision,c.samples
        FROM credentials c WHERE c.digest=?1 AND c.expires_at>?2
        AND NOT EXISTS(SELECT 1 FROM retired_users r WHERE r.user_id=c.user_id)";
    let session = "SELECT c.id,s.digest,c.subject,c.role,c.user_id,min(c.expires_at,s.expires_at),c.revision,c.samples
        FROM sessions s JOIN credentials c ON c.id=s.credential AND c.revision=s.revision
        WHERE s.digest=?1 AND s.legacy_digest IS NULL AND min(c.expires_at,s.expires_at)>?2
        AND NOT EXISTS(SELECT 1 FROM retired_users r WHERE r.user_id=c.user_id)";
    let time = clock_floor(db)?;
    let read = |sql| {
        db.query_row(sql, params![proof.digest, time], |row| {
            Ok(Row {
                id: row.get(0)?,
                digest: row.get(1)?,
                subject: row.get(2)?,
                role: row.get(3)?,
                user: row.get(4)?,
                expiry: row.get(5)?,
                revision: row.get(6)?,
                samples: row.get(7)?,
            })
        })
        .optional()
    };
    let row = read(session)?;
    let session = row.is_some();
    let row = if row.is_none() && !proof.cookie {
        read(credential)?
    } else {
        row
    };
    // An indexed lookup bounds the candidate set, and the candidate verification is constant-time.
    let verified = row
        .as_ref()
        .is_some_and(|row| matches(&proof.digest, &row.digest));
    if !verified {
        return Err(AuthError::Unauthorized);
    }
    let row = row.ok_or(AuthError::Unauthorized)?;
    let role = parse_role(&row.role).ok_or(AuthError::Unauthorized)?;
    if (row.user.is_some() && role != Role::Home) || row.revision == 0 {
        return Err(AuthError::Unauthorized);
    }
    let samples: Vec<Sample> =
        serde_json::from_str(&row.samples).map_err(|_| AuthError::Unauthorized)?;
    if samples.len() > 64 || (!samples.is_empty() && role != Role::Evaluator) {
        return Err(AuthError::Unauthorized);
    }
    let lease_id = sha256(&format!("nexa-lease-v1:{}", row.digest));
    if proof
        .binding
        .as_ref()
        .is_some_and(|(id, epoch)| id != &lease_id || *epoch != row.revision)
    {
        return Err(AuthError::Unauthorized);
    }
    Ok(Lease {
        id: row.id,
        subject: row.subject,
        role,
        user_id: row.user,
        expires_at: row.expiry,
        revision: row.revision,
        lease_id,
        samples,
        session,
    })
}

impl AuthStore {
    pub(crate) async fn validate(
        &self,
        proof: &Presented,
        required: &[Permission],
    ) -> Result<Lease, AuthError> {
        let proof = proof.clone();
        let lease = self.access(move |db| lookup(db, &proof)).await?;
        lease.require(required)?;
        Ok(lease)
    }

    pub(crate) async fn login(
        &self,
        raw: &str,
        previous: Option<Presented>,
    ) -> Result<(String, u64), AuthError> {
        let proof = Presented::new(raw, false).ok_or(AuthError::Unauthorized)?;
        let token = nonce("kzas_");
        let digest = sha256(&token);
        let max_age = self
            .write(move |db, time| {
                let lease = lookup(db, &proof)?;
                if lease.session {
                    return Err(AuthError::Unauthorized);
                }
                db.execute("DELETE FROM sessions WHERE expires_at<=?1", [time])?;
                if let Some(previous) = previous {
                    db.execute("DELETE FROM sessions WHERE digest=?1", [previous.digest])?;
                }
                let count: u32 =
                    db.query_row("SELECT count(*) FROM sessions", [], |row| row.get(0))?;
                if count >= SESSION_LIMIT {
                    return Err(AuthError::Unavailable);
                }
                let expiry = lease.expires_at.min(time + TTL);
                let age = (expiry - time).floor() as u64;
                if age == 0 {
                    return Err(AuthError::Unauthorized);
                }
                db.execute(
                    "INSERT INTO sessions VALUES (?,?,NULL,?,?,?)",
                    params![digest, lease.id, time, expiry, lease.revision],
                )?;
                Ok(age)
            })
            .await?;
        Ok((token, max_age))
    }

    pub(crate) async fn logout(&self, proof: Presented) -> Result<bool, AuthError> {
        self.write(move |db, _| {
            let lease = lookup(db, &proof)?;
            Ok(lease.session
                && db.execute("DELETE FROM sessions WHERE digest=?1", [proof.digest])? == 1)
        })
        .await
    }
}
