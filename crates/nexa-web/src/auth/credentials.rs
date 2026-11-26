use super::*;

impl AuthStore {
    pub async fn bootstrap_operator(&self, token: String) -> Result<(), AuthError> {
        if token.len() < 32 || token.len() > 4096 || token.starts_with("kzas_") {
            return Err(AuthError::Invalid);
        }
        self.seed(
            "bootstrap-operator",
            "dashboard-admin",
            Role::Operator,
            None,
            &token,
        )
        .await
    }

    pub(crate) async fn seed(
        &self,
        id: &str,
        subject: &str,
        role: Role,
        user_id: Option<&str>,
        token: &str,
    ) -> Result<(), AuthError> {
        let (id, subject, user_id, digest) = (
            id.to_owned(),
            subject.to_owned(),
            user_id.map(str::to_owned),
            sha256(token),
        );
        self.write(move |db, time| {
            if let Some(user) = &user_id {
                db.execute(
                    "INSERT OR IGNORE INTO nexa_accounts VALUES (?,?,'adult')",
                    params![user, subject],
                )?;
            }
            let existing: Option<(String, f64)> = db
                .query_row(
                    "SELECT digest,expires_at FROM credentials WHERE id=?1",
                    [&id],
                    |row| Ok((row.get(0)?, row.get(1)?)),
                )
                .optional()?;
            if existing.is_some_and(|(old, expiry)| matches(&old, &digest) && expiry > time) {
                return Ok(());
            }
            db.execute("DELETE FROM sessions WHERE credential=?1", [&id])?;
            db.execute(
                "INSERT INTO credentials VALUES (?,?,?,?,?,?,?,1,'[]')
                 ON CONFLICT(id) DO UPDATE SET digest=excluded.digest,issued_at=excluded.issued_at,
                 expires_at=excluded.expires_at,revision=credentials.revision+1",
                params![
                    id,
                    digest,
                    subject,
                    role_name(role),
                    user_id,
                    time,
                    time + TTL
                ],
            )?;
            Ok(())
        })
        .await
    }

    pub(crate) async fn credential_catalog(
        &self,
        after: String,
        limit: u32,
    ) -> Result<Value, AuthError> {
        self.access(move |db| {
            let observed = clock_floor(db)?;
            let mut query = db.prepare(
                "SELECT id,subject,role,user_id,issued_at,expires_at,revision,samples,
                 NOT EXISTS(SELECT 1 FROM retired_users r WHERE r.user_id=credentials.user_id)
                 FROM credentials WHERE id>?1 ORDER BY id LIMIT ?2",
            )?;
            let mut rows = query.query_map(params![after,limit+1], |row| {
                let expiry: f64 = row.get(5)?;
                let samples: Vec<session::Sample> = serde_json::from_str(&row.get::<_,String>(7)?)
                    .map_err(|error| rusqlite::Error::FromSqlConversionFailure(7,rusqlite::types::Type::Text,Box::new(error)))?;
                let samples: Vec<_> = samples.into_iter().map(|sample| json!({
                    "fingerprint_id":sample.fingerprint_id,"modalities":sample.modalities,"expires_at":sample.expires_at
                })).collect();
                Ok(json!({
                    "id":row.get::<_,String>(0)?, "subject":row.get::<_,String>(1)?,
                    "role":row.get::<_,String>(2)?, "user_id":row.get::<_,Option<String>>(3)?,
                    "issued_at":row.get::<_,f64>(4)?, "expires_at":expiry, "revision":row.get::<_,u64>(6)?,
                    "active":expiry>observed && row.get::<_,bool>(8)?, "pending":false, "samples":samples,
                }))
            })?.collect::<Result<Vec<_>, _>>()?;
            let next = if rows.len() > limit as usize {
                rows.pop();
                rows.last().map(|row| row["id"].clone())
            } else { None };
            Ok(json!({"rows":rows,"observed_at":observed,"next_cursor":next}))
        }).await
    }

    pub(crate) async fn credential_attempt(
        &self,
        proof: Presented,
        request_id: String,
    ) -> Result<Value, AuthError> {
        if !valid_attempt(&request_id) {
            return Err(AuthError::Invalid);
        }
        self.access(move |db| {
            let actor = lookup(db, &proof)?;
            actor.require(&[Permission::AuthManage])?;
            let receipt = db.query_row("SELECT receipt FROM nexa_credential_attempts WHERE request_id=?1 AND actor_id=?2",
                params![request_id,actor.id], |row| row.get::<_,String>(0)).optional()?.ok_or(AuthError::NotFound)?;
            serde_json::from_str(&receipt).map_err(|_| AuthError::Unavailable)
        }).await
    }

    pub(crate) async fn change_credential(
        &self,
        proof: Presented,
        action: &'static str,
        target: Option<String>,
        body: Value,
    ) -> Result<Value, AuthError> {
        let request_id = body
            .get("request_id")
            .and_then(Value::as_str)
            .filter(|id| valid_attempt(id))
            .ok_or(AuthError::Invalid)?
            .to_owned();
        let intent = sha256(&json!([action, target, body]).to_string());
        self.write(move |db, time| {
            let actor = lookup(db, &proof)?;
            actor.require(&[Permission::AuthManage])?;
            if let Some((owner, prior_intent, receipt)) = db.query_row(
                "SELECT actor_id,intent,receipt FROM nexa_credential_attempts WHERE request_id=?1", [&request_id],
                |row| Ok((row.get::<_,String>(0)?,row.get::<_,String>(1)?,row.get::<_,String>(2)?)),
            ).optional()? {
                if owner != actor.id || !matches(&prior_intent,&intent) {
                    return Err(AuthError::Conflict);
                }
                let receipt: Value = serde_json::from_str(&receipt).map_err(|_| AuthError::Unavailable)?;
                return Ok(json!({"receipt":receipt,"applied":false}));
            }
            let count: u32 = db.query_row("SELECT count(*) FROM nexa_credential_attempts", [], |row| row.get(0))?;
            if count >= ATTEMPT_LIMIT {
                return Err(AuthError::Unavailable);
            }
            let id = target.unwrap_or_else(|| request_id.clone());
            let mut revision = 1;
            let mut expected_revision = None;
            if action != "issue" {
                let expected = body.get("expected_revision").and_then(Value::as_u64).ok_or(AuthError::Invalid)?;
                expected_revision = Some(expected);
                let current: u64 = db.query_row("SELECT revision FROM credentials WHERE id=?1", [&id], |row| row.get(0))
                    .optional()?.ok_or(AuthError::NotFound)?;
                if expected != current || id == actor.id {
                    return Err(AuthError::Conflict);
                }
                revision = if action == "revoke" { current } else {
                    current.checked_add(1).filter(|value| *value <= i64::MAX as u64).ok_or(AuthError::Conflict)?
                };
            }
            let mut token = None;
            if action == "revoke" {
                db.execute("DELETE FROM sessions WHERE credential=?1", [&id])?;
                db.execute("DELETE FROM credentials WHERE id=?1", [&id])?;
            } else {
                let role = body.get("role").and_then(Value::as_str).and_then(parse_role).ok_or(AuthError::Invalid)?;
                let expires = body.get("expires_at").and_then(Value::as_f64).filter(|v| *v > time && *v <= time + TTL)
                    .ok_or(AuthError::Invalid)?;
                let user = match body.get("user_id") {
                    None | Some(Value::Null) => None,
                    Some(Value::String(user)) if role == Role::Home && valid_id(user) => Some(user),
                    _ => return Err(AuthError::Invalid),
                };
                if let Some(user) = user {
                    let exists = db.query_row(
                        "SELECT EXISTS(SELECT 1 FROM nexa_accounts WHERE id=?1)
                         AND NOT EXISTS(SELECT 1 FROM retired_users WHERE user_id=?1)", [user], |row| row.get::<_,bool>(0),
                    )?;
                    if !exists {
                        return Err(AuthError::NotFound);
                    }
                }
                if role == Role::Evaluator || body.get("selections").is_some_and(|value| value.as_array().is_none_or(|rows| !rows.is_empty())) {
                    return Err(AuthError::EvaluationUnavailable);
                }
                if action == "issue" {
                    db.execute("DELETE FROM sessions WHERE credential IN (SELECT id FROM credentials WHERE expires_at<=?1)", [time])?;
                    db.execute("DELETE FROM credentials WHERE expires_at<=?1", [time])?;
                    let count: u32 = db.query_row("SELECT count(*) FROM credentials", [], |row| row.get(0))?;
                    if count >= CREDENTIAL_LIMIT {
                        return Err(AuthError::Unavailable);
                    }
                    let subject = body.get("subject").and_then(Value::as_str).filter(|s| valid_id(s)).ok_or(AuthError::Invalid)?;
                    let secret = nonce("kzac_");
                    db.execute("INSERT INTO credentials VALUES (?,?,?,?,?,?,?,?,'[]')",
                        params![id,sha256(&secret),subject,role_name(role),user,time,expires,revision])?;
                    token = Some(secret);
                } else {
                    db.execute("UPDATE credentials SET role=?1,user_id=?2,issued_at=?3,expires_at=?4,revision=?5,samples='[]' WHERE id=?6",
                        params![role_name(role),user,time,expires,revision,id])?;
                    db.execute("DELETE FROM sessions WHERE credential=?1", [&id])?;
                }
            }
            let receipt = json!({"request_id":request_id,"credential_id":id,"action":action,
                                 "expected_revision":expected_revision,"outcome":"committed",
                                 "reason":"applied","result_revision":revision,"completed_at":time});
            db.execute("INSERT INTO nexa_credential_attempts VALUES (?,?,?,?)",params![request_id,actor.id,intent,receipt.to_string()])?;
            let mut response = json!({"receipt":receipt,"applied":true});
            if let Some(token) = token {
                response["token"] = json!(token);
            }
            Ok(response)
        }).await
    }
}
