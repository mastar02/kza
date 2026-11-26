use super::*;

#[derive(Clone, Copy)]
pub(super) enum Action {
    VoiceSwitch,
    VoiceSwitchWrite,
    VoiceProfiles,
    VoiceLink,
    VoiceReview(bool),
    VoiceCaptures,
    Capture,
    CaptureWrite,
    ForgetCarried,
    Ambient,
    AmbientWrite,
    EvaluationGrants,
    EvaluationSample,
}

pub(super) async fn execute(
    op: Action,
    state: &AppState,
    fields: &Fields,
    lease: &Lease,
    proof: &Presented,
    policy: Policy,
) -> Result<Response> {
    use Action::*;
    let brain = &state.clients.brain;
    let audio = &state.clients.audio;
    let memory = &state.clients.memory;
    let id = fields.id();
    let body = &fields.body;
    let actor = || Some(lease.actor());
    let actor_request = || ActorRequest { actor: actor() };
    let mut status = StatusCode::OK;
    let value = match op {
        VoiceSwitch => voice_switch(brain.get_voice_global_switch(actor_request()).await?),
        VoiceSwitchWrite => {
            exact_fields(body, &["enabled", "expected_revision"])?;
            voice_switch(
                brain
                    .set_voice_global_switch(SetVoiceGlobalSwitchRequest {
                        enabled: boolean(body, "enabled")?,
                        expected_revision: fields.revision()?,
                        actor: actor(),
                    })
                    .await?,
            )
        }
        VoiceProfiles => {
            let include_person = optional_bool(&fields.query, "include_person")?.unwrap_or(false);
            if include_person {
                lease.require(&[AdminManage])?;
            } else if fields.query.contains_key("person_id") {
                return Err(INVALID);
            }
            let cursor = page_cursor(&fields.query, &["cursor"])?;
            let connection =
                state
                    .live
                    .authorize(&fields.headers, lease, ws::Channel::VoiceProfiles)?;
            let page = memory
                .list_voice_profiles(ListVoiceProfilesRequest {
                    include_person,
                    actor: actor(),
                    cursor: cursor.into(),
                })
                .await?;
            let rows = page
                .profiles
                .into_iter()
                .filter(|row| {
                    !include_person
                        || fields
                            .query
                            .get("person_id")
                            .is_none_or(|id| row.linked_person_id == *id)
                })
                .map(|row| voice_profile(row, include_person))
                .collect::<Vec<_>>();
            state.live.projection(
                connection,
                ws::Channel::VoiceProfiles,
                json!({
                    "rows": rows,
                    "next_cursor": (!page.next_cursor.is_empty()).then_some(page.next_cursor),
                }),
            )?
        }
        VoiceLink => {
            let row = memory
                .link_voice_profile_to_person(LinkVoiceProfileRequest {
                    profile_id: id.into(),
                    person_id: optional_string(body, "user_id")?.unwrap_or("").into(),
                    expected_profile_revision: unsigned(body, "expected_profile_revision")?,
                    expected_link_revision: unsigned(body, "expected_link_revision")?,
                    actor: actor(),
                })
                .await?;
            json!({"ok":true,"profile_id":row.profile_id,"link_revision":row.link_revision})
        }
        VoiceReview(transcript) => {
            let connection =
                state
                    .live
                    .authorize(&fields.headers, lease, ws::Channel::VoiceReview)?;
            let doc = projection(
                brain
                    .get_voice_capture_review(VoiceCaptureReviewRequest {
                        fingerprint_id: id.into(),
                        include_transcript: transcript,
                        actor: actor(),
                    })
                    .await?
                    .payload_json,
            )?;
            state
                .live
                .projection(connection, ws::Channel::VoiceReview, doc)?
        }
        VoiceCaptures => {
            if body.get("text").is_some_and(|value| !value.is_null()) {
                lease.require(&[AdminManage])?;
            }
            let connection =
                state
                    .live
                    .authorize(&fields.headers, lease, ws::Channel::VoiceCaptures)?;
            let query = if fields.query.is_empty() {
                body.clone()
            } else if body == &json!({}) {
                json!(fields.query)
            } else {
                return Err(INVALID);
            };
            let doc = projection(
                brain
                    .query_voice_captures(VoiceCaptureQueryRequest {
                        query_json: query.to_string(),
                        actor: actor(),
                    })
                    .await?
                    .payload_json,
            )?;
            state
                .live
                .projection(connection, ws::Channel::VoiceCaptures, doc)?
        }
        Capture => capture(audio.get_capture_control(Empty {}).await?),
        CaptureWrite => {
            exact_fields(body, &["muted", "expected_revision"])?;
            let doc = audio
                .set_capture_mute(SetCaptureMuteRequest {
                    muted: boolean(body, "muted")?,
                    expected_revision: fields.revision()?,
                    actor: actor(),
                })
                .await?;
            status = StatusCode::ACCEPTED;
            capture(doc)
        }
        ForgetCarried => {
            exact_fields(body, &["expected_carried", "no_capture_running"])?;
            if !boolean(body, "no_capture_running")? {
                return Err(INVALID);
            }
            let receipt = audio
                .forget_carried_owners(ForgetCarriedOwnersRequest {
                    expected_carried: unsigned(body, "expected_carried")?
                        .try_into()
                        .map_err(|_| INVALID)?,
                    no_capture_running: true,
                    actor: actor(),
                })
                .await?;
            json!({"status":capture(receipt.status.ok_or(BAD_UPSTREAM)?),"forgotten_capture_owners":receipt.forgotten_capture_owners})
        }
        Ambient => ambient(memory.get_ambient_control(lease.actor()).await?),
        AmbientWrite => {
            exact_fields(body, &["paused", "expected_revision"])?;
            let value = memory
                .set_ambient_pause(SetAmbientPauseRequest {
                    paused: boolean(body, "paused")?,
                    expected_revision: fields.revision()?,
                    actor: actor(),
                })
                .await?;
            status = StatusCode::ACCEPTED;
            ambient(value)
        }
        EvaluationGrants => {
            require_evaluator(lease)?;
            lease.evaluation_grants()
        }
        EvaluationSample => {
            require_evaluator(lease)?;
            let modality = fields.query("modality");
            if fields.query.len() != 1 || !["audio_preview", "transcript_full"].contains(&modality)
            {
                return Err(INVALID);
            }
            let sources = lease.evaluation_sources(id, modality)?;
            let connection =
                state
                    .live
                    .authorize(&fields.headers, lease, ws::Channel::Evaluation)?;
            let doc = projection(
                brain
                    .get_evaluation_sample(EvaluationSampleRequest {
                        fingerprint_id: id.into(),
                        actor: actor(),
                    })
                    .await?
                    .payload_json,
            )?;
            let current = state.auth.validate(proof, policy.permissions).await?;
            current.evaluation_sources(id, modality)?;
            state.live.projection(
                connection,
                ws::Channel::Evaluation,
                verified_sample(doc, id, modality, sources)?,
            )?
        }
    };
    Ok((status, Json(value)).into_response())
}
