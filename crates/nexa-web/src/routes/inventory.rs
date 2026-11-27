use super::*;

pub(crate) fn router(state: AppState) -> Router {
    let mut router = Router::new();
    macro_rules! routes {
        ($($verb:ident $path:literal => $op:expr, $perms:expr;)*) => {
            $(router = add(router, &state, MethodFilter::$verb, $path, $op, $perms);)*
        };
    }
    use Op::{Health, Login, Logout, Whoami};
    routes! {
        GET "/api/health" => Health, &[];
        POST "/api/admin/auth/login" => Login, &[];
        POST "/api/admin/auth/logout" => Logout, &[];
        GET "/api/admin/auth/whoami" => Whoami, &[];

        GET "/api/home/catalog" => Op::Home(home::Action::Catalog), &[HomeRead];
        GET "/api/home/microphones" => Op::Home(home::Action::Microphones), &[HomeRead, DiagnosticsRead];
        POST "/api/home/actions" => Op::Home(home::Action::Execute), &[HomeControl];
        GET "/api/home/actions" => Op::Home(home::Action::HomeReceipt), &[HomeRead];
        GET "/api/home/model" => Op::Home(home::Action::Model), &[HomeRead];
        GET "/api/home/plan" => Op::Home(home::Action::Plan), &[HomeRead];

        GET "/api/routines" => Op::Routines(routines::Action::Routines), &[HomeRead];
        GET "/api/routines/{id}" => Op::Routines(routines::Action::Routine), &[HomeRead];
        POST "/api/routines/preview" => Op::Routines(routines::Action::RoutinePreview), &[RoutinesManage];
        POST "/api/routines" => Op::Routines(routines::Action::RoutineUpsert), &[RoutinesManage];
        PUT "/api/routines/{id}" => Op::Routines(routines::Action::RoutineUpsert), &[RoutinesManage];
        DELETE "/api/routines/{id}" => Op::Routines(routines::Action::RoutineDelete), &[RoutinesManage];
        POST "/api/routines/{id}/execute" => Op::Routines(routines::Action::RoutineExecute), &[RoutinesExecute];
        POST "/api/routines/{id}/enable" => Op::Routines(routines::Action::RoutineEnabled(true)), &[RoutinesManage];
        POST "/api/routines/{id}/disable" => Op::Routines(routines::Action::RoutineEnabled(false)), &[RoutinesManage];
        GET "/api/routines/{id}/runs/{run_id}" => Op::Routines(routines::Action::RoutineRun), &[HomeRead];
        GET "/api/routines/{id}/attempts" => Op::Routines(routines::Action::RoutineRuns), &[HomeRead];

        GET "/api/lists" => Op::Household(household::Action::Lists), &[TasksRead];
        POST "/api/lists" => Op::Household(household::Action::ListMutation("create_list")), &[TasksWrite];
        POST "/api/lists/sync" => Op::Household(household::Action::ListSync), &[TasksWrite];
        GET "/api/lists/{id}/items" => Op::Household(household::Action::ListItems), &[TasksRead];
        POST "/api/lists/{id}/items" => Op::Household(household::Action::ListMutation("add_item")), &[TasksWrite];
        DELETE "/api/lists/{id}/items/{item_id}" => Op::Household(household::Action::ListMutation("delete_item")), &[TasksWrite];

        GET "/api/reminders" => Op::Household(household::Action::Reminders), &[TasksRead];
        POST "/api/reminders" => Op::Household(household::Action::ReminderMutation("create")), &[TasksWrite];
        PUT "/api/reminders/{id}" => Op::Household(household::Action::ReminderMutation("update")), &[TasksWrite];
        DELETE "/api/reminders/{id}" => Op::Household(household::Action::ReminderMutation("cancel")), &[TasksWrite];
        GET "/api/reminders/notices" => Op::Household(household::Action::ReminderNotices), &[TasksRead];
        POST "/api/reminders/notices/dismiss" => Op::Household(household::Action::ReminderMutation("dismiss_notice")), &[TasksWrite];
        GET "/api/reminders/status" => Op::Household(household::Action::ReminderStatus), &[DiagnosticsRead];

        GET "/api/memory" => Op::Memory(memory::Action::Memory("summary")), &[MemoryRead];
        GET "/api/memory/facts" => Op::Memory(memory::Action::Memory("facts")), &[MemoryRead];
        GET "/api/memory/proposals" => Op::Memory(memory::Action::Memory("proposals")), &[MemoryRead];
        GET "/api/memory/returned" => Op::Memory(memory::Action::Memory("returned")), &[MemoryRead, MemoryAssign];
        GET "/api/memory/owners/counts" => Op::Memory(memory::Action::Memory("owners_counts")), &[MemoryRead, MemoryAssign];
        POST "/api/memory/facts/{id}/owner" => Op::Memory(memory::Action::MemoryMutation("assign_owner")), &[MemoryAssign];
        POST "/api/memory/proposals/{id}/confirm" => Op::Memory(memory::Action::MemoryMutation("confirm_proposal")), &[MemoryReview];
        POST "/api/memory/proposals/{id}/reject" => Op::Memory(memory::Action::MemoryMutation("reject_proposal")), &[MemoryReview];
        DELETE "/api/memory/facts/{id}" => Op::Memory(memory::Action::DeleteFact), &[MemoryReview];
        GET "/api/memory/mine" => Op::Memory(memory::Action::Memory("mine")), &[MemoryOwnRead];
        DELETE "/api/memory/mine/facts/{id}" => Op::Memory(memory::Action::DeleteFact), &[MemoryOwnWrite];
        POST "/api/memory/mine/facts/{id}/return" => Op::Memory(memory::Action::MemoryMutation("return_fact")), &[MemoryOwnWrite];
        GET "/api/memory/mine/voice" => Op::Memory(memory::Action::Memory("voice_mine")), &[MemoryOwnVoice];
        POST "/api/memory/mine/voice/consent" => Op::Memory(memory::Action::MemoryMutation("voice_consent_grant")), &[MemoryOwnVoice];
        DELETE "/api/memory/mine/voice/consent" => Op::Memory(memory::Action::MemoryMutation("voice_consent_revoke")), &[MemoryOwnVoice];
        POST "/api/memory/mine/voice/verification" => Op::Memory(memory::Action::MemoryMutation("voice_verification")), &[MemoryOwnVoice];
        GET "/api/memory/voice/consents" => Op::Memory(memory::Action::Memory("voice_consents")), &[MemoryRead, MemoryAssign];
        POST "/api/memory/voice/consents/{id}/reopen" => Op::Memory(memory::Action::MemoryMutation("reopen_voice_consent")), &[MemoryAssign];
        GET "/api/memory/voice/switch" => Op::Voices(voices::Action::VoiceSwitch), &[MemoryRead, MemoryAssign];
        POST "/api/memory/voice/switch" => Op::Voices(voices::Action::VoiceSwitchWrite), &[MemoryAssign, AdminManage];

        GET "/api/voices" => Op::Voices(voices::Action::VoiceCaptures), &[VoicesRead, VoicesReview];
        GET "/api/voices/profiles" => Op::Voices(voices::Action::VoiceProfiles), &[VoicesRead, VoicesReview];
        POST "/api/voices/profiles" => Op::Memory(memory::Action::MemoryMutation("confirm_voice_profile")), &[VoicesReview];
        DELETE "/api/voices/profiles/{id}" => Op::Memory(memory::Action::MemoryMutation("delete_voice_profile")), &[VoicesReview];
        PUT "/api/voices/profiles/{id}/person" => Op::Voices(voices::Action::VoiceLink), &[VoicesReview, AdminManage];
        POST "/api/voices/profiles/{id}/attribution/resume" => Op::Memory(memory::Action::MemoryMutation("resume_attribution")), &[VoicesReview, AdminManage, MemoryAssign];
        POST "/api/voices/groups/corrections" => Op::Memory(memory::Action::MemoryMutation("correct_voice_group")), &[VoicesReview];
        GET "/api/voices/captures/{id}/review" => Op::Voices(voices::Action::VoiceReview(false)), &[VoicesRead, VoicesReview];
        GET "/api/voices/captures/{id}/transcript" => Op::Voices(voices::Action::VoiceReview(true)), &[VoicesRead, VoicesReview];
        POST "/api/voices/captures/query" => Op::Voices(voices::Action::VoiceCaptures), &[VoicesRead, VoicesReview];

        GET "/api/capture/control" => Op::Voices(voices::Action::Capture), &[AdminManage];
        POST "/api/capture/control" => Op::Voices(voices::Action::CaptureWrite), &[AdminManage];
        POST "/api/capture/control/forget-carried" => Op::Voices(voices::Action::ForgetCarried), &[AdminManage];
        GET "/api/ambient/control" => Op::Voices(voices::Action::Ambient), &[AmbientControl];
        POST "/api/ambient/control" => Op::Voices(voices::Action::AmbientWrite), &[AmbientControl];

        GET "/api/evaluation/grants" => Op::Voices(voices::Action::EvaluationGrants), &[EvaluationRead];
        GET "/api/evaluation/samples/{id}" => Op::Voices(voices::Action::EvaluationSample), &[EvaluationRead];
        GET "/api/zones" => Op::Obs(obs::Action::Zones), &[DiagnosticsRead];
        PUT "/api/zones/{id}/volume" => Op::Obs(obs::Action::ZoneVolume), &[HomeControl];
        GET "/api/conversations" => Op::Obs(obs::Action::Conversations), &[ConversationsRead];
        GET "/api/ha/entities" => Op::Obs(obs::Action::HaEntities), &[DiagnosticsRead];
        GET "/api/ha/actions" => Op::Obs(obs::Action::HaActions), &[DiagnosticsRead];
        GET "/api/llm/endpoints" => Op::Obs(obs::Action::Llm), &[DiagnosticsRead];
        POST "/api/llm/endpoints/{id}/clear-cooldown" => Op::Obs(obs::Action::Cooldown), &[AdminManage];
        GET "/api/users" => Op::Admin(admin::Action::Users), &[DiagnosticsRead];

        GET "/api/alerts" => Op::Household(household::Action::Alerts), &[HomeRead];
        GET "/api/alerts/zones" => Op::Household(household::Action::AlertZones), &[HomeRead];
        PUT "/api/alerts/zones" => Op::Household(household::Action::AlertZonesWrite), &[AdminManage];
        GET "/api/system/gpus" => Op::Obs(obs::Action::Gpus), &[DiagnosticsRead];
        GET "/api/system/services" => Op::Obs(obs::Action::Services), &[DiagnosticsRead];
        GET "/api/system/daemons" => Op::Obs(obs::Action::Daemons), &[DiagnosticsRead];
        POST "/api/admin/ml/slots/{id}/reload" => Op::Obs(obs::Action::ReloadMlSlot), &[AdminManage];
        GET "/api/admin/users" => Op::Admin(admin::Action::Users), &[AdminManage];
        POST "/api/admin/users" => Op::Admin(admin::Action::UserWrite), &[AdminManage];
        PUT "/api/admin/users/{id}/permissions" => Op::Admin(admin::Action::UserWrite), &[AdminManage];
        DELETE "/api/admin/users/{id}" => Op::Admin(admin::Action::UserRetire), &[AdminManage, AuthManage];
        POST "/api/admin/alerts/{id}/ack" => Op::Household(household::Action::AlertAck), &[AdminManage];
        DELETE "/api/admin/alerts/{id}" => Op::Household(household::Action::AlertDelete), &[AdminManage];
        POST "/api/admin/services/{id}/restart" => Op::Obs(obs::Action::Restart), &[AdminManage];
        GET "/api/admin/cloud-consent" => Op::Obs(obs::Action::Cloud), &[AdminManage];
        POST "/api/admin/cloud-consent/withdraw" => Op::Obs(obs::Action::CloudWithdraw), &[AdminManage];

        GET "/api/admin/auth/credentials" => Op::Admin(admin::Action::Credentials), &[AuthManage];
        POST "/api/admin/auth/credentials" => Op::Admin(admin::Action::CredentialMutation("issue")), &[AuthManage];
        PUT "/api/admin/auth/credentials/{id}" => Op::Admin(admin::Action::CredentialMutation("revise")), &[AuthManage];
        DELETE "/api/admin/auth/credentials/{id}" => Op::Admin(admin::Action::CredentialMutation("revoke")), &[AuthManage];
        GET "/api/admin/auth/credential-attempts/{id}" => Op::Admin(admin::Action::CredentialAttempt), &[AuthManage];
    }
    router
        .route("/", get(index))
        .route("/obs", get(index))
        .route("/assets/{*file}", get(public_asset))
        .route("/favicon.ico", get(favicon))
        .with_state(state)
}

fn add(
    router: Router<AppState>,
    state: &AppState,
    method: MethodFilter,
    path: &str,
    op: Op,
    permissions: &'static [Permission],
) -> Router<AppState> {
    let policy = Policy {
        op,
        permissions,
        public: matches!(op, Op::Health | Op::Login),
    };
    let handler =
        move |State(state): State<AppState>, request: Request| dispatch(state, policy, request);
    let handler = handler.layer(middleware::from_fn_with_state(
        (state.clone(), policy),
        authorize,
    ));
    router.route(path, on(method, handler))
}
