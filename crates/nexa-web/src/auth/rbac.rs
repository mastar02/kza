use super::*;

pub fn role_name(role: Role) -> &'static str {
    match role {
        Role::Operator => "operator",
        Role::Home => "home",
        Role::Evaluator => "evaluator",
        Role::Unspecified => "unspecified",
    }
}

pub(super) fn parse_role(value: &str) -> Option<Role> {
    [Role::Operator, Role::Home, Role::Evaluator]
        .into_iter()
        .find(|role| role_name(*role) == value)
}

pub fn permission_name(permission: Permission) -> &'static str {
    use Permission::*;
    match permission {
        Unspecified => "",
        HomeRead => "home:read",
        HomeControl => "home:control",
        TasksRead => "tasks:read",
        TasksWrite => "tasks:write",
        TasksPersonal => "tasks:personal",
        RoutinesExecute => "routines:execute",
        RoutinesManage => "routines:manage",
        DiagnosticsRead => "diagnostics:read",
        ConversationsRead => "conversations:read",
        MemoryRead => "memory:read",
        MemoryReview => "memory:review",
        MemoryAssign => "memory:assign",
        MemoryOwnRead => "memory:own:read",
        MemoryOwnWrite => "memory:own:write",
        MemoryOwnVoice => "memory:own:voice",
        VoicesRead => "voices:read",
        VoicesReview => "voices:review",
        EvaluationRead => "evaluation:read",
        AmbientControl => "ambient:control",
        AdminManage => "admin:manage",
        AuthManage => "auth:manage",
    }
}

pub fn permissions(role: Role, personal: bool) -> Vec<Permission> {
    use Permission::*;
    let own = [TasksPersonal, MemoryOwnRead, MemoryOwnWrite, MemoryOwnVoice];
    match role {
        Role::Operator => (HomeRead as i32..=AuthManage as i32)
            .filter_map(|value| Permission::try_from(value).ok())
            .filter(|value| !own.contains(value))
            .collect(),
        Role::Home => {
            let mut result = vec![
                HomeRead,
                HomeControl,
                TasksRead,
                TasksWrite,
                RoutinesExecute,
            ];
            if personal {
                result.extend(own);
            }
            result
        }
        Role::Evaluator => vec![EvaluationRead],
        Role::Unspecified => vec![],
    }
}
