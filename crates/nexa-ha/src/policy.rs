use std::collections::{BTreeMap, BTreeSet, HashMap};

use nexa_config::HaControlPolicyConfig;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

use crate::{
    operations::Operation,
    state::{EntityState, domain},
};

const MAX_EFFECTS: usize = 128;
const MAX_GROUP_DEPTH: usize = 8;
const MAX_REFERENCES: usize = MAX_EFFECTS * 4;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ControlPolicyError {
    #[error("control denied by fail-closed policy for {entity_id}.{operation}")]
    Denied {
        entity_id: String,
        operation: String,
    },
}

pub fn authorize_action(
    policy: &HaControlPolicyConfig,
    entity_id: &str,
    operation: &str,
) -> Result<(), ControlPolicyError> {
    if nexa_config::supports_ha_operation(entity_id, operation)
        && policy.allows(entity_id, operation)
    {
        Ok(())
    } else {
        Err(ControlPolicyError::Denied {
            entity_id: entity_id.to_owned(),
            operation: operation.to_owned(),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EffectIntent {
    pub entity_id: String,
    pub operation: Operation,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum MembershipSource {
    #[default]
    HaState,
    ReviewedCopy,
}

#[derive(Debug, Clone)]
pub struct ReviewedGroup {
    pub members: Vec<String>,
    pub source: MembershipSource,
}

/// Trusted, static grants and reviewed effect definitions. Empty grants deny all actions.
#[derive(Debug, Clone, Default)]
pub struct ControlPolicy {
    grants: HaControlPolicyConfig,
    scenes: HashMap<String, Vec<EffectIntent>>,
    groups: HashMap<String, ReviewedGroup>,
}

impl ControlPolicy {
    pub fn new(grants: HaControlPolicyConfig) -> Self {
        Self {
            grants,
            ..Self::default()
        }
    }

    /// Attach server-owned definitions. Each leaf must still have its own exact grant.
    pub fn with_effects(
        mut self,
        scenes: HashMap<String, Vec<EffectIntent>>,
        groups: HashMap<String, ReviewedGroup>,
    ) -> Self {
        self.scenes = scenes;
        self.groups = groups;
        self
    }

    pub(crate) fn plan(&self, root: &EffectIntent) -> Result<EffectPlan, PlanError> {
        let mut plan = EffectPlan::default();
        let mut visits = 0;
        if root.operation == Operation::ActivateScene {
            self.authorize(root)?;
            let scene = self
                .scenes
                .get(&root.entity_id)
                .ok_or(PlanError::Unreviewed)?;
            let members = members(scene.iter().map(|effect| effect.entity_id.as_str()))?;
            plan.references.insert(
                root.entity_id.clone(),
                Reference {
                    members,
                    source: MembershipSource::HaState,
                },
            );
            for effect in scene {
                self.expand(effect, &mut plan, &mut Vec::new(), &mut visits)?;
            }
        } else {
            self.expand(root, &mut plan, &mut Vec::new(), &mut visits)?;
        }
        Ok(plan)
    }

    fn authorize(&self, intent: &EffectIntent) -> Result<(), PlanError> {
        authorize_action(&self.grants, &intent.entity_id, intent.operation.name())
            .map_err(|_| PlanError::Denied)?;
        intent
            .operation
            .validate()
            .map_err(|_| PlanError::InvalidParameters)?;
        if let Operation::ColorTempKelvin {
            brightness: Some(_),
            ..
        } = &intent.operation
        {
            authorize_action(&self.grants, &intent.entity_id, "brightness")
                .map_err(|_| PlanError::Denied)?;
        }
        Ok(())
    }

    fn expand(
        &self,
        intent: &EffectIntent,
        plan: &mut EffectPlan,
        path: &mut Vec<String>,
        visits: &mut usize,
    ) -> Result<(), PlanError> {
        *visits += 1;
        if *visits > MAX_REFERENCES
            || path.len() > MAX_GROUP_DEPTH
            || path.contains(&intent.entity_id)
        {
            return Err(PlanError::InvalidExpansion);
        }
        self.authorize(intent)?;
        if intent.operation == Operation::ActivateScene {
            return Err(PlanError::InvalidExpansion);
        }
        if let Some(group) = self.groups.get(&intent.entity_id) {
            if domain(&intent.entity_id) != Some("light")
                || group
                    .members
                    .iter()
                    .any(|member| domain(member) != Some("light"))
            {
                return Err(PlanError::InvalidExpansion);
            }
            let members = members(group.members.iter().map(String::as_str))?;
            plan.references.insert(
                intent.entity_id.clone(),
                Reference {
                    members,
                    source: group.source,
                },
            );
            path.push(intent.entity_id.clone());
            for entity_id in &group.members {
                let child = EffectIntent {
                    entity_id: entity_id.clone(),
                    operation: intent.operation.clone(),
                };
                self.expand(&child, plan, path, visits)?;
            }
            path.pop();
        } else if let Some(previous) = plan
            .leaves
            .iter()
            .find(|leaf| leaf.entity_id == intent.entity_id)
        {
            if !previous.operation.same_body(&intent.operation) {
                return Err(PlanError::InvalidExpansion);
            }
        } else {
            if plan.leaves.len() >= MAX_EFFECTS {
                return Err(PlanError::InvalidExpansion);
            }
            plan.leaves.push(intent.clone());
        }
        Ok(())
    }
}

fn members<'a>(ids: impl ExactSizeIterator<Item = &'a str>) -> Result<BTreeSet<String>, PlanError> {
    let count = ids.len();
    if !(1..=MAX_EFFECTS).contains(&count) {
        return Err(PlanError::InvalidExpansion);
    }
    let unique: BTreeSet<_> = ids.map(str::to_owned).collect();
    (unique.len() == count)
        .then_some(unique)
        .ok_or(PlanError::InvalidExpansion)
}

#[derive(Debug, Clone, Copy, Error, PartialEq, Eq)]
pub enum PlanError {
    #[error("control policy denied the operation")]
    Denied,
    #[error("invalid operation parameters")]
    InvalidParameters,
    #[error("scene effects have not been reviewed")]
    Unreviewed,
    #[error("invalid, conflicting or excessive effect expansion")]
    InvalidExpansion,
    #[error("live group or scene membership differs from the reviewed definition")]
    MembershipChanged,
}

#[derive(Default)]
pub(crate) struct EffectPlan {
    pub references: BTreeMap<String, Reference>,
    pub leaves: Vec<EffectIntent>,
}

pub(crate) struct Reference {
    members: BTreeSet<String>,
    source: MembershipSource,
}

impl Reference {
    pub fn validate(&self, state: &EntityState) -> Result<(), PlanError> {
        let Some(actual) = state
            .attributes()
            .get("entity_id")
            .filter(|value| !value.is_null())
        else {
            return (self.source == MembershipSource::ReviewedCopy)
                .then_some(())
                .ok_or(PlanError::MembershipChanged);
        };
        let actual = actual.as_array().ok_or(PlanError::MembershipChanged)?;
        if actual.len() != self.members.len() {
            return Err(PlanError::MembershipChanged);
        }
        let checked: Option<BTreeSet<_>> = actual.iter().map(Value::as_str).collect();
        if checked.is_some_and(|members| {
            members
                .into_iter()
                .eq(self.members.iter().map(String::as_str))
        }) {
            return Ok(());
        }
        Err(PlanError::MembershipChanged)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nested_groups_detect_cycles_and_conflicting_scene_effects() {
        let mut grants = HaControlPolicyConfig::default();
        for light in ["light.a", "light.b"] {
            grants
                .entities
                .insert(light.into(), vec!["turn_on".into(), "turn_off".into()]);
        }
        grants
            .entities
            .insert("scene.test".into(), vec!["activate_scene".into()]);
        let intent = |id: &str, operation| EffectIntent {
            entity_id: id.into(),
            operation,
        };
        let group = |id: &str| ReviewedGroup {
            members: vec![id.into()],
            source: MembershipSource::HaState,
        };
        let cycle = ControlPolicy::new(grants.clone()).with_effects(
            HashMap::new(),
            HashMap::from([
                ("light.a".into(), group("light.b")),
                ("light.b".into(), group("light.a")),
            ]),
        );
        assert!(matches!(
            cycle.plan(&intent("light.a", Operation::TurnOn)),
            Err(PlanError::InvalidExpansion)
        ));
        let overlap = ControlPolicy::new(grants).with_effects(
            HashMap::from([(
                "scene.test".into(),
                vec![
                    intent("light.a", Operation::TurnOn),
                    intent("light.b", Operation::TurnOff),
                ],
            )]),
            HashMap::from([("light.a".into(), group("light.b"))]),
        );
        assert!(matches!(
            overlap.plan(&intent("scene.test", Operation::ActivateScene)),
            Err(PlanError::InvalidExpansion)
        ));
    }
}
