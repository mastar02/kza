use nexa_nlu::household::{HouseholdFamily, HouseholdIntentClassifier, HouseholdOperation};

#[test]
fn python_household_precedence_corpus() {
    let classifier = HouseholdIntentClassifier::default();
    let mut mismatches = Vec::new();
    for line in include_str!("fixtures/household_precedence.tsv")
        .lines()
        .skip(1)
    {
        let row: Vec<_> = line.split('\t').collect();
        let text = if row[3] == "<empty>" { "" } else { row[3] };
        let claimed = classifier.classify(text);
        let actual = claimed.as_ref().map(|intent| {
            (
                family(intent.family),
                operation(intent.operation),
                intent.rule,
            )
        });
        let expected = (row[0] != "-").then_some((row[0], row[1], row[2]));
        if actual != expected {
            mismatches.push(format!(
                "{:?} expected {expected:?}, got {actual:?} ({:?})",
                row[3],
                claimed.and_then(|i| i.reason)
            ));
        }
    }
    assert!(mismatches.is_empty(), "{}", mismatches.join("\n"));
}

fn family(family: HouseholdFamily) -> &'static str {
    use HouseholdFamily::*;
    match family {
        Timer => "timer",
        Reminder => "reminder",
        List => "list",
        Routine => "routine",
        Intercom => "intercom",
        Notification => "notification",
        DeferredAction => "deferred_action",
        Compound => "compound",
        Energy => "energy",
        Calendar => "calendar",
        Media => "media",
        CameraHealth => "camera_health",
    }
}

fn operation(operation: HouseholdOperation) -> &'static str {
    use HouseholdOperation::*;
    match operation {
        Create => "create",
        Cancel => "cancel",
        Stop => "stop",
        Pause => "pause",
        Resume => "resume",
        Disable => "disable",
        Query => "query",
        AddItem => "add_item",
        RemoveItem => "remove_item",
        Execute => "execute",
        Propose => "propose",
        Clarify => "clarify",
    }
}

#[test]
fn python_energy_scope_corpus() {
    let classifier = HouseholdIntentClassifier::default();
    let mut mismatches = Vec::new();
    for line in include_str!("fixtures/household_energy.tsv")
        .lines()
        .skip(1)
    {
        let Some((expected, text)) = line.split_once('\t') else {
            panic!("malformed fixture: {line}");
        };
        let actual = classifier
            .classify(text)
            .filter(|intent| intent.family == HouseholdFamily::Energy);
        let actual = actual
            .as_ref()
            .map_or("-", |intent| intent.complement.as_str());
        if actual != expected {
            mismatches.push(format!("{text:?} expected {expected:?}, got {actual:?}"));
        }
    }
    assert!(mismatches.is_empty(), "{}", mismatches.join("\n"));
}
