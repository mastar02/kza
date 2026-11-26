//! Read-only routine and integration queries, plus TV controls.

use crate::grammar;
use crate::normalize::{has_phrase, words};

use super::HouseholdFamily::{Calendar, CameraHealth, Energy, Media, Routine};
use super::HouseholdOperation::{Execute, Query};
use super::utterance::Utterance;
use super::{HouseholdIntent, HouseholdIntentClassifier, energy};

impl HouseholdIntentClassifier {
    /// Recognize a read-only routine status/schedule question without a routine store.
    pub(crate) fn routine_read(&self, core: &str) -> Option<HouseholdIntent> {
        let plain = words(core);
        let m = ROUTINE_READ.captures(&plain)?;
        let name = ["before", "after"].iter().find_map(|group| m.name(group))?;
        let name = name.as_str();
        // Predicates and further clauses cannot stand for a saved routine name.
        if name.split_whitespace().any(|word| {
            matches!(word, "si" | "cuando" | "porque" | "que") || grammar::action(word).is_some()
        }) {
            return None;
        }
        Some(HouseholdIntent::new(
            Routine,
            Query,
            "R1",
            name.to_owned(),
            "",
        ))
    }
}

pub(super) fn integration(utt: &Utterance<'_>) -> Option<HouseholdIntent> {
    let text = utt.text();
    if utt.has(&["camara", "camaras"]) && !VIDEO_DOORBELL.is_match(&text) {
        let health = CAMERA_HEALTH_QUERY.is_match(&text);
        let media = CAMERA_MEDIA_QUERY.is_match(&text);
        if health || media {
            let kind = if media
                || utt.has(&[
                    "video",
                    "foto",
                    "imagen",
                    "grabando",
                    "grabacion",
                    "movimiento",
                    "stream",
                ]) {
                "media"
            } else {
                "health"
            };
            return Some(HouseholdIntent::new(
                CameraHealth,
                Query,
                "P5",
                kind.into(),
                "",
            ));
        }
    }
    if utt.has(&["calendario", "agenda"]) && CALENDAR_QUERY.is_match(&text) {
        let day = if has_phrase(&text, "manana") {
            "tomorrow"
        } else {
            "today"
        };
        return Some(HouseholdIntent::new(Calendar, Query, "P5", day.into(), ""));
    }
    if let Some(kind) = energy::measurement(utt, &text) {
        return Some(HouseholdIntent::new(Energy, Query, "P5", kind.into(), ""));
    }
    let m = TV_QUERY_OR_CONTROL.captures(&text)?;
    let room = m.name("room").map_or("", |m| m.as_str());
    if m.name("query").is_some() {
        return Some(HouseholdIntent::new(Media, Query, "P5", room.into(), ""));
    }
    if grammar::has_deferral_time(&text) {
        return None;
    }
    let verb = if m.name("pause").is_some() {
        "pause"
    } else if m.name("up").is_some() {
        "volume_up"
    } else {
        "volume_down"
    };
    Some(HouseholdIntent::new(
        Media,
        Execute,
        "P5",
        room.into(),
        verb,
    ))
}

regex_patterns! {
    VIDEO_DOORBELL => r"\bvideo ?porteros?\b";
    CAMERA_HEALTH_QUERY => r"^(?:como (?:esta|estan|andan|funcionan)|estado (?:tecnico )?de|salud (?:tecnica )?de) ";
    CAMERA_MEDIA_QUERY => r"^(?:mostra(?:me)?|muestra(?:me)?|ver|quiero ver|dame|mandame|(?:esta|estan) grabando) ";
    CALENDAR_QUERY => r"^(?:que (?:hay|tenemos|tengo)(?: (?:hoy|manana))?(?: en (?:el|la|mi|nuestro) (?:calendario|agenda)(?: (?:del hogar|compartid[oa]|personal))?)?|(?:mostrame|decime|lee) (?:el|la|mi) (?:calendario|agenda)|(?:calendario|agenda)(?: (?:del hogar|compartid[oa]))?)(?: (?:hoy|manana))?$";
    ROUTINE_READ => concat!(
        r"^(?:(?:decime|dime|contame) )?(?:(?:como esta|estado de|",
        r"(?:a que hora|cuando) (?:corre|se ejecuta|arranca)|",
        r"(?:esta|sigue|quedo) (?:activa(?:da)?|desactivada|programada|pausada)) ",
        r"(?:la |mi |nuestra )?rutina (?P<before>.+)|",
        r"(?:la |mi |nuestra )?rutina (?P<after>.+?) (?:(?:esta|sigue|quedo) ",
        r"(?:activa(?:da)?|desactivada|programada|pausada)|(?:se ejecuto|funciono|fallo|corrio)(?: hoy| ayer)?))$"
    );
    TV_QUERY_OR_CONTROL => r"^(?:(?P<query>que (?:esta pasando|hay|estan dando) en|como esta|estado de)|(?P<pause>(?:pone|pon|poneme) (?:en )?pausa(?: en| a)?|pausa|pausame)|(?P<up>(?:subi|sube|subime) (?:el )?volumen(?: de| en)?)|(?P<down>(?:baja|bajame) (?:el )?volumen(?: de| en)?)) (?:la |el )?(?:televisor|television|tele|tv)(?:(?: del| de la| de| en el| en la| en) (?P<room>[a-z ]+))?$";
}
