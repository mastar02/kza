//! Reasons a device utterance cannot authorize an immediate action.

use crate::normalize::{fold, words};

use super::lexicon::{action, entity_at};
use super::light::brightness;
use super::{Domain, GrammarBlock};

pub(crate) fn has_deferral_time(text: &str) -> bool {
    DEFERRAL.is_match(text)
}

pub(crate) fn security_order(text: &str) -> bool {
    let plain = words(text);
    LOCK_ORDER.is_match(&plain) || ALARM_ORDER.is_match(&plain)
}

/// Determine why speech with device evidence cannot authorize an immediate action.
pub(crate) fn non_immediate(core: &str) -> Option<GrammarBlock> {
    let folded = words(core);
    let has_device = folded.split_whitespace().any(|w| entity_at(w).is_some());
    let has_action = folded.split_whitespace().any(|w| action(w).is_some());
    if folded
        .split_whitespace()
        .any(|w| entity_at(w) == Some(Domain::Light))
        && brightness(&fold(core)).is_err()
    {
        return Some(GrammarBlock::InvalidBrightness);
    }
    if !has_device && !has_action {
        return None;
    }
    let checks = [
        (GrammarBlock::Negation, &*NEGATION),
        (GrammarBlock::Conditional, &*CONDITIONAL),
        (GrammarBlock::ReportedSpeech, &*REPORTED_SPEECH),
    ];
    for (reason, pattern) in checks {
        if pattern.is_match(if reason == GrammarBlock::ReportedSpeech {
            core
        } else {
            &folded
        }) {
            return Some(reason);
        }
    }
    if PAST_ACTION.is_match(core)
        || PERFECT_ACTION.is_match(&folded)
        || PAST_CONTEXT.is_match(&folded)
    {
        return Some(GrammarBlock::Past);
    }
    let core_lower = core.to_lowercase();
    if FUTURE_INFLECTED.is_match(&core_lower) || FUTURE_PERIPHRASTIC.is_match(&folded) {
        return Some(GrammarBlock::Future);
    }
    if INFORMATION.is_match(&folded) {
        return Some(GrammarBlock::Information);
    }
    (has_device && has_deferral_time(&folded)).then_some(GrammarBlock::Deferred)
}

regex_patterns! {
    LOCK_ORDER => concat!(
        r"\b(?:cerr(?:a|ar|e|en|ala|alo|arla|arlo)|cierr(?:a|e|en|ala|alo)|abr(?:i|ir|a|an|ila|ilo|irla|irlo)",
        r"|(?:des)?trab(?:a|ar|e|en|ala|alo|arla|arlo)|(?:des)?bloque(?:a|ar|e|en))\b(?: \S+){0,4}?",
        r" +(?:puertas?|porton(?:es)?|cerraduras?|cerrojos?|candados?|rejas?|garag(?:e|es)|garaj(?:e|es)|llave)\b",
        r"|\b(?:ech|pas)(?:a|ar|e|ale|arle) (?:la )?llave\b"
    );
    ALARM_ORDER => concat!(
        r"\b(?:(?:des)?activ|(?:des)?arm|(?:des)?conect)(?:a|ar|e|en|ala|arla)\b(?: \S+){0,3}? alarmas?\b",
        r"|\b(?:encend|prend)(?:e|er|ela|erla)\b(?: \S+){0,3}? alarmas?\b"
    );
    DEFERRAL => r"\b(?:manana|mas tarde|despues|luego|en (?:\w+ ){1,5}(?:segundos?|minutos?|horas?)|a las? (?:[0-9]+|una?|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez|once|doce))\b";
    NEGATION => r"\b(?:no|nunca|jamas|tampoco) \w+";
    CONDITIONAL => r"\b(?:ojala|si|cuando|aunque|en caso de que) \w+";
    REPORTED_SPEECH => r"\b(?:dije|dijo|dijiste|dijeron|coment[oé]|comentaste|comentaron|pregunt[oé]|preguntaste|preguntaron|avis[oé]|explic[oé])\b";
    PAST_ACTION => r"\b(?:\w+(?:aste|iste|aron|ieron|asteis|isteis)|apagué|prendí|encendí|bajé|cerré|cambié|pausé|frené|puse|puso|pusimos|apag[oó]|prendi[oó]|encendi[oó]|subi[oó]|bajó|cerró|abri[oó]|(?:ajust|configur|dej|activ|desactiv|par|cambi|levant)[éó]|reprodujo|apagaba[ns]?|prend[ií]a[ns]?|encend[ií]a[ns]?|sub[ií]a[ns]?|bajaba[ns]?|cerraba[ns]?|abr[ií]a[ns]?|pon[ií]a[ns]?)\b";
    PERFECT_ACTION => r"\b(?:he|has|ha|hemos|han|habia[ns]?|habiamos|hubo|habras?) (?:\w+ ){0,2}(?:apagado|prendido|encendido|subido|bajado|cerrado|abierto|puesto|activado|desactivado|cambiado)\b";
    PAST_CONTEXT => r"\b(?:ayer|anteayer|anoche|recien|antes|(?:ya|yo) (?:subi|abri))\b";
    FUTURE_INFLECTED => r"\b(?:apagar|prender|encender|subir|bajar|cerrar|abrir|activar|desactivar|poner)(?:é|ás|á|emos|án)\b";
    FUTURE_PERIPHRASTIC => r"\b(?:voy|vas|va|vamos|van) a (?:apagar|prender|encender|subir|bajar|cerrar|abrir|activar|desactivar|poner)\b";
    INFORMATION => r"^(?:(?:decime|dime|contame|cuentame|explicame) )?(?:que|como|cuando|donde|quien|por que|para que)\b|^(?:(?:la luz|las luces|el aire|las? persianas?)(?: \w+){0,4} )?(?:esta|estas|estan|sigue|siguen|quedo|quedaron|tengo|tenes)\b";
}
