//! Device nouns and command verbs.

use super::Domain;

pub(crate) fn entity_at(word: &str) -> Option<Domain> {
    Some(match word {
        "luz" | "luces" | "lampara" | "lamparas" | "foco" | "focos" | "bombilla" | "bombillas"
        | "led" | "leds" | "velador" | "veladores" | "iluminacion" | "brillo" => Domain::Light,
        "aire" | "ac" | "clima" | "climatizador" | "calefaccion" | "calefactor" | "estufa"
        | "termotanque" | "termostato" | "temperatura" => Domain::Climate,
        "persiana" | "persianas" | "cortina" | "cortinas" | "blackout" | "toldo" | "toldos"
        | "estor" | "estores" => Domain::Cover,
        "escena" | "escenas" => Domain::Scene,
        "ventilador" | "ventiladores" | "extractor" | "extractores" => Domain::Fan,
        "tele" | "tv" | "television" | "televisor" | "musica" | "parlante" | "parlantes"
        | "bocina" | "bocinas" | "altavoz" | "altavoces" | "radio" | "volumen" | "cancion"
        | "tema" | "temas" | "playlist" => Domain::MediaPlayer,
        _ => return None,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Verb {
    On,
    Off,
    Set,
    Raise,
    Lower,
    Open,
    Close,
    Stop,
    Play,
    Pause,
    Next,
}

pub(crate) fn action(word: &str) -> Option<Verb> {
    Some(match word {
        "prende" | "prenda" | "prendan" | "prender" | "prendela" | "prendelo" | "prenderla"
        | "prenderlo" | "prendeme" | "encende" | "encender" | "enciende" | "encienda"
        | "encendela" | "encendelo" | "encenderla" | "encenderlo" | "activa" | "activar"
        | "active" | "ilumina" => Verb::On,
        "apaga" | "apagar" | "apague" | "apaguen" | "apagala" | "apagalo" | "apagarla"
        | "apagarlo" | "apagame" | "corta" | "corte" | "desactiva" | "desactivar" | "desactive" => {
            Verb::Off
        }
        "pone" | "poner" | "pon" | "ponga" | "poneme" | "ponela" | "ponelo" | "ajusta"
        | "ajustar" | "ajuste" | "deja" | "dejar" | "deje" | "configura" | "configurar"
        | "configure" | "cambia" | "cambiar" | "cambie" => Verb::Set,
        "subi" | "sube" | "subir" | "suba" | "subila" | "subilo" | "levanta" | "levantar" => {
            Verb::Raise
        }
        "baja" | "bajar" | "baje" | "bajen" | "bajala" | "bajalo" => Verb::Lower,
        "abri" | "abre" | "abrir" | "abra" => Verb::Open,
        "cerra" | "cierra" | "cerrar" | "cierre" => Verb::Close,
        "para" | "parar" | "pare" | "detene" | "detener" | "deten" | "detenga" | "frena"
        | "frenar" | "frene" => Verb::Stop,
        "reproduci" | "reproduce" | "reproducir" | "segui" | "sigue" => Verb::Play,
        "pausa" | "pause" | "pausar" | "silencia" => Verb::Pause,
        "siguiente" | "salta" => Verb::Next,
        _ => return None,
    })
}
