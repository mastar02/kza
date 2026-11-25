//! Accent-folded vocabulary from the household classifier.

pub(super) const ENCLITICS: &[&str] = &[
    "noslo", "nosla", "melo", "mela", "selo", "sela", "me", "te", "le", "lo", "la", "nos", "les",
    "los", "las",
];
pub(super) const CANCEL: u16 = 1 << 0;
pub(super) const CLEAR: u16 = 1 << 1;
pub(super) const STOP: u16 = 1 << 2;
pub(super) const STOP_RINGING: u16 = 1 << 3;
pub(super) const PAUSE: u16 = 1 << 4;
pub(super) const RESUME: u16 = 1 << 5;
pub(super) const DISABLE: u16 = 1 << 6;
pub(super) const CREATE: u16 = 1 << 7;
pub(super) const TIMER_CREATE: u16 = 1 << 8;
pub(super) const ADD: u16 = 1 << 9;
pub(super) const REMOVE: u16 = 1 << 10;
pub(super) const ROUTINE_EXEC: u16 = 1 << 11;
pub(super) const ROUTINE_PROPOSE: u16 = 1 << 12;
pub(super) const AUTOMATE: u16 = 1 << 13;
pub(super) const INTERCOM: u16 = 1 << 14;
pub(super) const DOMOTICS: u16 = 1 << 15;
pub(super) const ALARM_NOUNS: &[&str] = &["alarma", "alarmas", "despertador", "despertadores"];
pub(super) const SECURITY_ALARM_VERBS: &[&str] = &[
    "activa",
    "active",
    "activar",
    "desactiva",
    "desactive",
    "desactivar",
    "arma",
    "arme",
    "armar",
    "desarma",
    "desarme",
    "desarmar",
];
pub(super) const DETERMINERS: &[&str] = &[
    "el", "la", "los", "las", "un", "una", "mi", "mis", "ese", "esa", "esos", "esas", "este",
    "esta", "estos", "estas", "del", "al", "todos", "todas", "nuestro", "nuestra", "nuestros",
    "nuestras", "otro", "otra",
];
pub(super) const ADJECTIVES: &[&str] = &["nuevo", "nueva", "nuevos", "nuevas"];
pub(super) const CONJUNCTIONS: &[&str] = &["y", "tambien", "despues", "luego"];
pub(super) const CONNECTORS: &[&str] = &["despues", "luego", "tambien", "ademas"];
pub(super) const CONTENT_MARKERS: &[&str] = &["que", "cuando", "si", "porque"];
pub(super) const MUSIC_WORDS: &[&str] = &[
    "musica",
    "cancion",
    "canciones",
    "tema",
    "temas",
    "playlist",
    "spotify",
    "reproduccion",
    "album",
    "disco",
];
pub(super) const QUERY_HEADS: &[&str] = &["que", "cuales", "cuantos", "cuantas"];
pub(super) const HAVE_HEADS: &[&str] = &["hay", "tengo", "tenemos"];
pub(super) const SOME: &[&str] = &["algun", "alguna", "algunos", "algunas"];
pub(super) const QUERY_VERBS: &[&str] = &[
    "hay",
    "tengo",
    "tenemos",
    "quedan",
    "queda",
    "son",
    "tiene",
    "estan",
    "activos",
    "activas",
    "pendientes",
    "corriendo",
    "anda",
    "andan",
];
pub(super) const READ_HEADS: &[&str] = &[
    "decime", "dime", "leeme", "lee", "mostrame", "listame", "nombrame", "contame",
];
pub(super) const ALARM_VERBS: &[&str] = &[
    "pone",
    "pon",
    "ponga",
    "poner",
    "programa",
    "programe",
    "programar",
    "crea",
    "crear",
    "configura",
    "configure",
    "configurar",
    "agenda",
    "agendar",
];
pub(super) const INDEFINITE: &[&str] = &["un", "una", "unos", "unas", "otro", "otra"];
pub(super) const ROUTINE_CREATE_FORMS: &[&str] = &[
    "crea",
    "cree",
    "crear",
    "crees",
    "creas",
    "crearias",
    "arma",
    "arme",
    "armar",
    "armes",
    "armas",
    "armarias",
    "genera",
    "genere",
    "generar",
    "generes",
    "generas",
    "generarias",
    "defini",
    "define",
    "defina",
    "definir",
    "definas",
    "definis",
    "definirias",
    "inventa",
    "invente",
    "inventes",
    "inventas",
    "inventarias",
    "disena",
    "disene",
    "disenar",
    "disenes",
    "disenas",
    "disenarias",
    "escribi",
    "escribe",
    "escriba",
    "escribir",
    "escribas",
    "escribis",
    "escribirias",
];
pub(super) const ROUTINE_NEW_ONLY_FORMS: &[&str] = &[
    "hace",
    "haz",
    "haga",
    "hacer",
    "hagas",
    "haces",
    "agrega",
    "agregue",
    "agregar",
    "agregues",
    "agregas",
    "agregarias",
    "suma",
    "sume",
    "sumar",
    "sumes",
    "sumas",
    "sumarias",
    "anadi",
    "anade",
    "anada",
    "anadir",
    "anadas",
    "programa",
    "programe",
    "programar",
    "programes",
    "programas",
    "programarias",
    "configura",
    "configure",
    "configurar",
    "configures",
    "configuras",
    "configurarias",
    "agenda",
    "agende",
    "agendar",
    "agendes",
    "agendas",
    "activa",
    "active",
    "activar",
    "actives",
    "activas",
    "prepara",
    "prepare",
    "preparar",
    "prepares",
    "preparas",
    "prepararias",
    "tener",
    "automatiza",
    "automatice",
    "automatizar",
];
pub(super) const ROUTINE_SLOT_EXCLUDED: &[&str] = &[
    "que",
    "ver",
    "mostra",
    "mostrar",
    "muestra",
    "muestre",
    "muestres",
    "mostras",
    "mostrarias",
    "mira",
    "mire",
    "mirar",
    "leer",
    "deci",
    "decir",
    "conta",
    "contar",
    "cuenta",
    "cuente",
    "saber",
    "pondria",
    "pondrias",
];

pub(super) fn verb_classes(word: &str) -> u16 {
    match word {
        "cancela" | "cancele" | "cancelen" | "cancelar" | "borren" | "anula" | "anule"
        | "anular" => CANCEL,
        "borra" | "borre" | "borrar" | "elimina" | "elimine" | "eliminar" | "saca" | "saque"
        | "sacar" | "quita" | "quite" | "quitar" => CANCEL | REMOVE,
        "vacia" | "vacie" | "vaciar" | "limpia" | "limpie" | "limpiar" => CLEAR,
        "pare" | "parar" | "detene" | "deten" | "detenga" | "detener" | "frena" | "frene"
        | "frenar" => STOP,
        "apaga" | "apague" | "apagar" => STOP_RINGING | DOMOTICS,
        "calla" | "calle" | "callar" | "silencia" | "silencie" | "silenciar" => STOP_RINGING,
        "pausa" | "pause" | "pausar" => PAUSE | DOMOTICS,
        "segui" | "sigue" => RESUME | DOMOTICS,
        "seguir" | "continua" | "continue" | "continuar" | "reanuda" | "reanude" | "reanudar" => {
            RESUME
        }
        "desactiva" | "desactive" | "desactivar" => DISABLE | DOMOTICS,
        "pone" | "pon" | "ponga" | "poner" => CREATE | ROUTINE_EXEC | DOMOTICS,
        "crea" | "crear" | "programa" | "programe" | "programar" | "configura" | "configure"
        | "configurar" | "agenda" | "agendar" => CREATE | ROUTINE_PROPOSE,
        "arranca" | "arranque" | "arrancar" | "inicia" | "inicie" | "iniciar" => {
            CREATE | ROUTINE_EXEC
        }
        "activa" | "active" | "activar" => CREATE | ROUTINE_PROPOSE | DOMOTICS,
        "hace" | "haga" => TIMER_CREATE | ROUTINE_PROPOSE,
        "mete" | "meta" | "tira" | "tire" => TIMER_CREATE,
        "agrega" | "agregue" | "agregar" | "anadi" | "anade" | "anada" | "anadir" | "suma"
        | "sume" | "sumar" | "anota" | "anote" | "anotar" => ADD,
        "tacha" | "tache" | "tachar" => REMOVE,
        "ejecuta" | "ejecute" | "ejecutar" | "corre" | "corra" | "correr" | "lanza" | "lance"
        | "lanzar" => ROUTINE_EXEC,
        "arma" | "arme" | "armar" | "hacer" => ROUTINE_PROPOSE,
        "automatiza" | "automatice" | "automatizar" => AUTOMATE,
        "anuncia" | "anuncie" | "anunciar" | "avisa" | "avise" | "comunica" | "comunique"
        | "deci" | "di" | "diga" => INTERCOM,
        "prende" | "prenda" | "prendan" | "prender" | "encende" | "enciende" | "encienda"
        | "encender" | "apaguen" | "sube" | "subi" | "suba" | "subir" | "baja" | "baje"
        | "bajen" | "bajar" | "abri" | "abre" | "abra" | "abrir" | "cierra" | "cerra"
        | "cierre" | "cerrar" | "cambia" | "cambie" | "cambiar" | "reproduci" | "reproduce"
        | "reproducir" | "ilumina" | "corta" | "corte" => DOMOTICS,
        _ => 0,
    }
}
pub(super) const ROUTINE_NOUNS: &[&str] = &[
    "rutina",
    "rutinas",
    "regla",
    "reglas",
    "automatizacion",
    "automatizaciones",
];
pub(super) const DESIRE_HEADS: &[&str] = &[
    "quiero",
    "quisiera",
    "queria",
    "querria",
    "necesito",
    "necesitaria",
    "preciso",
    "me gustaria",
    "nos gustaria",
    "me encantaria",
];

pub(super) fn known_command_word(word: &str) -> bool {
    if word.is_empty() || ROUTINE_SLOT_EXCLUDED.contains(&word) || verb_classes(word) != 0 {
        return true;
    }
    // Inflected command verbs must not be interpreted as novel creation verbs.
    const ROOTS: &[&str] = &[
        "activ", "desactiv", "prend", "apag", "encend", "abr", "cerr", "sub", "baj", "pon", "quit",
        "sac", "cancel", "borr", "elimin", "ejecut", "lanz", "corr", "paus", "busc", "mostr", "le",
        "mir", "revis", "cambi", "deten", "modific", "list", "consult",
    ];
    ROOTS.iter().any(|root| {
        word.strip_prefix(root).is_some_and(|tail| {
            matches!(
                tail,
                "a" | "e"
                    | "i"
                    | "ar"
                    | "er"
                    | "ir"
                    | "as"
                    | "es"
                    | "is"
                    | "amos"
                    | "emos"
                    | "imos"
                    | "aria"
                    | "arias"
                    | "eria"
                    | "erias"
                    | "iria"
                    | "irias"
                    | "an"
                    | "en"
                    | "ame"
                    | "eme"
                    | "ime"
            )
        })
    })
}
