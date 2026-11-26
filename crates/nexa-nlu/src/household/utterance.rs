//! Folded tokens with spans into the original speech.

use crate::normalize::{WAKE_WORDS, fold, normalize_utterance};

use super::lexicon::{CANCEL, ENCLITICS, STOP, verb_classes};

#[derive(Debug, Clone)]
struct Token {
    spoken: String,
    start: usize,
    end: usize,
    quoted: bool,
}

#[derive(Debug, Clone)]
pub(super) struct Utterance<'a> {
    pub(super) source: &'a str,
    tokens: Vec<Token>,
    pub(super) folded: Vec<String>,
}

impl<'a> Utterance<'a> {
    pub(super) fn tokenize(source: &'a str) -> Self {
        let mut quotes = quote_boundaries(source).into_iter().peekable();
        let mut quoted = false;
        let tokens: Vec<_> = TOKEN
            .find_iter(source)
            .map(|m| {
                while quotes.peek().is_some_and(|&i| i < m.start()) {
                    quotes.next();
                    quoted = !quoted;
                }
                Token {
                    spoken: normalize_utterance(m.as_str()),
                    start: m.start(),
                    end: m.end(),
                    quoted,
                }
            })
            .collect();
        let folded = tokens.iter().map(|token| fold(&token.spoken)).collect();
        Self {
            source,
            tokens,
            folded,
        }
    }

    pub(super) fn strip_fillers(&self, wakes: &[String]) -> Self {
        let mut start = 0;
        let mut end = self.len();
        for _ in 0..6 {
            if start >= end || self.is_quoted(start) {
                break;
            }
            let mut next = None;
            for wake in wakes
                .iter()
                .map(String::as_str)
                .chain(WAKE_WORDS.iter().copied())
            {
                let mut tokens = self.folded[start..end].iter();
                if wake
                    .split_whitespace()
                    .all(|word| tokens.next().is_some_and(|t| t == word))
                {
                    next = Some(start + wake.split_whitespace().count());
                    break;
                }
            }
            let word = self.at(start);
            let paused_no = word == "no"
                && self.tokens.get(start + 1).is_some_and(|next| {
                    self.source[self.tokens[start].end..next.start]
                        .contains([',', ';', ':', '.', '!'])
                });
            start = match next {
                Some(next) => next,
                None if word == "por" && self.at(start + 1) == "favor" => start + 2,
                None if paused_no
                    || matches!(
                        word,
                        "che"
                            | "ey"
                            | "oye"
                            | "hola"
                            | "porfa"
                            | "podes"
                            | "podrias"
                            | "puedes"
                            | "podras"
                            | "bueno"
                            | "bue"
                            | "dale"
                    ) =>
                {
                    start + 1
                }
                None => break,
            };
        }
        while start < end && !self.is_quoted(end - 1) {
            if self.at(end - 1) == "porfa" {
                end -= 1;
            } else if end - start >= 2 && self.folded[end - 2..end] == ["por", "favor"] {
                end -= 2;
            } else {
                break;
            }
        }
        let core = self.slice(start, end);
        if core.at(1) == "ya" {
            let without = core.slice(0, 1).join(&core.slice(2, core.len()));
            if verb_at(&without, 0, false).is_some_and(|verb| verb.has(CANCEL)) {
                return without;
            }
        }
        core
    }

    pub(super) fn is_quoted(&self, index: usize) -> bool {
        self.tokens.get(index).is_some_and(|token| token.quoted)
    }

    pub(super) fn len(&self) -> usize {
        self.tokens.len()
    }
    pub(super) fn at(&self, index: usize) -> &str {
        self.folded.get(index).map_or("", String::as_str)
    }
    pub(super) fn spoken_at(&self, index: usize) -> &str {
        self.tokens
            .get(index)
            .map_or("", |token| token.spoken.as_str())
    }

    pub(super) fn text(&self) -> String {
        self.folded.join(" ")
    }
    pub(super) fn has(&self, vocab: &[&str]) -> bool {
        self.folded
            .iter()
            .any(|token| vocab.contains(&token.as_str()))
    }
    pub(super) fn rest(&self, start: usize) -> String {
        self.tokens
            .iter()
            .skip(start)
            .map(|token| token.spoken.as_str())
            .collect::<Vec<_>>()
            .join(" ")
    }
    pub(super) fn heard(&self) -> String {
        self.rest(0)
    }

    pub(super) fn literal(&self, noun: usize) -> String {
        self.literal_after(noun).map_or_else(
            || self.rest(noun + 1),
            |(start, end)| self.source[start..end].trim().to_owned(),
        )
    }

    pub(super) fn literal_after(&self, token: usize) -> Option<(usize, usize)> {
        let first = self.tokens.get(token)?;
        let last = self.tokens.last()?;
        let next = NEXT_WORD.find_at(self.source, last.end);
        let end = next.map_or(self.source.len(), |m| {
            last.end
                + self.source[last.end..m.start()]
                    .trim_end_matches([' ', '\t', ',', ';', ':'])
                    .len()
        });
        (first.end <= end).then_some((first.end, end))
    }

    pub(super) fn slice(&self, start: usize, end: usize) -> Self {
        Self {
            source: self.source,
            tokens: self.tokens[start..end].to_vec(),
            folded: self.folded[start..end].to_vec(),
        }
    }
    pub(super) fn join(mut self, other: &Self) -> Self {
        self.tokens.extend(other.tokens.iter().cloned());
        self.folded.extend(other.folded.iter().cloned());
        self
    }

    pub(super) fn grammar_text(&self) -> String {
        let mut text = String::new();
        if self.is_quoted(0) {
            text.push('«');
        }
        for (i, token) in self.tokens.iter().enumerate() {
            if i > 0 {
                let previous = &self.tokens[i - 1];
                if let Some(between) = self.source.get(previous.end..token.start)
                    && !between.chars().any(char::is_alphanumeric)
                {
                    text.push_str(between);
                } else {
                    text.push(' ');
                }
            }
            text.push_str(&token.spoken);
        }
        if let Some(last) = self.tokens.last() {
            // A percent immediately after the segment's last token belongs to the segment.
            if self.source[last.end..].trim_start().starts_with('%') {
                text.push('%');
            }
        }
        text
    }
}

// Ignore apostrophes within words and keep a bounded stack for nested spoken quotes.
fn quote_boundaries(text: &str) -> Vec<usize> {
    let mut boundaries = Vec::new();
    let mut stack = Vec::new();
    let mut chars = text.char_indices().peekable();
    let mut previous = None;
    while let Some((index, ch)) = chars.next() {
        let next = chars.peek().map(|&(_, ch)| ch);
        let apostrophe = ch == '\''
            && previous.is_some_and(char::is_alphanumeric)
            && next.is_some_and(char::is_alphanumeric);
        if !apostrophe {
            if stack.last() == Some(&ch) {
                stack.pop();
                if stack.is_empty() {
                    boundaries.push(index);
                }
            } else if stack.len() < 6 {
                let closer = match ch {
                    '«' => Some('»'),
                    '“' => Some('”'),
                    '"' | '\'' => Some(ch),
                    _ => None,
                };
                if let Some(closer) = closer {
                    if stack.is_empty() {
                        boundaries.push(index);
                    }
                    stack.push(closer);
                }
            }
        }
        previous = Some(ch);
    }
    boundaries
}

#[derive(Debug, Clone, Copy)]
pub(super) struct Verb<'a> {
    pub(super) form: &'a str,
    classes: u16,
    pub(super) end: usize,
    pub(super) infinitive: bool,
}

impl Verb<'_> {
    pub(super) fn has(self, mask: u16) -> bool {
        self.classes & mask != 0
    }
}

pub(super) fn verb_at<'a>(utt: &'a Utterance<'_>, index: usize, strict: bool) -> Option<Verb<'a>> {
    if utt.is_quoted(index) {
        return None;
    }
    let word = utt.at(index);
    if word == "olvidate" && matches!(utt.at(index + 1), "de" | "del") {
        return Some(Verb {
            form: word,
            classes: CANCEL,
            end: index + if utt.at(index + 1) == "de" { 2 } else { 1 },
            infinitive: false,
        });
    }
    if word == "para" {
        let accented = utt.spoken_at(index) == "pará";
        if !accented
            && (strict || ((index + 1)..utt.len()).any(|i| verb_at(utt, i, true).is_some()))
        {
            return None;
        }
        return Some(Verb {
            form: word,
            classes: STOP,
            end: index + 1,
            infinitive: false,
        });
    }
    let forms = std::iter::once(word).chain(
        ENCLITICS
            .iter()
            .filter_map(|suffix| word.strip_suffix(suffix).filter(|form| form.len() >= 2)),
    );
    for form in forms {
        let classes = verb_classes(form);
        if classes == 0 {
            continue;
        }
        let infinitive = ["ar", "er", "ir"]
            .iter()
            .any(|ending| form.ends_with(ending));
        if strict && infinitive {
            return None;
        }
        return Some(Verb {
            form,
            classes,
            end: index + 1,
            infinitive,
        });
    }
    None
}

regex_patterns! {
    TOKEN => r"[0-9]+(?:[:.][0-9]{2}|[.,][0-9]+)?|[\p{L}\p{M}]+";
    NEXT_WORD => r"[0-9]|[\p{L}\p{M}]";
}
