//! Incremental filter retaining only an ambiguous delimiter prefix.
#[derive(Default, Debug)]
pub struct ThinkTagStripper {
    pending: String,
    hidden: bool,
}

impl ThinkTagStripper {
    pub fn push(&mut self, chunk: &str) -> String {
        let mut output = String::new();
        for c in chunk.chars() {
            let tag = if self.hidden { "</think>" } else { "<think>" };
            if c == '<' {
                if !self.hidden {
                    output.push_str(&self.pending);
                }
                self.pending.clear();
            }
            self.pending.push(c);
            if self.pending == tag {
                self.hidden = !self.hidden;
                self.pending.clear();
            } else if !tag.starts_with(&self.pending) {
                if !self.hidden {
                    output.push_str(&self.pending);
                }
                self.pending.clear();
            }
        }
        output
    }

    pub fn finish(self) -> String {
        if self.hidden {
            String::new()
        } else {
            self.pending
        }
    }
}
