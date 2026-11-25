use std::{
    collections::{HashMap, VecDeque},
    fs::File,
    io,
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, SystemTime},
};

use crate::LlmError;
use serde::{Deserialize, Serialize};
use tempfile::NamedTempFile;
use tokio::{sync::Mutex, time::Instant};

const BACKOFF: [u64; 4] = [60, 300, 1500, 3600];
#[derive(Clone, Debug, Serialize, Deserialize)]
struct Saved {
    failures: u32,
    until: SystemTime,
}

#[derive(Clone, Debug)]
struct Entry {
    saved: Saved,
    deadline: Instant,
}

#[derive(Debug)]
pub(crate) struct Cooldowns {
    state: Arc<Mutex<State>>,
    path: Option<PathBuf>,
}

#[derive(Debug, Default)]
struct State {
    entries: HashMap<String, Entry>,
    dirty: bool,
}

impl Cooldowns {
    pub async fn load(path: Option<PathBuf>) -> Result<Self, LlmError> {
        let mut entries = HashMap::new();
        if let Some(path) = path.clone() {
            let saved = tokio::task::spawn_blocking(move || Self::read_saved(&path))
                .await
                .map_err(io::Error::other)??;
            for (id, saved) in saved {
                let remaining = saved
                    .until
                    .duration_since(SystemTime::now())
                    .unwrap_or_default()
                    .min(Duration::from_secs(3600));
                entries.insert(
                    id,
                    Entry {
                        saved,
                        deadline: Instant::now() + remaining,
                    },
                );
            }
        }
        Ok(Self {
            state: Arc::new(Mutex::new(State {
                entries,
                ..State::default()
            })),
            path,
        })
    }

    fn read_saved(path: &Path) -> io::Result<HashMap<String, Saved>> {
        let file = match File::open(path) {
            Ok(file) => file,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(HashMap::new()),
            Err(error) => return Err(error),
        };
        match serde_json::from_reader(file) {
            Ok(saved) => Ok(saved),
            Err(error) if error.is_io() => Err(error.into()),
            Err(error) => {
                let mut quarantine = path.as_os_str().to_owned();
                quarantine.push(".corrupt");
                let rename_error = std::fs::rename(path, quarantine).err();
                tracing::warn!(
                    line = error.line(),
                    column = error.column(),
                    quarantine_error = ?rename_error.map(|error| error.kind()),
                    "Ignoring invalid LLM cooldown JSON"
                );
                Ok(HashMap::new())
            }
        }
    }

    pub async fn remaining(&self, id: &str) -> Duration {
        self.state
            .lock()
            .await
            .entries
            .get(id)
            .map(|entry| entry.deadline.saturating_duration_since(Instant::now()))
            .unwrap_or_default()
    }

    pub async fn consecutive_errors(&self, id: &str) -> u32 {
        self.state
            .lock()
            .await
            .entries
            .get(id)
            .map_or(0, |entry| entry.saved.failures)
    }

    pub async fn update(&self, id: &str, failed: bool) -> io::Result<()> {
        let mut state = self.state.clone().lock_owned().await;
        let changed = if failed {
            let failures = state
                .entries
                .get(id)
                .map_or(1, |entry| entry.saved.failures.saturating_add(1));
            let delay = Duration::from_secs(BACKOFF[(failures as usize).min(BACKOFF.len()) - 1]);
            state.entries.insert(
                id.to_owned(),
                Entry {
                    saved: Saved {
                        failures,
                        until: SystemTime::now() + delay,
                    },
                    deadline: Instant::now() + delay,
                },
            );
            true
        } else {
            state.entries.remove(id).is_some()
        };
        let Some(path) = &self.path else {
            return Ok(());
        };
        state.dirty |= changed;
        if !state.dirty {
            return Ok(());
        }

        // Serialize writes in update order. A failed write leaves the current
        // in-memory state dirty so a later update can persist it again.
        let path = path.clone();
        tokio::task::spawn_blocking(move || {
            Self::persist(&path, &state.entries)?;
            state.dirty = false;
            Ok(())
        })
        .await
        .map_err(io::Error::other)?
    }

    fn persist(path: &Path, entries: &HashMap<String, Entry>) -> io::Result<()> {
        let dir = path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        std::fs::create_dir_all(dir)?;
        let mut tmp = NamedTempFile::new_in(dir)?;
        let saved: HashMap<_, _> = entries
            .iter()
            .map(|(id, entry)| (id, &entry.saved))
            .collect();
        serde_json::to_writer(&mut tmp, &saved)?;
        tmp.as_file().sync_all()?;
        tmp.persist(path)?;
        Ok(())
    }
}

#[derive(Debug, PartialEq)]
pub struct Metrics {
    pub calls: usize,
    pub completion_tokens: u64,
    pub tokens_per_second: Option<f64>,
    pub median_latency: Duration,
}

#[derive(Debug, Default)]
pub(crate) struct Samples(VecDeque<(Instant, Duration, Option<u64>)>);

impl Samples {
    pub fn record(&mut self, elapsed: Duration, tokens: Option<u64>) {
        if self.0.len() == 200 {
            self.0.pop_front();
        }
        self.0.push_back((Instant::now(), elapsed, tokens));
    }

    pub fn snapshot(&self) -> Option<Metrics> {
        let mut latencies = Vec::new();
        let mut metered = Duration::ZERO;
        let mut tokens: u64 = 0;
        for (ts, elapsed, count) in &self.0 {
            if ts.elapsed() > Duration::from_secs(300) {
                continue;
            }
            latencies.push(*elapsed);
            if let Some(count) = count {
                tokens = tokens.saturating_add(*count);
                metered += *elapsed;
            }
        }
        if latencies.is_empty() {
            return None;
        }
        latencies.sort_unstable();
        let n = latencies.len();
        Some(Metrics {
            calls: n,
            completion_tokens: tokens,
            tokens_per_second: (!metered.is_zero()).then(|| tokens as f64 / metered.as_secs_f64()),
            median_latency: (latencies[(n - 1) / 2] + latencies[n / 2]) / 2,
        })
    }
}
