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
    failures: usize,
    until: SystemTime,
}

#[derive(Clone, Debug)]
struct Entry {
    saved: Saved,
    deadline: Instant,
}

#[derive(Debug)]
pub(crate) struct Cooldowns {
    entries: Arc<Mutex<HashMap<String, Entry>>>,
    path: Option<PathBuf>,
}

impl Cooldowns {
    pub async fn load(path: Option<PathBuf>) -> Result<Self, LlmError> {
        let mut entries = HashMap::new();
        if let Some(path) = path.clone() {
            let saved =
                tokio::task::spawn_blocking(move || -> io::Result<HashMap<String, Saved>> {
                    match File::open(path) {
                        Ok(f) => Ok(serde_json::from_reader(f)?),
                        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(HashMap::new()),
                        Err(e) => Err(e),
                    }
                })
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
            entries: Arc::new(Mutex::new(entries)),
            path,
        })
    }

    pub async fn remaining(&self, id: &str) -> Duration {
        self.entries
            .lock()
            .await
            .get(id)
            .map(|entry| entry.deadline.saturating_duration_since(Instant::now()))
            .unwrap_or_default()
    }

    pub async fn update(&self, id: &str, failed: bool) -> Result<(), LlmError> {
        let mut entries = self.entries.clone().lock_owned().await;
        if !failed && !entries.contains_key(id) {
            return Ok(());
        }
        let next = failed.then(|| {
            let failures = entries
                .get(id)
                .map_or(1, |e| e.saved.failures.saturating_add(1))
                .min(4);
            let delay = Duration::from_secs(BACKOFF[failures - 1]);
            Entry {
                saved: Saved {
                    failures,
                    until: SystemTime::now() + delay,
                },
                deadline: Instant::now() + delay,
            }
        });
        let id = id.to_owned();
        let commit = move |path: Option<&Path>| -> io::Result<()> {
            if let Some(path) = path {
                let dir = path
                    .parent()
                    .filter(|p| !p.as_os_str().is_empty())
                    .unwrap_or(Path::new("."));
                std::fs::create_dir_all(dir)?;
                let mut tmp = NamedTempFile::new_in(dir)?;
                let mut saved: HashMap<_, _> = entries.iter().map(|(k, v)| (k, &v.saved)).collect();
                saved.remove(&id);
                if let Some(e) = &next {
                    saved.insert(&id, &e.saved);
                }
                serde_json::to_writer(&mut tmp, &saved)?;
                tmp.as_file().sync_all()?;
                tmp.persist(path)?;
            }
            match next {
                Some(entry) => {
                    entries.insert(id, entry);
                }
                None => {
                    entries.remove(&id);
                }
            }
            Ok(())
        };
        if let Some(path) = self.path.clone() {
            tokio::task::spawn_blocking(move || commit(Some(&path)))
                .await
                .map_err(io::Error::other)??;
        } else {
            commit(None)?;
        }
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
