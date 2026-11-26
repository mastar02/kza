//! Read-only GPU observations and allowlisted systemd user operations.

use serde::Serialize;
use tonic::async_trait;

const RESTARTABLE: &[&str] = &[
    "nexa-audiod",
    "nexa-brain",
    "nexa-memoryd",
    "nexa-ml",
    "nexa-llm-fast",
];

#[cfg(any(target_os = "linux", test))]
fn monitored_services() -> impl Iterator<Item = &'static str> {
    RESTARTABLE
        .iter()
        .copied()
        .chain(["nexa-web", "nexa-dashboard-proxy"])
}

pub fn restartable(name: &str) -> bool {
    RESTARTABLE.contains(&name)
}

#[derive(Debug, Serialize)]
pub struct Gpu {
    pub id: u32,
    pub name: String,
    pub utilization: Option<u32>,
    pub memory_used_mb: Option<u64>,
    pub memory_total_mb: Option<u64>,
    pub temperature: Option<u32>,
    pub power_w: Option<f64>,
}

#[derive(Debug, Serialize)]
pub struct Service {
    pub name: String,
    pub status: String,
    pub substate: String,
}

#[derive(Debug, thiserror::Error)]
pub enum MonitorError {
    #[error("service_restart_denied")]
    Denied,
    #[error("system_monitor_unavailable")]
    Unavailable,
    #[error("restart_timeout")]
    Timeout,
    #[error("restart_command_failed")]
    Failed,
    #[error("observability_demo_mode")]
    Demo,
}

#[async_trait]
pub trait SystemMonitor: Send + Sync {
    async fn gpus(&self) -> Vec<Gpu>;
    async fn services(&self) -> Vec<Service>;
    async fn restart(&self, name: &str) -> Result<(), MonitorError>;
}

#[derive(Debug)]
pub struct PlatformMonitor;

#[async_trait]
impl SystemMonitor for PlatformMonitor {
    async fn gpus(&self) -> Vec<Gpu> {
        #[cfg(target_os = "linux")]
        {
            tokio::task::spawn_blocking(linux::gpus)
                .await
                .unwrap_or_default()
        }
        #[cfg(not(target_os = "linux"))]
        Vec::new()
    }

    async fn services(&self) -> Vec<Service> {
        #[cfg(target_os = "linux")]
        {
            tokio::time::timeout(std::time::Duration::from_secs(3), linux::services())
                .await
                .ok()
                .and_then(Result::ok)
                .unwrap_or_default()
        }
        #[cfg(not(target_os = "linux"))]
        Vec::new()
    }

    async fn restart(&self, name: &str) -> Result<(), MonitorError> {
        if !restartable(name) {
            return Err(MonitorError::Denied);
        }
        #[cfg(target_os = "linux")]
        {
            tokio::time::timeout(std::time::Duration::from_secs(15), linux::restart(name))
                .await
                .map_err(|_| MonitorError::Timeout)?
        }
        #[cfg(not(target_os = "linux"))]
        Err(MonitorError::Unavailable)
    }
}

#[cfg(target_os = "linux")]
mod linux {
    use super::{Gpu, MonitorError, Service, monitored_services};
    use futures_util::StreamExt;
    use nvml_wrapper::{Nvml, enum_wrappers::device::TemperatureSensor};
    use zbus::{Connection, Proxy, zvariant::OwnedObjectPath};

    const SYSTEMD: &str = "org.freedesktop.systemd1";
    const MANAGER: &str = "org.freedesktop.systemd1.Manager";

    pub fn gpus() -> Vec<Gpu> {
        let Ok(nvml) = Nvml::init() else {
            return Vec::new();
        };
        (0..nvml.device_count().unwrap_or_default())
            .filter_map(|id| {
                let device = nvml.device_by_index(id).ok()?;
                let memory = device.memory_info().ok();
                Some(Gpu {
                    id,
                    name: device.name().ok()?,
                    utilization: device.utilization_rates().ok().map(|rates| rates.gpu),
                    memory_used_mb: memory.as_ref().map(|info| info.used / (1024 * 1024)),
                    memory_total_mb: memory.as_ref().map(|info| info.total / (1024 * 1024)),
                    temperature: device.temperature(TemperatureSensor::Gpu).ok(),
                    power_w: device.power_usage().ok().map(|mw| f64::from(mw) / 1000.0),
                })
            })
            .collect()
    }

    async fn manager(connection: &Connection) -> zbus::Result<Proxy<'_>> {
        Proxy::new(connection, SYSTEMD, "/org/freedesktop/systemd1", MANAGER).await
    }

    pub async fn services() -> zbus::Result<Vec<Service>> {
        let connection = Connection::session().await?;
        let manager = manager(&connection).await?;
        let mut result = Vec::new();
        for name in monitored_services() {
            let unit_name = format!("{name}.service");
            let path: OwnedObjectPath = match manager.call("GetUnit", &(unit_name,)).await {
                Ok(path) => path,
                Err(_) => continue,
            };
            let unit =
                Proxy::new(&connection, SYSTEMD, path, "org.freedesktop.systemd1.Unit").await?;
            if let (Ok(status), Ok(substate)) = (
                unit.get_property::<String>("ActiveState").await,
                unit.get_property::<String>("SubState").await,
            ) {
                result.push(Service {
                    name: name.into(),
                    status,
                    substate,
                });
            }
        }
        Ok(result)
    }

    pub async fn restart(name: &str) -> Result<(), MonitorError> {
        let connection = Connection::session()
            .await
            .map_err(|_| MonitorError::Unavailable)?;
        let manager = manager(&connection)
            .await
            .map_err(|_| MonitorError::Unavailable)?;
        let mut jobs = manager
            .receive_signal("JobRemoved")
            .await
            .map_err(|_| MonitorError::Unavailable)?;
        manager
            .call::<_, _, ()>("Subscribe", &())
            .await
            .map_err(|_| MonitorError::Unavailable)?;
        let unit_name = format!("{name}.service");
        let path: OwnedObjectPath = manager
            .call("RestartUnit", &(unit_name, "replace"))
            .await
            .map_err(|_| MonitorError::Failed)?;
        while let Some(message) = jobs.next().await {
            let Ok((_, job, _, result)) = message
                .body()
                .deserialize::<(u32, OwnedObjectPath, String, String)>()
            else {
                continue;
            };
            if job == path {
                return if result == "done" {
                    Ok(())
                } else {
                    Err(MonitorError::Failed)
                };
            }
        }
        Err(MonitorError::Unavailable)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn systemd_inventory_preserves_http_services_during_restart() {
        let (allowed, read_only): (Vec<_>, Vec<_>) =
            monitored_services().partition(|name| restartable(name));
        assert_eq!(
            allowed,
            [
                "nexa-audiod",
                "nexa-brain",
                "nexa-memoryd",
                "nexa-ml",
                "nexa-llm-fast"
            ]
        );
        assert_eq!(read_only, ["nexa-web", "nexa-dashboard-proxy"]);
    }
}
