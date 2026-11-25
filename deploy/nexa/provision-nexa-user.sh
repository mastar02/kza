#!/usr/bin/env bash
# One-time privileged server bootstrap for the dedicated non-sudo `nexa` runtime account.
# Run ONLY on the Linux host during an authorized provisioning window (`sudo ./provision-nexa-user.sh`).
# Zero containers/pods: configures native systemd --user linger, minimal hardware groups,
# immutable /opt/nexa releases dir, strict udev rules, and per-UID 30s stop timeout.
set -euo pipefail

NEXA_USER="nexa"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${EUID}" -ne 0 ]]; then
  echo "ERROR: Run this one-time provisioning script with sudo/root on the Linux server." >&2
  exit 1
fi

if ! id -u "${NEXA_USER}" >/dev/null 2>&1; then
  useradd \
    --create-home \
    --home-dir "/home/${NEXA_USER}" \
    --shell /bin/bash \
    --comment "Nexa v2 Voice Assistant (non-sudo runtime)" \
    "${NEXA_USER}"
  passwd -l "${NEXA_USER}" >/dev/null
  echo "Created dedicated non-sudo user '${NEXA_USER}' (password locked)."
else
  echo "User '${NEXA_USER}' already exists; verifying groups and directories."
fi

# Strictly forbid privileged groups.
for forbidden in sudo adm wheel docker; do
  if id -nG "${NEXA_USER}" | tr ' ' '\n' | grep -qx "${forbidden}"; then
    echo "ERROR: User '${NEXA_USER}' must NOT belong to privileged group '${forbidden}'!" >&2
    exit 1
  fi
done

# Grant ONLY minimal hardware access: ALSA audio, MA1260 serial, CUDA GPUs.
for grp in audio dialout video render; do
  if getent group "${grp}" >/dev/null 2>&1; then
    usermod -aG "${grp}" "${NEXA_USER}"
  fi
done

loginctl enable-linger "${NEXA_USER}"

NEXA_UID="$(id -u "${NEXA_USER}")"

# Immutable release binaries & models (read-only for nexa; writable only by release installer).
install -d -o root -g "${NEXA_USER}" -m 0750 /opt/nexa /opt/nexa/releases /opt/nexa/models
# Validated TOML config (read-only for nexa) & owner-only secrets.
install -d -o root -g "${NEXA_USER}" -m 0750 /etc/nexa
install -d -o "${NEXA_USER}" -g "${NEXA_USER}" -m 0700 /etc/nexa/secrets
# Mutable WAL SQLite & runtime state owned exclusively by nexa.
install -d -o "${NEXA_USER}" -g "${NEXA_USER}" -m 0700 \
  /var/lib/nexa \
  /var/lib/nexa/.cache \
  /var/lib/nexa/proxy \
  /var/lib/nexa/memory \
  /var/lib/nexa/household \
  /var/lib/nexa/auth \
  /var/lib/nexa/clips \
  /var/lib/nexa/private_home
install -d -o "${NEXA_USER}" -g "${NEXA_USER}" -m 0750 \
  "/home/${NEXA_USER}/.config" \
  "/home/${NEXA_USER}/.config/systemd" \
  "/home/${NEXA_USER}/.config/systemd/user"

# Restrict XVF3800 USB vendor control to group `nexa` (0660, replacing legacy 0666).
install -m 0644 "${SCRIPT_DIR}/udev/99-xvf3800-nexa.rules" /etc/udev/rules.d/99-xvf3800-nexa.rules
udevadm control --reload-rules
udevadm trigger --subsystem-match=usb || true

# Raise Ubuntu's default 5s user@.service stop timeout ONLY for nexa's UID (Rust daemons stop in <2s; 30s safety cap).
DROPIN_DIR="/etc/systemd/system/user@${NEXA_UID}.service.d"
install -d -o root -g root -m 0755 "${DROPIN_DIR}"
cat > "${DROPIN_DIR}/zz-nexa-stop-timeout.conf" <<'EOF'
[Service]
TimeoutStopSec=30s
EOF
chmod 0644 "${DROPIN_DIR}/zz-nexa-stop-timeout.conf"
systemctl daemon-reload

echo ""
echo "=== Provisioning complete ==="
id "${NEXA_USER}"
loginctl show-user "${NEXA_USER}" -p Linger -p UID
echo "Next steps:"
echo "  1. Copy config/nexa/{nexa.toml,rooms.example.toml,ha-controls.example.toml} to /etc/nexa/ and edit."
echo "  2. Populate /etc/nexa/secrets/*.env (chmod 0600, chown nexa:nexa)."
echo "  3. Deploy compiled release via deploy/nexa/install-release.sh."
