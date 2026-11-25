#!/usr/bin/env bash
# Atomic compiled binary release installer & instant rollback for Nexa v2 (no pods, no .pyc slots).
# Usage:
#   sudo ./install-release.sh install <release-dir-with-SHA256SUMS> <version>
#   sudo ./install-release.sh rollback <previous-version>
set -euo pipefail

NEXA_USER="nexa"
RELEASES_ROOT="/opt/nexa/releases"
CURRENT_LINK="/opt/nexa/current"

run_as_nexa() {
  local uid
  uid="$(id -u "${NEXA_USER}")"
  sudo -u "${NEXA_USER}" env "XDG_RUNTIME_DIR=/run/user/${uid}" "DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/${uid}/bus" "$@"
}

restart_and_verify() {
  run_as_nexa systemctl --user daemon-reload
  run_as_nexa systemctl --user restart nexa.target
  for svc in nexa-audiod nexa-ml nexa-memoryd nexa-brain nexa-web; do
    if ! run_as_nexa systemctl --user is-active --quiet "${svc}.service"; then
      echo "ERROR: ${svc}.service failed to become active after switch." >&2
      run_as_nexa systemctl --user status "${svc}.service" --no-pager || true
      exit 1
    fi
  done
  echo "All Nexa v2 services active under user '${NEXA_USER}'."
}

cmd="${1:-}"
case "${cmd}" in
  install)
    src_dir="${2:?missing release directory}"
    version="${3:?missing release version (e.g. v2.0.0-abc1234)}"
    if [[ ! -f "${src_dir}/SHA256SUMS" ]]; then
      echo "ERROR: ${src_dir}/SHA256SUMS missing." >&2
      exit 1
    fi
    (cd "${src_dir}" && sha256sum -c SHA256SUMS)

    dest="${RELEASES_ROOT}/${version}"
    if [[ -e "${dest}" ]]; then
      echo "ERROR: Release directory ${dest} already exists (releases are immutable)." >&2
      exit 1
    fi

    install -d -o root -g "${NEXA_USER}" -m 0750 "${dest}/bin"
    for bin in nexa-audiod nexa-brain nexa-memoryd nexa-web; do
      install -o root -g "${NEXA_USER}" -m 0750 "${src_dir}/bin/${bin}" "${dest}/bin/${bin}"
    done
    if [[ -d "${src_dir}/frontend" ]]; then
      cp -a "${src_dir}/frontend" "${dest}/frontend"
      chown -R "root:${NEXA_USER}" "${dest}/frontend"
      chmod -R u=rX,g=rX,o= "${dest}/frontend"
    fi
    if [[ -d "${src_dir}/systemd" ]]; then
      install -m 0644 -o "${NEXA_USER}" -g "${NEXA_USER}" \
        "${src_dir}/systemd"/*.{service,target} "/home/${NEXA_USER}/.config/systemd/user/"
    fi

    # Atomic symlink swap.
    ln -sfn "${dest}" "${CURRENT_LINK}.next"
    mv -Tf "${CURRENT_LINK}.next" "${CURRENT_LINK}"
    echo "Switched ${CURRENT_LINK} -> ${dest}"
    restart_and_verify
    ;;

  rollback)
    prev="${2:?missing previous version}"
    target="${RELEASES_ROOT}/${prev}"
    if [[ ! -d "${target}/bin" ]]; then
      echo "ERROR: Previous release ${target} not found." >&2
      exit 1
    fi
    ln -sfn "${target}" "${CURRENT_LINK}.next"
    mv -Tf "${CURRENT_LINK}.next" "${CURRENT_LINK}"
    echo "Rolled back ${CURRENT_LINK} -> ${target}"
    restart_and_verify
    ;;

  *)
    echo "Usage: $0 {install <release-dir> <version> | rollback <previous-version>}" >&2
    exit 2
    ;;
esac
