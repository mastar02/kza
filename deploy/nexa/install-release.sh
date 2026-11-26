#!/usr/bin/env bash
# [--user] install <release-dir> <version> | rollback <previous-version>
set -euo pipefail
export LC_ALL=C

die() { echo "ERROR: $*" >&2; exit 1; }

user_mode=0
if [[ ${1:-} == --user ]]; then
  shift
  [[ $EUID != 0 ]] || die "--user requires an unprivileged account"
  [[ $(id -un) == "${NEXA_USER:-nexa}" ]] || die "--user requires the Nexa account"
  user_mode=1
  export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
  export DBUS_SESSION_BUS_ADDRESS="${DBUS_SESSION_BUS_ADDRESS:-unix:path=$XDG_RUNTIME_DIR/bus}"
fi
if [[ ${1:-} == --rollback ]]; then shift; set -- rollback "$@"; fi
case "${1:-}" in
  install) [[ $# == 3 ]] || die "expected install <release-dir> <version>"; version=$3 ;;
  rollback) [[ $# == 2 ]] || die "expected rollback <previous-version>"; version=$2 ;;
  *) die "expected install or rollback" ;;
esac
[[ $version =~ ^[A-Za-z0-9._-]+$ && $version != *..* && $version != . ]] || die "invalid version"

prefix=${NEXA_INSTALL_TEST_ROOT:-}
if [[ -n $prefix || -n ${NEXA_SKIP_SYSTEMCTL:-} || -n ${NEXA_SKIP_CHOWN:-} ]]; then
  [[ $prefix == /* && ${NEXA_SKIP_SYSTEMCTL:-} == 1 && ${NEXA_SKIP_CHOWN:-} == 1 ]] ||
    die "test mode requires an absolute test directory and both skip flags"
  prefix=$(cd "$prefix" && pwd -P)
  [[ -n $prefix && $prefix != / ]] || die "unsafe test root"
elif (( ! user_mode )); then
  [[ $EUID == 0 ]] || die "use --user as the Nexa account or run as root"
  [[ ! -L /opt/nexa ]] || die "expected a physical deployment directory"
  case $(stat -c %U /opt/nexa) in
    nexa)
      # After handoff, drop privileges before accessing the user-controlled tree.
      exec sudo -H -u nexa env -u XDG_RUNTIME_DIR -u DBUS_SESSION_BUS_ADDRESS NEXA_USER=nexa bash "$0" --user "$@" ;;
    root)
      [[ ! -L /opt/nexa/releases && $(stat -c %U /opt/nexa/releases) == root ]] ||
        die "expected root-owned releases before handoff" ;;
    *) die "unexpected owner of /opt/nexa" ;;
  esac
fi

base="$prefix/opt/nexa"
releases="$base/releases"
units="$prefix/home/nexa/.config/systemd/user"
if (( user_mode )) && [[ -z $prefix ]]; then units="$HOME/.config/systemd/user"; fi
dest="$releases/$version"
if [[ -n $prefix ]]; then mkdir -p "$releases" "$units"; fi

# Serialize install/rollback. All temporary state resides on the destination filesystem.
lock="$base/.install-lock"
mkdir -m 0700 "$lock" || die "install lock unavailable: $lock"
pending_release=""
cleanup() {
  if [[ -n $pending_release && -d $pending_release ]]; then
    chmod -R u+w "$pending_release"
    rm -rf "$pending_release"
  fi
  if (( user_mode )) && [[ -d $lock/release ]]; then
    chmod -R u+w "$lock/release"
  fi
  rm -rf "$lock"
}
trap cleanup EXIT
trap 'exit 1' HUP INT TERM

as_nexa() {
  if (( user_mode )) || [[ -n $prefix ]]; then "$@"; else
    local uid
    uid=$(id -u nexa)
    sudo -u nexa env "XDG_RUNTIME_DIR=/run/user/$uid" "DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/$uid/bus" "$@"
  fi
}

stage_tool="$(dirname "$0")/stage-release.py"
if [[ $1 == install ]]; then
  [[ ! -e $dest && ! -L $dest ]] || die "release already exists: $version"
  stage="$lock/release"
  python3 "$stage_tool" "$2" "$stage" "$version" "$prefix"
else
  stage="$dest"
  python3 "$stage_tool" --verify-installed "$stage" "$version" "$prefix"
fi
(
  cd "$stage"
  if command -v sha256sum >/dev/null; then sha256sum -c SHA256SUMS
  else shasum -a 256 -c SHA256SUMS; fi
)
for bin in nexa-audiod nexa-brain nexa-memoryd nexa-web; do
  [[ -f $stage/bin/$bin && -x $stage/bin/$bin ]] || die "missing executable: $bin"
done
if [[ $1 == install ]]; then
  if (( ! user_mode )) && [[ -z $prefix ]]; then chown -R root:nexa "$stage"; fi
  chmod -R u=rX,g=rX,o= "$stage"
  find "$stage" -type d -exec chmod 0750 {} +
  for bin in nexa-audiod nexa-brain nexa-memoryd nexa-web; do
    chmod 0750 "$stage/bin/$bin"
  done
  if (( user_mode )); then pending_release=$dest; fi
  mv "$stage" "$dest"
  if (( user_mode )); then
    chmod -R a-w "$dest"
    pending_release=""
  fi
fi

shopt -s nullglob
for unit in "$dest"/systemd/*.{service,target}; do
  as_nexa install -m 0644 "$unit" "$units/${unit##*/}"
done
ln -s "releases/$version" "$lock/current"
python3 -c 'import os, sys; os.replace(*sys.argv[1:])' "$lock/current" "$base/current"

if [[ -z $prefix ]]; then
  as_nexa systemctl --user daemon-reload
  deadline=$((SECONDS + 130))
  as_nexa timeout 130s systemctl --user restart nexa.target
  for svc in nexa-memoryd nexa-llm-fast nexa-audiod nexa-ml nexa-brain nexa-web nexa-dashboard-proxy; do
    while true; do
      state=$(as_nexa systemctl --user is-active "$svc.service" 2>/dev/null || true)
      case "$state" in
        active) break ;;
        activating|reloading)
          (( SECONDS < deadline )) || die "$svc timed out in state $state after switch"
          sleep 1 ;;
        *) die "$svc state=$state after switch" ;;
      esac
    done
  done
fi
echo "Switched $base/current -> releases/$version"
