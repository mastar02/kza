#!/usr/bin/env bash
# install <release-dir> <version> | rollback <previous-version>
set -euo pipefail
export LC_ALL=C

die() { echo "ERROR: $*" >&2; exit 1; }

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
elif [[ $EUID != 0 ]]; then
  die "production install requires root"
fi

base="$prefix/opt/nexa"
releases="$base/releases"
units="$prefix/home/nexa/.config/systemd/user"
dest="$releases/$version"
if [[ -n $prefix ]]; then mkdir -p "$releases" "$units"; fi

# Serialize install/rollback. All temporary state resides on the destination filesystem.
lock="$base/.install-lock"
mkdir -m 0700 "$lock" || die "install lock unavailable: $lock"
trap 'rm -rf "$lock"' EXIT
trap 'exit 1' HUP INT TERM

as_nexa() {
  if [[ -n $prefix ]]; then "$@"; else
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
  if [[ -z $prefix ]]; then chown -R root:nexa "$stage"; fi
  chmod -R u=rX,g=rX,o= "$stage"
  find "$stage" -type d -exec chmod 0750 {} +
  for bin in nexa-audiod nexa-brain nexa-memoryd nexa-web; do
    chmod 0750 "$stage/bin/$bin"
  done
  mv "$stage" "$dest"
fi

shopt -s nullglob
for unit in "$dest"/systemd/*.{service,target}; do
  as_nexa install -m 0644 "$unit" "$units/${unit##*/}"
done
ln -s "releases/$version" "$lock/current"
python3 -c 'import os, sys; os.replace(*sys.argv[1:])' "$lock/current" "$base/current"

if [[ -z $prefix ]]; then
  as_nexa systemctl --user daemon-reload
  as_nexa systemctl --user restart nexa.target
  deadline=$((SECONDS + 130))
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
