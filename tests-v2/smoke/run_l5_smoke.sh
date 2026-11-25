#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
exec "${PYTHON:-python3}" -m pytest tests-v2/smoke -q "$@"
