#!/usr/bin/env bash
# Install a pinned GMAT release for the golden drift check (design doc §23.1).
#
# GMAT is a ~300 MB NASA desktop tool shipped as a per-release Linux tarball —
# there is no apt/pip/conda package. This is NOT part of the normal build: only
# the nightly `golden` CI lane and local devs who want to run the GMAT drift test
# (tests/tools/test_gmat_drift.py) need it.
#
# Prints the resolved GmatConsole path to stdout (logs go to stderr) so callers
# can do:  GMAT_CONSOLE="$(bash tools/gmat/install_gmat.sh)"
#
# ponytail: verify GMAT_VERSION/GMAT_URL against the releases page before trusting
# a run — https://sourceforge.net/projects/gmat/files/GMAT/ . GMAT's Linux tarball
# name and target Ubuntu version change per release; override via env if they move.
# R2026a's ubuntu-x64 tarball was built on Ubuntu 22.04 LTS.
set -euo pipefail

GMAT_VERSION="${GMAT_VERSION:-R2026a}"
GMAT_URL="${GMAT_URL:-https://downloads.sourceforge.net/project/gmat/GMAT/GMAT-${GMAT_VERSION}/gmat-ubuntu-x64-${GMAT_VERSION}.tar.gz}"
GMAT_DIR="${GMAT_DIR:-${HOME}/.cache/polaris-gmat/${GMAT_VERSION}}"

find_console() { find "${GMAT_DIR}" -name GmatConsole -type f 2>/dev/null | head -1; }

console="$(find_console || true)"
if [[ -n "${console}" ]]; then
  echo "GMAT ${GMAT_VERSION} already present" >&2
  echo "${console}"
  exit 0
fi

mkdir -p "${GMAT_DIR}"
tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT
echo "Downloading GMAT ${GMAT_VERSION} from ${GMAT_URL} ..." >&2
curl -fsSL "${GMAT_URL}" -o "${tmp}/gmat.tar.gz"
echo "Extracting to ${GMAT_DIR} ..." >&2
tar -xzf "${tmp}/gmat.tar.gz" -C "${GMAT_DIR}" --strip-components=1

console="$(find_console || true)"
if [[ -z "${console}" ]]; then
  echo "GmatConsole not found under ${GMAT_DIR} after extract" >&2
  exit 1
fi
chmod +x "${console}"
echo "${console}"
