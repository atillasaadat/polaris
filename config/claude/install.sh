#!/usr/bin/env bash
# Install the committed Claude Code development environment onto this machine.
#
# Copies the global-config snapshot in this directory into ~/.claude so a fresh
# machine reproduces the same environment (rules, agents, commands, hooks,
# CLAUDE.md/RTK.md, settings.json). Existing settings.json is backed up, not
# clobbered. Project-level config (.claude/ at the repo root) travels with the
# repo itself and needs no install.
#
# Usage: ./config/claude/install.sh
set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DST="${HOME}/.claude"
mkdir -p "${DST}"

# Directory trees: replace wholesale so removals propagate too.
for d in rules agents commands hooks; do
  rm -rf "${DST}/${d}"
  cp -r "${SRC}/${d}" "${DST}/${d}"
done
chmod +x "${DST}/hooks/"*.sh 2>/dev/null || true

# Top-level instruction files.
cp "${SRC}/CLAUDE.md" "${SRC}/RTK.md" "${DST}/"

# settings.json: back up whatever is there before overwriting.
if [ -f "${DST}/settings.json" ]; then
  cp "${DST}/settings.json" "${DST}/settings.json.bak.pre-install"
fi
cp "${SRC}/settings.json" "${DST}/settings.json"

echo "Installed Claude config to ${DST} (previous settings.json saved as settings.json.bak.pre-install)."
echo "Machine-specific bits to check by hand:"
echo "  - statusLine points at ~/.kickbacks/vibe-ads-statusline.mjs (remove or install it)"
echo "  - Notification hook uses powershell.exe (WSL only; harmless elsewhere)"
echo "  - rtk must be on PATH for the rtk-rewrite.sh Bash hook (or delete that hook)"
echo "  - plugins/marketplaces in settings.json install themselves on first Claude Code start"
