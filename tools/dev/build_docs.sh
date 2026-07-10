#!/usr/bin/env bash
# Build the unified Polaris docs site (design doc §21.2).
# Doc build is a CI gate: -W turns warnings (broken docstrings, missing
# citations, uncovered baselined requirements) into errors.
#
# Usage:
#   tools/dev/build_docs.sh           # full build into docs/_build/html
#   tools/dev/build_docs.sh -E        # pass extra args to sphinx-build
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DOCS="$HERE/docs"
GEN="$DOCS/_generated"

# Ensure sphinx-needs external-needs JSON exists even before any tests run.
mkdir -p "$GEN"
for f in verif_pytest verif_gtest; do
  if [ ! -f "$GEN/$f.json" ]; then
    printf '{"current_version":"1.0","versions":{"1.0":{"needs":{}}}}\n' > "$GEN/$f.json"
  fi
done

# C++ API XML for Breathe (no-op until lib/ has sources).
if command -v doxygen >/dev/null 2>&1; then
  ( cd "$DOCS" && doxygen Doxyfile >/dev/null )
else
  echo "warning: doxygen not found; C++ API will be absent from this build" >&2
fi

sphinx-build -W --keep-going -b html "$DOCS" "$DOCS/_build/html" "$@"
echo "Docs built: $DOCS/_build/html/index.html"
