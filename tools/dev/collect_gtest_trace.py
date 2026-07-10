#!/usr/bin/env python3
"""GoogleTest -> sphinx-needs traceability collector (design doc §22.2).

C++ tests declare coverage with RecordProperty::

    TEST(MomentumManager, DesatThreshold) {
      RecordProperty("verifies",   "REQ-ACTL-014");   // ';'-separated for many
      RecordProperty("margin_pct", margin);
      EXPECT_GE(margin, 20.0);
    }

Run the tests with JSON output, then convert::

    ./test_binary --gtest_output=json:build/gtest.json
    python tools/dev/collect_gtest_trace.py build/gtest.json [more.json ...]

Writes ``docs/_generated/verif_gtest.json`` (sphinx-needs external needs). With no
input files it writes an empty-but-valid file so the docs build still resolves.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_OUT = _REPO / "docs" / "_generated" / "verif_gtest.json"


def _split_ids(value: str) -> list[str]:
    return [r.strip() for r in str(value).replace(",", ";").split(";") if r.strip()]


def main(argv: list[str]) -> int:
    needs: dict[str, dict] = {}
    for path in argv[1:]:
        data = json.loads(Path(path).read_text())
        for suite in data.get("testsuites", []):
            sname = suite.get("name", "suite")
            for tc in suite.get("testsuite", []):
                verifies = tc.get("verifies")
                if not verifies:
                    continue
                name = tc.get("name", "test")
                need_id = f"TEST_GT_{sname}_{name}"
                passed = not tc.get("failures") and tc.get("status", "RUN") == "RUN"
                needs[need_id] = {
                    "id": need_id,
                    "type": "test",
                    "title": f"{sname}.{name}",
                    "status": "passed" if passed else "failed",
                    "verifies": _split_ids(verifies),
                    "margin_achieved": str(tc.get("margin_pct", "")),
                }
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    _OUT.write_text(
        json.dumps(
            {
                "current_version": "1.0",
                "project": "Polaris gtest verification",
                "versions": {"1.0": {"needs": needs}},
            },
            indent=2,
        )
    )
    print(f"wrote {_OUT} ({len(needs)} verifying tests)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
