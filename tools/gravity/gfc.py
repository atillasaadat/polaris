"""Fetch / truncate the EGM2008 ICGEM ``.gfc`` gravity coefficient file.

Ground-side only (design doc §5.2, §3.7). ICGEM (icgem.gfz-potsdam.de) publishes
EGM2008 in the fixed `.gfc` text format. The full model runs to degree 2190
(~100 MB) — far too large to commit — so we keep a **degree-truncated** copy in
the same native `.gfc` format (a degree-window of the original, header
`max_degree` rewritten to match). Per §3.7 that trimming is a derivation *from*
the original, in the original format; the committed file is still parsed as-is by
`sim/world/egm2008.*`. CI never downloads.

`.gfc` layout: a header block ending in `end_of_head` (carrying
`earth_gravity_constant`, `radius`, `max_degree`, `norm fully_normalized`), then
`gfc  L  M  Cbar  Sbar  [sigmaC sigmaS]` data lines. Some distributions use a
Fortran `D` exponent; parsing tolerates it.
"""

from __future__ import annotations

import urllib.request

# ICGEM static-model endpoint for the full EGM2008 (tide-free). Kept as the
# provenance/source; the committed file is this, truncated to --max-degree.
DEFAULT_URL = "https://icgem.gfz-potsdam.de/getmodel/gfc/c50128797a9cb62e936337c890e4425f03f0461d7329b09a8cc8561504465340/EGM2008.gfc"


def fetch(url: str = DEFAULT_URL) -> str:
    """Download the raw `.gfc`. Ground-side only; never called from CI."""
    with urllib.request.urlopen(url, timeout=300) as resp:  # noqa: S310 (trusted ICGEM host)
        return resp.read().decode("ascii", errors="replace")


def _degree_of(line: str) -> int | None:
    """Degree L of a `gfc`/`gfct` data line, or None if it is not one."""
    parts = line.split()
    if len(parts) < 3 or parts[0] not in ("gfc", "gfct"):
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None


def truncate(text: str, max_degree: int) -> str:
    """Return native `.gfc` text keeping only degrees <= max_degree.

    The header is preserved verbatim except `max_degree`, which is rewritten to
    the truncation degree so the file is self-consistent. Coefficient lines with
    L > max_degree are dropped; everything else (order of lines, formatting,
    exponents) is untouched.
    """
    out: list[str] = []
    in_head = True
    for line in text.splitlines():
        if in_head:
            tok = line.split()[:1]
            if tok == ["max_degree"]:
                out.append(f"max_degree {max_degree}")
            else:
                out.append(line)
            # The terminator's first token is `end_of_head`; some ICGEM files pad
            # it with `=` (`end_of_head ====...`), so match the token, not the line.
            if tok == ["end_of_head"]:
                in_head = False
            continue
        deg = _degree_of(line)
        if deg is None or deg <= max_degree:
            out.append(line)
    return "\n".join(out) + "\n"


def count_coeffs(text: str) -> int:
    """Number of `gfc`/`gfct` coefficient lines — a cheap sanity metric."""
    return sum(1 for line in text.splitlines() if _degree_of(line) is not None)


def _self_check() -> None:
    """Truncate a synthetic `.gfc` and verify header + degree filtering."""
    gfc = (
        "earth_gravity_constant 3.986004415E+14\nradius 6378136.3\n"
        "max_degree 2190\nnorm fully_normalized\n"
        "end_of_head ==============================\n"  # padded terminator, as ICGEM ships it
        "gfc 0 0 1.0d0 0.0d0\ngfc 2 0 -4.84e-4 0.0\ngfc 2 2 2.4e-6 -1.4e-6\n"
        "gfc 3 0 9.5e-7 0.0\ngfc 5 0 2.0e-7 0.0\n"
    )
    trimmed = truncate(gfc, 2)
    assert "max_degree 2" in trimmed, trimmed
    assert "max_degree 2190" not in trimmed, trimmed
    assert count_coeffs(trimmed) == 3, trimmed  # degrees 0, 2, 2 kept; 3 and 5 dropped
    assert "gfc 3 0" not in trimmed and "gfc 5 0" not in trimmed, trimmed
    print("gravity.gfc self-check: ok")


if __name__ == "__main__":
    _self_check()
