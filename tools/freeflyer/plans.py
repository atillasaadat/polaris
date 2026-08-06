"""Generate FreeFlyer Mission Plans from FreeFlyer-script bodies.

A ``.MissionPlan`` is an XML project file whose executable content is one
FreeFlyer-script block. Everything the V&V and visualization sides need can be
expressed as script — spacecraft, force models, ``ApiLabel`` handshake points —
so this module owns exactly one concern: wrapping a script body in a project
file the loader accepts.

The wrapper is not authored here. The 7.10 loader rejects hand-minimised
project XML ("could not be converted to the latest version"), so the wrapper
is taken *from the installation itself* — the vendor's ``PropagateState``
example — and only its script blocks are replaced. That keeps the scaffold
byte-compatible with whatever FreeFlyer version is installed and keeps vendor
content out of this repository.
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

from .locate import FreeFlyerInstall

_TEMPLATE_RELPATH = (
    Path("Runtime API") / "examples_common" / "PropagateState.MissionPlan"
)
_CDATA = re.compile(r"<!\[CDATA\[.*?\]\]>", re.DOTALL)


def write_mission_plan(
    install: FreeFlyerInstall, script: str, title: str, directory: Path | None = None
) -> Path:
    """Write *script* as a Mission Plan file and return its path.

    Parameters
    ----------
    install : FreeFlyerInstall
        The installation whose project-file scaffold to reuse.
    script : str
        FreeFlyer-script body. Must not contain ``]]>``.
    title : str
        File stem for the generated plan.
    directory : Path, optional
        Target directory; a fresh temporary directory when omitted. Callers
        that pass one own its cleanup.
    """
    if "]]>" in script:
        raise ValueError("FreeFlyer script must not contain a CDATA terminator")
    template_path = install.install_dir / _TEMPLATE_RELPATH
    template = template_path.read_text(encoding="utf-8")
    # The scaffold carries the script twice (FreeForm block + ProjectScript);
    # both must agree or the loader runs the stale one.
    body, n = _CDATA.subn(lambda _: f"<![CDATA[{script}]]>", template)
    if n < 2:
        raise RuntimeError(
            f"unexpected scaffold layout in {template_path} ({n} script blocks)"
        )
    if directory is None:
        directory = Path(tempfile.mkdtemp(prefix="polaris-ff-"))
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{title}.MissionPlan"
    path.write_text(body, encoding="utf-8")
    return path
