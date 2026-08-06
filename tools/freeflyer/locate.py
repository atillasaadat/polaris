"""Find a usable FreeFlyer installation and report its license state.

Discovery order (first hit wins):

1. ``POLARIS_FF_DIR`` — explicit install directory (the one containing ``ff``
   on Linux or ``FF.exe`` on Windows).
2. The RPM's system location, ``/usr/share/a.i. solutions, Inc/FreeFlyer *``.
3. A no-root local extraction, ``~/freeflyer-*/usr/share/a.i. solutions, Inc/…``
   (the RPM unpacked with ``bsdtar``; see ``tools/freeflyer/README.md``).
4. A Windows installation visible from WSL under
   ``/mnt/c/Program Files/a.i. solutions, Inc/FreeFlyer *``.

A Windows install found from WSL is reported but marked not *runnable*: the
Runtime API client loads a native library into the current process, and a
Linux Python cannot load a Windows DLL. It is still useful — the GUI
visualization can be hosted by a Windows-side Python against it.

Non-RHEL Linux hosts (Ubuntu/WSL) lack the exact shared-library versions the
el9 build links (ICU 67, GLU, glvnd). Those are expected in a sibling ``deps``
directory next to the extraction root (``~/freeflyer-*/deps/usr/lib/…``) or in
``POLARIS_FF_DEPS``; :func:`tools.freeflyer.engine.open_engine` preloads them.
"""

from __future__ import annotations

import glob
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

#: Version-sorted glob patterns for install roots, most authoritative first.
_LINUX_GLOBS = (
    "/usr/share/a.i. solutions, Inc/FreeFlyer *",
    str(Path.home() / "freeflyer-*/usr/share/a.i. solutions, Inc/FreeFlyer *"),
)
_WINDOWS_GLOB = "/mnt/c/Program Files/a.i. solutions, Inc/FreeFlyer *"


@dataclass(frozen=True)
class FreeFlyerInstall:
    """One discovered FreeFlyer installation.

    Attributes
    ----------
    install_dir : Path
        Directory holding the engine binary and the ``Runtime API`` tree.
    platform : str
        ``"linux"`` or ``"windows"``.
    runnable : bool
        True when the Runtime API client can be loaded by *this* Python
        process (Linux install under Linux Python).
    licensed : bool
        True when ``ff -rli`` reports a valid license.
    license_info : dict
        Parsed ``-rli`` fields (``key``, ``type``, ``tier``, ``expires`` …);
        empty when unlicensed.
    deps_dir : Path | None
        Directory of locally extracted shared-library dependencies to preload
        on non-RHEL hosts, or None when the system provides them.
    """

    install_dir: Path
    platform: str
    runnable: bool
    licensed: bool
    license_info: dict = field(default_factory=dict)
    deps_dir: Path | None = None

    @property
    def engine_binary(self) -> Path:
        return self.install_dir / ("FF.exe" if self.platform == "windows" else "ff")

    @property
    def python_client_dir(self) -> Path:
        """The vendor Python Runtime API package root (contains ``aisolutions``)."""
        return self.install_dir / "Runtime API" / "python" / "src"


def _deps_dir_for(install_dir: Path) -> Path | None:
    env = os.environ.get("POLARIS_FF_DEPS")
    if env:
        return Path(env)
    # ~/freeflyer-*/usr/share/... -> ~/freeflyer-*/deps/usr/lib/x86_64-linux-gnu
    for parent in install_dir.parents:
        candidate = parent / "deps" / "usr" / "lib" / "x86_64-linux-gnu"
        if candidate.is_dir():
            return candidate
    return None


def _child_env(install: FreeFlyerInstall) -> dict:
    """Environment for running the engine binary as a subprocess."""
    env = dict(os.environ)
    if install.platform == "linux":
        parts = [str(install.install_dir)]
        if install.deps_dir is not None:
            parts.insert(0, str(install.deps_dir))
        prior = env.get("LD_LIBRARY_PATH")
        if prior:
            parts.append(prior)
        env["LD_LIBRARY_PATH"] = ":".join(parts)
    return env


def _probe_license(install: FreeFlyerInstall) -> dict:
    """Run ``ff -rli`` and parse its key/value report. {} means unlicensed."""
    try:
        out = subprocess.run(
            [str(install.engine_binary), "-rli"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(install.install_dir),
            env=_child_env(install),
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        return {}
    info: dict = {}
    for line in out.splitlines():
        if ":" not in line:
            continue
        name, _, value = line.partition(":")
        key = name.strip().lower().replace(".", "").replace(" ", "_")
        info[key] = value.strip()
    return info if "license_key" in info else {}


def _classify(install_dir: Path, platform: str) -> FreeFlyerInstall:
    runnable = platform == "linux" and os.name == "posix"
    base = FreeFlyerInstall(
        install_dir=install_dir,
        platform=platform,
        runnable=runnable,
        licensed=False,
        deps_dir=_deps_dir_for(install_dir) if platform == "linux" else None,
    )
    info = _probe_license(base)
    return FreeFlyerInstall(
        install_dir=base.install_dir,
        platform=base.platform,
        runnable=base.runnable,
        licensed=bool(info),
        license_info=info,
        deps_dir=base.deps_dir,
    )


def find_installs() -> list[FreeFlyerInstall]:
    """Every discoverable installation, discovery order, license state probed."""
    roots: list[tuple[Path, str]] = []
    env = os.environ.get("POLARIS_FF_DIR")
    if env and Path(env).is_dir():
        platform = "windows" if (Path(env) / "FF.exe").exists() else "linux"
        roots.append((Path(env), platform))
    for pattern in _LINUX_GLOBS:
        for hit in sorted(glob.glob(pattern), reverse=True):
            roots.append((Path(hit), "linux"))
    for hit in sorted(glob.glob(_WINDOWS_GLOB), reverse=True):
        roots.append((Path(hit), "windows"))

    seen: set[Path] = set()
    installs = []
    for root, platform in roots:
        if (
            root in seen
            or not (root / ("FF.exe" if platform == "windows" else "ff")).exists()
        ):
            continue
        seen.add(root)
        installs.append(_classify(root, platform))
    return installs


def find_runnable_licensed() -> FreeFlyerInstall | None:
    """The first installation this process can actually drive, or None.

    This is the gate the V&V tests skip on: no install, an unlicensed install,
    or only a Windows install visible from WSL all return None.
    """
    for install in find_installs():
        if install.runnable and install.licensed:
            return install
    return None
