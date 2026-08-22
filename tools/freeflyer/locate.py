"""Find a usable FreeFlyer installation and report its license state.

Discovery order (first hit wins):

1. ``POLARIS_FF_DIR`` — explicit install directory (the one containing ``ff``
   on Linux or ``ff.exe`` on Windows).
2. The RPM's system location, ``/usr/share/a.i. solutions, Inc/FreeFlyer *``.
3. A no-root local extraction, ``~/freeflyer-*/usr/share/a.i. solutions, Inc/…``
   (the RPM unpacked with ``bsdtar``; see ``tools/freeflyer/README.md``).
4. A Windows installation visible from WSL under
   ``/mnt/c/Program Files/a.i. solutions, Inc/FreeFlyer *``.

*Runnable* means the Runtime API client can load this install's native library
into **this** process, which is a per-platform fact: the client does
``ctypes.CDLL`` on ``libffrtapi.so`` or ``ffrtapi.dll``, so a Linux Python
drives only the Linux install and a Windows Python only the Windows one. Both
are normal here — the V&V lane runs headless under Linux, and the visualization
runs under Windows Python for the GPU (see ``winhost.py``), against the same
stream file both sides can see.

**The SDK may live in a different install than the engine.** The Windows
installer treats the Runtime API SDK — the pure-Python client, the examples and
the Mission Plan scaffold — as an optional component, while shipping
``ffrtapi.dll`` with the engine regardless. When the SDK is missing beside an
otherwise usable engine, :attr:`FreeFlyerInstall.client_source` falls back to
another discovered install **of the same build**, which is what makes a
SDK-less Windows engine drivable. The build equality is the whole safety
argument: the client is generated against a specific engine ABI, so pairing
across builds is exactly the drift the "never vendor the client" rule exists to
prevent, and it is refused rather than guessed at.

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
#: From WSL, the Windows installs are visible under the drive mount; running
#: natively on Windows they are at the same place without it.
_WINDOWS_GLOB = "/mnt/c/Program Files/a.i. solutions, Inc/FreeFlyer *"
_WINDOWS_NATIVE_GLOB = "C:/Program Files/a.i. solutions, Inc/FreeFlyer *"


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
        process — the install's platform matches the interpreter's.
    licensed : bool
        True when ``ff -rli`` reports a valid license.
    license_info : dict
        Parsed ``-rli`` fields (``key``, ``type``, ``tier``, ``expires`` …);
        empty when unlicensed.
    deps_dir : Path | None
        Directory of locally extracted shared-library dependencies to preload
        on non-RHEL hosts, or None when the system provides them.
    sdk_dir : Path | None
        This install's own ``Runtime API`` tree, or None when the component was
        not installed. See :attr:`client_source`.
    """

    install_dir: Path
    platform: str
    runnable: bool
    licensed: bool
    license_info: dict = field(default_factory=dict)
    deps_dir: Path | None = None
    sdk_dir: Path | None = None

    @property
    def engine_binary(self) -> Path:
        return self.install_dir / ("ff.exe" if self.platform == "windows" else "ff")

    @property
    def build(self) -> str:
        """The build stamp out of the install directory name, e.g. ``7.10.1.60799527``.

        Empty when the directory is not the vendor's ``FreeFlyer <build> (…)``
        layout, which makes :func:`client_source_for` refuse to pair it rather
        than pair it blindly.
        """
        name = self.install_dir.name
        marker = "FreeFlyer "
        if not name.startswith(marker):
            return ""
        return name[len(marker) :].split(" ")[0]

    @property
    def python_client_dir(self) -> Path:
        """The vendor Python Runtime API package root (contains ``aisolutions``)."""
        return (self.sdk_dir or (self.install_dir / "Runtime API")) / "python" / "src"


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


def _sdk_override() -> FreeFlyerInstall | None:
    """``POLARIS_FF_SDK`` as an SDK-providing install, or None.

    The Windows-hosted renderer's route to a client: the SDK component is
    optional in the Windows installer but always present in the WSL-side
    extraction, and the child reaches it over the ``\\wsl.localhost`` share.
    :mod:`tools.freeflyer.winhost` sets this (and lists it in ``WSLENV``, since
    nothing else crosses that boundary). Pointed at an install root; the build
    check in :func:`client_source_for` still applies.
    """
    raw = os.environ.get("POLARIS_FF_SDK")
    if not raw:
        return None
    root = Path(raw)
    sdk = root / "Runtime API"
    if not (sdk / "python" / "src").is_dir():
        return None
    return FreeFlyerInstall(
        install_dir=root,
        platform="windows" if os.name == "nt" else "linux",
        runnable=False,
        licensed=False,
        sdk_dir=sdk,
    )


def host_platform() -> str:
    """The platform name this interpreter can drive an engine on."""
    return "windows" if os.name == "nt" else "linux"


def _classify(install_dir: Path, platform: str) -> FreeFlyerInstall:
    sdk = install_dir / "Runtime API"
    base = FreeFlyerInstall(
        install_dir=install_dir,
        platform=platform,
        runnable=platform == host_platform(),
        licensed=False,
        deps_dir=_deps_dir_for(install_dir) if platform == "linux" else None,
        sdk_dir=sdk if (sdk / "python" / "src").is_dir() else None,
    )
    info = _probe_license(base)
    return FreeFlyerInstall(
        install_dir=base.install_dir,
        platform=base.platform,
        runnable=base.runnable,
        licensed=bool(info),
        license_info=info,
        deps_dir=base.deps_dir,
        sdk_dir=base.sdk_dir,
    )


def find_installs() -> list[FreeFlyerInstall]:
    """Every discoverable installation, discovery order, license state probed."""
    roots: list[tuple[Path, str]] = []
    env = os.environ.get("POLARIS_FF_DIR")
    if env and Path(env).is_dir():
        platform = "windows" if (Path(env) / "ff.exe").exists() else "linux"
        roots.append((Path(env), platform))
    if os.name == "nt":
        for hit in sorted(glob.glob(_WINDOWS_NATIVE_GLOB), reverse=True):
            roots.append((Path(hit), "windows"))
    else:
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
            or not (root / ("ff.exe" if platform == "windows" else "ff")).exists()
        ):
            continue
        seen.add(root)
        installs.append(_classify(root, platform))
    return installs


def client_source_for(
    install: FreeFlyerInstall, others: list[FreeFlyerInstall] | None = None
) -> FreeFlyerInstall:
    """The install whose Runtime API SDK should drive *install*'s engine.

    Itself when its own SDK component is present. Otherwise a same-build
    sibling that has one — the Windows installer makes the SDK optional but
    always ships ``ffrtapi.dll``, so a Windows engine is perfectly drivable
    with the client out of the Linux tree the WSL side already has.

    Raises
    ------
    RuntimeError
        When no SDK is reachable, or only one of a different build. The client
        is generated against a specific engine ABI; pairing across builds is
        the drift that vendoring the client would have caused, so it is refused
        with both builds named rather than silently attempted.
    """
    if install.sdk_dir is not None:
        return install
    build = install.build
    candidates = list(others or find_installs())
    override = _sdk_override()
    if override is not None:
        candidates.insert(0, override)
    candidates = [i for i in candidates if i.sdk_dir is not None]
    if not candidates:
        raise RuntimeError(
            f"FreeFlyer at {install.install_dir} has no Runtime API SDK, and no "
            "other installation provides one. Re-run the vendor installer with "
            "the Runtime API component selected."
        )
    for candidate in candidates:
        if build and candidate.build == build:
            return candidate
    raise RuntimeError(
        f"FreeFlyer at {install.install_dir} (build {build or 'unknown'}) has no "
        "Runtime API SDK, and the only SDK available is build "
        f"{candidates[0].build or 'unknown'} at {candidates[0].install_dir}. "
        "The client is generated against one engine ABI; install the Runtime "
        "API component for the matching build rather than crossing them."
    )


def find_runnable_licensed() -> FreeFlyerInstall | None:
    """The first installation this process can actually drive, or None.

    This is the gate the V&V tests skip on: no install, an unlicensed install,
    or only an install of the other platform all return None.
    """
    for install in find_installs():
        if install.runnable and install.licensed:
            return install
    return None


def find_render_host() -> FreeFlyerInstall | None:
    """The licensed install to *render* with, whatever platform this is.

    Windows first, and not as a preference: FreeFlyer renders through EGL, and
    Mesa's EGL path under WSLg offers only the CPU rasteriser (the hardware
    d3d12 driver is GLX-only), so a Linux-hosted window is software-rendered by
    construction. The Windows engine reports ``Renderer: NVIDIA`` against
    WSLg's ``Software``. Returns None when nothing licensed is installed.
    """
    installs = [i for i in find_installs() if i.licensed]
    for install in installs:
        if install.platform == "windows":
            return install
    return installs[0] if installs else None
