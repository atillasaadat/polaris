"""Engine lifecycle: load the vendor Runtime API client and open engines.

The vendor client (``aisolutions.freeflyer.runtimeapi``) ships inside every
FreeFlyer installation and is imported from there — never vendored — so the
client and engine versions cannot drift apart.

Two host quirks this module owns so nothing else has to:

* **Dependency preloading.** The el9 build links ICU 67 / GLU / glvnd, which
  Ubuntu-family hosts do not provide at those sonames. ``LD_LIBRARY_PATH``
  cannot help a process that has already started (glibc reads it once at
  startup), so the locally extracted libraries are preloaded here with
  ``ctypes.CDLL(..., RTLD_GLOBAL)`` by absolute path before the Runtime API
  library loads. The spawned ``ff`` child *does* honour a fresh environment,
  so the same directories are also exported for it.
* **sys.path injection.** The client is put on ``sys.path`` exactly once, from
  the located install.
"""

from __future__ import annotations

import ctypes
import os
import sys
from contextlib import contextmanager
from pathlib import Path

from .locate import FreeFlyerInstall

#: Sonames the el9 engine needs that Ubuntu-family hosts do not ship, in
#: dependency order — each entry may be a dependency of the ones after it
#: (libGLU needs libOpenGL), and dlopen of an absolute path still resolves
#: *its* dependencies by soname search, which only finds already-loaded ones.
_PRELOAD_SONAMES = (
    "libicudata.so.67",
    "libicuuc.so.67",
    "libicui18n.so.67",
    "libOpenGL.so.0",
    "libGLU.so.1",
)

_prepared: set[Path] = set()


def _prepare(install: FreeFlyerInstall) -> None:
    """Preload missing-soname deps and expose the vendor client. Idempotent."""
    if install.install_dir in _prepared:
        return
    if not install.runnable:
        raise RuntimeError(
            f"FreeFlyer at {install.install_dir} is not runnable from this "
            "process (Windows install under a Linux Python?)"
        )
    if install.deps_dir is not None:
        for soname in _PRELOAD_SONAMES:
            lib = install.deps_dir / soname
            if lib.exists():
                ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)
        # The engine subprocess resolves the same sonames through its own
        # environment; children inherit os.environ at spawn time.
        parts = [str(install.deps_dir), str(install.install_dir)]
        prior = os.environ.get("LD_LIBRARY_PATH")
        if prior:
            parts.append(prior)
        os.environ["LD_LIBRARY_PATH"] = ":".join(parts)
    client = str(install.python_client_dir)
    if client not in sys.path:
        sys.path.insert(0, client)
    _prepared.add(install.install_dir)


def runtime_api(install: FreeFlyerInstall):
    """Import and return the vendor ``runtimeapi`` package for *install*."""
    _prepare(install)
    import aisolutions.freeflyer.runtimeapi as rtapi  # noqa: PLC0415 — vendor import after path injection

    return rtapi


@contextmanager
def open_engine(install: FreeFlyerInstall, windowed: bool = False):
    """Yield a live ``RuntimeApiEngine``, guaranteed closed on exit.

    Parameters
    ----------
    install : FreeFlyerInstall
        A runnable, licensed installation from :mod:`tools.freeflyer.locate`.
    windowed : bool
        Generate interactive output windows (visualization host) instead of
        running fully headless (V&V). Headless is the default: a licensed
        engine at ``Max. Instances: 2`` is a scarce resource and tests must
        never contend for a display.
    """
    _prepare(install)
    from aisolutions.freeflyer.runtimeapi.ConsoleOutputProcessingMethod import (
        ConsoleOutputProcessingMethod,
    )
    from aisolutions.freeflyer.runtimeapi.RuntimeApiEngine import RuntimeApiEngine
    from aisolutions.freeflyer.runtimeapi.WindowedOutputMode import WindowedOutputMode

    mode = WindowedOutputMode.GenerateOutputWindows if windowed else None
    with RuntimeApiEngine(
        str(install.install_dir),
        consoleOutputProcessingMethod=ConsoleOutputProcessingMethod.RedirectToRuntimeApi,
        windowedOutputMode=mode,
    ) as engine:
        yield engine
