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
  whichever install actually carries the SDK — which is not always the one
  being driven, since the Windows installer makes the Runtime API component
  optional while always shipping ``ffrtapi.dll``
  (:func:`tools.freeflyer.locate.client_source_for`).

Rendering is **Windows-hosted**. FreeFlyer draws through EGL, and Mesa's EGL
path under WSLg offers only the CPU rasteriser, so a window opened by the Linux
engine is software-rendered by construction — measured as ``Renderer: Software``
against the Windows engine's ``Renderer: NVIDIA``. Windowed output is therefore
refused on Linux and :mod:`tools.freeflyer.winhost` re-enters the same command
under Windows Python. The Linux engine keeps the headless work it is good at:
the V&V cross-check, which never opens a window.
"""

from __future__ import annotations

import ctypes
import os
import sys
from contextlib import contextmanager
from pathlib import Path

from .locate import FreeFlyerInstall, client_source_for

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
    if install.platform == "linux" and install.deps_dir is not None:
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
    client = str(client_source_for(install).python_client_dir)
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

    if windowed and install.platform == "linux":
        raise RuntimeError(
            "windowed FreeFlyer output is not supported on Linux/WSL: the "
            "renderer falls back to Mesa's CPU rasteriser (measured "
            '"Renderer: Software" against the Windows engine\'s '
            '"Renderer: NVIDIA"). Run the visualization through '
            "`python -m freeflyer viz`, which hosts it on Windows "
            "automatically, or pass --headless for a smoke run."
        )
    mode = WindowedOutputMode.GenerateOutputWindows if windowed else None
    engine = RuntimeApiEngine(
        str(install.install_dir),
        consoleOutputProcessingMethod=ConsoleOutputProcessingMethod.RedirectToRuntimeApi,
        windowedOutputMode=mode,
    )
    # Manual lifecycle, not the vendor context manager: FreeFlyer's failure
    # mode is a hang, and a clean destroyEngine() against a wedged engine
    # blocks forever inside a ctypes call — which is also where a Ctrl-C
    # (KeyboardInterrupt) gets deferred indefinitely, hanging the terminal.
    # Any abnormal exit therefore force-kills the engine process instead of
    # negotiating with it; each kill costs nothing but the engine.
    try:
        yield engine
    except BaseException:
        try:
            engine.killEngine()
        except Exception:
            pass
        raise
    else:
        try:
            engine.destroyEngine()
        except Exception:
            try:
                engine.killEngine()
            except Exception:
                pass
