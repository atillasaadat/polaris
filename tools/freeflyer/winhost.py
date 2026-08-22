"""Re-enter a visualization command under Windows Python, for the GPU.

Why this module exists
----------------------
The Runtime API client loads the engine's native library into the *calling*
process (``ctypes.CDLL`` on ``libffrtapi.so`` / ``ffrtapi.dll``), so the
process that renders must be the same platform as the engine that draws. There
is no remote-engine mode to lean on: driving the Windows engine means running
this package under Windows Python.

That is worth the trouble because the Linux window is software-rendered by
construction, not by misconfiguration — FreeFlyer draws through EGL, and Mesa's
EGL path under WSLg offers only zink (no Vulkan driver for the WSL vGPU) or the
CPU rasteriser, while the hardware d3d12 driver is GLX-only. Measured on this
machine: ``ff -rr`` reports ``Renderer: Software`` on WSL and
``Renderer: NVIDIA`` on Windows.

What crosses the boundary
-------------------------
Only paths and arguments. The sim keeps running in WSL and keeps writing its
JSONL stream where it always did; Windows reads that same file through the
``\\\\wsl.localhost\\<distro>\\…`` share, so there is one stream file and no copy
to fall out of date. The FreeFlyer side never propagates anything either way —
it is purely a display — so hosting it elsewhere cannot change what is shown.

Every path handed to the child is translated with ``wslpath -w``: the stream
file, and ``tools/``, which the child runs *in* rather than receiving as
``PYTHONPATH``. Two reasons, both learned the hard way. WSL passes only
``WSLENV``-listed variables across to a Windows process, so a ``PYTHONPATH``
exported here simply does not arrive; and the repository root carries a
``freeflyer/`` vendor directory (the installer and license, gitignored) that
would otherwise shadow ``tools/freeflyer`` on the child's ``sys.path`` — which
is exactly what it did, resolving the package to a folder of RPMs. Running from
``tools/`` puts the right package first and the vendor directory nowhere.

The child is told it is the child (``POLARIS_FF_WINHOST=1``) so a mis-detection
cannot recurse.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

#: Set in the child's environment; its presence means "do not re-enter".
_MARKER = "POLARIS_FF_WINHOST"

#: Interpreters to try, in order, when POLARIS_WIN_PYTHON is not set. ``py`` is
#: the Windows launcher and resolves whatever version is registered.
_WIN_PYTHONS = ("python.exe", "py.exe")


def running_under_wsl() -> bool:
    """True on a WSL Linux host — where a Windows interpreter is reachable."""
    if os.name != "posix":
        return False
    if os.environ.get("WSL_DISTRO_NAME"):
        return True
    return Path("/proc/sys/fs/binfmt_misc/WSLInterop").exists()


def is_child() -> bool:
    """True when this process was launched by :func:`relaunch`."""
    return bool(os.environ.get(_MARKER))


def to_windows_path(path: str | Path) -> str:
    """Translate a WSL path to the Windows form Windows FreeFlyer can open.

    ``wslpath`` is the authority here rather than string surgery: it knows the
    distro name in the ``\\\\wsl.localhost`` share and the drive mounts, and both
    change per machine.
    """
    out = subprocess.run(
        ["wslpath", "-w", str(path)], capture_output=True, text=True, timeout=30
    )
    if out.returncode != 0:
        raise RuntimeError(f"wslpath failed for {path}: {out.stderr.strip()}")
    return out.stdout.strip()


def find_windows_python() -> str | None:
    """The Windows interpreter to host the viz, or None if none is reachable."""
    explicit = os.environ.get("POLARIS_WIN_PYTHON")
    if explicit:
        return explicit
    for name in _WIN_PYTHONS:
        found = shutil.which(name)
        if found:
            return found
    return None


def _translate_args(argv: list[str], path_flags: tuple[str, ...]) -> list[str]:
    """Copy *argv*, rewriting the value of each flag in *path_flags*."""
    out = list(argv)
    for i, token in enumerate(out):
        for flag in path_flags:
            if token == flag and i + 1 < len(out):
                out[i + 1] = to_windows_path(out[i + 1])
            elif token.startswith(f"{flag}="):
                out[i] = f"{flag}={to_windows_path(token[len(flag) + 1 :])}"
    return out


def relaunch(argv: list[str], path_flags: tuple[str, ...] = ("--stream",)) -> int:
    """Run ``python -m freeflyer <argv>`` under Windows Python; return its code.

    *argv* is the subcommand and its arguments (no ``python -m freeflyer``
    prefix). Output is inherited, so the child's prints and its Ctrl-C land in
    the caller's terminal exactly as a local run would.
    """
    interpreter = find_windows_python()
    if interpreter is None:
        raise RuntimeError(
            "no Windows Python found to host the visualization. Install "
            "Python on Windows (python.org), or set POLARIS_WIN_PYTHON to its "
            "path — e.g. /mnt/c/Python311/python.exe. The sim keeps running in "
            "WSL either way; only the renderer crosses over."
        )
    tools_dir = Path(__file__).resolve().parent.parent
    env = dict(os.environ)
    env[_MARKER] = "1"
    # The Windows installer makes the Runtime API SDK optional, so tell the
    # child where the WSL-side one is; it reads it over the share. WSLENV is
    # the only channel that crosses into a Windows process at all, and the
    # value is already in Windows form, so it is passed through untranslated.
    sdk = _sdk_root_for_child()
    if sdk is not None:
        env["POLARIS_FF_SDK"] = sdk
        prior = env.get("WSLENV", "")
        env["WSLENV"] = f"{prior}:POLARIS_FF_SDK" if prior else "POLARIS_FF_SDK"
    # WSL exports its own PYTHONHOME/VIRTUAL_ENV; a Windows interpreter that
    # honours them would look for its standard library inside the Linux venv.
    for leak in ("PYTHONHOME", "VIRTUAL_ENV", "PYTHONPATH"):
        env.pop(leak, None)
    # Translate before changing directory: wslpath resolves relative paths
    # against the *current* working directory.
    # -u: the child's stdout is a pipe, so it would otherwise block-buffer and
    # the panel URL / per-frame notes would not appear until it exited.
    command = [interpreter, "-u", "-m", "freeflyer", *_translate_args(argv, path_flags)]
    print(
        f"[winhost] {Path(interpreter).name} -m freeflyer {argv[0] if argv else ''} "
        f"(rendering on Windows; sim stays in WSL)",
        flush=True,
    )
    try:
        # A Linux path: the chdir happens in this (Linux) parent before exec,
        # and WSL interop maps the working directory to its UNC form for the
        # Windows child.
        return subprocess.call(command, env=env, cwd=str(tools_dir))
    except KeyboardInterrupt:
        return 130


def _sdk_root_for_child() -> str | None:
    """Windows path of an install root carrying a Runtime API SDK, or None."""
    from .locate import find_installs  # noqa: PLC0415 — avoid import cycle at module load

    for install in find_installs():
        if install.sdk_dir is not None:
            return to_windows_path(install.install_dir)
    return None


def should_relaunch() -> bool:
    """True when this command ought to be hosted on Windows instead.

    Deliberately narrow: only a WSL host that is *not* already the child, and
    only when a Windows install is what would render. A Linux-only machine
    keeps the old behaviour and gets :mod:`engine`'s refusal, which names the
    reason, rather than a confusing relaunch failure.
    """
    if is_child() or not running_under_wsl():
        return False
    from .locate import find_render_host  # noqa: PLC0415 — avoid import cycle at module load

    host = find_render_host()
    return host is not None and host.platform == "windows"
