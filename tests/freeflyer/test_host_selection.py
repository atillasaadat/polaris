"""Install classification and the Windows render-host handoff.

Engine-free by construction: every case builds :class:`FreeFlyerInstall`
records directly instead of discovering real ones, so this runs on a machine
with no FreeFlyer, no license and no Windows — which is most CI runners. What
is under test is the routing logic that decides *where* a window opens and
*which* tree supplies the client, and that logic is exactly what a machine
without an install cannot exercise by accident.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from freeflyer import winhost
from freeflyer.locate import FreeFlyerInstall, client_source_for

WIN_ROOT = Path(
    "/mnt/c/Program Files/a.i. solutions, Inc/FreeFlyer 7.10.1.60799527 (64-Bit)"
)
LIN_ROOT = Path(
    "/home/u/freeflyer-7.10.1/usr/share/a.i. solutions, Inc/FreeFlyer 7.10.1.60799527 (64-Bit)"
)
OLD_ROOT = Path("/opt/a.i. solutions, Inc/FreeFlyer 7.9.0.123 (64-Bit)")


def _install(root: Path, platform: str, *, sdk: bool) -> FreeFlyerInstall:
    return FreeFlyerInstall(
        install_dir=root,
        platform=platform,
        runnable=False,
        licensed=True,
        sdk_dir=(root / "Runtime API") if sdk else None,
    )


def test_the_build_stamp_comes_off_the_install_directory():
    assert _install(WIN_ROOT, "windows", sdk=False).build == "7.10.1.60799527"
    assert _install(Path("/somewhere/else"), "linux", sdk=True).build == ""


def test_an_sdkless_engine_borrows_a_same_build_sibling():
    """The Windows installer makes the Runtime API SDK optional but always
    ships ffrtapi.dll, so the engine is drivable with the WSL tree's client."""
    windows = _install(WIN_ROOT, "windows", sdk=False)
    linux = _install(LIN_ROOT, "linux", sdk=True)

    assert client_source_for(windows, [windows, linux]).install_dir == LIN_ROOT
    # An install with its own SDK never borrows one.
    assert client_source_for(linux, [windows, linux]).install_dir == LIN_ROOT


def test_a_different_build_is_refused_rather_than_paired():
    """The client is generated against one engine ABI. Crossing builds is the
    drift that vendoring the client would have caused, so it must not happen
    silently — it is the whole reason the borrow is allowed at all."""
    windows = _install(WIN_ROOT, "windows", sdk=False)
    older = _install(OLD_ROOT, "linux", sdk=True)

    with pytest.raises(RuntimeError, match="7.9.0.123"):
        client_source_for(windows, [windows, older])


def test_no_sdk_anywhere_names_the_fix():
    windows = _install(WIN_ROOT, "windows", sdk=False)
    with pytest.raises(RuntimeError, match="Runtime API component"):
        client_source_for(windows, [windows])


def test_stream_paths_are_translated_and_other_arguments_are_not(monkeypatch):
    """Only path-valued flags cross the boundary rewritten; a bare number that
    happens to follow one must not be mangled."""
    monkeypatch.setattr(winhost, "to_windows_path", lambda p: rf"W:\{Path(p).name}")

    argv = ["viz", "--stream", "/tmp/run.jsonl", "--fps", "8", "--view", "orbit"]
    assert winhost._translate_args(argv, ("--stream",)) == [
        "viz",
        "--stream",
        r"W:\run.jsonl",
        "--fps",
        "8",
        "--view",
        "orbit",
    ]
    # The --flag=value spelling is the same flag.
    assert winhost._translate_args(["viz", "--stream=/tmp/a.jsonl"], ("--stream",)) == [
        "viz",
        r"--stream=W:\a.jsonl",
    ]


def test_the_child_never_relaunches_itself(monkeypatch):
    """A mis-detection in the child would fork bomb across the boundary."""
    monkeypatch.setenv("POLARIS_FF_WINHOST", "1")
    assert winhost.is_child()
    assert not winhost.should_relaunch()


def test_a_plain_linux_host_does_not_relaunch(monkeypatch):
    monkeypatch.delenv("POLARIS_FF_WINHOST", raising=False)
    monkeypatch.setattr(winhost, "running_under_wsl", lambda: False)
    assert not winhost.should_relaunch()
