"""Interactive FreeFlyer visualization of Polaris truth-state streams.

The sim side is the producer: any closed-loop run writes one JSON line per
macro boundary when ``POLARIS_SIM_STREAM`` names a file (``sim/io/
closed_loop.cpp``), carrying TAI epoch, ECI position/velocity, the Body←ECI
quaternion and the body rate. This module is the consumer: it opens a
FreeFlyer engine with interactive output windows, builds a spacecraft plus a
3D orbit view and a close-up attitude view, and pushes each streamed state
into the display — live while the run executes (``follow``) or afterwards
from the same file (``replay``).

The engine does no propagation at all here: FreeFlyer is purely the display,
and every state it renders is the Polaris truth state. That is what makes the
window trustworthy — nothing on the FreeFlyer side can disagree with the sim.

Frames and conventions: stream positions are ECI [m] → FreeFlyer ICRF [km]
(the frame bias is centimetres at LEO — invisible at display scale); stream
quaternions are JPL scalar-first Body←ECI → FreeFlyer vector-first,
scalar-last against the ICRF attitude reference; stream epochs are TAI ns
since the Unix epoch → FreeFlyer's native epoch, TAI since 1941-01-05 12:00
(the GSFC MJD base) — pure TAI arithmetic on both sides, no leap seconds
anywhere.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterator

from .engine import open_engine
from .locate import FreeFlyerInstall
from .plans import write_mission_plan

#: FreeFlyer's epoch base (1941-01-05T12:00:00 TAI) relative to the Unix epoch
#: date on the TAI scale, in seconds. Both are calendar instants on one scale,
#: so this is plain calendar arithmetic: -10588.5 days.
_FF_EPOCH_BASE_UNIX_TAI_S = -10588.5 * 86400.0

_VIZ_HEADER = """Spacecraft Polaris;
Polaris.AttitudeRefFrame = "ICRF";
Polaris.AttitudeSystem = "Quaternion";

// Truth geometry vectors, drawn in every window. All three are FreeFlyer
// object-bound vectors, so they track the spacecraft state automatically as
// each frame updates it: sun = Object-to-Object (type 9) at the Sun, nadir =
// Object-to-Object at the Earth, velocity = Body Velocity (type 7).
Vector sunVec;
sunVec.BuildVector(9, Polaris, Sun);
sunVec.Color = ColorTools.Yellow;
Vector nadirVec;
nadirVec.BuildVector(9, Polaris, Earth);
nadirVec.Color = ColorTools.Cyan;
Vector velVec;
velVec.BuildVector(7, Polaris);
velVec.Color = ColorTools.Magenta;
"""

_VIZ_ORBIT = """
// Whole-Earth 3D orbit view, with the vehicle named and its body axes drawn.
ViewWindow orbitView({Polaris});
orbitView.WindowTitle = "Polaris - orbit (truth)";
orbitView.SetShowName(Polaris.ObjectId, 1);
orbitView.SetShowAxis(Polaris.ObjectId, 1);
// Bounded trajectory trail. Without a bound every Update appends history the
// software rasteriser must redraw, so frames slow steadily until one exceeds
// its budget and the window "freezes" — the long-replay failure mode. 900
// points at the 2 fps default is a comfortable visual arc.
orbitView.SetTailLength(Polaris.ObjectId, 900);
orbitView.AddObject(sunVec);
orbitView.AddObject(nadirVec);
orbitView.AddObject(velVec);
"""

_VIZ_CLOSE = """
// Body-fixed close-up: a chase camera parked a few metres off the vehicle,
// where the drawn body axes make the attitude motion readable. The viewpoint
// numbers are the ones the reference handlers flew.
ViewWindow closeView({Polaris});
closeView.WindowTitle = "Polaris - attitude close-up (truth)";
Viewpoint closeUp;
closeUp.ViewpointName = "CloseUp";
closeUp.ViewpointType = "view";
closeUp.ThreeDView.ReferenceFrame = "body fixed";
closeUp.ThreeDView.Source = Polaris.ObjectId;
closeUp.ThreeDView.Target = Polaris.ObjectId;
closeUp.ThreeDView.Declination = 105;
closeUp.ThreeDView.RightAscension = 0.6;
closeUp.ThreeDView.Radius = 0.002;
closeView.AddViewpoint(closeUp);
closeView.ActivateViewpoint(closeUp.ViewpointName);
closeView.SetShowName(Polaris.ObjectId, 1);
closeView.SetShowAxis(Polaris.ObjectId, 1);
closeView.SetTailLength(Polaris.ObjectId, 100);
closeView.AddObject(sunVec);
closeView.AddObject(nadirVec);
closeView.AddObject(velVec);
"""


def _build_script(view: str) -> str:
    """Assemble the viz script for ``view`` ("orbit", "close", or "both").

    One window is a real performance lever, not a preference: every Update is
    a full software-rasterised redraw, so two windows are twice the frame
    cost on exactly the machine that is struggling.
    """
    parts = [_VIZ_HEADER]
    updates = []
    if view in ("orbit", "both"):
        parts.append(_VIZ_ORBIT)
        updates.append("\tUpdate orbitView;")
    if view in ("close", "both"):
        parts.append(_VIZ_CLOSE)
        updates.append("\tUpdate closeView;")
    if not updates:
        raise ValueError(f"unknown view {view!r} (orbit, close, or both)")
    parts.append(
        '\nWhile (1);\n\tApiLabel "Frame";\n' + "\n".join(updates) + "\nEnd;\n"
    )
    return "".join(parts)


def _synchronize_patiently(engine, total_s: float, slice_ms: int = 2000) -> bool:
    """Wait for the engine in short slices so Ctrl-C stays responsive.

    ``synchronize`` blocks inside a C call for its whole timeout, and Python
    cannot deliver a KeyboardInterrupt mid-call — one long wait is exactly the
    hung terminal this module exists to prevent. Short slices return control
    every couple of seconds (where the interrupt fires and routes to the
    engine-kill path) while still tolerating minutes of legitimate stall —
    window interaction shares FreeFlyer's one render thread, so a user
    dragging the view pauses Update for as long as they drag.
    """
    deadline = time.monotonic() + total_s
    while time.monotonic() < deadline:
        if engine.synchronize(slice_ms):
            return True
    return False


def replay(stream_path: Path) -> Iterator[dict]:
    """Every state currently in *stream_path*, in order."""
    with stream_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def follow(
    stream_path: Path,
    poll_s: float = 0.05,
    idle_stop_s: float = 30.0,
    max_fps: float = 4.0,
) -> Iterator[dict]:
    """Tail *stream_path* live, yielding the **newest** state at most *max_fps*.

    Waits **indefinitely** for the file to appear — a SITL run spends minutes
    in configuration and atmosphere setup before its loop starts marching, and
    giving up during that window is how a viewer reports "0 frames" on a
    perfectly healthy run (Ctrl-C to abandon). Once data has flowed, a silence
    of *idle_stop_s* means the run finished, and the tail returns.

    Coalescing is what keeps the display honest: the sim emits ~10 states/s
    while a software-rendered FreeFlyer frame costs a multiple of that, so a
    viewer that renders *every* line falls steadily behind until the window
    looks frozen (and a backlogged engine's failure mode is a hang). Skipping
    to the newest complete line keeps the window showing *now*, at a frame
    rate the renderer actually sustains.
    """
    while not stream_path.exists():
        time.sleep(max(poll_s, 0.5))
    deadline = time.monotonic() + idle_stop_s
    min_interval = 1.0 / max_fps if max_fps > 0 else 0.0
    last_yield = 0.0
    with stream_path.open() as f:
        pending: str | None = None
        buffer = ""
        while True:
            chunk = f.readline()
            if chunk:
                deadline = time.monotonic() + idle_stop_s
                buffer += chunk
                if not buffer.endswith("\n"):
                    continue  # partial line: the producer flushes whole lines, but be safe
                if buffer.strip():
                    pending = buffer.strip()  # newest complete line wins
                buffer = ""
                continue  # drain everything available before rendering
            now = time.monotonic()
            if pending is not None and now - last_yield >= min_interval:
                last_yield = now
                state = json.loads(pending)
                pending = None
                yield state
                continue
            if now > deadline:
                if pending is not None:  # the run's final state still renders
                    yield json.loads(pending)
                return
            time.sleep(poll_s)


def _push_state(engine, state: dict) -> None:
    """Write one stream record into the FreeFlyer spacecraft (units per module docstring)."""
    from aisolutions.freeflyer.runtimeapi.RuntimeApiEngine import (  # noqa: PLC0415 — vendor import after path injection
        FFTimeSpan,
    )

    ff_epoch_s = state["tai_ns"] / 1.0e9 - _FF_EPOCH_BASE_UNIX_TAI_S
    whole = int(ff_epoch_s)
    frac_ns = int(round((ff_epoch_s - whole) * 1.0e9))
    engine.setExpressionTimeSpan(
        "Polaris.Epoch",
        FFTimeSpan.fromWholeSecondsAndNanoseconds(whole, frac_ns),
    )
    engine.setExpressionArray(
        "Polaris.Position", [x / 1000.0 for x in state["r_eci_m"]]
    )
    engine.setExpressionArray(
        "Polaris.Velocity", [v / 1000.0 for v in state["v_eci_m_s"]]
    )
    q0, q1, q2, q3 = state["q_body_eci"]  # JPL scalar-first
    engine.setExpressionArray("Polaris.Quaternion", [q1, q2, q3, q0])  # FF scalar-last


def _execute_frame(engine, frame_no: int, budget_s: float = 300.0) -> None:
    """Run one window Update (the ``Frame`` stop) behind sliced waits.

    The frame execution is where a wedged renderer shows up, so it runs
    asynchronously: Ctrl-C lands within a couple of seconds and routes to
    open_engine's kill path, while legitimate stalls get minutes of patience —
    the first frame carries window creation, and a user *interacting* with a
    window (rotating, zooming) parks Update for the duration of the drag,
    since FreeFlyer has one render thread. A stall past the budget is a dead
    engine, killed.
    """
    started = time.monotonic()
    engine.executeUntilApiLabelAsync("Frame")
    if not _synchronize_patiently(engine, budget_s):
        raise RuntimeError(
            f"FreeFlyer stopped responding while rendering frame {frame_no} "
            f"(waited {budget_s:.0f} s); the engine was killed. "
            "(The Linux build is officially headless — interactive "
            "windows over WSLg are best-effort. Lower --fps or use "
            "--view orbit, or replay when the run is done.)"
        )
    cost = time.monotonic() - started
    if cost > 3.0:
        print(f"[viz] frame {frame_no} took {cost:.1f} s", flush=True)


def run_viz(
    install: FreeFlyerInstall,
    states: Iterator[dict],
    pace: float | None = 1.0,
    windowed: bool = True,
    max_fps: float = 2.0,
    view: str = "both",
) -> int:
    """Render *states* in interactive FreeFlyer windows; return the frame count.

    Parameters
    ----------
    install : FreeFlyerInstall
        A runnable, licensed installation.
    states : iterator of dict
        Stream records (:func:`replay` or :func:`follow`).
    pace : float or None
        Playback rate for replay: 1.0 = real time, 10.0 = 10× faster, None =
        as fast as the display draws. Live following is naturally paced by the
        producer, so None is the right choice there.
    windowed : bool
        False renders headless (used by the smoke test; nothing to look at).
    max_fps : float
        Ceiling on engine render calls. Sim time still advances at *pace* —
        states between render slots are simply not drawn. The software
        renderer sustains a couple of frames per second; pushing it faster is
        how the window ends up frozen. 0 disables the ceiling.
    """
    plan = write_mission_plan(install, _build_script(view), "polaris_viz")
    frames = 0
    last_t: float | None = None
    min_interval = 1.0 / max_fps if max_fps > 0 else 0.0
    last_render = 0.0
    with open_engine(install, windowed=windowed) as engine:
        engine.loadMissionPlanFromFile(str(plan))
        engine.prepareMissionPlan()
        # Execute the declarations and park at the first Frame stop; only then
        # does the spacecraft exist for setExpression to write into. Each later
        # stop runs one Update and comes back around.
        engine.executeUntilApiLabel("Frame")
        for state in states:
            if pace is not None and last_t is not None:
                time.sleep(max(0.0, (state["t_s"] - last_t) / pace))
            last_t = state["t_s"]
            # Skip states arriving faster than the renderer's sustainable
            # rate; sim-time pacing above still ran, so the clock is honest.
            now = time.monotonic()
            if min_interval > 0.0 and now - last_render < min_interval:
                continue
            last_render = now
            _push_state(engine, state)
            _execute_frame(engine, frames)
            frames += 1
    return frames


def run_viz_panel(
    install: FreeFlyerInstall,
    states: list[dict],
    playback,
    windowed: bool = True,
    max_fps: float = 2.0,
    view: str = "both",
) -> int:
    """Render *states* under a :class:`freeflyer.panel.Playback` cursor.

    Unlike :func:`run_viz`'s one-way march, this loop is index-driven: each
    tick asks the cursor which record to draw, so the browser panel's
    play/pause/seek land here. Pausing renders nothing (the frame on screen is
    already right), and a seek is just a different index — every stream record
    is a complete truth state, so there is no propagator to rewind. Runs until
    Ctrl-C; parking at the final frame keeps the windows alive for seeking
    back. Returns the frame count.
    """
    plan = write_mission_plan(install, _build_script(view), "polaris_viz")
    frames = 0
    min_interval = 1.0 / max_fps if max_fps > 0 else 0.05
    with open_engine(install, windowed=windowed) as engine:
        engine.loadMissionPlanFromFile(str(plan))
        engine.prepareMissionPlan()
        engine.executeUntilApiLabel("Frame")
        last = time.monotonic()
        while True:
            now = time.monotonic()
            idx = playback.tick(now - last)
            last = now
            if idx is None:
                time.sleep(0.05)  # paused or between frames: no engine work
                continue
            _push_state(engine, states[idx])
            _execute_frame(engine, frames)
            frames += 1
            time.sleep(min_interval)  # the renderer's sustainable rate
    return frames
