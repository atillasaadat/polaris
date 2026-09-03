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

import itertools
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


_VIZ_TARGET = """
// A tracked secondary object (design doc 8.4.1). It is a bare Spacecraft whose
// position this client writes every frame from the stream: FreeFlyer never
// propagates it, exactly as it never propagates Polaris. The position on the
// wire came from the *sim truth side* (sim/world/tracked_object), propagated
// with the scenario's own spherical-harmonic field for a state vector and with
// SGP4 + TEME->ECI for a TLE -- deliberately not the onboard two-body + J2 the
// camera is being aimed by. That is what makes the picture evidence: if the
// onboard propagation were wrong, the target would sit off the boresight here
// rather than being dragged along with it.
Spacecraft {name};
{name}.Color = ColorTools.Orange;

// Line of sight from the vehicle to the target, so "is it tracking?" is one
// glance in the orbit view rather than an inference from two dots.
Vector {name}Vec;
{name}Vec.BuildVector(9, Polaris, {name});
{name}Vec.Color = ColorTools.Orange;
"""

_VIZ_CAMERA = """
// The payload camera, as the sim models it: boresight and field of view come
// off Vehicle::payload_sensors through the stream's meta record, so this draws
// the instrument the truth side actually flies rather than a second definition
// of it that could drift from it.
// The boresight is NOT set here. FreeFlyer's script parser rejects an array
// literal assignment to BoresightUnitVector (measured: every other statement in
// this fragment loads, that one is ErrorFailedToParseScript in both the comma
// and semicolon forms), so it is written once through the Runtime API after the
// plan loads -- which is what the vendor's own example does. Body axes: {bx},
// {by}, {bz}.
Polaris.AddSensor("{sensor}");
// FreeFlyer's sensor cone is conic and the imager's field is rectangular
// (half-angles {hx} deg by {hy} deg), so the drawn cone is the circumscribing
// one. It over-states the corners and never under-states the field, which is
// the safe direction for a "is the target inside?" glance.
Polaris.Sensors[{index}].ConeHalfAngle = {cone};
"""

_VIZ_STVIEW = """
// Through-the-king-star-tracker view. The camera window answers "is it looking
// at the right thing"; this one answers "can it know where it is looking",
// which on this vehicle is the question that actually limits pointing. The
// trackers sit 45 deg off body -Z, so aiming the payload on +Z at a target near
// the vehicle's zenith sweeps them across the Earth -- and the Earth filling
// this window is what a coarse-mode knowledge error looks like from outside.
ViewWindow stView({{Polaris}});
stView.WindowTitle = "Polaris - king star tracker POV ({sensor})";
Viewpoint stPov;
stPov.ViewpointName = "StPOV";
stPov.ViewpointType = "sensorview";
stPov.SensorView.Source = Polaris.Sensors[{index}].ObjectId;
stPov.SensorView.FieldOfView = {view_fov};
stView.AddViewpoint(stPov);
stView.ActivateViewpoint(stPov.ViewpointName);
"""

_VIZ_CAMVIEW = """
// Through-the-camera view. This is the window that answers the question the
// pointing rows exist to ask: the target should sit at the centre of the frame
// and stay there through the slew and the track.
ViewWindow camView({{Polaris{targets}}});
camView.WindowTitle = "Polaris - camera POV ({sensor})";
Viewpoint camPov;
camPov.ViewpointName = "CameraPOV";
camPov.ViewpointType = "sensorview";
camPov.SensorView.Source = Polaris.Sensors[{index}].ObjectId;
// Drawn wider than the instrument so a target *outside* the field is still
// visible approaching it. A view clipped to the field would show an empty
// frame for a near miss and an empty frame for a wild miss, which are the two
// cases most worth telling apart.
camPov.SensorView.FieldOfView = {view_fov};
camView.AddViewpoint(camPov);
camView.ActivateViewpoint(camPov.ViewpointName);
"""


def _sanitize(name: str) -> str:
    """A FreeFlyer object name from a stream name: letters, digits, underscore."""
    cleaned = "".join(c if c.isalnum() else "_" for c in name)
    return "Tgt_" + cleaned if not cleaned[:1].isalpha() else cleaned


def _build_script(view: str, meta: dict | None = None) -> str:
    """Assemble the viz script for ``view`` ("orbit", "close", or "both").

    Window count stopped being a performance lever when rendering moved to the
    Windows GPU: measured at 56.4 ms per frame for one window against 58.6 ms
    for two, because what a frame costs is the engine round-trip, not the
    rasterising. It was a real lever under WSLg's CPU rasteriser, and the
    choice survives as what it always should have been — which view you want.
    """
    meta = meta or {}
    target_names = [_sanitize(n) for n in meta.get("targets", [])]
    cameras = meta.get("cameras", [])

    parts = [_VIZ_HEADER]
    for name in target_names:
        parts.append(_VIZ_TARGET.format(name=name))
    # One camera gets a POV window: more than one would be more windows than
    # frames-per-second, and the pointing rows aim exactly one instrument.
    # Every sensor in the meta is declared, in the meta's order, because that
    # order *is* the Sensors[] index the POV windows reference. The sim emits
    # payload cameras first, then star trackers.
    for i, cam in enumerate(cameras):
        bx, by, bz = cam["boresight_body"]
        hx = cam.get("half_fov_x_deg", 0.0)
        hy = cam.get("half_fov_y_deg", 0.0)
        parts.append(
            _VIZ_CAMERA.format(
                sensor=cam["name"],
                index=i,
                bx=bx,
                by=by,
                bz=bz,
                hx=hx,
                hy=hy,
                cone=max(hx, hy) or 5.0,
            )
        )
    cam_index = next(
        (i for i, cam in enumerate(cameras) if not cam.get("star_tracker")), None
    )
    st_index = next(
        (i for i, cam in enumerate(cameras) if cam.get("star_tracker")), None
    )

    updates = []
    added = "".join(f", {n}" for n in target_names)
    if view in ("orbit", "both"):
        parts.append(_VIZ_ORBIT.replace("({Polaris})", "({Polaris" + added + "})"))
        for n in target_names:
            parts.append(
                f"orbitView.AddObject({n}Vec);\n"
                f"orbitView.SetShowName({n}.ObjectId, 1);\n"
                f"orbitView.SetTailLength({n}.ObjectId, 900);\n"
            )
        updates.append("\tUpdate orbitView;")
    if view in ("close", "both"):
        parts.append(_VIZ_CLOSE)
        updates.append("\tUpdate closeView;")
    if cam_index is not None:
        c = cameras[cam_index]
        view_fov = (
            max(c.get("half_fov_x_deg", 0.0), c.get("half_fov_y_deg", 0.0), 5.0) * 6.0
        )
        parts.append(
            _VIZ_CAMVIEW.format(
                targets=added, sensor=c["name"], index=cam_index, view_fov=view_fov
            )
        )
        updates.append("\tUpdate camView;")
    if st_index is not None:
        c = cameras[st_index]
        # Drawn far wider than the tracker's own field, because the question
        # this window answers is what is *in the way* — the Earth limb arriving
        # — not how the star field looks.
        view_fov = max(c.get("half_fov_x_deg", 0.0), 10.0) * 6.0
        parts.append(
            _VIZ_STVIEW.format(sensor=c["name"], index=st_index, view_fov=view_fov)
        )
        updates.append("\tUpdate stView;")
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
    if not stream_path.exists():
        # Say so, once. The wait is unbounded by design, and a silent one is
        # indistinguishable from a hung viewer — especially since FreeFlyer
        # opens its view windows on the *first* Update, so a viewer waiting
        # here shows a running engine and no visualization at all. That is the
        # shape of "the sim never wrote this path", which is the usual cause.
        print(
            f"[viz] waiting for {stream_path} to appear "
            "(the windows open on the first state; Ctrl-C to abandon)",
            flush=True,
        )
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
                text = buffer.strip()
                buffer = ""
                if text:
                    # The meta record is written once, ahead of every sample, and
                    # is what the scene is built from — so it is the one line
                    # coalescing must not drop. Yield it immediately; the newest
                    # *sample* still wins among samples.
                    if '"meta"' in text:
                        yield json.loads(text)
                    else:
                        pending = text  # newest complete line wins
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


def _push_camera(engine, meta: dict | None) -> None:
    """Write every sensor boresight into the loaded plan, once, and verify it.

    Separate from the scene script because the parser will not take an array
    literal for it (see _VIZ_CAMERA). Synchronous rather than queued: it happens
    once at setup, and a boresight that had not landed before the first frame
    would draw the first frame through the wrong instrument.
    """
    cameras = (meta or {}).get("cameras", [])
    if not cameras:
        return
    for i, cam in enumerate(cameras):
        engine.setExpressionArray(
            f"Polaris.Sensors[{i}].BoresightUnitVector", list(cam["boresight_body"])
        )
    # Read back, because this is the second silent-write bug in this file's
    # short life: a boresight that fails to land leaves the sensor at
    # FreeFlyer's default (+Z, the parent body's z-axis), which is *also* where
    # the payload camera points -- so the failure renders as two instruments
    # perfectly aligned, which looks like a plausible spacecraft rather than
    # like a bug. The check costs three round-trips once per run.
    for i, cam in enumerate(cameras):
        got = engine.getExpressionArray(f"Polaris.Sensors[{i}].BoresightUnitVector")
        want = list(cam["boresight_body"])
        if max(abs(a - b) for a, b in zip(got, want)) > 1e-6:
            raise RuntimeError(
                f"FreeFlyer kept boresight {got} for sensor {cam['name']} "
                f"(index {i}); {want} was written and did not land"
            )


def split_meta(states):
    """Pull a leading ``meta`` record off *states*, if the stream has one.

    Returns ``(meta_or_None, states)`` where the returned iterator still yields
    every truth sample. The meta record has to be consumed *before* the scene is
    built, because what it carries -- how many tracked objects there are, which
    body vector is the camera -- is what the scene declares. A stream written
    before this record existed simply has none, and the scene is the old one.
    """
    it = iter(states)
    for first in it:
        if first.get("meta"):
            return first, it
        return None, itertools.chain([first], it)
    return None, iter(())


def _push_targets(engine, state: dict, names: list[str]) -> None:
    """Queue the tracked objects' truth positions for this frame.

    Async like the vehicle writes, and for the same reason: the cost of a frame
    is the engine round-trip, not the drawing. A stream sample that carries
    fewer positions than the scene has objects leaves the extras where they
    were, which is the honest picture of "the truth side stopped answering for
    that one" -- better than snapping it to the origin.
    """
    positions = state.get("targets_eci_m") or []
    for i, name in enumerate(names):
        if i < len(positions):
            engine.setExpressionArrayAsync(
                f"{name}.Position", [x / 1000.0 for x in positions[i]]
            )


def _push_state(engine, state: dict) -> None:
    """Queue one stream record into the FreeFlyer spacecraft (units per module docstring).

    The four writes are **asynchronous**, and that is the difference between a
    watchable replay and a slideshow. Each synchronous ``setExpression*`` is a
    blocking round-trip to the engine process, and the round-trip — not the
    drawing — is what a frame costs: measured on the Windows/NVIDIA host,
    four sequential writes plus the render came to 188 ms per frame (5.3 fps),
    while queueing the same four and letting the single post-``execute``
    ``synchronize`` drain them came to 59 ms (17.1 fps). Ordering is what makes
    it safe: the engine consumes queued commands in submission order, so the
    render that :func:`_execute_frame` queues next necessarily sees all four
    writes, and its synchronize is the one wait that covers everything.

    Nothing here is dropped or approximated by going async — the same values
    arrive in the same order; only the number of times Python stops to wait for
    an acknowledgement changes.
    """
    from aisolutions.freeflyer.runtimeapi.RuntimeApiEngine import (  # noqa: PLC0415 — vendor import after path injection
        FFTimeSpan,
    )

    ff_epoch_s = state["tai_ns"] / 1.0e9 - _FF_EPOCH_BASE_UNIX_TAI_S
    whole = int(ff_epoch_s)
    frac_ns = int(round((ff_epoch_s - whole) * 1.0e9))
    engine.setExpressionTimeSpanAsync(
        "Polaris.Epoch",
        FFTimeSpan.fromWholeSecondsAndNanoseconds(whole, frac_ns),
    )
    engine.setExpressionArrayAsync(
        "Polaris.Position", [x / 1000.0 for x in state["r_eci_m"]]
    )
    engine.setExpressionArrayAsync(
        "Polaris.Velocity", [v / 1000.0 for v in state["v_eci_m_s"]]
    )
    q0, q1, q2, q3 = state["q_body_eci"]  # JPL scalar-first
    engine.setExpressionArrayAsync(
        "Polaris.Quaternion", [q1, q2, q3, q0]
    )  # FF scalar-last


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
            "(A window left mid-drag parks the single render thread; "
            "otherwise lower --fps, or replay once the run has finished.)"
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
        states between render slots are simply not drawn. The Windows-hosted
        engine sustains ~17 fps on this scene (measured), so the ceiling is
        there to keep a fast ``--pace`` from queueing frames faster than the
        engine retires them, not to protect a struggling rasteriser as it was
        under WSLg. 0 disables it.
    """
    meta, states = split_meta(states)
    target_names = [_sanitize(n) for n in (meta or {}).get("targets", [])]
    plan = write_mission_plan(install, _build_script(view, meta), "polaris_viz")
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
        _push_camera(engine, meta)
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
            _push_targets(engine, state, target_names)
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
    meta: dict | None = None,
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
    # The meta record is stripped by the caller here, not inside: the playback
    # cursor is built from the same list by the caller, so a record removed on
    # one side and not the other shifts every index by one.
    target_names = [_sanitize(n) for n in (meta or {}).get("targets", [])]
    plan = write_mission_plan(install, _build_script(view, meta), "polaris_viz")
    frames = 0
    min_interval = 1.0 / max_fps if max_fps > 0 else 0.05
    with open_engine(install, windowed=windowed) as engine:
        engine.loadMissionPlanFromFile(str(plan))
        engine.prepareMissionPlan()
        engine.executeUntilApiLabel("Frame")
        _push_camera(engine, meta)
        last = time.monotonic()
        while True:
            now = time.monotonic()
            idx = playback.tick(now - last)
            last = now
            if idx is None:
                time.sleep(0.05)  # paused or between frames: no engine work
                continue
            _push_state(engine, states[idx])
            _push_targets(engine, states[idx], target_names)
            _execute_frame(engine, frames)
            frames += 1
            time.sleep(min_interval)  # the renderer's sustainable rate
    return frames
