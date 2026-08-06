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

_VIZ_SCRIPT = """Spacecraft s;
s.AttitudeRefFrame = "ICRF";
s.AttitudeSystem = "Quaternion";

// One whole-Earth 3D view. Deliberately minimal: every decoration (3D model
// files, viewpoints, star maps, sensor views) layers on top of this loop
// without changing its shape, and the loop is the part that has to be right.
ViewWindow orbitView({s});
orbitView.WindowTitle = "Polaris truth state";

While (1);
	ApiLabel "Frame";
	Update orbitView;
End;
"""


def replay(stream_path: Path) -> Iterator[dict]:
    """Every state currently in *stream_path*, in order."""
    with stream_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def follow(
    stream_path: Path, poll_s: float = 0.05, idle_stop_s: float = 30.0
) -> Iterator[dict]:
    """Tail *stream_path* live, yielding states as the sim appends them.

    Starts before the file exists (waits for it), survives the producer being
    slower than the display, and returns once the producer has been silent for
    *idle_stop_s* — a finished run, not an error.
    """
    deadline = time.monotonic() + idle_stop_s
    while not stream_path.exists():
        if time.monotonic() > deadline:
            return
        time.sleep(poll_s)
    with stream_path.open() as f:
        buffer = ""
        while True:
            chunk = f.readline()
            if not chunk:
                if time.monotonic() > deadline:
                    return
                time.sleep(poll_s)
                continue
            deadline = time.monotonic() + idle_stop_s
            buffer += chunk
            if not buffer.endswith("\n"):
                continue  # partial line: the producer flushes whole lines, but be safe
            line = buffer.strip()
            buffer = ""
            if line:
                yield json.loads(line)


def run_viz(
    install: FreeFlyerInstall,
    states: Iterator[dict],
    pace: float | None = 1.0,
    windowed: bool = True,
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
    """
    plan = write_mission_plan(install, _VIZ_SCRIPT, "polaris_viz")
    frames = 0
    last_t: float | None = None
    with open_engine(install, windowed=windowed) as engine:
        from aisolutions.freeflyer.runtimeapi.RuntimeApiEngine import (  # noqa: PLC0415 — vendor import after path injection
            FFTimeSpan,
        )

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

            ff_epoch_s = state["tai_ns"] / 1.0e9 - _FF_EPOCH_BASE_UNIX_TAI_S
            whole = int(ff_epoch_s)
            frac_ns = int(round((ff_epoch_s - whole) * 1.0e9))
            engine.setExpressionTimeSpan(
                "s.Epoch", FFTimeSpan.fromWholeSecondsAndNanoseconds(whole, frac_ns)
            )
            engine.setExpressionArray(
                "s.Position", [x / 1000.0 for x in state["r_eci_m"]]
            )
            engine.setExpressionArray(
                "s.Velocity", [v / 1000.0 for v in state["v_eci_m_s"]]
            )
            q0, q1, q2, q3 = state["q_body_eci"]  # JPL scalar-first
            engine.setExpressionArray(
                "s.Quaternion", [q1, q2, q3, q0]
            )  # FF scalar-last
            engine.executeUntilApiLabel("Frame")
            frames += 1
    return frames
