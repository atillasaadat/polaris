"""A progress bar for the phase where a SITL scenario is simulating.

The signal is the truth stream the sim is already writing (`POLARIS_SIM_STREAM`,
see `sim/io/README.md`): one JSON line per macro boundary, each carrying `t_s`,
led by a `meta` record carrying the run's planned `duration_s`. So progress is
exact rather than a spinner -- the sim is the only party that knows how long the
run is, and it now says so.

Two properties this has to have, and both come from the same rule: **the bar is
never allowed to be the reason a run fails or stalls.**

  - It polls a file the producer flushes per line. It never opens a pipe to the
    simulation, never blocks it, and a malformed or half-written line is skipped
    rather than raised on.
  - It draws only to a TTY. Under CI, a pipe, or a redirect it prints nothing at
    all, because a carriage-return bar in a log file is thousands of lines of
    noise around the one line that mattered.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path


def _fmt_hms(seconds: float) -> str:
    seconds = max(0.0, seconds)
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"


def _read_progress(stream: Path) -> tuple[float, float]:
    """(sim seconds written, planned duration) from *stream*; zeros if unknown.

    Reads the whole file each poll. That is O(size) per tick and deliberately
    so: the alternative is holding an open handle and tracking offsets across a
    file the producer is appending to and may have recreated, which is more
    state than a progress bar has any business owning. A long run's stream is a
    few MB and the poll is twice a second.
    """
    try:
        text = stream.read_text()
    except (OSError, UnicodeDecodeError):
        return 0.0, 0.0
    duration = 0.0
    latest = 0.0
    for line in text.splitlines():
        if not line or not line.endswith("}"):
            continue  # a half-written tail line; the next poll will have it
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("meta"):
            # Per phase: a scenario with several ClosedLoop phases emits one
            # meta each, and the last is the phase now being written.
            duration = float(rec.get("duration_s") or 0.0)
        else:
            latest = float(rec.get("t_s") or 0.0)
    return latest, duration


def watch(stream: Path, is_running, poll_s: float = 0.5, out=None) -> None:
    """Draw a bar for *stream* until ``is_running()`` returns False.

    *is_running* is a callable rather than a process handle so the caller keeps
    ownership of the subprocess -- this function must not be able to reap, kill
    or wait on it.
    """
    out = out or sys.stdout
    if not out.isatty():
        return  # see the module docstring
    started = time.monotonic()
    width = max(20, min(shutil.get_terminal_size((80, 24)).columns - 42, 48))
    last = ""
    while is_running():
        t_s, duration = _read_progress(stream)
        elapsed = time.monotonic() - started
        if duration > 0.0:
            frac = min(1.0, t_s / duration)
            filled = int(round(frac * width))
            # An ETA is only offered once there is enough of a rate to make one
            # honest; before that the number would swing by minutes per tick.
            eta = ""
            if frac > 0.02 and elapsed > 1.0:
                eta = f" eta {_fmt_hms(elapsed * (1.0 - frac) / frac)}"
            bar = "#" * filled + "-" * (width - filled)
            line = (
                f"[run] [{bar}] {frac * 100:5.1f}%  {t_s:.0f}/{duration:.0f} s sim{eta}"
            )
        else:
            # No meta yet: the run is still in configuration and table loading,
            # which on a SITL row is minutes before the first macro step. Saying
            # so beats an empty bar that looks stuck at zero.
            spin = "|/-\\"[int(elapsed * 4) % 4]
            line = f"[run] {spin} starting up (compiling config, loading tables)  {_fmt_hms(elapsed)}"
        if line != last:
            out.write("\r" + line.ljust(len(last)))
            out.flush()
            last = line
        time.sleep(poll_s)
    if last:
        out.write("\r" + " " * len(last) + "\r")
        out.flush()
