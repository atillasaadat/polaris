"""The simulate-phase progress bar (`tools/freeflyer/progress.py`).

The bar reads a file another process is appending to, which is the whole reason
it needs tests: every interesting case is a partial or surprising read, and the
one thing it must never do is raise or block the run it is reporting on.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "tools")

from freeflyer import progress  # noqa: E402


def _write(path: Path, *records: dict, tail: str = "") -> None:
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n" + tail)


def test_reads_the_planned_duration_from_the_meta_record(tmp_path: Path) -> None:
    stream = tmp_path / "s.jsonl"
    _write(
        stream,
        {"meta": 1, "duration_s": 900.0, "rate_hz": 10.0},
        {"t_s": 0.0, "tai_ns": 1},
        {"t_s": 321.5, "tai_ns": 2},
    )
    assert progress._read_progress(stream) == (321.5, 900.0)


def test_a_half_written_last_line_is_skipped_not_raised_on(tmp_path: Path) -> None:
    # The producer flushes per line, but a poll can still land mid-write. The
    # previous complete sample is the right answer; an exception is never one.
    stream = tmp_path / "s.jsonl"
    _write(
        stream,
        {"meta": 1, "duration_s": 900.0},
        {"t_s": 100.0},
        tail='{"t_s":110.0,"tai',
    )
    assert progress._read_progress(stream) == (100.0, 900.0)


def test_a_missing_or_empty_stream_reports_nothing_rather_than_failing(
    tmp_path: Path,
) -> None:
    assert progress._read_progress(tmp_path / "absent.jsonl") == (0.0, 0.0)
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    assert progress._read_progress(empty) == (0.0, 0.0)


def test_the_last_meta_wins_so_a_multi_phase_run_tracks_the_current_phase(
    tmp_path: Path,
) -> None:
    # A scenario that runs several ClosedLoop phases in one process streams them
    # as one arc with a meta record each. The bar should follow the phase being
    # written, not the first one.
    stream = tmp_path / "s.jsonl"
    _write(
        stream,
        {"meta": 1, "duration_s": 450.0},
        {"t_s": 450.0},
        {"meta": 1, "duration_s": 900.0},
        {"t_s": 12.0},
    )
    assert progress._read_progress(stream) == (12.0, 900.0)


def test_it_draws_nothing_at_all_when_the_output_is_not_a_terminal(
    tmp_path: Path,
) -> None:
    # A carriage-return bar in a CI log is thousands of lines of noise around
    # the one line that mattered, so the non-TTY path must be silent -- not
    # quieter, silent.
    stream = tmp_path / "s.jsonl"
    _write(stream, {"meta": 1, "duration_s": 10.0}, {"t_s": 1.0})

    class Refuses:
        def isatty(self) -> bool:
            return False

        def write(self, _text: str) -> int:
            raise AssertionError("the bar wrote to a non-terminal")

        def flush(self) -> None:
            pass

    progress.watch(stream, lambda: True, out=Refuses())


def test_it_stops_when_the_run_stops_and_leaves_the_line_clean(
    tmp_path: Path,
) -> None:
    stream = tmp_path / "s.jsonl"
    _write(stream, {"meta": 1, "duration_s": 100.0}, {"t_s": 50.0})

    class Tty:
        def __init__(self) -> None:
            self.written: list[str] = []

        def isatty(self) -> bool:
            return True

        def write(self, text: str) -> int:
            self.written.append(text)
            return len(text)

        def flush(self) -> None:
            pass

    out = Tty()
    ticks = iter([True, True, False])
    progress.watch(stream, lambda: next(ticks), poll_s=0.0, out=out)
    assert any("50%" in w or "50.0%" in w for w in out.written), out.written
    # The last write erases the bar, so whatever the caller prints next starts
    # on a clean line rather than inside a half-overwritten one.
    assert out.written[-1].startswith("\r") and out.written[-1].strip() == ""
