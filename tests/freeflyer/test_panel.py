"""The replay control panel: cursor math and the HTTP transport surface.

Runs everywhere — the panel never touches the FreeFlyer engine (that is
``run_viz_panel``'s side of the seam), so unlike the V&V suite these tests
need no license and no install.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

import pytest

from freeflyer.panel import Playback, serve

TIMES = [0.0, 10.0, 20.0, 30.0, 40.0]


def test_playing_cursor_advances_by_paced_wall_time():
    pb = Playback(TIMES, pace=10.0)
    assert pb.tick(0.0) == 0  # first tick renders the initial frame
    assert pb.tick(1.0) == 1  # +10 s sim → second sample
    assert pb.tick(0.0) is None  # same frame: no engine work
    assert pb.tick(1.05) == 2


def test_pause_seek_and_park_at_end():
    pb = Playback(TIMES)
    pb.tick(0.0)
    pb.command({"action": "pause"})
    assert pb.tick(100.0) is None  # paused: time does not move
    pb.command({"action": "seek", "t": 25.0})
    assert pb.tick(0.0) == 2  # seek renders once, floor sample
    pb.command({"action": "seek", "t": 1e9})
    assert pb.tick(0.0) == 4  # clamped to the last frame
    pb.command({"action": "play"})  # play at the end restarts
    assert pb.snapshot()["t"] == 0.0
    pb.command({"action": "start"})
    assert pb.tick(0.0) == 0
    pb.tick(1e9)  # run off the end while playing
    assert pb.snapshot()["playing"] is False  # parked, windows stay alive


def test_junk_commands_are_refused():
    pb = Playback(TIMES)
    with pytest.raises(ValueError):
        pb.command({"action": "explode"})
    with pytest.raises(ValueError):
        pb.command({"action": "pace", "pace": 1e9})
    with pytest.raises(KeyError):
        pb.command({"action": "seek"})
    with pytest.raises(ValueError):
        Playback([])


def test_http_roundtrip():
    pb = Playback(TIMES)
    server = serve(pb, port=0)
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        page = urllib.request.urlopen(f"{base}/").read()
        assert b"Polaris" in page
        req = urllib.request.Request(
            f"{base}/cmd", data=json.dumps({"action": "seek", "t": 30.0}).encode()
        )
        assert urllib.request.urlopen(req).status == 200
        state = json.loads(urllib.request.urlopen(f"{base}/state").read())
        assert state["t"] == 30.0 and state["n"] == 5
        bad = urllib.request.Request(f"{base}/cmd", data=b'{"action":"explode"}')
        with pytest.raises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(bad)
        assert err.value.code == 400
    finally:
        server.shutdown()
