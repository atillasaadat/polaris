"""Browser control panel for a seekable FreeFlyer replay.

``python -m freeflyer panel`` serves a single-page transport control —
play/pause, a seek slider, jump-to-start, jump-to-timestamp, and a playback
pace — over a stdlib HTTP server, and the render loop in
:func:`freeflyer.viz.run_viz_panel` polls the shared :class:`Playback` cursor
each tick. Seeking is trivial by construction: every stream record is a
complete truth state and FreeFlyer never propagates (``viz.py``), so "seek"
is nothing more than choosing which record gets pushed next.

The page is deliberately embeddable: it is one self-contained HTML document
with no external assets, so a Grafana dashboard can carry it in an iframe
panel next to the telemetry charts (§21). The FreeFlyer 3D windows themselves
stay native (WSLg) — the browser controls the display, it does not host it.

The server binds 127.0.0.1 by default and trusts nothing it receives: every
command is validated and clamped before it touches the cursor.
"""

from __future__ import annotations

import json
import threading
from bisect import bisect_right
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

#: Accepted playback rates. Wide enough for "crawl through a burn" and
#: "skim a day of GEO"; clamped so a typo cannot spin the render loop.
_PACE_MIN, _PACE_MAX = 0.01, 1000.0


def stream_times(states: list[dict]) -> list[float]:
    """Panel timeline for *states*: seconds since stream start, from ``tai_ns``.

    Not ``t_s``: each ClosedLoop phase counts its own sim time from zero, so a
    multi-phase arc appended into one stream (detumble, then sun acquisition —
    the whole point of the tap's append mode) restarts ``t_s`` at every phase
    boundary. The TAI epoch is continuous across the arc, and it is also the
    key a Grafana time axis joins on.
    """
    t0 = states[0]["tai_ns"]
    return [(s["tai_ns"] - t0) / 1.0e9 for s in states]


class Playback:
    """Thread-safe playback cursor over the stream's sample times.

    The render loop advances it with :meth:`tick`; the HTTP handlers mutate it
    with :meth:`command`. A tick returns the index to draw, or ``None`` when
    the frame on screen is already the right one — pausing therefore costs no
    engine work at all.
    """

    def __init__(self, times_s: list[float], pace: float = 1.0):
        if not times_s:
            raise ValueError("empty stream: nothing to replay")
        self._lock = threading.Lock()
        self._times = times_s
        self._sim_t = times_s[0]
        self._idx = 0
        self._playing = True
        self._pace = min(max(pace, _PACE_MIN), _PACE_MAX)
        self._dirty = True  # current index not yet rendered

    def _seek_locked(self, t: float) -> None:
        self._sim_t = min(max(t, self._times[0]), self._times[-1])
        self._idx = max(bisect_right(self._times, self._sim_t) - 1, 0)
        self._dirty = True

    def tick(self, wall_dt_s: float) -> int | None:
        """Advance sim time by *wall_dt_s* of wall clock; index to render or None."""
        with self._lock:
            if self._playing:
                self._sim_t = min(self._sim_t + wall_dt_s * self._pace, self._times[-1])
                if self._sim_t >= self._times[-1]:
                    self._playing = False  # park on the final frame
                idx = max(bisect_right(self._times, self._sim_t) - 1, 0)
                if idx != self._idx or self._dirty:
                    self._idx = idx
                    self._dirty = False
                    return idx
                return None
            if self._dirty:
                self._dirty = False
                return self._idx
            return None

    def command(self, cmd: dict) -> None:
        """Apply a validated transport command; raise ValueError on junk."""
        action = cmd.get("action")
        with self._lock:
            if action == "play":
                if self._sim_t >= self._times[-1]:
                    self._seek_locked(self._times[0])  # play at the end restarts
                self._playing = True
            elif action == "pause":
                self._playing = False
            elif action == "start":
                self._seek_locked(self._times[0])
            elif action == "seek":
                self._seek_locked(float(cmd["t"]))
            elif action == "pace":
                pace = float(cmd["pace"])
                if not _PACE_MIN <= pace <= _PACE_MAX:
                    raise ValueError(f"pace {pace} outside [{_PACE_MIN}, {_PACE_MAX}]")
                self._pace = pace
            else:
                raise ValueError(f"unknown action {action!r}")

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "t": self._sim_t,
                "t0": self._times[0],
                "t1": self._times[-1],
                "idx": self._idx,
                "n": len(self._times),
                "playing": self._playing,
                "pace": self._pace,
            }


_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Polaris — FreeFlyer replay</title>
<style>
  body { background:#101418; color:#d8dee6; font:14px/1.5 system-ui, sans-serif;
         max-width:640px; margin:2rem auto; padding:0 1rem; }
  h1 { font-size:1.1rem; font-weight:600; color:#8fb8ff; }
  .row { display:flex; gap:.6rem; align-items:center; margin:.8rem 0; flex-wrap:wrap; }
  button { background:#1c2733; color:#d8dee6; border:1px solid #33465c;
           border-radius:6px; padding:.45rem .9rem; font-size:1rem; cursor:pointer; }
  button:hover { background:#243447; }
  input[type=range] { flex:1; min-width:200px; }
  input[type=number], select { background:#1c2733; color:#d8dee6;
           border:1px solid #33465c; border-radius:6px; padding:.35rem .5rem; }
  #clock { font-variant-numeric:tabular-nums; color:#9fd6a5; }
  #meta { color:#7d8a99; font-size:.85rem; }
</style></head><body>
<h1>Polaris — FreeFlyer replay</h1>
<div class="row">
  <button id="start" title="jump to start">&#9198;</button>
  <button id="toggle">&#9199;</button>
  <span id="clock">t = 0.0 s</span>
</div>
<div class="row"><input type="range" id="seek" min="0" max="0" step="any"></div>
<div class="row">
  <label>jump to <input type="number" id="jump" step="any" style="width:7rem"> s</label>
  <button id="go">Go</button>
  <label>pace <select id="pace">
    <option>0.1</option><option>0.5</option><option selected>1</option>
    <option>2</option><option>5</option><option>10</option>
    <option>25</option><option>100</option>
  </select>&times;</label>
</div>
<div id="meta"></div>
<script>
const $ = id => document.getElementById(id);
let dragging = false, playing = true;
async function cmd(body) {
  await fetch('/cmd', {method:'POST', headers:{'Content-Type':'application/json'},
                       body: JSON.stringify(body)});
}
async function poll() {
  try {
    const s = await (await fetch('/state')).json();
    playing = s.playing;
    $('toggle').innerHTML = playing ? '&#9208;' : '&#9205;';
    $('clock').textContent = 't = ' + s.t.toFixed(1) + ' s';
    $('meta').textContent = 'frame ' + (s.idx + 1) + ' / ' + s.n +
      ' — stream spans ' + s.t0.toFixed(1) + ' … ' + s.t1.toFixed(1) +
      ' s — pace ' + s.pace + '\\u00d7';
    if (!dragging) { $('seek').min = s.t0; $('seek').max = s.t1; $('seek').value = s.t; }
  } catch (e) { $('meta').textContent = 'viewer not responding'; }
}
$('toggle').onclick = () => cmd({action: playing ? 'pause' : 'play'}).then(poll);
$('start').onclick = () => cmd({action:'start'}).then(poll);
$('go').onclick = () => cmd({action:'seek', t:+$('jump').value}).then(poll);
$('pace').onchange = () => cmd({action:'pace', pace:+$('pace').value});
$('seek').oninput = () => { dragging = true; };
$('seek').onchange = () => { dragging = false; cmd({action:'seek', t:+$('seek').value}); };
setInterval(poll, 500); poll();
</script></body></html>
"""


def serve(playback: Playback, host: str = "127.0.0.1", port: int = 8765):
    """Start the panel HTTP server on a daemon thread; return the server.

    ``port=0`` picks a free port (``server.server_address[1]`` reports it).
    Shut down with ``server.shutdown()`` — or just let the daemon thread die
    with the process, which is the normal Ctrl-C path.
    """

    class Handler(BaseHTTPRequestHandler):
        def _reply(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802 — BaseHTTPRequestHandler API
            if self.path == "/":
                self._reply(200, _PAGE.encode(), "text/html; charset=utf-8")
            elif self.path == "/state":
                body = json.dumps(playback.snapshot()).encode()
                self._reply(200, body, "application/json")
            else:
                self._reply(404, b"not found", "text/plain")

        def do_POST(self):  # noqa: N802 — BaseHTTPRequestHandler API
            if self.path != "/cmd":
                self._reply(404, b"not found", "text/plain")
                return
            length = int(self.headers.get("Content-Length", 0))
            try:
                playback.command(json.loads(self.rfile.read(min(length, 4096))))
            except (ValueError, KeyError, TypeError) as err:
                self._reply(400, str(err).encode(), "text/plain")
                return
            self._reply(200, b"ok", "text/plain")

        def log_message(self, *_args):  # keep the terminal for the viz itself
            pass

    server = ThreadingHTTPServer((host, port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server
