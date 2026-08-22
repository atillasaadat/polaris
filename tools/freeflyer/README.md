# `tools/freeflyer/` — FreeFlyer V&V cross-check and visualization

FreeFlyer (a.i. solutions) is Polaris's third, fully independent astrodynamics
implementation, used two ways:

1. **Cross-validation (REQ-VV-006).** `vv.py` replays the GMAT golden
   propagation cases (`tests/golden/gmat_propagation.json`) through the
   FreeFlyer Runtime API; `tests/freeflyer/` compares the sampled states.
   Three-way by construction: the C++ stack verifies against the same fixture
   in `polaris_golden_tests`, so a disagreement isolates the odd
   implementation out. Measured on 7.10.1: every case sub-metre (worst 0.27 m,
   the GEO SRP-flux convention), attitude spinner < 1e-6 deg.
2. **Visualization, rendered on the Windows GPU.** Any closed-loop run streams
   truth states as JSONL when `POLARIS_SIM_STREAM=<path>` is set
   (`sim/io/closed_loop.cpp`); `viz.py` renders that stream in interactive
   FreeFlyer windows, live (`--follow`) or replayed. FreeFlyer never
   propagates here — it is purely the display, so the window cannot disagree
   with the sim. `panel.py` makes the replay *seekable*: `python -m freeflyer
   panel` serves a browser transport control (play/pause, seek slider,
   jump-to-start/timestamp, pace) and drives the windows from it — seeking is
   free because every stream record is a complete truth state, so "seek" is
   just choosing which record gets pushed next. The page is one self-contained
   HTML document, built so a Grafana dashboard (§21) can embed it in an iframe
   panel next to the telemetry charts.

   **The sim stays in WSL; only the renderer crosses.** You still run
   `python -m freeflyer viz` from the repo root in WSL: it re-enters itself
   under Windows Python (`winhost.py`), translates the stream path to its
   `\\wsl.localhost\…` form, and drives the Windows engine, which renders on
   the GPU (`ff -rr` → `Renderer: NVIDIA`, against WSLg's `Renderer: Software`).
   There is one stream file, read by both sides, so nothing can fall out of
   date. Windowed output on Linux is **refused**, not silently
   software-rendered — see *Why Windows hosts the windows* below.

```bash
python -m freeflyer status                       # installs, licenses, who renders
python -m freeflyer viz --stream run.jsonl       # replay at real time
python -m freeflyer viz --stream run.jsonl --follow   # watch a live run
python -m freeflyer viz --stream run.jsonl --pace 10  # 10x faster than real time
python -m freeflyer panel --stream run.jsonl     # seekable replay + browser panel
uv run --frozen --group analysis pytest tests/freeflyer   # the V&V suite
```

(Every command wants `PYTHONPATH=tools` from the repo root, or `uv run` which
inherits it from `pytest.ini` for the test lane.)

> **`No module named freeflyer.__main__; 'freeflyer' is a package and cannot be
> directly executed`** means exactly one thing: `PYTHONPATH=tools` was missing.
> Without it, `freeflyer` resolves to the gitignored **vendor** directory at the
> repository root — the installer, the RPM and the license — which is a
> namespace package with no `__main__.py`. It shadows `tools/freeflyer` only
> when nothing else puts the real package on the path first. The same shadowing
> bit the Windows child, which is why `winhost.py` runs it from `tools/`.

### Producing a stream to look at

Any closed-loop run writes one. The SITL integration rows are the ready-made
scenarios — set `POLARIS_SIM_STREAM` and pick a row with `--gtest_filter`:

```bash
STREAM=/tmp/polaris/detumble.jsonl
POLARIS_SIM_STREAM=$STREAM \
  ./build-fprime-automatic-native-ut/bin/Linux/polaris_integration_tests \
  --gtest_filter='SitlAttitudeControl.DetumblesFromFiveDegreesPerSecond'
PYTHONPATH=tools python -m freeflyer viz --stream $STREAM --pace 20
```

`--gtest_list_tests` on that binary lists every row. The ones worth watching:

| scenario | `--gtest_filter` | what you see |
|---|---|---|
| Detumble | `SitlAttitudeControl.DetumblesFromFiveDegreesPerSecond` | 5 °/s tumble bled off by the rods |
| Detumble → sun point | `SitlAttitudeControl.DetumblesThenAcquiresSunPointing` | the safe-mode sequence end to end |
| Slew at the rate limit | `SitlAttitudeControl.LatentStarTrackerStaysFusedThroughASlewAtTheRateLimit` | a 30° eigenaxis slew, tracker stays fused |
| Momentum dump | `SitlAttitudeControl.DesaturationDumpsMomentumWhilePointingHolds` | pointing held while the wheels unload |
| Stuck rod | `SitlAttitudeControl.StuckOnRodIsCaughtAndTheEstimatorSurvivesIt` | the FDIR case, attitude survives |
| Every source at once | `SitlFaultMatrix.GeometryMakesEverySourceAvailableAtOnce` | sun, mag and both trackers live |
| Dark start | `SitlFaultMatrix.DarkStartSeedsFromATrackerAndSurvivesSunrise` | eclipse start, sunrise transition |
| Both trackers lost | `SitlFaultMatrix.BothTrackersLostFallsToSunMagAndClimbsBackOnReturn` | the demotion and the climb back |
| GNSS outage | `SitlOdFault.GnssOutagePastTheFineHorizonIsDegradedNotDropped` | the orbit filter coasting |
| Burn in an outage | `SitlOdBurn.BurnInsideAnOutageIsCoastedOnTheCommandedThrust` | a finite burn flown blind on thrust |
| One orbit | `SitlOdFault.OneOrbitPeriodHoldsOneSolution` | a full period, one solution |

Live instead of replayed: start the run in one shell and, in another,
`python -m freeflyer viz --stream $STREAM --follow` — it waits for the file
(saying so), then tails it. **Create the directory first** (`mkdir -p` the
parent of `$STREAM`): the sim opens the path, it does not build the tree, and
a stream that never appears looks exactly like a broken viewer — a FreeFlyer
engine running with no window, because FreeFlyer opens its view windows on the
*first* frame and no state has arrived to draw. The seekable version of any of them is `panel` in place of
`viz`; open the URL it prints in a **Windows** browser (WSL has its own
network namespace, so that URL from inside WSL will not reach it).

## Installing FreeFlyer on this machine

The vendor ships Linux FreeFlyer as an el9 (RHEL 8/9) RPM, headless engine
only (`ff`). Two supported paths:

- **RHEL-family / container:** `sudo FF_ACCEPT_SLA=true yum install
  ./freeflyer_*.el9.x86_64.rpm`.
- **Ubuntu / WSL2, no root** (what the dev machine runs):

  ```bash
  mkdir -p ~/freeflyer-7.10.1/deps && cd ~/freeflyer-7.10.1
  bsdtar -xf /path/to/freeflyer_7.10.1.*.el9.x86_64.rpm
  # el9 links sonames Ubuntu doesn't ship; stage them locally:
  cd deps
  curl -fsSLO http://ftp.debian.org/debian/pool/main/i/icu/libicu67_67.1-7_amd64.deb
  curl -fsSLO http://archive.ubuntu.com/ubuntu/pool/universe/libg/libglu/libglu1-mesa_9.0.2-1.1build1_amd64.deb
  curl -fsSLO http://archive.ubuntu.com/ubuntu/pool/main/libg/libglvnd/libopengl0_1.7.0-1build1_amd64.deb
  for d in *.deb; do bsdtar -xOf "$d" data.tar.xz | bsdtar -xf - || \
                     bsdtar -xOf "$d" data.tar.zst | zstd -d | bsdtar -xf -; done
  ```

  `locate.py` finds the extraction automatically (`~/freeflyer-*` or
  `POLARIS_FF_DIR`), and `engine.py` preloads the staged sonames — no
  `LD_LIBRARY_PATH` gymnastics needed by callers.

The installer, license key, and vendor help files live in `freeflyer/` at the
repository root (gitignored, **never committed** — the RPM is 577 MB and the
key is a credential). Note that this directory would otherwise *shadow*
`tools/freeflyer` on `sys.path`; `winhost.py` runs the Windows child from
`tools/` for exactly that reason.

## Installing FreeFlyer on Windows (the renderer)

Run the vendor installer and activate with `ff.exe -al <key>`. The Runtime API
component is optional and **not** required — the WSL tree supplies the client
(above). The visualization also needs a Windows Python on `PATH`
(`python.exe`/`py.exe`) or `POLARIS_WIN_PYTHON` pointing at one; nothing else
is installed on the Windows side, and no Polaris code is copied there — the
child imports this package straight out of the WSL checkout.

## Licensing (read before touching)

Node-locked, LicenseSpring-backed, CLI-managed:

```bash
ff -al XXXX-XXXX-XXXX-XXXX   # activate (online)   ff -rli  # report
ff -dal                      # deactivate — returns the seat
```

Hard-won facts about this project's single Mission-tier key:

- **Both machines are licensed now** (2026-08-22): WSL holds one instance and
  Windows the other, of `max_instances: 2`. That is what makes the split work
  — the V&V lane drives the Linux engine headless while the visualization
  drives the Windows one — and `python -m freeflyer status` prints which
  install fills which role.
- **Two counters, and the one that blocks you is not the one `-rli` prints.**
  The report shows `max_instances: 2`; the vendor also meters **device
  transfers**, separately. Activation is refused on the transfer counter
  whether or not an instance slot is free, so `max_instances` tells you
  nothing about whether another machine can be licensed today. The transfer
  allowance was exhausted on 2026-08-05 (activate-WSL → deactivate →
  activate-Windows) and needed a reset from fflicense@ai-solutions.com;
  same-device reactivation does *not* count as a transfer.
- **Try the additive activation, never the deactivate-first one.** A refused
  activation costs nothing and leaves the working seat untouched; `ff -dal`
  gives up a seat that the transfer counter may not let you get back. The
  2026-08-05 loss came from deactivating first.
- The Runtime API needs the **Mission** tier (we have it; expires
  2027-01-15).
- Containers are licensed via a network license server only, per vendor
  policy; runner VMs activate directly.

The CI job (`freeflyer-vv` in `ci.yml`) stays dormant until the
`FREEFLYER_CI_ENABLED` repository variable is set — do that only after the
vendor blesses ephemeral-runner activation, since a hard-killed runner leaks
the seat until reset.

## FreeFlyer quirks this package encodes (so you don't rediscover them)

- Mission-plan XML cannot be hand-minimised ("could not be converted to the
  latest version"); `plans.py` reuses the installed vendor scaffold and swaps
  only the script CDATA.
- `Spacecraft.Position/Velocity` are **ICRF, km**; quaternions are
  **vector-first, scalar-last**; `AngularVelocity` is **deg/s**; epochs are
  **TAI days since 1941-01-05 12:00**.
- RK89 defaults to **fixed 300 s steps**; condition-targeted `Step … to
  (== t)` back-solves exactly, but **hangs forever** inside a loop when a
  kinematic attitude system is active (per-interval `StepSize` + plain `Step`
  is the workaround), and inequality targets stop a whole step late.
- Reading state through per-sample `ApiLabel` stops is off-by-one; collect
  into an FF-side `Matrix` and read once at a final label (`vv.py`).
- FreeFlyer's internal failure mode is frequently a **hang, not an error** —
  timeout every engine interaction you script.
- **Why Windows hosts the windows.** WSLg's accelerated GL fails the renderer
  probe (`ff -rr` → "Renderer: Unknown", zink/dri2 errors) and only Mesa's CPU
  rasteriser works ("Renderer: Software"). This is **structural, not a tuning
  choice**: FreeFlyer renders via EGL, and Mesa's EGL path on WSLg offers only
  zink (needs a Vulkan driver Ubuntu doesn't ship for the WSL vGPU — dead end)
  or the CPU rasteriser; the hardware d3d12 driver is GLX-only. The Windows
  engine reports `Renderer: NVIDIA`, so that is where windows open;
  `engine.py` refuses windowed output on Linux rather than quietly
  software-rendering it, and `winhost.py` re-enters the command under Windows
  Python. Headless Linux work — the whole V&V lane — is untouched.
- **The Windows installer makes the Runtime API SDK optional** while always
  shipping `ffrtapi.dll`. A Windows install without it still renders: the pure
  Python client and the Mission Plan scaffold are read from the WSL tree's SDK
  (`locate.client_source_for`, `POLARIS_FF_SDK`), which is legitimate only
  because the builds match — the client is generated against one engine ABI, so
  a build mismatch is refused with both builds named.
- **What a frame costs is the round-trip, not the drawing.** Each synchronous
  `setExpression*` blocks on the engine process, and four of them per frame
  dominated everything: measured 188 ms/frame (5.3 fps) sequential against
  59 ms/frame (17.1 fps) with the same four queued asynchronously and drained
  by the single post-`execute` synchronize — a 3.2x speedup for no change in
  what is drawn, since the engine consumes queued commands in order. Two
  consequences for anyone tuning this: window count is **not** a lever any more
  (56.4 ms one window vs 58.6 ms two — the old "prefer `--view close`" advice
  was a CPU-rasteriser fact), and the stream file is nowhere near one (reading
  all 4501 states of a detumble run over the `\\wsl.localhost` share takes
  36 ms, 8 µs a state, so preloading it would save nothing). The next lever, if
  one is ever needed, is batching frames engine-side through
  `setExpressionMatrix` + `setExpressionTimeSpanArray` so one round-trip
  carries many frames; the render itself is nearly free on the GPU.
- The engine process (`ff --api-mode`) outlives a killed Python parent;
  `pkill -f api-mode` cleans up leaked engines (each holds one of the two
  license instances).
