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
2. **Visualization.** Any closed-loop run streams truth states as JSONL when
   `POLARIS_SIM_STREAM=<path>` is set (`sim/io/closed_loop.cpp`); `viz.py`
   renders that stream in interactive FreeFlyer windows, live (`--follow`)
   or replayed. FreeFlyer never propagates here — it is purely the display,
   so the window cannot disagree with the sim. `panel.py` makes the replay
   *seekable*: `python -m freeflyer panel` serves a browser transport control
   (play/pause, seek slider, jump-to-start/timestamp, pace) on
   `http://127.0.0.1:8765` and drives the windows from it — seeking is free
   because every stream record is a complete truth state, so "seek" is just
   choosing which record gets pushed next. The page is one self-contained
   HTML document, built so a Grafana dashboard (§21) can embed it in an
   iframe panel next to the telemetry charts; the FreeFlyer 3D windows
   themselves stay native (WSLg).

```bash
python -m freeflyer status                       # discovered installs + licenses
python -m freeflyer viz --stream run.jsonl       # replay at real time
python -m freeflyer viz --stream run.jsonl --follow   # watch a live run
python -m freeflyer panel --stream run.jsonl     # seekable replay + browser panel
uv run --frozen --group analysis pytest tests/freeflyer   # the V&V suite
```

(Every command wants `PYTHONPATH=tools` from the repo root, or `uv run` which
inherits it from `pytest.ini` for the test lane.)

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

The installer, license key, and vendor help files live in `/freeflyer/`
(gitignored, **never committed** — the RPM is 577 MB and the key is a
credential).

## Licensing (read before touching)

Node-locked, LicenseSpring-backed, CLI-managed:

```bash
ff -al XXXX-XXXX-XXXX-XXXX   # activate (online)   ff -rli  # report
ff -dal                      # deactivate — returns the seat
```

Hard-won facts about this project's single Mission-tier key:

- **Two counters, and the one that blocks us is not the one `-rli` prints.**
  The report shows `max_instances: 2`; the vendor also meters **device
  transfers**, separately, and that budget is exhausted. Activation is refused
  on the transfer counter whether or not an instance slot is free, so
  `max_instances` tells you nothing about whether a second machine can be
  licensed today.
- **The transfer allowance is spent** (activate-WSL → deactivate →
  activate-Windows burned it, 2026-08-05). The seat lives on WSL. Re-confirmed
  2026-08-12 by attempting the Windows activation **without** deactivating WSL
  first: refused with *"This license has already been transferred the maximum
  number of times. Code: 9."* while the WSL seat stayed valid — so the
  refusal is not a seat-availability problem and not a stale binding. Moving
  the seat, or lighting up a second machine, needs a transfer reset from
  fflicense@ai-solutions.com. Same-device reactivation does *not* count as a
  transfer.
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
- WSLg's accelerated GL fails the renderer probe (`ff -rr` → "Renderer:
  Unknown", zink/dri2 errors); `LIBGL_ALWAYS_SOFTWARE=1` renders fine
  ("Renderer: Software") and `engine.py` sets it automatically for
  windowed engines on Linux. This is **structural, not a tuning choice**:
  FreeFlyer renders via EGL, and Mesa's EGL path on WSLg offers only zink
  (needs a Vulkan driver Ubuntu doesn't ship for the WSL vGPU — dead end)
  or the CPU rasteriser; the hardware d3d12 driver is GLX-only. Frame cost
  scales with window area — keep the windows small, prefer `--view close`.
  **Planned follow-on:** render on the *Windows* side (native GPU) while
  the sim runs in WSL — the stream file is visible to both — once the
  vendor resets the device-transfer allowance (see Licensing above).
- The engine process (`ff --api-mode`) outlives a killed Python parent;
  `pkill -f api-mode` cleans up leaked engines (each holds one of the two
  license instances).
