"""Cross-validate Polaris/GMAT golden propagation cases in FreeFlyer.

The golden fixture ``tests/golden/gmat_propagation.json`` is the project's
propagation truth set: GMAT-generated sample ephemerides that the C++ stack
already reproduces inside per-case tolerances (``polaris_golden_tests``). This
module runs the *same* cases through FreeFlyer — a third, fully independent
implementation — and returns FreeFlyer's state at every golden sample time.

What is compared, and against what tolerance, is the test's business
(``tests/freeflyer/``); this module only knows how to express a golden case as
a FreeFlyer script and harvest samples over the Runtime API.

Frame and unit notes
--------------------
* FreeFlyer ``Spacecraft.Position/Velocity`` are **ICRF, km**; the golden
  samples are GMAT ``EarthMJ2000Eq``, metres. The two frames differ by the
  constant frame bias (~23 mas, ≲1 m at LEO radius), which is inside every
  FreeFlyer-vs-GMAT tolerance band used here; the numeric state vectors are
  therefore compared directly, and the bias is part of each case's stated
  tolerance rationale.
* Golden epochs are UTC ISO-8601; FreeFlyer epochs are set with
  ``ParseCalendarDate`` on a UTC calendar string.
* The two-body case pins ``Earth.Mu`` to ``wgs84::kGM`` exactly as the GMAT
  script did, so nothing but integrator truncation and the frame bias
  separates the implementations there.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .engine import open_engine
from .locate import FreeFlyerInstall
from .plans import write_mission_plan

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_FIXTURE = REPO_ROOT / "tests" / "golden" / "gmat_propagation.json"

#: wgs84::kGM (lib/constants/constants.hpp), in FreeFlyer's km^3/s^2.
_WGS84_GM_KM3_S2 = 3.986004418e5

#: ForceModel per-planet array index for Earth.
_EARTH = 2
#: PlanetFieldType codes (ForceModel.PlanetFieldType docs).
_POINT_MASS = 0
_ZONAL_AND_TESSERAL = 2


def load_cases() -> dict[str, dict]:
    """Golden propagation cases by name."""
    fixture = json.loads(GOLDEN_FIXTURE.read_text())
    return {case["name"]: case for case in fixture["cases"]}


def _ff_epoch(epoch_utc: str) -> str:
    """ISO-8601 UTC -> FreeFlyer calendar-date string ("Jan 01 2026 00:00:00.000")."""
    stamp = datetime.fromisoformat(epoch_utc.replace("Z", "+00:00")).astimezone(
        timezone.utc
    )
    return stamp.strftime("%b %d %Y %H:%M:%S.") + f"{stamp.microsecond // 1000:03d}"


def _fmt(values: list[float]) -> str:
    return ", ".join(repr(float(v)) for v in values)


def build_case_script(case: dict) -> str:
    """FreeFlyer script that replays *case* and pauses at every sample time."""
    env = case["environment"]
    init = case["initial_state"]
    times = [s["t_s"] for s in case["samples"]]
    degree = int(env["gravity_degree"])
    # The golden convention: gravity_order == -1 means "square field"
    # (order = degree); GMAT scripts were generated that way.
    order = degree if int(env["gravity_order"]) < 0 else int(env["gravity_order"])

    lines = [
        "ForceModel fm;",
        "RK89 integ(fm);",
        "Spacecraft s(integ);",
        f's.Epoch = "{_ff_epoch(case["epoch_utc"])}".ParseCalendarDate();',
        f"s.Position = {{{_fmt([p / 1000.0 for p in init['position_m']])}}};",
        f"s.Velocity = {{{_fmt([v / 1000.0 for v in init['velocity_m_s']])}}};",
        # FreeFlyer masses are VehicleDryMass + tank contents; the golden
        # vehicles carry no propellant, so the whole mass is dry.
        f"s.VehicleDryMass = {case['spacecraft']['mass_kg']};",
        # Earth gravity: point mass for degree 0, square harmonic field otherwise.
        "fm.Earth = 1;",
    ]
    if degree <= 0:
        lines += [
            f"fm.PlanetFieldType[{_EARTH}] = {_POINT_MASS};",
            f"Earth.Mu = {_WGS84_GM_KM3_S2!r};",
        ]
    else:
        lines += [
            f"fm.PlanetFieldType[{_EARTH}] = {_ZONAL_AND_TESSERAL};",
            f"fm.PlanetFieldDegree[{_EARTH}] = {degree};",
            f"fm.PlanetFieldOrder[{_EARTH}] = {order};",
        ]
    third = {b.lower() for b in env["third_bodies"]}
    lines += [
        f"fm.Moon = {1 if 'moon' in third else 0};",
        f"fm.Sun = {1 if 'sun' in third else 0};",
        f"fm.Drag = {1 if env['drag_enabled'] else 0};",
        f"fm.SRP = {1 if env['srp_enabled'] else 0};",
    ]
    if env["srp_enabled"]:
        # The values the GMAT golden scripts flew (tools/gmat/scripts/prop_geo
        # .script: SRPArea 0.06 m^2, Cr 1.3 — the reference vehicle's own
        # srp_area_m2). FreeFlyer's solar flux is fixed at 1358 W/m^2 against
        # GMAT's scripted 1361; that 0.2 % is part of the case tolerance, not
        # something this script can configure away.
        lines += ["s.SRPArea = 0.06;", "s.Cr = 1.3;", "fm.SRPForceGeometry = 0;"]
    if _has_attitude(case):
        rate_deg = [r * 180.0 / 3.141592653589793 for r in init["body_rate_rad_s"]]
        q = init["attitude_quaternion"]  # golden: JPL scalar-first [q0, q1, q2, q3]
        lines += [
            # FreeFlyer's kinematic quaternion attitude system with a constant
            # AngularVelocity is the same model the GMAT golden flew (GMAT
            # "Spinner": constant inertial rotation) — the rate vector *is* the
            # rotation axis, so it is constant in both frames.
            's.AttitudeRefFrame = "ICRF";',
            's.AttitudeSystem = "Quaternion";',
            # FreeFlyer quaternions are vector-first, scalar-LAST.
            f"s.Quaternion = {{{_fmt([q[1], q[2], q[3], q[0]])}}};",
            f"s.AngularVelocity = {{{_fmt(rate_deg)}}};",  # deg/s
        ]
    # One Step statement per sample, deliberately unrolled. The natural For
    # loop over a sample-time array hangs this FreeFlyer build whenever a
    # kinematic attitude system is active (loop + condition-targeted Step +
    # attitude never converges, however the target is formed), while the same
    # Steps as sequential statements land exactly — and ">=" is no
    # alternative, since it stops a full integrator step (~300 s) past the
    # target. The scripts are generated, so unrolling costs nothing.
    # Samples are accumulated FreeFlyer-side inside the (single-statement)
    # sampling loop and read back in one shot at the final ApiLabel — the one
    # shape this FreeFlyer build executes correctly. Two failure modes ruled
    # the alternatives out, both verified empirically:
    #  * Per-sample ApiLabel handshakes (stop, read state over the API,
    #    resume) return off-by-one states once more than one label name or
    #    repeated stops are involved.
    #  * The same Steps written as sequential unrolled statements complete
    #    without ever advancing the spacecraft (every recorded time is 0).
    #    Only the For-loop form actually propagates between samples.
    # A zero-time first sample is benign here: the loop's Step targets an
    # equality edge that is already true, records the initial state, and the
    # engine moves to the next iteration.
    attitude = _has_attitude(case)
    cols = 11 if attitude else 7
    if attitude:
        # A kinematic attitude system breaks the condition-targeted Step (the
        # loop below never converges with one active), so the attitude case
        # advances by explicit per-interval fixed steps instead: exact for the
        # kinematic attitude under any step size, and far inside the carrier
        # orbit's tolerance for RK89 at these ≤60 s intervals.
        deltas = [times[0]] + [b - a for a, b in zip(times, times[1:])]
        step = (
            "\tIf (sampleDeltas[i] > 0.0);\n"
            "\t\tinteg.StepSize = TimeSpan.FromSeconds(sampleDeltas[i]);\n"
            "\t\tStep s;\n"
            "\tEnd;"
        )
        schedule = f"Array sampleDeltas = {{{_fmt(deltas)}}};"
        count = "sampleDeltas"
    else:
        step = "\tStep s to (s.ElapsedTime == TimeSpan.FromSeconds(sampleTimes[i]));"
        schedule = f"Array sampleTimes = {{{_fmt(times)}}};"
        count = "sampleTimes"
    lines += [
        schedule,
        f"Matrix out({len(times)}, {cols});",
        "Variable i;",
        f"For i = 0 to {count}.Dimension - 1;",
        step,
        "\tout[i, 0] = s.ElapsedTime.ToSeconds();",
        "\tout[i, 1] = s.Position[0];",
        "\tout[i, 2] = s.Position[1];",
        "\tout[i, 3] = s.Position[2];",
        "\tout[i, 4] = s.Velocity[0];",
        "\tout[i, 5] = s.Velocity[1];",
        "\tout[i, 6] = s.Velocity[2];",
    ]
    if attitude:
        lines += [f"\tout[i, {7 + j}] = s.Quaternion[{j}];" for j in range(4)]
    lines += [
        "End;",
        'ApiLabel "Done";',
    ]
    return "\n".join(lines) + "\n"


def _has_attitude(case: dict) -> bool:
    return "attitude_quaternion" in case["samples"][0]


def run_case(install: FreeFlyerInstall, case: dict) -> list[dict]:
    """Propagate *case* in FreeFlyer; return per-sample states in metres.

    Returns
    -------
    list of dict
        One entry per golden sample: ``t_s``, ``position_m``, ``velocity_m_s``
        (FreeFlyer ICRF state, converted from km).
    """
    plan = write_mission_plan(install, build_case_script(case), f"vv_{case['name']}")
    attitude = _has_attitude(case)
    with open_engine(install) as engine:
        engine.loadMissionPlanFromFile(str(plan))
        engine.prepareMissionPlan()
        engine.executeUntilApiLabel("Done")
        rows = engine.getExpressionMatrix("out")
        engine.executeRemainingStatements()

    out: list[dict] = []
    for row in rows:
        entry = {
            "t_s": row[0],
            "position_m": [p * 1000.0 for p in row[1:4]],
            "velocity_m_s": [v * 1000.0 for v in row[4:7]],
        }
        if attitude:
            # FreeFlyer quaternions are vector-first, scalar-last -> JPL scalar-first.
            entry["attitude_quaternion"] = [row[10], row[7], row[8], row[9]]
        out.append(entry)
    return out
