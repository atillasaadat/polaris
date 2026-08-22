"""Reading the orbit-OD campaign driver's per-sample records.

The C++ driver (``tests/mc/orbit_od_mc.cpp``) writes one JSON object per line —
JSONL rather than CSV because the file interleaves two record kinds, because a
sample carries optional fields (a NIS only exists where a measurement was
judged), and because appending a line survives a campaign killed half way. The
file lands under ``build-artifacts/`` and is a derived artifact: never committed.

Two record kinds
----------------
``kind: "meta"`` opens each (run, scenario) and carries what a sample cannot: the
scenario's *intent*, written once in ``tests/mc/orbit_od_scenarios.hpp`` and
quoted by the report rather than transcribed here, plus the cadence and latency
that make a 6000-line shard legible as two minutes rather than a day.

``kind: "sample"`` is one GNC cycle. The driver emits one per cycle whether or
not a fix arrived, because the interesting rows are the ones where none did.

Why the refusal is a string
---------------------------
``refusal`` arrives as its own name (``"fix_implausible"``), not as the enum's
integer, and is **absent** on the overwhelming majority of rows where there was
nothing to refuse. Names because an integer would oblige this module to carry a
copy of ``gnc::OrbitOdRefusal``, which is the same list written twice and drifts
the first time a value is inserted; the driver calls ``gnc::refusalName``, which
lives beside the enum and is compiler-checked exhaustive. Which refusal fired is
not a detail — a fault caught by the wrong layer looks identical to one caught by
the right one unless the reason is recorded (see ``.claude/review-lessons.md``).

Units
-----
As recorded, and as the filter computes them: seconds, metres, metres per second.
NEES and NIS are dimensionless. No conversion happens here; the report is the
presentation boundary.

References
----------
Design doc §8.3 (onboard OD), §9.2 (GNSS FDIR), §13 (analysis tools), §23.2
(Monte Carlo).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = ["Campaign", "ScenarioRun", "load_campaign"]


@dataclass(frozen=True)
class ScenarioRun:
    """One scenario flown once, as columns.

    Stored column-wise rather than as a list of per-sample objects: a 7-day run
    at the 10 s cadence is 60480 samples and every statistic here is a reduction
    over one column, so the array form is both the natural shape and the one that
    does not spend a second per run in the interpreter.

    Attributes
    ----------
    run : int
        Campaign run index, which is also the dispersion substream. Run 0 is
        undispersed by construction, so it is the one reproducible trajectory
        every other run is read against.
    scenario : str
        Scenario name, from ``orbit_od_scenarios.hpp``.
    intent : str
        The scenario's one-line rationale, carried from the driver so the report
        quotes the source rather than a copy of it.
    cycle_period_s : float
        GNC cycle period this run flew [s].
    fix_latency_s : float
        Receiver fix latency armed for this run [s]. Zero on the long arcs, whose
        cadence cannot resolve the datasheet's 50 ms; see the driver's header.
    duration_s : float
        Arc length requested [s].
    coast_horizon_s, degraded_horizon_s : float
        The filter's two coast horizons as the driver flew them [s], carried in
        the meta record so this module never keeps its own copy of a number the
        filter owns. Zero on shards written before Push 74 recorded them.
    t_s : numpy.ndarray
        Sample times from epoch [s].
    pos_err_m, vel_err_mps : numpy.ndarray
        Norm of the estimate minus truth, at the sample epoch [m], [m/s].
    pos_sigma_m, vel_sigma_mps : numpy.ndarray
        ``sqrt(trace)`` of the corresponding covariance block — the filter's own
        claimed 1σ, which is what the error is read against [m], [m/s].
    err_ric_m, sigma_ric_m : numpy.ndarray
        Position error and the filter's own 1σ, resolved in the truth RIC frame,
        shaped ``(samples, 3)`` as radial / in-track / cross-track [m]. Signed,
        unlike the norms above, because the ensemble covariance built across runs
        needs the sign. Recorded per axis rather than as a trace because orbit
        uncertainty is overwhelmingly in-track: a covariance with the right size
        and the wrong split between these three is wrong in the way that matters,
        and no scalar can see it.
    nees : numpy.ndarray
        Normalised estimation error squared over the full 6-state, against truth.
        Zero where the filter is uninitialised.
    nis : numpy.ndarray
        Normalised innovation squared of the position update, ``nan`` on cycles
        where no measurement was judged.
    solution_valid, fix_valid, fix_accepted : numpy.ndarray of bool
        Whether the filter published a solution, whether the receiver delivered a
        fix, and whether that fix was folded in.
    age_s : numpy.ndarray
        Seconds since the last accepted fix — the coast clock [s].
    quality : list of str
        The filter's own coast verdict per sample: ``"fine"``, ``"degraded"`` or
        ``"none"``, from ``gnc::qualityName``. A fix refused while the solution
        is already degraded is the filter saying it is coasting, not the gate
        raising a false alarm, which is why the clean-fix rejection rate is
        measured over fine samples only. Empty string on pre-Push-74 shards.
    rejected_total : numpy.ndarray
        Cumulative count of rejected fixes.
    refusal : list of str
        Per-sample refusal name, empty string where there was none.
    regime : list of str
        Which fault, if any, was armed at the sample: ``"nominal"``, ``"outage"``,
        ``"spoof"``, ``"jam"``, ``"clock_jump"``, ``"radius_jump"``,
        ``"sigma_degrade"``.
    """

    run: int
    scenario: str
    intent: str
    cycle_period_s: float
    fix_latency_s: float
    duration_s: float
    coast_horizon_s: float
    degraded_horizon_s: float
    t_s: np.ndarray
    pos_err_m: np.ndarray
    vel_err_mps: np.ndarray
    pos_sigma_m: np.ndarray
    vel_sigma_mps: np.ndarray
    err_ric_m: np.ndarray
    sigma_ric_m: np.ndarray
    #: Semi-major-axis error (estimate − truth) [m] and the filter's SMA 1σ [m]
    #: (NASA/TP-2018-219822 §2.1.2, Eq. 2.23); NaN on shards written before
    #: Push 73 recorded them.
    sma_err_m: np.ndarray
    sma_sigma_m: np.ndarray
    nees: np.ndarray
    nis: np.ndarray
    solution_valid: np.ndarray
    fix_valid: np.ndarray
    fix_accepted: np.ndarray
    age_s: np.ndarray
    quality: list[str]
    rejected_total: np.ndarray
    refusal: list[str]
    regime: list[str]

    @property
    def samples(self) -> int:
        """Number of GNC cycles recorded."""
        return int(self.t_s.size)

    def mask(self, regime: str) -> np.ndarray:
        """Boolean mask selecting the samples belonging to one regime."""
        return np.array([r == regime for r in self.regime], dtype=bool)


@dataclass(frozen=True)
class Campaign:
    """Every run of every scenario in one campaign.

    Attributes
    ----------
    runs : tuple of ScenarioRun
        In file order, which is the driver's order: scenario-major within a run.
    paths : tuple of pathlib.Path
        The shards read, for provenance. A campaign is normally sharded across
        cores and reassembled here rather than by concatenating files.
    truncated : tuple of pathlib.Path
        Shards whose final record was a partial write — the campaign was still
        flying, or its driver was killed mid-line. The half-record is dropped;
        this names the shards it happened in so a report can say the campaign
        is incomplete instead of quietly reporting a short one as finished.
    """

    runs: tuple[ScenarioRun, ...] = ()
    paths: tuple[Path, ...] = ()
    truncated: tuple[Path, ...] = ()

    @property
    def scenarios(self) -> tuple[str, ...]:
        """Scenario names, in first-seen order and without duplicates."""
        seen: dict[str, None] = {}
        for run in self.runs:
            seen.setdefault(run.scenario, None)
        return tuple(seen)

    def of(self, scenario: str) -> tuple[ScenarioRun, ...]:
        """Every run of one scenario, in file order."""
        return tuple(r for r in self.runs if r.scenario == scenario)


@dataclass
class _Accumulator:
    """Rows for one (run, scenario), gathered before being frozen into arrays."""

    run: int
    scenario: str
    intent: str = ""
    cycle_period_s: float = 0.0
    fix_latency_s: float = 0.0
    duration_s: float = 0.0
    coast_horizon_s: float = 0.0
    degraded_horizon_s: float = 0.0
    rows: list[dict] = field(default_factory=list)

    def freeze(self) -> ScenarioRun:
        """Build the immutable, column-wise record."""
        rows = self.rows

        def col(key: str, default: float = 0.0) -> np.ndarray:
            return np.array([float(r.get(key, default)) for r in rows], dtype=float)

        def flag(key: str) -> np.ndarray:
            return np.array([bool(r.get(key, 0)) for r in rows], dtype=bool)

        # Absent rather than zero: a NIS of zero is a perfect innovation, which
        # is a real and different thing from no measurement having been judged.
        # Averaging the two together would quietly drag every NIS statistic down
        # by the fraction of cycles with no fix.
        nis = np.array([float(r["nis"]) if "nis" in r else np.nan for r in rows])

        def triple(key: str) -> np.ndarray:
            # Absent on records written before the RIC decomposition existed;
            # NaN rather than zero so a stale shard is excluded from the
            # ensemble statistics instead of contributing three fake zeros.
            out = np.full((len(rows), 3), np.nan, dtype=float)
            for i, r in enumerate(rows):
                v = r.get(key)
                if isinstance(v, list) and len(v) == 3:
                    out[i] = [float(x) for x in v]
            return out

        return ScenarioRun(
            run=self.run,
            scenario=self.scenario,
            intent=self.intent,
            cycle_period_s=self.cycle_period_s,
            fix_latency_s=self.fix_latency_s,
            duration_s=self.duration_s,
            coast_horizon_s=self.coast_horizon_s,
            degraded_horizon_s=self.degraded_horizon_s,
            t_s=col("t_s"),
            pos_err_m=col("pos_err_m"),
            vel_err_mps=col("vel_err_mps"),
            pos_sigma_m=col("pos_sigma_m"),
            vel_sigma_mps=col("vel_sigma_mps"),
            err_ric_m=triple("err_ric_m"),
            sigma_ric_m=triple("sigma_ric_m"),
            sma_err_m=col("sma_err_m", np.nan),
            sma_sigma_m=col("sma_sigma_m", np.nan),
            nees=col("nees"),
            nis=nis,
            solution_valid=flag("solution_valid"),
            fix_valid=flag("fix_valid"),
            fix_accepted=flag("fix_accepted"),
            age_s=col("age_s"),
            quality=[str(r.get("quality", "")) for r in rows],
            rejected_total=col("rejected_total"),
            refusal=[str(r.get("refusal", "")) for r in rows],
            regime=[str(r.get("regime", "nominal")) for r in rows],
        )


def load_campaign(paths: str | Path | list[str | Path]) -> Campaign:
    """Read one or more JSONL shards into a :class:`Campaign`.

    Parameters
    ----------
    paths : str or pathlib.Path or list
        A shard, a directory of ``*.jsonl`` shards, or a list of either.

    Returns
    -------
    Campaign

    Raises
    ------
    FileNotFoundError
        No readable shard was found. A campaign that produced nothing is a
        harness failure and must not be reported as a campaign with no faults.
    ValueError
        A sample appeared before the ``meta`` line that opens its group, or a
        line is not valid JSON. Both mean a truncated or interleaved file, and
        both are refused rather than silently producing a partial verdict —
        which would read as a clean campaign that happened to be short.
    """
    shards = _resolve(paths)
    if not shards:
        raise FileNotFoundError(f"no JSONL shards found under {paths!r}")

    accumulators: list[_Accumulator] = []
    index: dict[tuple[int, str], _Accumulator] = {}
    truncated: list[str] = []

    for shard in shards:
        with shard.open(encoding="utf-8") as handle:
            for number, raw in enumerate(handle, start=1):
                line = raw.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    # A line with no terminating newline is the last line in the
                    # file, and the only one a partial write can have reached.
                    # That happens for an ordinary reason — the campaign is
                    # still flying, or its process was killed mid-record — so
                    # the half-record is dropped and counted rather than
                    # failing the read. Anywhere else, a malformed line means a
                    # damaged or interleaved shard and stays fatal, per the
                    # refusal this function documents.
                    if raw.endswith("\n"):
                        raise ValueError(f"{shard}:{number}: not valid JSON") from error
                    truncated.append(str(shard))
                    continue

                # Shards are flown by separate processes over disjoint run
                # ranges, so the key is (run, scenario) and never collides
                # across files. If it ever does, the campaign was flown with
                # overlapping --first-run ranges and the samples would be
                # silently interleaved; keyed this way they merge into one run,
                # which the sample count in the report makes visible.
                key = (int(row["run"]), str(row["scenario"]))
                if row.get("kind") == "meta":
                    group = _Accumulator(run=key[0], scenario=key[1])
                    group.intent = str(row.get("intent", ""))
                    group.cycle_period_s = float(row.get("cycle_period_s", 0.0))
                    group.fix_latency_s = float(row.get("fix_latency_s", 0.0))
                    group.duration_s = float(row.get("duration_s", 0.0))
                    group.coast_horizon_s = float(row.get("coast_horizon_s", 0.0))
                    group.degraded_horizon_s = float(row.get("degraded_horizon_s", 0.0))
                    accumulators.append(group)
                    index[key] = group
                    continue

                group = index.get(key)
                if group is None:
                    raise ValueError(
                        f"{shard}:{number}: sample for run {key[0]} scenario "
                        f"{key[1]!r} before its meta line"
                    )
                group.rows.append(row)

    return Campaign(
        runs=tuple(a.freeze() for a in accumulators if a.rows),
        paths=tuple(shards),
        truncated=tuple(Path(p) for p in dict.fromkeys(truncated)),
    )


def _resolve(paths: str | Path | list[str | Path]) -> list[Path]:
    """Expand shards, directories and lists into a sorted list of files."""
    candidates = paths if isinstance(paths, list) else [paths]
    found: list[Path] = []
    for candidate in candidates:
        path = Path(candidate)
        if path.is_dir():
            found.extend(sorted(path.glob("*.jsonl")))
        elif path.is_file():
            found.append(path)
    return found
