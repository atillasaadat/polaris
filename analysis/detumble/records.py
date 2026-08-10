"""Reading the campaign driver's per-run records.

The C++ driver (``tests/mc/detumble_mc.cpp``) writes one JSON object per line —
JSONL rather than CSV because a run carries a nested dispersion block and a
variable-length rate profile, and because appending a line under a mutex from
several worker threads is atomic enough to survive a campaign that is killed
half way. The file lands under ``build-artifacts/`` and is a derived artifact:
never committed.

Reproducibility
---------------
Every record carries the master ``seed`` substream it was drawn from, the
compiled ``config_hash`` of the vehicle it flew, and the full dispersion draw.
That is the ``{config, seed}`` invariant the repo requires: a run is re-flyable
from its own record with
``polaris_detumble_mc --first-run <run_index> --runs 1 --seed <master seed>``.

Units
-----
As recorded: seconds and degrees per second (the driver converts at its output
boundary, which is this file's input boundary). Angles in the dispersion block
are degrees; the rate axis is a unit vector in the body frame.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = ["RunRecord", "load_records"]


@dataclass(frozen=True)
class RunRecord:
    """One Monte Carlo run.

    Attributes
    ----------
    run_index : int
        Index within the campaign; also the RNG substream index.
    seed : int
        The run's derived seed, as drawn from ``{master seed, run_index}``.
    config_hash : str
        SHA-256 of the resolved vehicle configuration the run flew.
    healthy : bool
        The closed loop and the SITL wire completed without error. An unhealthy
        run is a harness failure, not a slow detumble, and is excluded from the
        statistics rather than counted as a non-convergence.
    note : str
        Failure reason when :attr:`healthy` is false; empty otherwise.
    t_engage_s : float
        Seconds from epoch to the first macro step on which the deployment
        scheduled a torque-rod on-window — B-dot's first commanded cycle. All
        other times are measured from here, matching REQ-ACTL-001's convention
        that the window opens at engagement and not at boot.
    t_exit_s : float
        Seconds from engagement to a confirmed ``DetumbleExitRadps`` completion,
        or ``-1`` when the flown arc ended first.
    t_first_below_s : float
        Seconds from engagement to the start of the confirming streak.
    rate_initial_deg_s, rate_final_deg_s, rate_min_deg_s : float
        Body-rate magnitude at the start, at the end, and its minimum over the
        run [deg/s], truth-side.
    rate_at_fast_phase_deg_s : float
        Body-rate magnitude 200 s after engagement — the quantity REQ-ACTL-001's
        fast-phase bound is written on.
    peak_rate_after_fast_phase_deg_s : float
        Largest rate magnitude at or after that instant, which is what the
        requirement's "shall not subsequently rise" clause is checked against.
    initial_spin_field_angle_deg, fast_phase_spin_field_angle_deg : float
        Angle between the spin axis and the local geomagnetic field at
        engagement and at the end of the fast phase [deg], or ``-1`` when the
        field resolver had no value there. B-dot is blind to the rate component
        along **B**, so the second of these is the geometric quantity the tail
        duration should track; the first is the draw, which it should not.
    exit_threshold_deg_s : float
        The deployment's committed ``DetumbleExitRadps`` the run was judged
        against [deg/s], carried with the record so nothing downstream has to
        transcribe it.
    confirm_cycles : int
        Its ``DetumbleConfirmCycles``.
    wall_s : float
        Wall-clock cost of the run, for sizing future campaigns.
    profile_t_s, profile_rate_deg_s : numpy.ndarray
        Sub-sampled rate history from engagement, for the ensemble figure.
    """

    run_index: int
    seed: int
    config_hash: str
    healthy: bool
    note: str
    t_engage_s: float
    t_exit_s: float
    t_first_below_s: float
    rate_initial_deg_s: float
    rate_at_fast_phase_deg_s: float
    rate_final_deg_s: float
    rate_min_deg_s: float
    peak_rate_after_fast_phase_deg_s: float
    initial_spin_field_angle_deg: float
    fast_phase_spin_field_angle_deg: float
    exit_threshold_deg_s: float
    confirm_cycles: int
    wall_s: float
    dispersion: dict
    profile_t_s: np.ndarray
    profile_rate_deg_s: np.ndarray

    @property
    def converged(self) -> bool:
        """The run reached a confirmed completion inside the arc it flew."""
        return self.healthy and self.t_exit_s >= 0.0


def load_records(path: str | Path) -> list[RunRecord]:
    """Read a campaign JSONL file (or every ``*.jsonl`` in a directory).

    Parameters
    ----------
    path : str or pathlib.Path
        The file the driver wrote, or a directory of them — a campaign run as
        several shards produces one file per shard.

    Returns
    -------
    list of RunRecord
        Sorted by ``run_index``. Worker threads interleave their writes, so file
        order is not run order and anything downstream that pairs a record with
        its draw would otherwise be reading a different run's geometry.

    Raises
    ------
    FileNotFoundError
        The path does not exist, or a directory contains no ``*.jsonl``.
    ValueError
        A line is not valid JSON, or is missing a required field. A silently
        skipped malformed record would shrink the sample the confidence bound is
        computed from without saying so.
    """
    target = Path(path)
    if target.is_dir():
        files = sorted(target.glob("*.jsonl"))
        if not files:
            raise FileNotFoundError(f"no *.jsonl campaign records under {target}")
    else:
        if not target.is_file():
            raise FileNotFoundError(f"no campaign records at {target}")
        files = [target]

    records: list[RunRecord] = []
    for file in files:
        for lineno, line in enumerate(file.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                records.append(_from_json(raw))
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ValueError(
                    f"{file}:{lineno}: malformed campaign record: {exc}"
                ) from exc
    return sorted(records, key=lambda r: r.run_index)


def _from_json(raw: dict) -> RunRecord:
    """Build a :class:`RunRecord` from one decoded JSONL line."""
    return RunRecord(
        run_index=int(raw["run_index"]),
        seed=int(raw["seed"]),
        config_hash=str(raw["config_hash"]),
        healthy=bool(raw["healthy"]),
        note=str(raw.get("note", "")),
        t_engage_s=float(raw["t_engage_s"]),
        t_exit_s=float(raw["t_exit_s"]),
        t_first_below_s=float(raw["t_first_below_s"]),
        rate_initial_deg_s=float(raw["rate_initial_deg_s"]),
        rate_at_fast_phase_deg_s=float(raw["rate_at_fast_phase_deg_s"]),
        rate_final_deg_s=float(raw["rate_final_deg_s"]),
        rate_min_deg_s=float(raw["rate_min_deg_s"]),
        peak_rate_after_fast_phase_deg_s=float(raw["peak_rate_after_fast_phase_deg_s"]),
        initial_spin_field_angle_deg=float(
            raw.get("initial_spin_field_angle_deg", -1.0)
        ),
        fast_phase_spin_field_angle_deg=float(
            raw.get("fast_phase_spin_field_angle_deg", -1.0)
        ),
        exit_threshold_deg_s=float(raw["exit_threshold_deg_s"]),
        confirm_cycles=int(raw["confirm_cycles"]),
        wall_s=float(raw.get("wall_s", 0.0)),
        dispersion=dict(raw["dispersion"]),
        profile_t_s=np.asarray(raw.get("profile_t_s", []), dtype=float),
        profile_rate_deg_s=np.asarray(raw.get("profile_rate_deg_s", []), dtype=float),
    )
