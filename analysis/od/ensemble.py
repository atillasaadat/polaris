"""The one consistency check that does not ask the filter about itself.

NEES and NIS (:mod:`analysis.od.statistics`) normalise an error by the very
covariance under test. That makes them cheap — one run is enough to compute one
— and it makes them blind in one specific direction: a filter whose error and
whose covariance are wrong by the same factor passes both. The failure is not
exotic. A process-noise PSD off by a decade, a variance stored where a standard
deviation was meant, a units slip between the propagator and the update: each
scales the reported covariance and the realised error together, and each sails
through a self-normalised test.

The ensemble check closes that gap by estimating the covariance a second way,
from **truth alone**. Fly the same filter over many independent runs, look at
where the estimate actually landed relative to truth each time, and the spread
of those errors *is* the covariance — no model, no filter opinion, just the
sample second moment of a quantity the filter never sees. Holding the reported
1σ up against that spread is the check, and the ratio is the result: one means
the filter's stated uncertainty is the uncertainty it actually has.

Resolved in RIC, not ECI
------------------------
Orbit uncertainty is not isotropic — it is overwhelmingly in-track, because a
small along-track velocity error integrates into a growing position error while
the radial and cross-track components stay bounded and oscillate at orbit rate.
A single scalar spread would average a large number against two small ones and
report a covariance as "right" while its shape was wrong. The three axes are
therefore checked separately, and the frame is built from **truth** (see
``ricFromEci`` in ``tests/mc/orbit_od_mc.cpp``) so that a filter wrong about
where the vehicle is is not also allowed to be wrong about which way in-track
points.

What it costs
-------------
Runs. The precision of a sample standard deviation is roughly
:math:`1/\\sqrt{2(N-1)}` — about 13 % at 30 runs, 35 % at 5 — so this is a check
on the *size* of the covariance to within tens of percent and never a precision
measurement. The acceptance band in :mod:`analysis.od.report` is set to match,
wide enough to pass a healthy filter at the campaign's run count and narrow
enough to catch the factor-of-two-and-worse errors above.

Units
-----
Metres. The ratio is dimensionless.

References
----------
Bar-Shalom, Li & Kirubarajan, *Estimation with Applications to Tracking and
Navigation*, Wiley, 2001, §5.4 [barshalom2001] — Monte Carlo evaluation of
filter consistency, of which this is the covariance-matching half.
Design doc §8.3 (onboard OD), §23.2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from analysis.od.records import ScenarioRun

__all__ = ["RIC_AXES", "EnsembleAxis", "ensemble_covariance"]

#: Axis names of the RIC decomposition, in the order the driver records them.
RIC_AXES = ("radial", "in_track", "cross_track")


@dataclass(frozen=True)
class EnsembleAxis:
    """The ensemble spread against the reported spread, on one RIC axis.

    Attributes
    ----------
    axis : str
        ``"radial"``, ``"in_track"`` or ``"cross_track"``.
    ensemble_sigma_m : float
        The **truth-derived** 1σ: the spread of the actual error across
        independent runs, pooled over the nominal stretch. Computed from truth
        alone — the filter's covariance is never consulted.
    reported_sigma_m : float
        The mean 1σ the filter claimed over the same samples.
    ratio : float
        ``ensemble_sigma_m / reported_sigma_m``. One is a covariance that
        matches reality. Above one the filter is optimistic — the real spread
        is wider than it admits, which is the unsafe direction. Below one it is
        pessimistic.
    runs : int
        Independent runs the ensemble spread was estimated from.
    """

    axis: str
    ensemble_sigma_m: float
    reported_sigma_m: float
    ratio: float
    runs: int


def ensemble_covariance(runs: tuple[ScenarioRun, ...]) -> tuple[EnsembleAxis, ...]:
    """Compare the spread of the true error across runs against the reported 1σ.

    See the module docstring for why this exists alongside NEES. Restricted to
    the **nominal** regime and to cycles with a valid solution: a covariance is
    only meaningful where the filter is claiming one, and the fault stretches
    are deliberately not drawn from the distribution the covariance describes.

    Parameters
    ----------
    runs : tuple of ScenarioRun
        Every run of one scenario. Runs are the independent samples; cycles
        within a run are correlated over the filter's own time constants.

    Returns
    -------
    tuple of EnsembleAxis
        One entry per RIC axis. Empty when fewer than two runs carry usable RIC
        records — a spread cannot be estimated from one sample, and a stale
        shard written before the decomposition existed carries NaN.
    """
    usable = [r for r in runs if r.err_ric_m.size and np.isfinite(r.err_ric_m).any()]
    if len(usable) < 2:
        return ()

    out: list[EnsembleAxis] = []
    for index, axis in enumerate(RIC_AXES):
        errors: list[np.ndarray] = []
        sigmas: list[np.ndarray] = []
        for run in usable:
            keep = run.mask("nominal") & run.solution_valid
            err = run.err_ric_m[keep, index]
            sig = run.sigma_ric_m[keep, index]
            good = np.isfinite(err) & np.isfinite(sig) & (sig > 0.0)
            errors.append(err[good])
            sigmas.append(sig[good])

        pooled_err = np.concatenate(errors) if errors else np.array([])
        pooled_sigma = np.concatenate(sigmas) if sigmas else np.array([])
        if pooled_err.size < 2 or pooled_sigma.size == 0:
            continue

        # The ensemble spread about zero rather than about the sample mean: the
        # error is *supposed* to be zero-mean, and a filter with a real bias
        # must be caught by this rather than have the bias subtracted out
        # before it is looked at.
        ensemble = float(np.sqrt(np.mean(pooled_err**2)))
        reported = float(np.sqrt(np.mean(pooled_sigma**2)))
        out.append(
            EnsembleAxis(
                axis=axis,
                ensemble_sigma_m=ensemble,
                reported_sigma_m=reported,
                ratio=ensemble / reported if reported > 0.0 else float("nan"),
                runs=len(usable),
            )
        )
    return tuple(out)
