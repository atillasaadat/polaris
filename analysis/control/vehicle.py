"""As-flown vehicle description, read from the committed spacecraft YAML.

Every number the linear-analysis toolkit uses comes from
``config/spacecraft/*.yaml`` through the config compiler's own loader
(:func:`configc.compiler.load_config`), so the models analysed here are the
models the flight software is tuned with. Nothing in this module carries a
default: a transcription of a catalog value is a copy, and every copy is a
place for the analysis and the vehicle to disagree in the direction that
passes.

Units and frames
----------------
SI throughout. Inertia is the body-frame tensor [kg·m²]; wheel spin axes and
magnetorquer dipole axes are unit vectors in the body frame; gains are in the
units the ``flight.attitudeController.*`` parameters declare (N·m/rad,
N·m/(rad·s), N·m/(rad/s)); the orbit is the scenario's osculating Keplerian
set. Quaternions do not appear — this module reads matrices, not attitudes.

Notes
-----
``tools/`` must be importable (``pytest.ini`` puts it on ``pythonpath``; a
script run by hand needs ``PYTHONPATH=tools``).

References
----------
Design doc §19.1 (spacecraft configuration), §19.3 (the config pipeline),
§19.4 (no vehicle constants in code).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from configc.compiler import load_config, load_hardware_library

#: Largest per-unit array the FSW parameter set carries (``flight.GncMaxUnits``);
#: the flat ``*AxesBody`` parameters are always this many 3-vectors, with the
#: uninstalled slots written as the zero vector.
MAX_UNITS = 8


@dataclass(frozen=True)
class PidGains:
    """The shipped ``AttitudePid`` tuning (``lib/gnc/attitude_pid.hpp``).

    Attributes
    ----------
    kp_nm_per_rad : float
        Proportional gain on the error rotation vector [N·m/rad].
    ki_nm_per_rad_s : float
        Integral gain on the accumulated error rotation [N·m/(rad·s)].
    kd_nm_per_radps : float
        Derivative gain on the rate error [N·m/(rad/s)].
    max_integral_rad_s : float
        Per-axis integrator clamp [rad·s].
    max_torque_nm : float
        Magnitude limit on the commanded body torque [N·m].
    """

    kp_nm_per_rad: float
    ki_nm_per_rad_s: float
    kd_nm_per_radps: float
    max_integral_rad_s: float
    max_torque_nm: float


@dataclass(frozen=True)
class SensorNoise:
    """Estimator noise budget, as tuned on this vehicle (§8.1, §8.2).

    The sun figure is the *total* the estimator forms per cycle, white and
    systematic in quadrature, in the nominal case (albedo correction running,
    DE440 tables active). The magnetic figure is the uncalibrated total the
    vehicle launches with — the case the observability study is worth running,
    since a post-calibration magnetometer only improves it.

    Attributes
    ----------
    sigma_sun_rad : float
        Sun-direction 1σ [rad].
    sigma_mag_rad : float
        Magnetic-direction 1σ [rad].
    gyro_arw_rad_s_sqrt : float
        Gyro angle random walk [rad·s^(-1/2)].
    min_sin_angle : float
        TRIAD geometry gate, ``sin`` of the smallest admitted separation between
        the two reference directions [-].
    """

    sigma_sun_rad: float
    sigma_mag_rad: float
    gyro_arw_rad_s_sqrt: float
    min_sin_angle: float


@dataclass(frozen=True)
class Orbit:
    """The scenario's initial osculating orbit (design doc §19.1).

    Attributes
    ----------
    sma_m : float
        Semi-major axis [m].
    ecc : float
        Eccentricity [-].
    inc_rad : float
        Inclination [rad].
    raan_rad : float
        Right ascension of the ascending node [rad].
    epoch_utc : str
        Scenario epoch, ISO-8601 UTC. Carried for provenance; the dipole field
        model in :mod:`analysis.control.field` is evaluated at the IGRF epoch
        the coefficient table names, not at this one.
    """

    sma_m: float
    ecc: float
    inc_rad: float
    raan_rad: float
    epoch_utc: str

    @property
    def mean_motion_rad_s(self) -> float:
        """Keplerian mean motion :math:`\\sqrt{\\mu/a^3}` [rad/s]."""
        mu_earth_m3_s2 = 3.986004418e14  # WGS84 GM [m³/s²]
        return float(np.sqrt(mu_earth_m3_s2 / self.sma_m**3))

    @property
    def period_s(self) -> float:
        """Keplerian orbital period [s]."""
        return float(2.0 * np.pi / self.mean_motion_rad_s)


@dataclass(frozen=True)
class Vehicle:
    """Everything the linear models need, resolved from one config file.

    Attributes
    ----------
    name : str
        Vehicle name from the config.
    inertia_kgm2 : numpy.ndarray
        Body-frame inertia tensor, shape ``(3, 3)`` [kg·m²].
    wheel_spin_axes : numpy.ndarray
        Installed wheel **spin** axes as columns, shape ``(3, N)`` [-]. These are
        the sim's ``W`` matrix; see :meth:`wheel_torque_axes` for the sign the
        controller applies.
    wheel_max_torque_nm : float
        Per-wheel commanded-torque limit [N·m].
    wheel_max_momentum_nms : float
        Per-wheel momentum capacity [N·m·s], from the hardware catalog entry the
        wheel's ``model_id`` resolves to. Not an FSW parameter — the flight
        controller has no momentum management yet (Push 56) — but the number the
        SISO validity boundary is drawn against, so it is read from the catalog
        rather than assumed.
    mtq_axes : numpy.ndarray
        Installed rod dipole axes as columns, shape ``(3, M)`` [-].
    mtq_max_dipole_am2 : float
        Per-rod rated moment [A·m²].
    mtq_duty_factor : float
        Fraction of the control period the rods are energised [-].
    alloc_min_conditioning : float
        The flight allocator's three-axis-span gate,
        :math:`\\lambda_{\\min}/\\lambda_{\\max}` of :math:`AA^\\top` [-]. Carried
        so the controllability study can compare its own conditioning against
        the value the vehicle would actually refuse at, rather than against a
        number chosen here.
    control_period_s : float
        GNC rate-group period [s] — the sampling period every discrete-time
        result in this package is computed at.
    pid : PidGains
        The shipped pointing gains.
    sensors : SensorNoise
        The estimator noise budget.
    orbit : Orbit
        The scenario orbit.
    """

    name: str
    inertia_kgm2: np.ndarray
    wheel_spin_axes: np.ndarray
    wheel_max_torque_nm: float
    wheel_max_momentum_nms: float
    mtq_axes: np.ndarray
    mtq_max_dipole_am2: float
    mtq_duty_factor: float
    alloc_min_conditioning: float
    control_period_s: float
    pid: PidGains
    sensors: SensorNoise
    orbit: Orbit

    def wheel_torque_axes(self, wheels: tuple[int, ...] | None = None) -> np.ndarray:
        """Body torque per unit commanded wheel torque, as columns.

        A wheel's reaction on the body is :math:`-I\\dot\\omega`, so the torque
        authority matrix is the **negated** spin axes — the sign
        ``flight::AttitudeController`` applies once, where the config is read
        (``lib/gnc/rw_allocation.hpp``). Getting it wrong yields a
        sign-inverted, perfectly plausible controller, which is why it is
        derived here rather than restated.

        Parameters
        ----------
        wheels : tuple of int, optional
            Column indices to keep, for wheel-failure subsets. ``None`` keeps
            every installed wheel.

        Returns
        -------
        numpy.ndarray
            Shape ``(3, k)`` [-], body frame.
        """
        axes = -self.wheel_spin_axes
        if wheels is None:
            return axes
        return axes[:, list(wheels)]

    def inertia_inverse(self) -> np.ndarray:
        """Inverse body-frame inertia :math:`J^{-1}` [1/(kg·m²)], shape ``(3, 3)``."""
        return np.linalg.inv(self.inertia_kgm2)

    @property
    def principal_moments_kgm2(self) -> np.ndarray:
        """Diagonal of the body-frame inertia tensor [kg·m²], shape ``(3,)``.

        The reference vehicle's tensor is diagonal, so these *are* its principal
        moments and the per-axis decoupled models are exact rather than
        approximate. :func:`load_vehicle` refuses a config whose products of
        inertia are non-zero rather than letting that assumption go unstated.
        """
        return np.diag(self.inertia_kgm2).copy()


def _axes_from_flat(flat: list[float], count: int, what: str) -> np.ndarray:
    """Unpack a flat ``3 * MAX_UNITS`` parameter into the installed columns."""
    if len(flat) != 3 * MAX_UNITS:
        raise ValueError(f"{what}: expected {3 * MAX_UNITS} values, got {len(flat)}")
    axes = np.asarray(flat, dtype=float).reshape(MAX_UNITS, 3).T
    installed = axes[:, :count]
    if np.any(np.linalg.norm(installed, axis=0) == 0.0):
        raise ValueError(f"{what}: a zero column inside the installed count {count}")
    if np.any(np.linalg.norm(axes[:, count:], axis=0) != 0.0):
        raise ValueError(f"{what}: a non-zero column past the installed count {count}")
    return installed


def _wheel_momentum_capacity(spacecraft, hardware_dir: Path) -> float:
    """Per-wheel momentum capacity [N·m·s] from the resolved hardware catalog.

    Read rather than restated: a transcribed catalog value is a copy, and every
    copy is a place for the analysis and the vehicle to disagree. The smallest
    capacity across the installed wheels is taken, since a mixed array is
    bounded by its weakest unit.
    """
    library = load_hardware_library(hardware_dir)
    capacities = [
        float(library[unit.model_id].params["max_momentum_nms"])
        for unit in spacecraft.actuators
        if unit.model_id in library and library[unit.model_id].kind == "reaction_wheel"
    ]
    if not capacities:
        raise KeyError(
            f"{hardware_dir}: no reaction-wheel catalog entry for this vehicle's "
            "actuators; the SISO validity boundary cannot be evaluated"
        )
    return min(capacities)


def load_vehicle(
    config_path: str | Path, hardware_dir: str | Path | None = None
) -> Vehicle:
    """Build a :class:`Vehicle` from a committed spacecraft config file.

    Parameters
    ----------
    config_path : str or pathlib.Path
        Path to a ``config/spacecraft/*.yaml`` file.
    hardware_dir : str or pathlib.Path, optional
        The hardware model library the ``model_id`` references resolve against.
        Defaults to ``config/hardware`` beside the given config.

    Returns
    -------
    Vehicle
        The as-flown model.

    Raises
    ------
    ValueError
        If the config carries products of inertia (the per-axis models in
        :mod:`analysis.control.plant` assume a diagonal tensor and say so), or
        if an axes parameter disagrees with its declared unit count.
    KeyError
        If a parameter this toolkit needs is absent. There are no flight
        defaults (§19.3) and there are none here either.
    """
    config_path = Path(config_path)
    hardware = (
        Path(hardware_dir)
        if hardware_dir is not None
        else config_path.parent.parent / "hardware"
    )
    cfg = load_config(config_path)
    sc = cfg.spacecraft
    fsw = sc.fsw_parameters
    inertia = sc.inertia_kgm2

    if inertia.ixy != 0.0 or inertia.ixz != 0.0 or inertia.iyz != 0.0:
        raise ValueError(
            "products of inertia are non-zero; the per-axis decoupled models in "
            "analysis.control.plant are stated for a diagonal tensor. Rotate the "
            "config into principal axes or extend the models before analysing it."
        )
    inertia_matrix = np.array(
        [
            [inertia.ixx, inertia.ixy, inertia.ixz],
            [inertia.ixy, inertia.iyy, inertia.iyz],
            [inertia.ixz, inertia.iyz, inertia.izz],
        ],
        dtype=float,
    )

    def param(key: str) -> float | int | list[float]:
        name = f"flight.attitudeController.{key}"
        if name not in fsw:
            raise KeyError(f"{config_path}: missing FSW parameter {name}")
        return fsw[name]

    def estimator_param(key: str) -> float:
        name = f"flight.attitudeEstimator.{key}"
        if name not in fsw:
            raise KeyError(f"{config_path}: missing FSW parameter {name}")
        return float(fsw[name])

    wheel_count = int(param("WheelCount"))
    mtq_count = int(param("MtqCount"))
    orbit = cfg.scenario.initial_state.orbit

    # Sun total, nominal case: white plus the corrected albedo dispersion and the
    # DE440-grade ephemeris term, in quadrature — the composition the estimator
    # performs per cycle (§8.1, and the derivation in the config's own comments).
    sigma_sun = float(
        np.hypot(
            estimator_param("SigmaSunWhiteRad"),
            np.hypot(
                estimator_param("SigmaSunAlbedoRad"),
                estimator_param("SigmaSunEphemPreciseRad"),
            ),
        )
    )
    sigma_mag = float(
        np.hypot(estimator_param("SigmaMagWhiteRad"), estimator_param("SigmaMagSysRad"))
    )

    return Vehicle(
        name=sc.name,
        inertia_kgm2=inertia_matrix,
        wheel_spin_axes=_axes_from_flat(
            list(param("WheelAxesBody")), wheel_count, "WheelAxesBody"
        ),
        wheel_max_torque_nm=float(param("WheelMaxTorqueNm")),
        wheel_max_momentum_nms=_wheel_momentum_capacity(sc, hardware),
        mtq_axes=_axes_from_flat(list(param("MtqAxesBody")), mtq_count, "MtqAxesBody"),
        mtq_max_dipole_am2=float(param("BdotMaxDipoleAm2")),
        mtq_duty_factor=float(param("MtqDutyFactor")),
        alloc_min_conditioning=float(param("AllocMinConditioning")),
        control_period_s=float(param("ControlPeriodSec")),
        pid=PidGains(
            kp_nm_per_rad=float(param("PidKpNmPerRad")),
            ki_nm_per_rad_s=float(param("PidKiNmPerRadS")),
            kd_nm_per_radps=float(param("PidKdNmPerRadps")),
            max_integral_rad_s=float(param("PidMaxIntegralRadS")),
            max_torque_nm=float(param("PidMaxTorqueNm")),
        ),
        sensors=SensorNoise(
            sigma_sun_rad=sigma_sun,
            sigma_mag_rad=sigma_mag,
            gyro_arw_rad_s_sqrt=estimator_param("GyroArw"),
            min_sin_angle=estimator_param("MinSinAngle"),
        ),
        orbit=Orbit(
            sma_m=orbit.sma_km * 1000.0,
            ecc=orbit.ecc,
            inc_rad=float(np.deg2rad(orbit.inc_deg)),
            raan_rad=float(np.deg2rad(orbit.raan_deg)),
            epoch_utc=cfg.scenario.epoch_utc,
        ),
    )
