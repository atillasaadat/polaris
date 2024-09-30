# satellite.py

import datetime as dt
from typing import Optional, Union
import requests
from dataclasses import dataclass, field

import numpy as np
from numba import njit
import pytz
from skyfield.api import EarthSatellite, load, wgs84
from skyfield.api import Time as SkyfieldTime
from skyfield.sgp4lib import Satrec
from orbit_propagation.integrators import propagate_rk89


@dataclass
class StateVector:
    position: tuple  # (x, y, z) in km
    velocity: tuple  # (vx, vy, vz) in km/s
    epoch: dt.datetime


class Satellite:
    """
    A class to represent a satellite with multiple initialization options.

    Attributes
    ----------
    name : str
        The name of the satellite.
    norad_id : Optional[int]
        The NORAD ID of the satellite.
    epoch_utc_dt : Optional[dt.datetime]
        The epoch UTC time of the satellite.
    state_vector : StateVector
        The current state vector of the satellite.
    """

    TLE_URL = "https://celestrak.com/NORAD/elements/gp.php?CATNR={norad_id}&FORMAT=TLE"

    def __init__(
        self,
        name: Optional[str] = None,
        norad_id: Optional[int] = None,
        epoch_utc_dt: Optional[dt.datetime] = None,
        tle: Optional[str] = None,
        keplerian: Optional[dict] = None,
        state_vector: Optional[StateVector] = None,
        auto_download: bool = False,
    ) -> None:
        """
        Initializes a Satellite instance with various initialization options.

        Parameters
        ----------
        name : str, optional
            The name of the satellite.
        norad_id : Optional[int], optional
            The NORAD ID of the satellite, by default None.
        epoch_utc_dt : Optional[datetime.datetime], optional
            The epoch UTC datetime of the satellite, by default None.
        tle : Optional[str], optional
            Two-Line Element (TLE) set as a single string, by default None.
        keplerian : Optional[dict], optional
            Dictionary containing Keplerian elements, by default None.
        state_vector : Optional[StateVector], optional
            Initial state vector, by default None.
        auto_download : bool, optional
            If True and norad_id is provided without a TLE, automatically download TLE, by default False.

        Raises
        ------
        ValueError
            If insufficient initialization parameters are provided.
        """
        if name is None and norad_id is not None:
            name = str(norad_id)
        if name is None and norad_id is None:
            raise ValueError(
                "Initialization requires at least a name or norad_id."
            )

        self.name = name
        self.norad_id = norad_id
        self.epoch_utc_dt = epoch_utc_dt

        # Initialize state_vector
        self.state_vector: Optional[StateVector] = state_vector

        # Initialize satellite data based on provided parameters
        if tle:
            self._init_from_tle(tle)
        elif norad_id:
            if auto_download:
                tle = self._download_tle(norad_id)
                if tle:
                    self._init_from_tle(tle)
                else:
                    raise ValueError(f"Could not download TLE for NORAD ID {norad_id}")
            else:
                raise ValueError(
                    "NORAD ID provided without TLE and auto_download is False."
                )
        elif keplerian:
            self._init_from_keplerian(keplerian)
        elif state_vector:
            self._init_from_state_vector(state_vector)
        else:
            raise ValueError(
                "Initialization requires TLE, keplerian elements, or state vector."
            )

    def _init_from_tle(self, tle: str):
        lines = tle.strip().split('\n')
        if len(lines) != 2 and len(lines) != 3:
            raise ValueError("TLE must have 2 or 3 lines.")

        if len(lines) == 3:
            # Assume first line is name
            self.name = lines[0].strip()
            tle_lines = lines[1:]
        else:
            tle_lines = lines

        self.satellite = EarthSatellite(tle_lines[0], tle_lines[1], self.name)
        self.epoch_utc_dt = self.satellite.epoch.utc_datetime()

        # Initialize state vector
        self.update_state_vector()

    def _init_from_keplerian(self, keplerian: dict):
        """
        Initialize the satellite from Keplerian elements.
        Keplerian elements should include:
        - semi_major_axis (km)
        - eccentricity
        - inclination (degrees)
        - raan (degrees)
        - arg_of_perigee (degrees)
        - true_anomaly (degrees)
        - epoch (dt.datetime)
        """
        required_elements = [
            'semi_major_axis',
            'eccentricity',
            'inclination',
            'raan',
            'arg_of_perigee',
            'true_anomaly',
            'epoch',
        ]
        for elem in required_elements:
            if elem not in keplerian:
                raise ValueError(f"Missing Keplerian element: {elem}")

        ts = load.timescale()
        epoch = keplerian['epoch']
        t = ts.utc(epoch.year, epoch.month, epoch.day, epoch.hour, epoch.minute, epoch.second)

        # Convert Keplerian elements to state vector
        state_vec = self._keplerian_to_state_vector(keplerian)
        self.state_vector = StateVector(
            position=state_vec['position'],
            velocity=state_vec['velocity'],
            epoch=epoch
        )
        # Note: For full integration, you'd need to create an EarthSatellite or similar object
        # Placeholder: Implement as needed

    def _init_from_state_vector(self, state_vector: StateVector):
        """
        Initialize the satellite from a state vector.
        """
        self.state_vector = state_vector
        # Create EarthSatellite or similar object based on state vector
        # Placeholder: Implement as needed

    def _download_tle(self, norad_id: int) -> Optional[str]:
        """
        Downloads the TLE for a given NORAD ID from Celestrak.

        Parameters
        ----------
        norad_id : int
            The NORAD ID of the satellite.

        Returns
        -------
        Optional[str]
            The TLE as a string if successful, None otherwise.
        """
        url = self.TLE_URL.format(norad_id=norad_id)
        response = requests.get(url)
        if response.status_code == 200:
            return response.text
        else:
            return None

    def _keplerian_to_state_vector(self, keplerian: dict) -> dict:
        """
        Converts Keplerian elements to a state vector using optimized numerical methods.

        Parameters
        ----------
        keplerian : dict
            Dictionary containing Keplerian elements.

        Returns
        -------
        dict
            Dictionary with 'position' and 'velocity' tuples.
        """
        # Unpack Keplerian elements
        a = keplerian['semi_major_axis']  # km
        e = keplerian['eccentricity']
        i = keplerian['inclination']  # degrees
        raan = keplerian['raan']  # degrees
        arg_perigee = keplerian['arg_of_perigee']  # degrees
        true_anomaly = keplerian['true_anomaly']  # degrees

        # Convert angles to radians
        i_rad = np.radians(i)
        raan_rad = np.radians(raan)
        arg_perigee_rad = np.radians(arg_perigee)
        true_anomaly_rad = np.radians(true_anomaly)

        # Gravitational parameter for Earth (km^3/s^2)
        mu = 398600.4418

        # Compute distance
        r = a * (1 - e**2) / (1 + e * np.cos(true_anomaly_rad))

        # Position in orbital plane
        x_orb = r * np.cos(true_anomaly_rad)
        y_orb = r * np.sin(true_anomaly_rad)
        z_orb = 0.0

        # Specific angular momentum
        h = np.sqrt(mu * a * (1 - e**2))

        # Velocity in orbital plane
        vx_orb = -mu / h * np.sin(true_anomaly_rad)
        vy_orb = mu / h * (e + np.cos(true_anomaly_rad))
        vz_orb = 0.0

        # Rotation matrices as NumPy arrays
        R_z_raan = rotation_matrix_z(raan_rad)
        R_x_i = rotation_matrix_x(i_rad)
        R_z_arg_perigee = rotation_matrix_z(arg_perigee_rad)

        # Combined rotation matrix
        rotation_matrix = R_z_raan @ R_x_i @ R_z_arg_perigee

        # Position and velocity in ECI frame using NumPy dot product
        position_eci = rotation_matrix @ np.array([x_orb, y_orb, z_orb])
        velocity_eci = rotation_matrix @ np.array([vx_orb, vy_orb, vz_orb])

        return {
            'position': tuple(position_eci),
            'velocity': tuple(velocity_eci)
        }

    def update_state_vector(self):
        """
        Updates the state vector based on the current epoch using propagation.

        This method should be called after each propagation step.
        """
        if not hasattr(self, 'satellite'):
            raise AttributeError("Satellite object not initialized with TLE.")
        
        ts = load.timescale()
        current_time = ts.utc(self.epoch_utc_dt)
        geocentric = self.satellite.at(current_time)
        position = geocentric.position.km
        velocity = geocentric.velocity.km_per_s
        self.state_vector = StateVector(
            position=tuple(position),
            velocity=tuple(velocity),
            epoch=self.epoch_utc_dt
        )

    def propagate(self, duration: dt.timedelta, method: str = 'rk89', dt_step: float = 10.0):
        """
        Propagates the satellite's orbit by the specified duration.

        Parameters
        ----------
        duration : dt.timedelta
            The duration to propagate the orbit.
        method : str, optional
            The numerical integration method to use, by default 'rk89'.
        dt_step : float, optional
            Time step for integration in seconds, by default 10.0.
        """
        if method == 'rk89':
            if self.state_vector is None:
                raise ValueError("State vector is not initialized.")

            # Convert position and velocity to NumPy arrays
            position = np.array(self.state_vector.position)
            velocity = np.array(self.state_vector.velocity)

            # Total duration in seconds
            duration_sec = duration.total_seconds()

            # Perform propagation using the optimized RK89 integrator
            final_position, final_velocity = propagate_rk89(position, velocity, duration_sec, dt_step)

            # Update epoch
            self.epoch_utc_dt += duration

            # Update state vector
            self.state_vector = StateVector(
                position=tuple(final_position),
                velocity=tuple(final_velocity),
                epoch=self.epoch_utc_dt
            )
        else:
            raise NotImplementedError(f"Propagation method {method} is not implemented.")

    def __str__(self) -> str:
        return f"Satellite {self.name}, NORAD ID: {self.norad_id or 'Unknown'}"

    def __repr__(self) -> str:
        return (
            f"Satellite(name={self.name!r}, "
            f"norad_id={self.norad_id!r}, "
            f"epoch_utc_dt={self.epoch_utc_dt!r}, "
            f"state_vector={self.state_vector!r})"
        )

    def get_epoch_iso_string(self, sep: Optional[str] = "T") -> str:
        return self.epoch_utc_dt.isoformat(sep=sep)
