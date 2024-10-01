# orbital_dynamics.py

import numpy as np
from datetime import datetime, timedelta
import pandas as pd
from scipy import constants
from pymsis import MSIS
from astropy.time import Time
from astropy.coordinates import get_body_barycentric, solar_system_ephemeris
import astropy.units as u
from integrators import propagate_rk89  # Ensure integrators.py is in the same directory or in the Python path

# Constants from scipy.constants for precision
MU_EARTH = constants.G * 5.972e24 / 1e9**3  # Earth's gravitational parameter in km^3/s^2
R_EARTH = 6378.137  # Earth's radius in km
J2 = 1.08262668e-3  # Earth's J2 coefficient
SOLAR_CONSTANT = 1361  # Solar constant in W/m^2
AU = constants.au / 1e3  # Astronomical Unit in km
SOLAR_PRESSURE = SOLAR_CONSTANT / constants.c * 1e-3  # Convert to N/(m^2 * m/s) -> N/m^3

# Maximum degree and order for gravitational harmonics
MAX_DEGREE = 64
MAX_ORDER = 64

class OrbitalDynamics:
    """
    Class to handle the orbital dynamics of a satellite, including various perturbations.
    """

    def __init__(self, initial_position, initial_velocity, initial_time, area_over_mass, drag_coeff=1.458e-6, reflectivity=1.5):
        """
        Initializes the OrbitalDynamics object.

        Parameters
        ----------
        initial_position : np.ndarray
            Initial position vector in ECI coordinates (km).
        initial_velocity : np.ndarray
            Initial velocity vector in ECI coordinates (km/s).
        initial_time : datetime.datetime
            Initial epoch time.
        area_over_mass : float
            Area-to-mass ratio of the satellite (m^2/kg).
        drag_coeff : float, optional
            Drag coefficient (dimensionless). Default is 1.458e-6.
        reflectivity : float, optional
            Reflectivity coefficient for SRP (dimensionless). Default is 1.5.
        """
        self.initial_position = initial_position
        self.initial_velocity = initial_velocity
        self.initial_time = initial_time
        self.area_over_mass = area_over_mass  # m^2/kg
        self.drag_coeff = drag_coeff
        self.reflectivity = reflectivity
        self.msis = MSIS()  # Initialize the MSIS-2000 model

        # Initialize gravitational harmonics coefficients (C and S)
        # For demonstration, only up to degree and order 10 are initialized.
        # Full implementation would require coefficients up to degree and order 64.
        self.C = {}  # Dictionary to hold C coefficients
        self.S = {}  # Dictionary to hold S coefficients
        self._initialize_gravitational_harmonics()

    def _initialize_gravitational_harmonics(self):
        """
        Initializes the gravitational harmonics coefficients.
        In practice, these coefficients should be loaded from a reliable data source.
        Here, only a few coefficients are initialized for demonstration.
        """
        # Example coefficients (degree, order): value
        # These are placeholder values and should be replaced with accurate data.
        self.C[(2, 0)] = -J2  # C20
        self.C[(3, 0)] = 0.0   # C30
        self.S[(3, 1)] = 0.0   # S31
        # Add more coefficients as needed up to degree and order 64
        # ...
        pass  # Placeholder for actual coefficient initialization

    def _compute_gravitational_acceleration(self, position):
        """
        Computes the gravitational acceleration including central gravity and higher-order harmonics.

        Parameters
        ----------
        position : np.ndarray
            Position vector in ECI coordinates (km).

        Returns
        -------
        np.ndarray
            Gravitational acceleration vector in ECI coordinates (km/s^2).
        """
        r = np.linalg.norm(position)
        accel = -MU_EARTH / r**3 * position  # Central gravity

        # Add higher-order gravitational harmonics
        # Placeholder implementation: Only J2 is added
        # Full implementation requires summing spherical harmonics up to degree and order 64
        accel += self._compute_j2_acceleration(position)

        # TODO: Implement higher-order harmonics up to degree and order 64
        # This requires spherical harmonic calculations with C and S coefficients

        return accel

    def _compute_j2_acceleration(self, position):
        """
        Computes the acceleration due to Earth's J2 perturbation.

        Parameters
        ----------
        position : np.ndarray
            Position vector in ECI coordinates (km).

        Returns
        -------
        np.ndarray
            J2 acceleration vector in ECI coordinates (km/s^2).
        """
        x, y, z = position
        r = np.linalg.norm(position)
        factor = (3/2) * J2 * MU_EARTH * (R_EARTH**2) / (r**5)
        accel_x = factor * x * (5 * (z**2) / r**2 - 1)
        accel_y = factor * y * (5 * (z**2) / r**2 - 1)
        accel_z = factor * z * (5 * (z**2) / r**2 - 3)
        return np.array([accel_x, accel_y, accel_z])

    def _compute_drag_acceleration(self, position, velocity, current_time):
        """
        Computes the acceleration due to atmospheric drag using the MSIS-2000 model.

        Parameters
        ----------
        position : np.ndarray
            Position vector in ECI coordinates (km).
        velocity : np.ndarray
            Velocity vector in ECI coordinates (km/s).
        current_time : datetime.datetime
            Current epoch time.

        Returns
        -------
        np.ndarray
            Drag acceleration vector in ECI coordinates (km/s^2).
        """
        # Calculate altitude
        r = np.linalg.norm(position)
        altitude = r - R_EARTH  # in km

        if altitude < 1000:  # MSIS-2000 is valid up to ~1000 km
            # Convert current_time to components
            year = current_time.year
            month = current_time.month
            day = current_time.day
            hour = current_time.hour
            minute = current_time.minute
            second = current_time.second

            # Geodetic latitude and longitude (simplified as 0, assuming equatorial orbit)
            latitude = 0.0
            longitude = 0.0

            # Get density in kg/m^3 from MSIS-2000
            density = self.msis.run(year, month, day, hour, minute, second, altitude*1000, latitude, longitude, 0)[0]  # 0: total density

            # Convert density to kg/km^3
            density = density * 1e9  # 1 m^3 = 1e-9 km^3

            # Compute relative velocity (assuming atmosphere is non-rotating for simplicity)
            rel_velocity = velocity  # Simplification: atmospheric velocity neglected

            v_rel = np.linalg.norm(rel_velocity)
            if v_rel == 0:
                return np.zeros(3)

            drag_accel_mag = -0.5 * density * v_rel * self.drag_coeff * self.area_over_mass
            drag_accel = drag_accel_mag * (rel_velocity / v_rel)
            return drag_accel
        else:
            return np.zeros(3)

    def _compute_srp_acceleration(self, position, current_time):
        """
        Computes the acceleration due to Solar Radiation Pressure (SRP).

        Parameters
        ----------
        position : np.ndarray
            Position vector in ECI coordinates (km).
        current_time : datetime.datetime
            Current epoch time.

        Returns
        -------
        np.ndarray
            SRP acceleration vector in ECI coordinates (km/s^2).
        """
        # Vector from satellite to Sun
        # Using astropy to get Sun's position in ECI frame
        with solar_system_ephemeris.set('builtin'):
            t = Time(current_time)
            sun_pos = get_body_barycentric('sun', t).xyz.to(u.km).value  # Sun position in km
            earth_pos = get_body_barycentric('earth', t).xyz.to(u.km).value  # Earth position in km

        # Satellite position in barycentric ECI
        sat_bary_pos = position + earth_pos

        # Vector from satellite to Sun
        r_sun = sun_pos - sat_bary_pos
        distance_sun = np.linalg.norm(r_sun)
        unit_r_sun = r_sun / distance_sun

        # SRP acceleration
        # F = P * A * C_r / m
        # a = F / m = P * A * C_r / m = (Solar Constant) / c * (A/m) * C_r
        # Adjust for distance from Sun (inverse square law)
        srp_accel_mag = (SOLAR_CONSTANT / constants.c) * self.area_over_mass * self.reflectivity * (AU / distance_sun)**2  # km/s^2
        srp_accel = srp_accel_mag * unit_r_sun
        return srp_accel

    def _compute_third_body_acceleration(self, position, current_time):
        """
        Computes the gravitational acceleration due to third-body effects (Sun and Moon).

        Parameters
        ----------
        position : np.ndarray
            Position vector in ECI coordinates (km).
        current_time : datetime.datetime
            Current epoch time.

        Returns
        -------
        np.ndarray
            Third-body acceleration vector in ECI coordinates (km/s^2).
        """
        third_body_accel = np.zeros(3)

        # Define third bodies: Sun and Moon
        third_bodies = ['sun', 'moon']
        with solar_system_ephemeris.set('builtin'):
            t = Time(current_time)
            earth_pos = get_body_barycentric('earth', t).xyz.to(u.km).value  # Earth position in km

            for body in third_bodies:
                body_pos = get_body_barycentric(body, t).xyz.to(u.km).value  # Body position in km
                r_body = body_pos - position - earth_pos  # Vector from satellite to third body
                distance_body = np.linalg.norm(r_body)
                accel_body = constants.G * 5.972e24 / (distance_body**3) * r_body  # Simplified formula
                third_body_accel += accel_body

        return third_body_accel

    def _compute_solid_tides(self, position, current_time):
        """
        Computes the acceleration due to solid Earth tides.

        Parameters
        ----------
        position : np.ndarray
            Position vector in ECI coordinates (km).
        current_time : datetime.datetime
            Current epoch time.

        Returns
        -------
        np.ndarray
            Solid tides acceleration vector in ECI coordinates (km/s^2).
        """
        # Solid Earth tides are complex and require detailed modeling.
        # Here, we include a simplified placeholder implementation.

        # Placeholder: Assume a small periodic perturbation
        tide_amplitude = 1e-6  # km/s^2, example value
        omega = 2 * np.pi / (12 * 3600)  # Tidal frequency (rad/s), example for semidiurnal tides
        accel_tides = tide_amplitude * np.array([np.sin(omega * (datetime.utcnow() - self.initial_time).total_seconds()),
                                                np.cos(omega * (datetime.utcnow() - self.initial_time).total_seconds()),
                                                0.0])
        return accel_tides

    def _compute_total_acceleration(self, position, velocity, current_time):
        """
        Computes the total acceleration on the satellite, including all perturbations.

        Parameters
        ----------
        position : np.ndarray
            Position vector in ECI coordinates (km).
        velocity : np.ndarray
            Velocity vector in ECI coordinates (km/s).
        current_time : datetime.datetime
            Current epoch time.

        Returns
        -------
        np.ndarray
            Total acceleration vector in ECI coordinates (km/s^2).
        """
        # Gravitational acceleration (central and higher-order harmonics)
        a_grav = self._compute_gravitational_acceleration(position)

        # Atmospheric drag
        a_drag = self._compute_drag_acceleration(position, velocity, current_time)

        # Solar Radiation Pressure
        a_srp = self._compute_srp_acceleration(position, current_time)

        # Third-body effects (Sun and Moon)
        a_third_body = self._compute_third_body_acceleration(position, current_time)

        # Solid Earth tides
        a_tides = self._compute_solid_tides(position, current_time)

        # Total acceleration
        total_accel = a_grav + a_drag + a_srp + a_third_body + a_tides

        return total_accel

    def propagate(self, target_times, dt_step=60.0):
        """
        Propagates the satellite's state to multiple target times accounting for all perturbations.

        Parameters
        ----------
        target_times : list, pd.Series, or np.ndarray of datetime.datetime
            Target times to propagate to.
        dt_step : float, optional
            Integration time step in seconds. Default is 60 seconds.

        Returns
        -------
        tuple of np.ndarray
            - positions: Array of position vectors at each target time (shape: [N, 3]).
            - velocities: Array of velocity vectors at each target time (shape: [N, 3]).
        """
        # Convert target_times to a list if it's a pandas Series or NumPy array
        if isinstance(target_times, pd.Series):
            target_times = target_times.tolist()
        elif isinstance(target_times, np.ndarray):
            target_times = target_times.tolist()
        elif not isinstance(target_times, list):
            raise TypeError("target_times must be a list, pandas Series, or numpy array of datetime objects.")

        # Calculate durations in seconds from the initial_time
        durations = np.array([(t - self.initial_time).total_seconds() for t in target_times])

        # Initialize arrays to store positions and velocities
        num_targets = len(durations)
        positions = np.empty((num_targets, 3))
        velocities = np.empty((num_targets, 3))

        # Iterate over each duration and propagate the state
        for i, duration in enumerate(durations):
            if duration < 0:
                raise ValueError("Target times must be after the initial_time.")

            # Propagate using custom RK4 integrator
            pos, vel = self._custom_propagate(self.initial_position.copy(), self.initial_velocity.copy(),
                                              duration, dt_step)
            positions[i] = pos
            velocities[i] = vel

        return positions, velocities

    def _custom_propagate(self, position, velocity, duration_sec, dt_step):
        """
        Custom propagation using RK4 to include all perturbations.

        Parameters
        ----------
        position : np.ndarray
            Initial position vector (km).
        velocity : np.ndarray
            Initial velocity vector (km/s).
        duration_sec : float
            Total duration to propagate (s).
        dt_step : float
            Time step for integration (s).

        Returns
        -------
        tuple
            Final position and velocity vectors after propagation.
        """
        num_steps = int(duration_sec / dt_step)
        current_time = self.initial_time

        for _ in range(num_steps):
            # RK4 Integration
            k1_vel = self._compute_total_acceleration(position, velocity, current_time)
            k1_pos = velocity

            k2_time = current_time + timedelta(seconds=0.5 * dt_step)
            k2_vel = self._compute_total_acceleration(position + 0.5 * dt_step * k1_pos,
                                                     velocity + 0.5 * dt_step * k1_vel, k2_time)
            k2_pos = velocity + 0.5 * dt_step * k1_vel

            k3_time = current_time + timedelta(seconds=0.5 * dt_step)
            k3_vel = self._compute_total_acceleration(position + 0.5 * dt_step * k2_pos,
                                                     velocity + 0.5 * dt_step * k2_vel, k3_time)
            k3_pos = velocity + 0.5 * dt_step * k2_vel

            k4_time = current_time + timedelta(seconds=dt_step)
            k4_vel = self._compute_total_acceleration(position + dt_step * k3_pos,
                                                     velocity + dt_step * k3_vel, k4_time)
            k4_pos = velocity + dt_step * k3_vel

            # Update position and velocity
            position += (dt_step / 6.0) * (k1_pos + 2 * k2_pos + 2 * k3_pos + k4_pos)
            velocity += (dt_step / 6.0) * (k1_vel + 2 * k2_vel + 2 * k3_vel + k4_vel)

            # Update current_time
            current_time += timedelta(seconds=dt_step)

        return position, velocity
