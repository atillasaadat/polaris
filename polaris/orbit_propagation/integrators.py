from numba import njit
import numpy as np
import datetime as dt

# Constants
MU_EARTH = 398600.4418  # km^3/s^2


@njit(cache=True)
def compute_acceleration(position):
    """
    Computes the acceleration due to Earth's gravity.

    Parameters
    ----------
    position : np.ndarray
        Position vector in ECI coordinates (km).

    Returns
    -------
    np.ndarray
        Acceleration vector in ECI coordinates (km/s^2).
    """
    r = np.linalg.norm(position)
    accel = -MU_EARTH / r**3 * position
    return accel


@njit(cache=True)
def rk89_step(position, velocity, dt_step):
    """
    Performs a single Runge-Kutta 8/9 integration step.

    Parameters
    ----------
    position : np.ndarray
        Current position vector (km).
    velocity : np.ndarray
        Current velocity vector (km/s).
    dt_step : float
        Time step (s).

    Returns
    -------
    tuple
        Updated position and velocity vectors.
    """
    # RK89 coefficients would go here
    # For simplicity, using RK4 as a placeholder
    k1_vel = compute_acceleration(position)
    k1_pos = velocity

    k2_vel = compute_acceleration(position + 0.5 * dt_step * k1_pos)
    k2_pos = velocity + 0.5 * dt_step * k1_vel

    k3_vel = compute_acceleration(position + 0.5 * dt_step * k2_pos)
    k3_pos = velocity + 0.5 * dt_step * k2_vel

    k4_vel = compute_acceleration(position + dt_step * k3_pos)
    k4_pos = velocity + dt_step * k3_vel

    new_position = position + (dt_step / 6.0) * (k1_pos + 2*k2_pos + 2*k3_pos + k4_pos)
    new_velocity = velocity + (dt_step / 6.0) * (k1_vel + 2*k2_vel + 2*k3_vel + k4_vel)

    return new_position, new_velocity


@njit(cache=True)
def propagate_rk89(position, velocity, duration_sec, dt_step):
    """
    Propagates the state vector using Runge-Kutta 8/9 integration.

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
    for _ in range(num_steps):
        position, velocity = rk89_step(position, velocity, dt_step)
    return position, velocity
