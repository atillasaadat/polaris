# tests/test_satellite.py

import datetime as dt
from unittest.mock import patch

import pytest

from polaris.satellite import Satellite, StateVector

# Sample TLE data for testing
SAMPLE_TLE = """
ISS (ZARYA)             
1 25544U 98067A   23165.54791667  .00001264  00000-0  29603-4 0  9991
2 25544  51.6453 215.6543 0002213  82.7854  46.1218 15.50021279322248
"""

# Sample Keplerian elements for testing
SAMPLE_KEPLERIAN = {
    'semi_major_axis': 7000.0,      # km
    'eccentricity': 0.001,          # dimensionless
    'inclination': 98.7,            # degrees
    'raan': 0.0,                     # degrees
    'arg_of_perigee': 0.0,          # degrees
    'true_anomaly': 0.0,            # degrees
    'epoch': dt.datetime(2024, 4, 27, 12, 0, 0, tzinfo=dt.timezone.utc)
}

# Sample State Vector for testing
SAMPLE_STATE_VECTOR = StateVector(
    position=(7000.0, 0.0, 0.0),  # km
    velocity=(0.0, 7.5, 1.0),     # km/s
    epoch=dt.datetime(2024, 4, 27, 12, 0, 0, tzinfo=dt.timezone.utc)
)

def test_satellite_initialization_with_name_norad_epoch():
    """Test initialization with name, NORAD ID, and epoch."""
    epoch = dt.datetime(2023, 9, 28, 14, 30, tzinfo=dt.timezone.utc)
    satellite = Satellite(
        name="Sentinel-6",
        norad_id=12345,
        epoch_utc_dt=epoch
    )
    assert satellite.name == "Sentinel-6"
    assert satellite.norad_id == 12345
    assert satellite.epoch_utc_dt == epoch
    assert satellite.state_vector is None

def test_satellite_str():
    """Test the __str__ method."""
    satellite = Satellite(name="Sentinel-6", norad_id=12345)
    assert str(satellite) == "Satellite Sentinel-6, NORAD ID: 12345"

def test_satellite_repr():
    """Test the __repr__ method."""
    satellite = Satellite(name="Sentinel-6", norad_id=12345)
    expected_repr = "Satellite(name='Sentinel-6', norad_id=12345, epoch_utc_dt=None, state_vector=None)"
    assert repr(satellite) == expected_repr

def test_satellite_initialization_without_norad_id():
    """Test initialization without a NORAD ID."""
    satellite = Satellite(name="Sentinel-6")
    assert satellite.norad_id is None
    assert str(satellite) == "Satellite Sentinel-6, NORAD ID: Unknown"

def test_satellite_initialization_with_tle():
    """Test initialization using a TLE."""
    satellite = Satellite(tle=SAMPLE_TLE)
    assert satellite.name == "ISS (ZARYA)"
    assert satellite.norad_id == 25544
    assert satellite.epoch_utc_dt is not None
    assert satellite.state_vector is not None
    # Further assertions can be made on state_vector if expected values are known

def test_satellite_initialization_with_norad_id_auto_download():
    """Test initialization with NORAD ID and auto_download flag."""
    norad_id = 25544
    with patch('polaris.satellite.requests.get') as mock_get:
        # Mock the TLE download response
        mock_get.return_value.status_code = 200
        mock_get.return_value.text = SAMPLE_TLE.strip()

        satellite = Satellite(norad_id=norad_id, auto_download=True)
        assert satellite.name == "ISS (ZARYA)"
        assert satellite.norad_id == norad_id
        assert satellite.epoch_utc_dt is not None
        assert satellite.state_vector is not None
        mock_get.assert_called_once_with(
            f"https://celestrak.com/NORAD/elements/gp.php?CATNR={norad_id}&FORMAT=TLE"
        )

def test_satellite_initialization_with_norad_id_auto_download_failure():
    """Test initialization with NORAD ID and auto_download flag when download fails."""
    norad_id = 99999  # Assuming this NORAD ID does not exist
    with patch('polaris.satellite.requests.get') as mock_get:
        # Mock the TLE download response as failure
        mock_get.return_value.status_code = 404
        mock_get.return_value.text = ""

        with pytest.raises(ValueError, match=f"Could not download TLE for NORAD ID {norad_id}"):
            Satellite(norad_id=norad_id, auto_download=True)

def test_satellite_initialization_with_keplerian():
    """Test initialization using Keplerian elements."""
    satellite = Satellite(keplerian=SAMPLE_KEPLERIAN)
    assert satellite.name is None
    assert satellite.norad_id is None
    assert satellite.epoch_utc_dt == SAMPLE_KEPLERIAN['epoch']
    assert satellite.state_vector is not None
    assert satellite.state_vector.position == SAMPLE_STATE_VECTOR.position  # Depending on conversion
    assert satellite.state_vector.velocity == SAMPLE_STATE_VECTOR.velocity  # Depending on conversion

def test_satellite_initialization_with_state_vector():
    """Test initialization using a state vector."""
    satellite = Satellite(state_vector=SAMPLE_STATE_VECTOR)
    assert satellite.name is None
    assert satellite.norad_id is None
    assert satellite.epoch_utc_dt == SAMPLE_STATE_VECTOR.epoch
    assert satellite.state_vector == SAMPLE_STATE_VECTOR

def test_satellite_propagate_rk89():
    """Test the propagate method with RK89."""
    satellite = Satellite(state_vector=SAMPLE_STATE_VECTOR)
    original_position = satellite.state_vector.position
    original_velocity = satellite.state_vector.velocity
    original_epoch = satellite.state_vector.epoch

    # Propagate for 1 hour
    prop_duration = dt.timedelta(hours=1)
    satellite.propagate(duration=prop_duration, method='rk89', dt_step=10.0)

    # Check that epoch has been updated
    expected_epoch = original_epoch + prop_duration
    assert satellite.state_vector.epoch == expected_epoch

    # Check that state vector has been updated (positions and velocities should change)
    assert satellite.state_vector.position != original_position
    assert satellite.state_vector.velocity != original_velocity

def test_satellite_propagate_without_state_vector():
    """Test propagation raises error when state vector is not initialized."""
    satellite = Satellite(name="Sentinel-6", norad_id=12345)
    with pytest.raises(ValueError, match="State vector is not initialized."):
        satellite.propagate(duration=dt.timedelta(hours=1), method='rk89', dt_step=10.0)

def test_satellite_invalid_initialization():
    """Test that initializing Satellite without required parameters raises ValueError."""
    with pytest.raises(ValueError, match="Initialization requires at least a name or norad_id."):
        Satellite()

def test_satellite_initialization_with_insufficient_keplerian_elements():
    """Test initialization with incomplete Keplerian elements raises ValueError."""
    incomplete_keplerian = {
        'semi_major_axis': 7000.0,
        'eccentricity': 0.001,
        # Missing inclination, raan, arg_of_perigee, true_anomaly, epoch
    }
    with pytest.raises(ValueError, match="Missing Keplerian element: inclination"):
        Satellite(keplerian=incomplete_keplerian)

def test_get_epoch_iso_string():
    """Test the get_epoch_iso_string method."""
    epoch = dt.datetime(2024, 4, 27, 12, 0, 0, tzinfo=dt.timezone.utc)
    satellite = Satellite(state_vector=SAMPLE_STATE_VECTOR)
    assert satellite.get_epoch_iso_string() == epoch.isoformat(sep="T")

    # Test with different separator
    assert satellite.get_epoch_iso_string(sep=" ") == epoch.isoformat(sep=" ")

def test_satellite_state_vector_update_after_propagation():
    """Ensure that the state vector is updated correctly after propagation."""
    satellite = Satellite(state_vector=SAMPLE_STATE_VECTOR)
    original_state = satellite.state_vector

    # Propagate for 30 minutes
    prop_duration = dt.timedelta(minutes=30)
    satellite.propagate(duration=prop_duration, method='rk89', dt_step=10.0)

    # Ensure epoch has been updated
    expected_epoch = original_state.epoch + prop_duration
    assert satellite.state_vector.epoch == expected_epoch

    # Ensure position and velocity have changed
    assert satellite.state_vector.position != original_state.position
    assert satellite.state_vector.velocity != original_state.velocity

def test_satellite_initialization_from_state_vector_propagate():
    """Test initialization from state vector and immediate propagation."""
    satellite = Satellite(state_vector=SAMPLE_STATE_VECTOR)
    
    # Propagate for 15 minutes
    prop_duration = dt.timedelta(minutes=15)
    satellite.propagate(duration=prop_duration, method='rk89', dt_step=10.0)
    
    # Check that epoch is updated correctly
    expected_epoch = SAMPLE_STATE_VECTOR.epoch + prop_duration
    assert satellite.state_vector.epoch == expected_epoch
    
    # Check that position and velocity have changed
    assert satellite.state_vector.position != SAMPLE_STATE_VECTOR.position
    assert satellite.state_vector.velocity != SAMPLE_STATE_VECTOR.velocity

def test_satellite_propagate_invalid_method():
    """Test that propagating with an unsupported method raises NotImplementedError."""
    satellite = Satellite(state_vector=SAMPLE_STATE_VECTOR)
    with pytest.raises(NotImplementedError, match="Propagation method invalid_method is not implemented."):
        satellite.propagate(duration=dt.timedelta(hours=1), method='invalid_method', dt_step=10.0)
