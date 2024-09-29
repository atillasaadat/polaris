# tests/test_satellite.py
import datetime as dt

import pytest
from polaris.satellite import Satellite


def test_satellite_initialization():
    """Test the initialization of the Satellite class."""
    satellite = Satellite(
        name="Sentinel-6", norad_id=12345, epoch_utc_dt=dt.datetime(2023, 9, 28, 14, 30)
    )
    assert satellite.name == "Sentinel-6"
    assert satellite.norad_id == 12345
    assert satellite.epoch_utc_dt == dt.datetime(2023, 9, 28, 14, 30)


def test_satellite_str():
    """Test the __str__ method."""
    satellite = Satellite(name="Sentinel-6", norad_id=12345)
    assert str(satellite) == "Satellite Sentinel-6, NORAD ID: 12345"


def test_satellite_repr():
    """Test the __repr__ method."""
    satellite = Satellite(name="Sentinel-6", norad_id=12345)
    assert (
        repr(satellite)
        == "Satellite(name='Sentinel-6', norad_id=12345, epoch_utc_dt=None)"
    )


def test_satellite_without_norad_id():
    """Test initialization without a NORAD ID."""
    satellite = Satellite(name="Sentinel-6")
    assert satellite.norad_id is None
    assert str(satellite) == "Satellite Sentinel-6, NORAD ID: Unknown"
