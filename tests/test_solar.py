#!/usr/bin/env python
# Copyright Salient Predictions 2025


"""Tests for solar.py.

Usage:
```
python -s -m pytest tests/test_solar.py
```

"""

# import pytest

import unittest.mock as mock

import salientsdk as sk

FAST = True  # When in FAST mode, prefer mocked functions to real API calls

TEST_DESTINATION = "test_solar"
TEST_START = "2022-07-01"
TEST_END = "2022-07-31"
FORCE = False
VERBOSE = False


def run_data_timeseries_solar(session, loc: sk.Location):
    """Test the data_timeseries_solar and sun2power functions."""
    met = sk.solar.data_timeseries_solar(
        loc=loc,
        start=TEST_START,
        end=TEST_END,
        destination=TEST_DESTINATION,
        force=FORCE,
        verbose=VERBOSE,
        session=session,
    )
    check_met(met)
    pwr = sk.solar.run_pvlib_dataset(met)
    check_pwr(pwr)


def test_data_timeseries_solar_latlon(session):
    """Test a single lat/lon pair."""
    # Important to test a southern hemisphere location, since we automatically
    # set tilt and azimuth based on latitude.
    loc = sk.Location(lat=-12.9777, lon=-38.5016)  # Salvador Brazil
    run_data_timeseries_solar(session, loc=loc)


def test_data_timeseries_solar_location(session):
    """Test a location_file."""
    run_data_timeseries_solar(session, loc=solar_data_location())


def test_data_timeseries_solar_shapefile(session):
    """Test a shapefile."""
    run_data_timeseries_solar(session, loc=solar_data_shapefile())


def test_downscale_solar_location(session):
    """Test the data_timeseries_solar and sun2power functions.

    Though we're mocking the heavyweight downscale call, we still need a
    session object for the geo call inside downscale_solar.
    """
    sk.set_file_destination(TEST_DESTINATION)
    with mock.patch(
        "salientsdk.solar.downscale",
        new=sk.downscale_api._downscale_mock if FAST else sk.downscale_api.downscale,
    ):
        met = sk.solar.downscale_solar(
            loc=solar_data_location(),
            date=TEST_START,
            members=3,
            length=10,  # days
            session=session,
            force=FORCE,
        )
    check_met(met)
    pwr = sk.solar.run_pvlib_dataset(met)
    check_pwr(pwr)


# ============ utility functions =====================


def solar_data_shapefile():
    """Create a shapefile location to pass to Solar functions."""
    shp_file = sk.upload_bounding_box(
        north=32.75 + 0.5,
        south=32.75 - 0.5,
        east=-96.75 + 0.5,
        west=-96.75 - 0.5,
        geoname="solar_test_shapefile",
        destination=TEST_DESTINATION,
        force=FORCE,
    )
    return sk.Location(shapefile=shp_file)


def solar_data_location():
    """Create a location_file location to pass to Solar functions."""
    loc_name = "solar_test_locations"
    loc_file = sk.upload_location_file(
        lats=[44.3327, 31.2194, 34.830556],
        lons=[-69.781, -102.1922, -118.398056],
        names=["3 Corners ME", "Roadrunner TX", "Solar Star CA"],
        rated_capacity=[109, 400, 579],  # MW
        geoname=loc_name,
        destination=TEST_DESTINATION,
        force=FORCE,
    )
    return sk.Location(location_file=loc_file)


def check_met(met):
    """Validate the met data."""
    assert met["dni"].min().values >= 0
    assert met["dhi"].min().values >= 0
    assert met["tsi"].min().values >= 0
    assert met["wspd"].min().values >= 0
    assert "temp" in met


def check_pwr(pwr):
    """Validate the power data."""
    assert pwr["dc"].min().values >= 0
    assert pwr["ac"].min().values >= 0
    assert pwr["effective_irradiance"].min().values >= 0


def main():
    """Run the tests, without the overhead of the testing infrastructure.

    python tests/test_solar.py
    """
    sk.constants.URL = "https://api-beta.salientpredictions.com/"
    session = sk.login()
    # test_data_timeseries_solar_location(session)
    test_data_timeseries_solar_shapefile(session)
    # test_data_timeseries_solar_latlon(session)


if __name__ == "__main__":
    main()
