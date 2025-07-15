#!/usr/bin/env python
# Copyright Salient Predictions 2025

"""Tests for geo_api.

Usage:
```
python -m pytest -s -v tests/test_geo_api.py
```

"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import salientsdk as sk

DST_DIR = "test_geo"


def test_geo_shapefile(session):  # noqa: D103
    geo_name = "Colorado"
    geo_file = sk.upload_bounding_box(
        west=-109.1,
        east=-102.0,
        south=37.8,
        north=41.0,
        geoname=geo_name,
        destination=DST_DIR,
        session=session,
        force=False,
        verbose=False,
    )
    loc = sk.Location(shapefile=geo_file)

    var_names = ["elevation", "population"]
    var_file = sk.geo(loc=loc, variables=var_names, start="2020-01-01", end="2021-01-01")
    var_data = xr.open_dataset(var_file)

    # Test dataset has expected variables
    for name in var_names:
        assert name in var_data

    # Test dataset has expected times
    expected_times = np.arange(np.datetime64("2020", "Y"), np.datetime64("2022", "Y")).astype(
        "datetime64[ns]"
    )
    actual_times = var_data.time.data
    np.testing.assert_array_equal(actual_times, expected_times)

    # Test with invalid variable names:
    with pytest.raises(AssertionError):
        sk.geo(loc=loc, variables=["invalid_name", "another invalid name"])

    with pytest.raises(AssertionError):
        sk.geo(loc=loc, variables="invalid_name1,invalid_name2")


def test_geo_location_file(session):  # noqa: D103
    lats = [37.7749, 33.9416, 32.7336]
    lons = [-122.4194, -118.4085, -117.1897]
    iata = ["SFO", "LAX", "SAN"]
    name = "CA_Airports"

    loc_file = sk.upload_location_file(
        lats=lats,
        lons=lons,
        names=iata,
        geoname=name,
        destination=DST_DIR,
        session=session,
        force=True,
    )
    loc = sk.Location(location_file=loc_file)

    elv_name = "elevation"
    elv_file = sk.geo(loc=loc, variables=[elv_name])
    elv_data = xr.open_dataset(elv_file)

    assert elv_name in elv_data


def test_geo_latlon(session):  # noqa: D103
    # SFO:
    lat = 37.7749
    lon = -122.4194

    loc = sk.Location(lat=lat, lon=lon)

    elv_name = "elevation"
    elv_file = sk.geo(loc=loc, variables=[elv_name])
    elv_data = xr.open_dataset(elv_file)

    assert elv_name in elv_data


def test_geo_format(session):  # noqa: D103
    # SFO:
    lat = 37.7749
    lon = -122.4194

    loc = sk.Location(lat=lat, lon=lon)

    elv_name = "elevation"
    elv_file = sk.geo(loc=loc, variables=[elv_name], format="csv")
    elv_data = pd.read_csv(elv_file)

    assert "Elevation (m)" in elv_data.columns


def run_add_geo(loc, **kwargs):
    """Run add_geo for a single location, hourly daily historicals."""
    start = "2022-07-01"
    end = "2022-07-31"
    var = "temp"
    fld = "vals"

    meth = sk.data_timeseries(
        loc=loc, variable=var, field=fld, start=start, end=end, frequency="hourly", **kwargs
    )
    metd = sk.data_timeseries(
        loc=loc, variable=var, field=fld, start=start, end=end, frequency="daily", **kwargs
    )
    meth = xr.open_dataset(meth)
    metd = xr.open_dataset(metd)

    metha = sk.geo_api.add_geo(meth, loc=loc, variables="elevation", **kwargs)
    metda = sk.geo_api.add_geo(metd, loc=loc, variables="elevation", **kwargs)

    # Make sure that elevation doesn't have any NA values
    assert metha.elevation.notnull().all()
    assert metda.elevation.notnull().all()

    # Make sure that add_elevation doesn't change the lat/lon values
    assert metha.lat.equals(meth.lat)
    assert metda.lat.equals(metd.lat)

    assert metha.lon.equals(meth.lon)
    assert metda.lon.equals(metd.lon)


def test_add_geo(session):
    """Test add_geo function."""
    dst = "test_add_geo"

    force = False
    locp = sk.Location(lat=42.1, lon=-73.1)
    locs = sk.Location(
        shapefile=sk.upload_bounding_box(
            north=32.75 + 0.5,
            south=32.75 - 0.5,
            east=-96.75 + 0.5,
            west=-96.75 - 0.5,
            geoname="test_add_geo_shp",
            destination=dst,
            session=session,
            force=force,
        )
    )
    locl = sk.Location(
        location_file=sk.upload_location_file(
            lats=[44.3327, 31.2194, 34.830556],
            lons=[-69.781, -102.1922, -118.398056],
            names=["3 Corners ME", "Roadrunner TX", "Solar Star CA"],
            geoname="test_add_geo_loc",
            destination=dst,
            session=session,
            force=force,
        )
    )

    run_add_geo(locp, destination=dst, session=session)
    run_add_geo(locs, destination=dst, session=session)
    run_add_geo(locl, destination=dst, session=session)


def local_test():
    """Execute functions without the overhead of pytest.

    python tests/test_geo_api.py
    """
    # sk.constants.URL = "https://api-beta.salientpredictions.com/"
    session = sk.login()
    sk.set_file_destination(DST_DIR)
    test_add_geo(session)


if __name__ == "__main__":
    local_test()
