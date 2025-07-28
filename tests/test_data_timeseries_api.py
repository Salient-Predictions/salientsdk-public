#!/usr/bin/env python
# Copyright Salient Predictions 2025

"""Test for the data_timeseries module.

Usage example:
```
python -m pytest -s -v tests/test_data_timeseries_api.py
```

"""

import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import salientsdk as sk
from tests.conftest import get_test_dir

VERBOSE = False

TEST_START = "2020-01-01"
TEST_DEST = get_test_dir(__file__)


def test_data_timeseries_success(session, force=True):  # noqa: D103
    loc = sk.Location(lat=42, lon=-73)
    fil = sk.data_timeseries(
        loc=loc,
        variable="temp",
        field="all",
        start=TEST_START,
        end="2020-12-31",
        format="nc",
        frequency="daily",
        destination=TEST_DEST,
        force=force,
        session=session,
        verbose=VERBOSE,
    )

    assert os.path.exists(fil)
    assert str(fil).startswith(str(TEST_DEST))

    ts = xr.open_dataset(fil)
    assert "time" in ts.coords
    assert "anom" in ts.data_vars

    # to test stack_observed, we need a synthetic forecast timeseries
    # with forecast_date and lead defined so we can match it.
    fcst = mock_stacked_forecast(TEST_START)
    stk = sk.stack_history(fil, fcst.forecast_date, fcst.lead)

    # Test 1: Coordinate dimensions match
    assert set(fcst.forecast_date.values) == set(
        stk.forecast_date.values
    ), "forecast_date coordinates don't match"
    assert set(fcst.lead.values) == set(stk.lead.values), "lead coordinates don't match"

    # Test 2: Check stk's anom dimensions
    assert set(stk.anom.dims) == set(
        ["forecast_date", "lead", "location"]
    ), "stk.anom has incorrect dimensions"

    # Test 3: Check stk's time coordinates
    assert set(stk.time.dims) == set(
        ["forecast_date", "lead"]
    ), "stk.time has incorrect dimensions"

    # Test 4: Check climatology variables dimensions
    assert set(stk.stdv.dims) == set(["dayofyear", "location"]), "stdv has incorrect dimensions"
    assert "forecast_date" not in stk.stdv.dims, "stdv should not have forecast_date dimension"
    assert "lead" not in stk.stdv.dims, "stdv should not have lead dimension"

    # Test 5: Check clim dimensions
    assert set(stk.clim.dims) == set(["dayofyear", "location"]), "clim has incorrect dimensions"

    # Test 6: Check trend dimensions
    assert set(stk.trend.dims) == set(["dayofyear", "location"]), "trend has incorrect dimensions"

    # Test 7: Check anom_qnt dimensions
    assert set(stk.anom_qnt.dims) == set(
        ["quantile", "dayofyear", "location"]
    ), "anom_qnt has incorrect dimensions"

    # Test 8: Check time values are correctly derived from forecast_date + lead
    for i, fd in enumerate(stk.forecast_date.values):
        for j, ld in enumerate(stk.lead.values):
            expected_time = pd.Timestamp(fd) + pd.Timedelta(ld) - pd.Timedelta("1D")
            actual_time = stk.time.values[i, j]
            assert (
                pd.Timestamp(actual_time) == expected_time
            ), f"time at {i},{j} doesn't match forecast_date + lead"

    # Test 9: Check location coordinates are properly set
    assert "lat" in stk.coords, "lat coordinate missing in stacked dataset"
    assert "lon" in stk.coords, "lon coordinate missing in stacked dataset"

    # Test 10: Check data completeness - no missing values in key variables
    assert not np.isnan(stk.anom.values).any(), "Missing values in anom variable"
    assert not np.isnan(stk.clim.values).any(), "Missing values in clim variable"


def mock_stacked_forecast(start_date):
    """Create a mock forecast dataset with forecast_date and lead dimensions."""
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d")

    forecast_dates = [start_date, start_date + timedelta(days=7), start_date + timedelta(days=14)]
    leads = [timedelta(days=i) for i in range(1, 16)]  # 1 to 15 days
    lats = np.linspace(30, 45, 10)
    lons = np.linspace(-120, -100, 12)

    anom = np.random.normal(15, 5, size=(len(forecast_dates), len(leads), len(lats), len(lons)))
    mock_forecast = xr.Dataset(
        data_vars={
            "anom": (["forecast_date", "lead", "lat", "lon"], anom),
        },
        coords={
            "forecast_date": forecast_dates,
            "lead": leads,
            "lat": lats,
            "lon": lons,
        },
    )

    return mock_forecast


def test_data_timeseries_multi_success(session):
    """Test vectorization in data_timeseries."""
    loc = sk.Location(lat=42, lon=-73)
    fil = sk.data_timeseries(
        loc=loc,
        variable="temp,precip",
        field="all",
        start=TEST_START,
        end="2020-12-31",
        format="nc",
        frequency="weekly",
        destination=TEST_DEST,
        force=False,
        session=session,
        verbose=VERBOSE,
    )

    assert "temp" in fil["variable"].values
    assert "precip" in fil["variable"].values

    for fn in fil["file_name"]:
        assert os.path.exists(fn)
        assert str(fn).startswith(str(TEST_DEST))

    fld = ["anom", "vals"]
    mds = sk.load_multihistory(fil, fld)
    assert "temp_anom" in mds
    assert "precip_anom" in mds
    assert "temp" in mds
    assert "precip" in mds
    assert "time" in mds


def test_data_timeseries_apikey():  # noqa: D103
    """Test with API key, and no pre-existing session."""
    sk.set_current_session(None)

    loc = sk.Location(lat=42, lon=-73)
    fil = sk.data_timeseries(
        loc=loc,
        variable="temp",
        field="anom",
        start=TEST_START,
        end="2020-12-31",
        format="nc",
        frequency="monthly",
        destination=TEST_DEST,
        force=True,
        session=None,
        apikey="SALIENT_APIKEY",
        verbose=VERBOSE,
    )

    assert os.path.exists(fil)
    assert str(fil).startswith(str(TEST_DEST))

    ts = xr.open_dataset(fil)
    assert "time" in ts.coords
    assert "anom" in ts.data_vars


def test_data_timeseries_custom_quantity(session, force=False):
    """Make sure the custom quantity field works."""
    # To get a list of valid custom quantities:
    # sk.user_files(type="derived", session=session, destination=TEST_DEST)

    loc = sk.Location(lat=42, lon=-73)
    qnt = "cold_spell"
    fil = sk.data_timeseries(
        loc=loc,
        custom_quantity=qnt,
        start=TEST_START,
        end="2020-12-31",
        format="nc",
        frequency="daily",
        destination=TEST_DEST,
        force=force,
        session=session,
        verbose=VERBOSE,
    )

    assert os.path.exists(fil)
    assert str(fil).startswith(str(TEST_DEST))

    ts = xr.open_dataset(fil)

    assert qnt in ts

    if VERBOSE:
        print(ts)


def test_data_timeseries_failure(session):  # noqa: D103
    loc = sk.Location(lat=42, lon=-73)
    with pytest.raises(AssertionError) as ae:
        fil = sk.data_timeseries(
            loc=loc,
            variable="temp",
            field="invalid_field",
            start=TEST_START,
            end="2020-12-31",
            format="nc",
            frequency="daily",
            force=True,
            destination=TEST_DEST,
            session=session,
            verbose=VERBOSE,
        )
    assert "invalid_field not in" in str(ae.value)

    with pytest.raises(AssertionError) as ve:
        fil = sk.data_timeseries(
            loc=loc,
            variable="temp,invalid_var",
            field="vals",
            start=TEST_START,
            end="2020-12-31",
            format="nc",
            frequency="hourly",
            force=True,
            session=session,
            destination=TEST_DEST,
            verbose=VERBOSE,
        )
    assert "invalid_var" in str(ve.value)

    with pytest.raises(ValueError) as he:
        fil = sk.data_timeseries(
            loc=loc,
            variable="temp",
            field="anom",  # anom not available for hourly
            start=TEST_START,
            end="2020-12-31",
            format="nc",
            frequency="hourly",
            force=True,
            session=session,
            destination=TEST_DEST,
            verbose=VERBOSE,
        )
    assert "hourly" in str(he.value)


def test_extrapolate_trend(session):
    """Extrapolate."""
    # The variables to project into the future:
    sk.set_file_destination(TEST_DEST)

    start_date = "2020-01-01"
    end_date = "1YE"
    force = False
    # force = True
    var = ["tmax", "precip"]
    loc = sk.Location(
        location_file=[
            sk.upload_location_file(
                lats=[55.364, 51.565, 46.698, 35.389],
                lons=[25.252, 25.273, 24.966, 31.722],
                names=["DXB", "DOH", "RUH", "AMM"],
                geoname="airports_mena",
                destination=TEST_DEST,
                force=force,
                verbose=VERBOSE,
                session=session,
            ),
            # TODO: upload_location_file should check for out-of-bounds lats
            sk.upload_location_file(
                lats=[19.436, 8.979, 23.034, 18.430],
                lons=[-99.072, -79.383, -82.409, -69.669],
                names=["MEX", "PTY", "HAV", "SDQ"],
                geoname="airports_cac",
                destination=TEST_DEST,
                force=force,
                verbose=VERBOSE,
                session=session,
            ),
        ]
    )
    loc_data = loc.load_location_file()

    trend = sk.data_timeseries_api.extrapolate_trend(
        loc=loc,
        variable=var,
        start=start_date,
        end=end_date,
        force=force,
        stdv_mult=0,
        destination=TEST_DEST,
        session=session,
        verbose=VERBOSE,
    )

    # Do the same thing, but at +2sd to identify extremes
    trend_stdv = sk.data_timeseries_api.extrapolate_trend(
        loc=loc,
        variable=var,
        start=start_date,
        end=end_date,
        force=force,
        stdv_mult=2,
        destination=TEST_DEST,
        session=session,
        verbose=VERBOSE,
    )

    # Verify dataset structure
    assert isinstance(trend, xr.Dataset)
    assert set(trend.dims) == {"location", "time"}
    assert set(trend.coords) == {"location", "time", "lat", "lon"}
    assert set(trend.data_vars) == set(var)

    # Verify dimensions
    assert trend.sizes["location"] == len(loc_data)
    assert trend.sizes["time"] == 366  # Full year (2020 was leap year)

    # Verify coordinates
    assert set(trend.location.values) == set(loc_data["name"])
    assert pd.Timestamp(trend.time.values[0]) == pd.Timestamp("2020-01-01")
    assert pd.Timestamp(trend.time.values[-1]) == pd.Timestamp("2020-12-31")

    # Verify data variables exist and have correct shape
    for v in var:
        assert trend[v].dims == ("time", "location")
        assert not trend[v].isnull().any().values  # No missing values

    # by definition, the elements of the +2sd trend should be greater than the mean
    trend_dif = trend_stdv - trend
    assert (trend_dif > 0).all().values, "Expected all elements of trend_dif to be positive"


def local_test():
    """Execute functions without the overhead of pytest.

    python tests/test_data_timeseries_api.py
    """
    # sk.constants.URL = "https://api-beta.salientpredictions.com/"
    session = sk.login()
    # test_extrapolate_trend(session)
    test_data_timeseries_success(session, force=False)
    # test_data_timeseries_custom_quantity(session, force=False)


if __name__ == "__main__":
    local_test()
