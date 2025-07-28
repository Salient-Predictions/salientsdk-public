#!/usr/bin/env python
# Copyright Salient Predictions 2025

"""Test for the forecast_timeseries_api module.

Usage example:
```
python -m pytest -s -v tests/test_forecast_timeseries_api.py
```

"""

import os
from unittest.mock import patch

import dask.array as da
import numpy as np
import pytest
import xarray as xr

import salientsdk as sk
from salientsdk.__main__ import main
from tests.conftest import get_test_dir

VERBOSE = False
FORCE = False


TEST_DATE = "2024-01-01"
TEST_DATE2 = "2024-02-01"
TEST_LAT = 42.0
TEST_LON = -73.0
TEST_TIMESCALE = "seasonal"
TEST_DEST = get_test_dir(__file__)


def test_forecast_timeseries_success(session):  # noqa: D103
    loc = sk.Location(lat=TEST_LAT, lon=TEST_LON)
    fil = sk.forecast_timeseries(
        loc=loc,
        variable="temp",
        field="anom",
        timescale=TEST_TIMESCALE,
        date=TEST_DATE,
        force=FORCE,
        session=session,
        verbose=VERBOSE,
        destination=TEST_DEST,
    )

    assert os.path.exists(fil)

    ts = xr.open_dataset(fil, decode_timedelta=True)
    assert "forecast_date_monthly" in ts.coords
    assert "anom_monthly" in ts.data_vars


def test_forecast_timeseries_custom_quantity(session):
    """Make sure custom_quantity argument works."""
    # Get a list of valid custom quantities:
    # src = sk.user_files(type="derived", session=session, destination=TEST_DEST)

    loc = sk.Location(lat=TEST_LAT, lon=TEST_LON)
    qnt = "cold_spell"
    fil = sk.forecast_timeseries(
        loc=loc,
        # variable="temp",  # System should be smart enough to ignore default value
        custom_quantity=qnt,
        model="gem",
        # field="anom", # System should be smart enough to ignore default value
        timescale="daily",
        date=TEST_DATE,
        force=FORCE,
        session=session,
        verbose=VERBOSE,
        destination=TEST_DEST,
    )
    assert os.path.exists(fil)
    ts = xr.open_dataset(fil, decode_timedelta=True)

    assert qnt in ts

    if VERBOSE:
        print(ts)


def test_forecast_timeseries_multi_success(session):
    """Vectorize a call and then stack by forecast_date."""

    def verify_stacked_forecast(stk):
        """Verify the structure of a stacked forecast dataset."""
        # Check basic structure
        assert "forecast_date" in stk.dims
        assert "lead" in stk.dims
        assert "ensemble" in stk.dims
        assert len(stk.forecast_date) == 2
        assert len(stk.ensemble) > 1  # GEFS is 31
        assert len(stk.lead) > 1  # GEFS is 35
        assert np.datetime64(TEST_DATE) in stk.forecast_date.values
        assert np.datetime64(TEST_DATE2) in stk.forecast_date.values

        # Verify data variable exists and has correct shape
        assert test_field in stk.data_vars
        assert list(stk[test_field].dims) == ["forecast_date", "lead", "ensemble"]

    loc = sk.Location(lat=TEST_LAT, lon=TEST_LON)
    test_field = "anom_ens"
    fil = sk.forecast_timeseries(
        loc=loc,
        variable="temp",
        field=test_field,
        model="noaa_gefs",
        timescale="daily",
        date=[TEST_DATE, TEST_DATE2],
        force=FORCE,
        session=session,
        verbose=VERBOSE,
        destination=TEST_DEST,
    )

    assert os.path.exists(fil["file_name"][0])
    assert os.path.exists(fil["file_name"][1])

    assert TEST_DATE in fil["date"].values
    assert TEST_DATE2 in fil["date"].values

    # Test computed in-memory version -----
    stk_comp = sk.stack_forecast(fil, compute=True)
    verify_stacked_forecast(stk_comp)

    # Test dask delayed version -----
    stk_dask = sk.stack_forecast(fil, compute=False)
    verify_stacked_forecast(stk_dask)

    # With compute=False, we must have a dask array chunked by forecast_date
    assert isinstance(stk_dask.anom_ens.data, da.Array), "compute=False should return a dask array"
    assert hasattr(stk_dask.anom_ens.data, "chunks"), "dask array must have chunks"

    # Get the index of forecast_date dimension
    dims = list(stk_dask[test_field].dims)
    idx = dims.index("forecast_date")
    dim = (1, 1)

    assert stk_dask.anom_ens.data.chunks[idx] == dim, "forecast_date should be chunked (1,1)"


def test_forecast_timeseries_failure(session):  # noqa: D103
    loc = sk.Location(lat=TEST_LAT, lon=TEST_LON)
    with pytest.raises(AssertionError) as ae:
        fil = sk.forecast_timeseries(
            loc=loc,
            variable="temp",
            field="invalid_field",
            date=TEST_DATE,
            timescale=TEST_TIMESCALE,
            force=FORCE,
            session=session,
            verbose=VERBOSE,
            destination=TEST_DEST,
        )


@patch(
    "sys.argv",
    [
        "salientsdk",
        "forecast_timeseries",
        "--lat",
        "42",
        "--lon",
        "-73",
        "--variable",
        "temp",
        "--timescale",
        TEST_TIMESCALE,
        "--date",
        TEST_DATE,
        "--verbose",  # needed to test the output
    ],
)
def test_command_line(capsys):  # noqa: D103
    main()

    captured = capsys.readouterr()
    assert "Data variables:" in captured.out


def run_local():
    """Run the tests, without the overhead of the testing infrastructure.

    python tests/test_forecast_timeseries_api.py
    """
    sk.constants.URL = "https://api-beta.salientpredictions.com/"
    session = sk.login()
    # test_crps_vectorized(session=session, comprehensive=True)
    # test_forecast_timeseries_success(session)
    test_forecast_timeseries_multi_success(session)
    # test_forecast_timeseries_custom_quantity(session)


if __name__ == "__main__":
    run_local()
