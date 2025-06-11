#!/usr/bin/env python
# Copyright Salient Predictions 2025

"""Test for the forecast_timeseries_api module.

Usage example:
```
python -m pytest -s tests/test_forecast_timeseries_api.py
```

"""

import os
from unittest.mock import patch

import pytest
import xarray as xr

import salientsdk as sk
from salientsdk.__main__ import main

verbose = False


TEST_DATE = "2024-01-01"
TEST_DATE2 = "2024-02-01"
TEST_LAT = 42.0
TEST_LON = -73.0
TEST_TIMESCALE = "seasonal"
TEST_DEST = "test_forecast_timeseries"


def test_forecast_timeseries_success(session):  # noqa: D103
    loc = sk.Location(lat=TEST_LAT, lon=TEST_LON)
    fil = sk.forecast_timeseries(
        loc=loc,
        variable="temp",
        field="anom",
        timescale=TEST_TIMESCALE,
        date=TEST_DATE,
        force=True,
        session=session,
        verbose=verbose,
        destination=TEST_DEST,
    )

    assert os.path.exists(fil)

    ts = xr.open_dataset(fil, decode_timedelta=True)
    assert "forecast_date_monthly" in ts.coords
    assert "anom_monthly" in ts.data_vars


def test_forecast_timeseries_multi_success(session):  # noqa: D103
    """Vectorize a call and then stack by forecast_date."""
    loc = sk.Location(lat=TEST_LAT, lon=TEST_LON)
    fil = sk.forecast_timeseries(
        loc=loc,
        variable="temp",
        field="anom_ens",
        # field="anom",
        model="noaa_gefs",
        # model="blend",
        timescale="daily",
        # timescale="long-range",
        date=[TEST_DATE, TEST_DATE2],
        force=True,
        session=session,
        verbose=verbose,
        destination=TEST_DEST,
    )

    assert os.path.exists(fil["file_name"][0])
    assert os.path.exists(fil["file_name"][1])

    assert TEST_DATE in fil["date"].values
    assert TEST_DATE2 in fil["date"].values

    # stk = sk.forecast_timeseries_api.stack_forecast(fil)
    # assert "forecast_date" in stk
    # assert len(stk.forecast_date) == 2


def test_forecast_timeseries_failure(session):  # noqa: D103
    loc = sk.Location(lat=TEST_LAT, lon=TEST_LON)
    with pytest.raises(AssertionError) as ae:
        fil = sk.forecast_timeseries(
            loc=loc,
            variable="temp",
            field="invalid_field",
            date=TEST_DATE,
            timescale=TEST_TIMESCALE,
            force=True,
            session=session,
            verbose=verbose,
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


if __name__ == "__main__":
    run_local()
