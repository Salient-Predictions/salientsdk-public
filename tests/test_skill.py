#!/usr/bin/env python
# Copyright Salient Predictions 2025

"""Tests for skill.py.

Usage:
```
python -s -m pytest tests/test_skill.py
```

"""

import numpy as np
import pandas as pd
import properscoring as ps
import xarray as xr
from scipy.stats import norm

import salientsdk as sk

DST = "test_skill"
VAR = "temp"


def test_crps_scientific(y_obs=0.3, mu=-0.4, sigma=0.8, n_quantiles=30):
    """Compare analytically computed CRPS for a Gaussian RV.

    Note that there is no "session" variable here since there is no API access.
    """
    quantiles = np.around(
        np.linspace(0 + 1 / (n_quantiles + 1), 1 - 1 / (n_quantiles + 1), n_quantiles),
        decimals=3,
    )

    # create a stub dataset with quantile predictions computed from a Gaussian
    y_pred = xr.DataArray(
        data=norm.ppf(quantiles, loc=mu, scale=sigma),
        dims=["quantiles"],
        coords=dict(quantiles=quantiles),
    )
    y_true = xr.DataArray(
        data=np.array(y_obs),
    )

    crps_analytic = ps.crps_gaussian(y_obs, mu=mu, sig=sigma)
    crps_qnt = sk.skill.crps(y_true, y_pred)

    np.testing.assert_allclose(crps_analytic, crps_qnt.values, rtol=1e-2)

    # Also calculate a "bad" CRPS for comparison
    y_bad = xr.DataArray(
        data=norm.ppf(quantiles, loc=2 * mu, scale=2 * sigma),
        dims=["quantiles"],
        coords=dict(quantiles=quantiles),
    )
    crps_bad = sk.skill.crps(y_bad, y_pred)

    assert crps_bad > crps_qnt

    crps_rel = sk.skill.crpss(crps_qnt, crps_bad)

    assert crps_rel > 0


def test_crps_ensemble():
    """Test the ensemble CRPS calculation against the properscoring version."""
    y_true = xr.DataArray(0.5)
    y_pred = xr.DataArray(np.random.normal(0, 1, 100), dims=["ensemble"])
    actual = sk.skill._crps_ensemble_core(y_true, y_pred)
    expected = ps.crps_ensemble(y_true.values, y_pred.values)
    np.testing.assert_allclose(actual, expected)


def test_crps_vectorized(session, comprehensive=False, force=False, verbose=False):
    """Test the CRPS calculation with actual API-returned data."""
    # classical lead-denominated forecast
    run_skill(session, "sub-seasonal", "daily", force, verbose)

    # calendar-locked time-denominated forecast
    run_skill(session, "weekly", "daily", force, verbose)

    # multi-lead all = sub-seasonal, seasonal, and long-range in one file
    run_skill(session, "all", "daily", force, verbose)

    if comprehensive:
        # This is a lot of data to download for a unit test,
        # so we resere these test cases for deep development testing.

        # classical lead-denominated forecast
        run_skill(session, "seasonal", "daily", force, verbose)
        run_skill(session, "long-range", "daily", force, verbose)

        # calendar-locked time-denominated forecast, daily
        run_skill(session, "monthly", "daily", force, verbose)
        run_skill(session, "quarterly", "daily", force, verbose)

        # calendar-locked time-denominated forecast, monthly
        run_skill(session, "monthly", "monthly", force, verbose)

        # calendar-locked time-denominated forecast:
        # run_skill(session, "quarterly", "monthly", force)


def run_skill(session, timescale="sub-seasonal", frequency="daily", force=False, verbose=False):
    """Instead of mocking data, use actual API calls to run end-to-end.

    This will test the full vectorization and alignment capabilities of `_calc_skill`.

    Args:
        session: The `session` object.  Provided by the test harness.
        timescale: Passed to `forecast_timeseries` to get the `forecasts`
        frequency: Passed to `data_timeseries` to get the `observations`
        force: download fresh files, even if they are cached
        verbose: print debugging/status information
    """
    lat = 37.7749
    lon = -122.4194
    args = {
        "loc": sk.Location(lat, lon),
        "variable": VAR,
        "field": "anom",
        "force": force,
        "verbose": force,
        "session": session,
        "destination": DST,
    }

    start_date = "2021-01-01"
    end_date = "2021-01-31"
    # There is no 2020 blend forecast, so some calls will fail if strict=True
    # start_date = "2020-01-01"
    # end_date = "2020-02-28"

    duration = {
        "sub-seasonal": 7 * 5 + 2,
        "weekly": 7 * 5 + 2,
        "seasonal": 31 * 3,
        "monthly": 31 * 3,
        "long-range": 366,
        "quarterly": 366,
        "all": 366,
    }[timescale]
    date_range = pd.date_range(start=start_date, end=end_date, freq="W").strftime("%Y-%m-%d")

    fcst = sk.forecast_timeseries(
        **args,
        date=date_range.tolist(),
        timescale=timescale,
        reference_clim="30_yr",  # this is the climatology used by data_timeseries
        strict=False,
    )

    hist = sk.data_timeseries(
        **args,
        start=np.datetime64(start_date) - np.timedelta64(3, "D"),
        end=np.datetime64(end_date) + np.timedelta64(duration, "D"),
        frequency=frequency,
    )
    hist = xr.open_dataset(hist)

    # minimal single-forecast test:
    # first_fcst = xr.open_dataset(fcst.file_name[0])
    # skill_fcst = sk.skill.crps(observations=hist, forecasts=first_fcst)

    # Vectorized test: multiple forecasts:
    skill_fcst = sk.skill.crps(observations=hist, forecasts=fcst)

    if verbose:
        print(f"--- CRPS {timescale} {frequency} forecast at {lat}, {lon} ---")
        print(skill_fcst)

    assert hist.lat == skill_fcst.lat
    assert hist.lon == skill_fcst.lon

    assert np.all(skill_fcst >= 0)

    assert sk.skill._find_coord(skill_fcst, "lead")

    if isinstance(skill_fcst, xr.DataArray):
        assert skill_fcst.name == "crps"

    return skill_fcst


def main():
    """Run the tests, without the overhead of the testing infrastructure.

    python tests/test_skill.py
    """
    # sk.constants.URL = "https://api-beta.salientpredictions.com/"
    # sk.constants.set_model_version("v9")
    # session = sk.login()
    # test_crps_vectorized(session)

    # run_skill(session, "weekly", "daily", force=False, verbose=True)
    test_crps_scientific()

    # test_crps_vectorized(session=session, comprehensive=True, force=False, verbose=True)
    # run_skill(session, "sub-seasonal", "daily", force=False, verbose=True)
    # run_skill(session, "all", "daily", verbose=True)


if __name__ == "__main__":
    main()
