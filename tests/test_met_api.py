#!/usr/bin/env python
# Copyright Salient Predictions 2025

"""Tests for met_api.py.

Usage:
```
python -s -m pytest tests/test_met_api.py
```

"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import salientsdk as sk

DST_DIR = "test_met_api"

ID_SINGLE = "USW00013874"
ID_VECTOR = [ID_SINGLE, "USW00014739"]

NAME_SINGLE = "ATL"
NAME_VECTOR = [NAME_SINGLE, "BOS"]

VERBOSE = False


def test_met(session, force=False):
    """Test met_stations AND met_observations."""
    # the "XXX" location is 200 mi offshore, and should have no met stations
    lats = [37.7749, 33.9416, 32.7336]  # , 37.7749]
    lons = [-122.4194, -118.4085, -117.1897]  # , -126.0]
    iata = ["SFO", "LAX", "SAN"]  # ,"XXX"]
    name = "met_locations"

    loc_request = sk.Location(
        location_file=sk.upload_location_file(
            lats=lats,
            lons=lons,
            names=iata,
            geoname=name,
            session=session,
            force=force,
            destination=DST_DIR,
            verbose=VERBOSE,
        )
    )

    variables = ["tmax", "tmin", "precip"]
    start = "2023-01-01"
    end = "2023-03-31"

    loc_stations_csv = sk.met_stations(
        loc=loc_request,
        variables=variables,
        start=start,
        end=end,
        destination=DST_DIR,
        session=session,
        verbose=VERBOSE,
        force=force,
    )

    stat = pd.read_csv(loc_stations_csv)

    # Assert that Lat and Lon columns have the same length as input coordinates
    assert len(stat.Latitude) == len(lats), f"Expected {len(lats)} rows, got {len(stat)}"
    assert len(stat.Longitude) == len(lons), f"Expected {len(lons)} rows, got {len(stat)}"

    # Assert that Lat and Lon columns have no NAs
    assert not stat["Latitude"].isna().any(), "Lat column should not contain any NAs"
    assert not stat["Longitude"].isna().any(), "Lon column should not contain any NAs"

    sk.upload_file(file=loc_stations_csv, verbose=VERBOSE, session=session)
    loc_stations = sk.Location(location_file=loc_stations_csv)

    obs_file = sk.met_observations(
        loc=loc_stations,
        variables=variables,
        start=start,
        end=end,
        destination=DST_DIR,
        session=session,
        verbose=VERBOSE,
        force=force,
    )

    obs = xr.load_dataset(obs_file)

    # Assert dataset dimensions match expectations
    assert obs.sizes["location"] == len(
        lats
    ), f"Expected {len(lats)} locations, got {obs.sizes['location']}"
    assert obs.sizes["time"] == 90, f"Expected 90 days (Jan-Mar), got {obs.sizes['time']}"

    # Assert all requested variables are present
    for var in variables:
        assert var in obs.data_vars, f"Variable '{var}' not found in dataset"

    # Assert coordinates are present and complete
    required_coords = ["location", "time", "lat", "lon", "station"]
    for coord in required_coords:
        assert coord in obs.coords, f"Coordinate '{coord}' not found"
        assert not obs[coord].isnull().any(), f"Coordinate '{coord}' contains null values"

    # Assert time range is correct
    expected_start = pd.Timestamp(start)
    expected_end = pd.Timestamp(end)
    assert (
        obs.time[0].values == expected_start
    ), f"Start time mismatch: expected {expected_start}, got {obs.time[0].values}"
    assert (
        obs.time[-1].values == expected_end
    ), f"End time mismatch: expected {expected_end}, got {obs.time[-1].values}"

    # Assert lat/lon coordinates are reasonable (within expected ranges)
    assert (
        obs.lat.min() >= 30.0 and obs.lat.max() <= 40.0
    ), f"Latitude out of expected range: {obs.lat.min()} to {obs.lat.max()}"
    assert (
        obs.lon.min() >= -125.0 and obs.lon.max() <= -115.0
    ), f"Longitude out of expected range: {obs.lon.min()} to {obs.lon.max()}"

    # Assert data variables have correct dimensions
    for var in variables:
        assert obs[var].dims == (
            "time",
            "location",
        ), f"Variable '{var}' has incorrect dimensions: {obs[var].dims}"

    # Assert we have some actual data (not all NaN)
    for var in variables:
        assert not obs[var].isnull().all(), f"Variable '{var}' is entirely NaN"

    # Assert temperature data is reasonable (if present)
    if "tmax" in obs.data_vars and "tmin" in obs.data_vars:
        assert (obs.tmax >= obs.tmin).all(), "tmax should always be >= tmin"

    # Observed dataset should be in the same format as ERA5 historicals ---------
    hist_src = sk.data_timeseries(
        loc=loc_stations,
        variable=variables,
        field="vals",
        start=start,
        end=end,
        format="nc",
        frequency="daily",
        destination=DST_DIR,
        session=session,
        verbose=VERBOSE,
        force=force,
    )
    era = sk.load_multihistory(hist_src)

    dif = obs - era

    # Check dimensions are the same across all datasets
    obs_dims = set(obs.dims)
    hist_dims = set(era.dims)
    dif_dims = set(dif.dims)

    assert obs_dims == hist_dims, f"Dimension mismatch: obs has {obs_dims}, hist has {hist_dims}"
    assert obs_dims == dif_dims, f"Dimension mismatch: obs has {obs_dims}, dif has {dif_dims}"

    # Check dimension sizes are the same
    for dim in obs_dims:
        assert (
            obs.sizes[dim] == era.sizes[dim]
        ), f"Size mismatch for dimension '{dim}': obs={obs.sizes[dim]}, hist={era.sizes[dim]}"
        assert (
            obs.sizes[dim] == dif.sizes[dim]
        ), f"Size mismatch for dimension '{dim}': obs={obs.sizes[dim]}, dif={dif.sizes[dim]}"

    # Check data variable names are the same
    obs_vars = set(obs.data_vars.keys())
    hist_vars = set(era.data_vars.keys())
    dif_vars = set(dif.data_vars.keys())

    assert (
        obs_vars == hist_vars
    ), f"Data variable mismatch: obs has {obs_vars}, hist has {hist_vars}"
    assert obs_vars == dif_vars, f"Data variable mismatch: obs has {obs_vars}, dif has {dif_vars}"

    # Check that each data variable has the same dimensions
    for var in obs_vars:
        assert (
            obs[var].dims == era[var].dims
        ), f"Dimension mismatch for variable '{var}': obs={obs[var].dims}, hist={era[var].dims}"
        assert (
            obs[var].dims == dif[var].dims
        ), f"Dimension mismatch for variable '{var}': obs={obs[var].dims}, dif={dif[var].dims}"

    if VERBOSE:
        print(obs)
        print(f"✓ Common dimensions: {list(obs_dims)}")
        print(f"✓ Common variables: {list(obs_vars)}")
        print(f"✓ Dataset sizes: {dict(obs.sizes)}")


def test_make_observed_ds(force: bool = False) -> None:
    """Test make_observed_ds."""
    var1 = "temp"
    var2 = "precip"

    ghcnd1 = mock_ghcnd(ID_SINGLE)
    ghcnd2 = mock_ghcnd(ID_VECTOR)
    # ghcnd1 = sk.met_api.get_ghcnd(ID_SINGLE, destination=DST, force=force)
    # ghcnd2 = sk.met_api.get_ghcnd(ID_VECTOR, destination=DST, force=force)

    obs1 = sk.met_api.make_observed_ds(ghcnd1, NAME_SINGLE, var1)
    obs2 = sk.met_api.make_observed_ds(ghcnd2, NAME_VECTOR, var2)

    # Test return types
    assert isinstance(ghcnd1, pd.DataFrame), "ghcnd1 should be a DataFrame"
    assert isinstance(ghcnd2, list), "ghcnd2 should be a list"
    assert all(
        isinstance(df, pd.DataFrame) for df in ghcnd2
    ), "all items in ghcnd2 should be DataFrames"
    assert isinstance(obs1, xr.Dataset), "obs1 should be a Dataset"
    assert isinstance(obs2, xr.Dataset), "obs2 should be a Dataset"

    # Test attributes
    assert obs1.attrs["short_name"] == var1, f"obs1 short_name should be {var1}"
    assert obs2.attrs["short_name"] == var2, f"obs2 short_name should be {var2}"

    # Test location coordinates
    assert list(obs1.location.values) == [NAME_SINGLE], f"obs1 location should be [{NAME_SINGLE}]"
    assert list(obs2.location.values) == NAME_VECTOR, f"obs2 location should be {NAME_VECTOR}"

    # Test dimensions and coordinates
    assert "time" in obs1.dims, "obs1 should have time dimension"
    assert "location" in obs1.dims, "obs1 should have location dimension"
    assert "time" in obs2.dims, "obs2 should have time dimension"
    assert "location" in obs2.dims, "obs2 should have location dimension"

    # Test data variables
    assert "vals" in obs1, "obs1 should have 'vals' variable"
    assert "vals" in obs2, "obs2 should have 'vals' variable"
    assert obs1.vals.dims == (
        "time",
        "location",
    ), "obs1.vals should have dimensions (time, location)"
    assert obs2.vals.dims == (
        "time",
        "location",
    ), "obs2.vals should have dimensions (time, location)"

    # Test for NaN handling
    assert not np.all(np.isnan(obs1.vals)), "obs1 should not be all NaN"
    assert not np.all(np.isnan(obs2.vals)), "obs2 should not be all NaN"


def test_observed_failures():
    """Test expected failures in observed functions."""
    # Test invalid GHCND ID
    # with pytest.raises(requests.exceptions.HTTPError, match="404 Client Error"):
    #    sk.met_api.get_ghcnd("INVALID123", destination=DST)

    # Test mismatched names for vectorized input
    # dfs = sk.met_api.get_ghcnd(ID_VECTOR, destination=DST)
    dfs = mock_ghcnd(ID_VECTOR)
    with pytest.raises(ValueError, match="When obs_df is a list"):
        sk.met_api.make_observed_ds(dfs, name=None, variable="temp")

    # Test mismatched lengths of dataframes and names
    with pytest.raises(AssertionError, match="Length mismatch"):
        sk.met_api.make_observed_ds(dfs, name=[NAME_SINGLE], variable="temp")

    # Test invalid variable name
    with pytest.raises(KeyError):
        sk.met_api.make_observed_ds(dfs[0], name=NAME_SINGLE, variable="not_a_variable")

    # Test empty dataframe
    empty_df = pd.DataFrame()
    with pytest.raises(KeyError):
        sk.met_api.make_observed_ds(empty_df, name=NAME_SINGLE, variable="temp")


def mock_ghcnd(ghcnd_id: str | list[str]):
    """Emulate the get_ghcnd function found in validate.ipynb."""
    if isinstance(ghcnd_id, list):
        return [mock_ghcnd(single_id) for single_id in ghcnd_id]
    elif ghcnd_id == "USW00013874":  # ATL
        return pd.DataFrame(
            [
                {
                    "ghcnd_id": "USW00013874",
                    "time": "2021-04-01",
                    "lat": 33.62972,
                    "lon": -84.44224,
                    "elev": 308.2,
                    "name": "ATLANTA HARTSFIELD JACKSON INTERNATIONAL AIRPORT",
                    "precip": 0.0,
                    "tmax": 11.7,
                    "tmin": 4.4,
                    "wspd": 8.8,
                    "temp": 8.4,
                    "cdd": 0.0,
                    "hdd": 10.3,
                },
                {
                    "ghcnd_id": "USW00013874",
                    "time": "2021-04-02",
                    "lat": 33.62972,
                    "lon": -84.44224,
                    "elev": 308.2,
                    "name": "ATLANTA HARTSFIELD JACKSON INTERNATIONAL AIRPORT",
                    "precip": 0.0,
                    "tmax": 13.9,
                    "tmin": 0.6,
                    "wspd": 4.0,
                    "temp": 6.7,
                    "cdd": 0.0,
                    "hdd": 11.1,
                },
                {
                    "ghcnd_id": "USW00013874",
                    "time": "2021-04-03",
                    "lat": 33.62972,
                    "lon": -84.44224,
                    "elev": 308.2,
                    "name": "ATLANTA HARTSFIELD JACKSON INTERNATIONAL AIRPORT",
                    "precip": 0.0,
                    "tmax": 17.8,
                    "tmin": 1.7,
                    "wspd": 1.3,
                    "temp": 9.1,
                    "cdd": 0.0,
                    "hdd": 8.6,
                },
                {
                    "ghcnd_id": "USW00013874",
                    "time": "2021-04-04",
                    "lat": 33.62972,
                    "lon": -84.44224,
                    "elev": 308.2,
                    "name": "ATLANTA HARTSFIELD JACKSON INTERNATIONAL AIRPORT",
                    "precip": 0.0,
                    "tmax": 22.8,
                    "tmin": 4.4,
                    "wspd": 2.5,
                    "temp": 13.8,
                    "cdd": 0.0,
                    "hdd": 4.7,
                },
                {
                    "ghcnd_id": "USW00013874",
                    "time": "2021-04-05",
                    "lat": 33.62972,
                    "lon": -84.44224,
                    "elev": 308.2,
                    "name": "ATLANTA HARTSFIELD JACKSON INTERNATIONAL AIRPORT",
                    "precip": 0.0,
                    "tmax": 25.6,
                    "tmin": 9.4,
                    "wspd": 2.5,
                    "temp": 17.5,
                    "cdd": 0.0,
                    "hdd": 0.8,
                },
            ]
        )
    elif ghcnd_id == "USW00014739":  # BOS
        return pd.DataFrame(
            [
                {
                    "ghcnd_id": "USW00014739",
                    "time": "2021-04-01",
                    "lat": 42.36057,
                    "lon": -71.00975,
                    "elev": 3.2,
                    "name": "BOSTON LOGAN INTERNATIONAL AIRPORT, MA US",
                    "precip": 23.4,
                    "tmax": 14.4,
                    "tmin": 1.7,
                    "wspd": 6.7,
                    "temp": 10.3,
                    "cdd": 0.0,
                    "hdd": 10.3,
                },
                {
                    "ghcnd_id": "USW00014739",
                    "time": "2021-04-02",
                    "lat": 42.36057,
                    "lon": -71.00975,
                    "elev": 3.2,
                    "name": "BOSTON LOGAN INTERNATIONAL AIRPORT, MA US",
                    "precip": 0.0,
                    "tmax": 5.0,
                    "tmin": -1.6,
                    "wspd": 5.9,
                    "temp": 1.7,
                    "cdd": 0.0,
                    "hdd": 16.6,
                },
                {
                    "ghcnd_id": "USW00014739",
                    "time": "2021-04-03",
                    "lat": 42.36057,
                    "lon": -71.00975,
                    "elev": 3.2,
                    "name": "BOSTON LOGAN INTERNATIONAL AIRPORT, MA US",
                    "precip": 0.0,
                    "tmax": 10.0,
                    "tmin": -1.0,
                    "wspd": 3.6,
                    "temp": 3.1,
                    "cdd": 0.0,
                    "hdd": 13.8,
                },
                {
                    "ghcnd_id": "USW00014739",
                    "time": "2021-04-04",
                    "lat": 42.36057,
                    "lon": -71.00975,
                    "elev": 3.2,
                    "name": "BOSTON LOGAN INTERNATIONAL AIRPORT, MA US",
                    "precip": 0.0,
                    "tmax": 16.7,
                    "tmin": 0.6,
                    "wspd": 4.8,
                    "temp": 8.6,
                    "cdd": 0.0,
                    "hdd": 9.7,
                },
                {
                    "ghcnd_id": "USW00014739",
                    "time": "2021-04-05",
                    "lat": 42.36057,
                    "lon": -71.00975,
                    "elev": 3.2,
                    "name": "BOSTON LOGAN INTERNATIONAL AIRPORT, MA US",
                    "precip": 0.0,
                    "tmax": 14.4,
                    "tmin": 8.3,
                    "wspd": 7.4,
                    "temp": 11.2,
                    "cdd": 0.0,
                    "hdd": 7.0,
                },
            ]
        )
    else:
        raise ValueError(f"unrecognized ghcn station {ghcnd_id}")


def main(force=True):
    """Run the tests, without the overhead of the testing infrastructure.

    python tests/test_met_api.py
    """
    sk.constants.URL = "https://api-beta.salientpredictions.com/"
    # test_make_observed_ds()
    session = sk.login()
    test_met(session, force=force)


if __name__ == "__main__":
    main()
