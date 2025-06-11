#!/usr/bin/env python

"""Tests for wind.py.

Usage:
```
python -s -m pytest tests/test_wind.py
```

Copyright Salient Predictions 2025
"""

import numpy as np
import pandas as pd
import xarray as xr

import salientsdk as sk


def mock_wind_data_location() -> xr.Dataset:
    """Generate a synthetic data input that ok to pass to wind functions."""
    locs = ["Escalade", "Twin Groves I", "Windy Point IIa"]
    n_location = len(locs)
    n_time = 24 * 31
    time = pd.date_range(start="2023-12-01", periods=n_time, freq="h")

    temp_val = 5 * np.random.randn(n_time, n_location) + 10
    wspd_val = np.random.exponential(scale=3.0, size=(n_time, n_location))
    sher_val = np.random.uniform(0.1, 0.4, size=(n_time, n_location))
    w100_val = wspd_val * (100 / 10) ** sher_val
    LOC = "location"
    coord = ("time", LOC)

    # Create the dataset
    ds = xr.Dataset(
        {
            "temp": (coord, temp_val),
            "wspd": (coord, wspd_val),
            "wspd100": (coord, w100_val),
            "hub_height": (LOC, [119, 80, 80]),
            "elevation": (LOC, [444, 260, 582]),
            "rated_capacity": (LOC, [5.6, 1.65, 2.3]),
            "turbine_count": (LOC, [65, 121, 24]),
            "turbine_model": (LOC, ["V162-5.6", "V82-1.65", "SWT-2.3-93"]),
            "turbine_manufacturer": (LOC, ["Vestas", "Vestas", "Siemens"]),
            "year_online": (LOC, [2022, 2007, 2009]),
        },
        coords={
            "time": time,
            LOC: locs,
            "lat": (LOC, [33.735371, 40.445091, 45.710293]),
            "lon": (LOC, [-99.646866, -88.696297, -120.848785]),
        },
    )

    return ds


def test_shear_wind():
    """Test the shear_wind function .

    Note that this test doesn't need the "session" variable since
    there is no API access.
    """
    met = mock_wind_data_location()

    met["wspdhh"] = sk.wind.shear_wind(met["wspd100"], met["wspd"], met["hub_height"])

    # Escalade has a hub height of 119, so we expect positive shear from 100m
    LOC_POS = "Escalade"
    # Twin Groves has a hub height of 80, so we expect negative shear from 100m
    LOC_NEG = "Twin Groves I"

    # Test with a vector of hub heights where each location's height is different.
    assert met["wspdhh"].shape == met["wspd100"].shape
    assert met["wspdhh"].long_name == "Wind Speed at hub height"
    wspd_dif = met["wspdhh"] - met["wspd100"]
    assert np.all(wspd_dif.sel(location=LOC_POS).values >= 0)
    assert np.all(wspd_dif.sel(location=LOC_NEG).values <= 0)

    # This should also work with a scalar hub height
    met["wspdhh2"] = sk.wind.shear_wind(met["wspd100"], met["wspd"], 120)
    assert met["wspdhh2"].shape == met["wspd100"].shape
    assert met["wspdhh2"].long_name == "Wind Speed at hub height"
    wspd_dif = met["wspdhh2"] - met["wspd100"]
    assert np.all(wspd_dif.values >= 0)

    # For the future: test to make sure that NaN values translate though
    # and aren't erroneously converted to zeros.


def test_correct_wind_density():
    """Test the correct_wind_density function.

    Note that this test doesn't need the "session" variable since
    there is no API access.
    """
    met = mock_wind_data_location()

    met["wspd_dc"] = sk.wind.correct_wind_density(
        wspd=met["wspd100"],
        dens=1.225,
        temp=met["temp"],
        elev=met["elevation"] + met["hub_height"],
    )

    assert met["wspd_dc"].shape == met["wspd100"].shape
    assert met["wspd_dc"].density_corrected
    assert not np.all(met["wspd_dc"].values == met["wspd100"].values)


def test_get_power_curve():
    """Test the get_power_curve function."""
    pc = sk.wind.get_power_curve(None)

    assert isinstance(pc, xr.DataArray)
    assert pc.name == "power_curve"
    assert "wind_speed" in pc.dims

    pc = sk.wind.get_power_curve("default")

    assert isinstance(pc, xr.DataArray)
    assert pc.name == "power_curve"
    assert "wind_speed" in pc.dims

    met = mock_wind_data_location()
    pc = sk.wind.get_power_curve(met["turbine_model"])

    assert isinstance(pc, xr.DataArray)
    assert pc.name == "power_curve"
    assert "wind_speed" in pc.dims


def test_calc_wind_power():
    """Test the calc_wind_power function."""
    met = mock_wind_data_location()

    wnd = met["wspd100"]
    pwr = sk.wind.calc_wind_power(wnd, met["turbine_model"])

    assert isinstance(pwr, xr.DataArray)
    assert pwr.attrs["units"] == "MW"
    assert pwr.dims == wnd.dims


def test_calc_wind_power_all():
    """Test the calc_wind_power function."""
    met = mock_wind_data_location()

    pwr = sk.wind.calc_wind_power_all(met)

    assert isinstance(pwr, xr.Dataset)

    assert pwr["wspdhh"].dims == met["wspd100"].dims
    assert pwr["power"].dims == met["wspd100"].dims
