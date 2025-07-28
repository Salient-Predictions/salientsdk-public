#!/usr/bin/env python
# Copyright Salient Predictions 2025

"""Tests for hydro.py.

Usage:
```
python -s -m pytest tests/test_hydro.py
```

"""

import datetime as dt
import os

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import salientsdk as sk
from tests.conftest import get_test_dir

VIC_TEST_START = "2020-01-01"
VIC_TEST_END = "2020-02-01"
VIC_TEST_REF_CLIM = 2
VIC_TEST_SOIL_DEPTHS = [0.05, 1, 2.0]


TEST_DEST = get_test_dir(__file__)


@pytest.fixture
def vic_file(session):
    """Generates a location file for testing VIC functions."""
    geo_name = "VIC Test"
    return sk.upload_bounding_box(
        west=-104,
        east=-102,
        south=38,
        north=40,
        geoname=geo_name,
        destination=TEST_DEST,
        session=session,
        force=False,
        verbose=False,
    )


@pytest.fixture
def vic_loc(vic_file):
    """Constructs a sk.Location object using vic_file as shapefile."""
    return sk.Location(shapefile=vic_file)


@pytest.fixture
def vic_destinations(vic_loc):
    """Returns dictionary of paths to VIC inputs and outputs for testing."""
    return sk.hydro._init_vic_destinations(
        destination=TEST_DEST,
        loc=vic_loc.shapefile,
        start=VIC_TEST_START,
        end=VIC_TEST_END,
        reference_clim=VIC_TEST_REF_CLIM,
    )


@pytest.fixture
def vic_salient_ds(session, vic_loc):
    """Fetches salient data used to generate VIC domain and parameter datasets."""
    return sk.hydro._get_salient_params(
        loc=vic_loc,
        start=VIC_TEST_START,
        end=VIC_TEST_END,
        session=session,
        destination=TEST_DEST,
    )


@pytest.fixture
def vic_domain_ds(vic_salient_ds, vic_destinations):
    """Generates the VIC domain dataset."""
    return sk.hydro._build_vic_domain(vic_salient_ds, vic_destinations["domain_path"])


def mock_source_data() -> xr.Dataset:
    """Generate a dataset that's ok to pass to calc_swe."""
    n_time = 60
    n_ensemble = 2
    n_location = 2
    time = pd.date_range(start="2023-12-01", periods=n_time, freq="D")
    locations = ["Hakuba", "Rusutsu"]
    latitudes = [36.69, 42.82]
    longitudes = [137.9, 140.7]
    coord = ("ensemble", "time", "location")
    temp_val = 5 * np.random.randn(n_ensemble, n_time, n_location) - 5
    precip_val = np.random.exponential(scale=3.0, size=(n_ensemble, n_time, n_location))

    # Create the dataset
    ds = xr.Dataset(
        {
            "temp": (coord, temp_val),
            "precip": (coord, precip_val),
        },
        coords={
            "time": time,
            "location": locations,
            "lat": ("location", latitudes),
            "lon": ("location", longitudes),
            "ensemble": np.arange(1, n_ensemble + 1),
        },
    )

    return ds


def test_calc_swe():
    """Test the calc_swe function.

    Note that this test doesn't need the "session" variable since
    there is no API access.
    """
    met = mock_source_data()
    met["swe"] = sk.hydro.calc_swe(met, "time")

    assert (
        met["swe"].attrs.get("long_name") == "Snow Water Equivalent"
    ), "Incorrect long_name for SWE"
    assert (met["swe"] >= 0).all(), "Not all SWE values are non-negative"


def test_build_vic_inputs(session, vic_file, vic_loc, vic_destinations):
    """Test the _build_vic_inputs functon.

    pytest tests/test_hydro.py::test_build_vic_inputs
    """
    # Ensure invalid location throws AssertionError
    invalid_loc = sk.Location(location_file=vic_file)
    with pytest.raises(AssertionError):
        sk.hydro._build_vic_inputs(
            loc=invalid_loc,
            destination=TEST_DEST,
            start=VIC_TEST_START,
            end=VIC_TEST_END,
            reference_clim=VIC_TEST_REF_CLIM,
            session=session,
        )

    invalid_loc = sk.Location(shapefile=[vic_file, vic_file])
    with pytest.raises(AssertionError):
        sk.hydro._build_vic_inputs(
            loc=invalid_loc,
            destination=TEST_DEST,
            start=VIC_TEST_START,
            end=VIC_TEST_END,
            reference_clim=VIC_TEST_REF_CLIM,
            session=session,
        )

    # Ensure invalid start/end throws AssertionError
    invalid_start = sk.hydro.VIC_VALID_DATE_RANGE[0] - dt.timedelta(days=1)
    with pytest.raises(AssertionError):
        sk.hydro._build_vic_inputs(
            loc=vic_loc,
            destination=TEST_DEST,
            start=str(invalid_start),
            end=str(sk.hydro.VIC_VALID_DATE_RANGE[1]),
            reference_clim=VIC_TEST_REF_CLIM,
            session=session,
        )

    invalid_end = sk.hydro.VIC_VALID_DATE_RANGE[1] + dt.timedelta(days=1)
    with pytest.raises(AssertionError):
        sk.hydro._build_vic_inputs(
            loc=vic_loc,
            destination=TEST_DEST,
            start=str(sk.hydro.VIC_VALID_DATE_RANGE[0]),
            end=str(invalid_end),
            reference_clim=VIC_TEST_REF_CLIM,
            session=session,
        )

    # Ensure end < start throws AssertionError
    invalid_end = sk.hydro.VIC_VALID_DATE_RANGE[1] + dt.timedelta(days=1)
    with pytest.raises(AssertionError):
        sk.hydro._build_vic_inputs(
            loc=vic_loc,
            destination=TEST_DEST,
            start=VIC_TEST_END,
            end=VIC_TEST_START,
            reference_clim=VIC_TEST_REF_CLIM,
            session=session,
        )

    actual_vic_destinations = sk.hydro._build_vic_inputs(
        loc=vic_loc,
        destination=TEST_DEST,
        start=VIC_TEST_START,
        end=VIC_TEST_END,
        reference_clim=VIC_TEST_REF_CLIM,
        session=session,
    )

    assert actual_vic_destinations == vic_destinations


def test_build_vic_domain(vic_domain_ds, vic_destinations):
    """Test the _build_vic_domain function."""
    # Ensure proper variables are returned for VIC domain dataset
    expected_vars = set(["elev", "mask", "area", "frac", "run_cell", "lats", "lons", "gridcell"])
    actual_vars = set(vic_domain_ds.data_vars)
    diff = expected_vars ^ actual_vars  # symmetric difference
    assert len(diff) == 0

    # Ensure domain dataset NetCDF exists
    assert os.path.exists(vic_destinations["domain_path"])


def test_build_vic_forcings(vic_salient_ds, vic_destinations):
    """Test the _build_vic_forcings function."""
    # Ensure proper variables are returned for VIC domain dataset
    forcings_ds = sk.hydro._build_vic_forcings(
        start=VIC_TEST_START,
        end=VIC_TEST_END,
        domain_path=vic_destinations["domain_path"],
        salient_ds=vic_salient_ds,
        out_path=vic_destinations["forcings_path"],
    )
    expected_vars = set(
        ["temp", "prec", "air_pressure", "shortwave", "longwave", "vapor_pressure", "wind"]
    )
    actual_vars = set(forcings_ds.data_vars)
    diff = expected_vars ^ actual_vars  # symmetric difference
    assert len(diff) == 0

    # Ensure proper time range is returned
    assert pd.Timestamp(forcings_ds.time.values[0]) == pd.Timestamp(VIC_TEST_START)
    assert pd.Timestamp(forcings_ds.time.values[-1]) == pd.Timestamp(f"{VIC_TEST_END}T18:00:00")

    # Ensure forcing dataset NetCDF exists
    year = VIC_TEST_START.split("-")[0]
    assert os.path.exists(f"{vic_destinations['forcings_path']}{year}.nc")


def test_build_vic_params(vic_salient_ds, vic_domain_ds, vic_destinations):
    """Test the _build_vic_params function.

    pytest tests/test_hydro.py::test_build_vic_params
    """
    # Ensure soil_depths outside soil variables depth domain throws AssertionError
    invalid_soil_depths = [0.01]
    with pytest.raises(AssertionError):
        sk.hydro._build_vic_params(
            salient_ds=vic_salient_ds,
            domain_ds=vic_domain_ds,
            start=VIC_TEST_START,
            end=VIC_TEST_END,
            soil_depths=invalid_soil_depths,
            out_path=vic_destinations["params_path"],
        )

    params_ds = sk.hydro._build_vic_params(
        salient_ds=vic_salient_ds,
        domain_ds=vic_domain_ds,
        start=VIC_TEST_START,
        end=VIC_TEST_END,
        soil_depths=VIC_TEST_SOIL_DEPTHS,
        out_path=vic_destinations["params_path"],
    )

    # Ensure proper variables are returned for VIC domain dataset
    expected_vars = set(
        [
            "layer",
            "mask",
            "run_cell",
            "gridcell",
            "lats",
            "lons",
            "infilt",
            "Ds",
            "Dsmax",
            "Ws",
            "c",
            "expt",
            "Ksat",
            "phi_s",
            "init_moist",
            "elev",
            "depth",
            "dp",
            "bubble",
            "quartz",
            "bulk_density",
            "soil_density",
            "off_gmt",
            "Wcr_FRACT",
            "Wpwp_FRACT",
            "rough",
            "snow_rough",
            "annual_prec",
            "avg_T",
            "resid_moist",
            "fs_active",
            "veg_descr",
            "Nveg",
            "Cv",
            "root_depth",
            "root_fract",
            "LAI",
            "overstory",
            "rarc",
            "rmin",
            "wind_h",
            "RGL",
            "rad_atten",
            "wind_atten",
            "trunk_ratio",
            "albedo",
            "veg_rough",
            "displacement",
        ]
    )
    actual_vars = set(params_ds.data_vars)
    diff = expected_vars ^ actual_vars  # symmetric difference
    assert len(diff) == 0, f"VIC params dataset differing vars: {diff}"

    # Ensure parameters dataset NetCDF exists
    assert os.path.exists(vic_destinations["params_path"])


def test_build_vic_global_params(vic_destinations):
    """Test the _build_vic_params function.

    pytest tests/test_hydro.py::test_build_vic_params
    """
    sk.hydro._build_vic_global_params(
        start=VIC_TEST_START,
        end=VIC_TEST_END,
        n_soil_layers=len(VIC_TEST_SOIL_DEPTHS),
        out_paths=vic_destinations,
    )

    # Ensure global parameters text file exists
    assert os.path.exists(vic_destinations["global_params_path"])
