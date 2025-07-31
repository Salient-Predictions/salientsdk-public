"""Tests for the derived module.

This module tests all functions in the derived module for calculating derived
meteorological variables from primary forecast variables.
"""
import numpy as np
import pytest
import xarray as xr

from salientsdk import derived


class TestTempBased:
    """Test temperature based derived functions."""

    def test_tmean_basic(self):
        """Test basic tmean calculation."""
        ds = xr.Dataset(
            {
                "tmax": xr.DataArray([25, 30, 35], dims=["time"]),
                "tmin": xr.DataArray([15, 20, 25], dims=["time"]),
            }
        )
        result = derived.tmean(ds)
        expected = xr.DataArray([20, 25, 30], dims=["time"])
        xr.testing.assert_equal(result, expected)
        assert result.attrs["long_name"] == "Average temperature at 2m"
        assert result.attrs["units"] == "degC"

    def test_heat_index_low_temps(self):
        """Test heat index with temperatures below 80°F."""
        ds = xr.Dataset(
            {
                "tmax": xr.DataArray([20, 25, 30], dims=["time"], attrs={"units": "degC"}),
                "rh": xr.DataArray([0.5, 0.7, 0.9], dims=["time"]),
            }
        )
        result = derived.heat_index(ds)
        assert not np.isnan(result).all()

    def test_heat_index_high_temps(self):
        """Test heat index with temperatures above 80°F."""
        ds = xr.Dataset(
            {
                "tmax": xr.DataArray([35, 40, 45], dims=["time"], attrs={"units": "degC"}),
                "rh": xr.DataArray([0.5, 0.7, 0.9], dims=["time"]),
            }
        )
        result = derived.heat_index(ds)
        assert not np.isnan(result).all()

    def test_wind_chill_basic(self):
        """Test basic wind chill calculation."""
        ds = xr.Dataset(
            {
                "tmin": xr.DataArray([5, 0, -5], dims=["time"]),
                "wspd": xr.DataArray([10, 15, 20], dims=["time"]),
            }
        )
        result = derived.wind_chill(ds)
        assert result.attrs["long_name"] == "Minimum daily wind chill"
        assert result.attrs["units"] == "degC"
        assert not np.isnan(result).all()

    def test_wind_chill_conditions(self):
        """Test wind chill only applies under specific conditions."""
        ds = xr.Dataset(
            {
                "tmin": xr.DataArray([15, 5, 0], dims=["time"]),  # Above 10°C, below 10°C
                "wspd": xr.DataArray([2, 10, 10], dims=["time"]),  # Below 4.8 m/s, above 4.8 m/s
            }
        )
        result = derived.wind_chill(ds)
        # First case: temp > 10°C, should return temp
        assert result.values[0] == 15
        # Second case: temp < 10°C and wind > 4.8 m/s, should compute wind chill
        assert result.values[1] < 5  # Wind chill should be less than temperature
        # Third case: temp < 10°C and wind > 4.8 m/s, should compute wind chill
        assert result.values[2] < 0  # Wind chill should be less than temperature

    def test_heating_degree_days_default_base(self):
        """Test heating degree days with default base temperature."""
        ds = xr.Dataset(
            {
                "tmax": xr.DataArray([20, 15, 10], dims=["time"]),
                "tmin": xr.DataArray([10, 5, 0], dims=["time"]),
            }
        )
        result = derived.heating_degree_days(ds)
        # tmean = [15, 10, 5], base = 18, so HDD = [3, 8, 13]
        expected = xr.DataArray([3, 8, 13], dims=["time"])
        xr.testing.assert_equal(result, expected)
        assert result.attrs["long_name"] == "Heating degree days"
        assert result.attrs["units"] == "HDD day**-1 degC"

    def test_heating_degree_days_custom_base(self):
        """Test heating degree days with custom base temperature."""
        ds = xr.Dataset(
            {
                "tmax": xr.DataArray([20, 15, 10], dims=["time"]),
                "tmin": xr.DataArray([10, 5, 0], dims=["time"]),
            }
        )
        result = derived.heating_degree_days(ds, base_temp=15)
        # tmean = [15, 10, 5], base = 15, so HDD = [0, 5, 10]
        expected = xr.DataArray([0, 5, 10], dims=["time"])
        xr.testing.assert_equal(result, expected)

    def test_heating_degree_days_no_heating_needed(self):
        """Test heating degree days when no heating is needed."""
        ds = xr.Dataset(
            {
                "tmax": xr.DataArray([25, 30, 35], dims=["time"]),
                "tmin": xr.DataArray([15, 20, 25], dims=["time"]),
            }
        )
        result = derived.heating_degree_days(ds)
        # tmean = [20, 25, 30], base = 18, so HDD = [0, 0, 0]
        expected = xr.DataArray([0, 0, 0], dims=["time"])
        xr.testing.assert_equal(result, expected)

    def test_cooling_degree_days_default_base(self):
        """Test cooling degree days with default base temperature."""
        ds = xr.Dataset(
            {
                "tmax": xr.DataArray([25, 20, 15], dims=["time"]),
                "tmin": xr.DataArray([15, 10, 5], dims=["time"]),
            }
        )
        result = derived.cooling_degree_days(ds)
        # tmean = [20, 15, 10], base = 18, so CDD = [2, 0, 0]
        expected = xr.DataArray([2, 0, 0], dims=["time"])
        xr.testing.assert_equal(result, expected)
        assert result.attrs["long_name"] == "Cooling degree days"
        assert result.attrs["units"] == "CDD day**-1 degC"

    def test_cooling_degree_days_custom_base(self):
        """Test cooling degree days with custom base temperature."""
        ds = xr.Dataset(
            {
                "tmax": xr.DataArray([25, 20, 15], dims=["time"]),
                "tmin": xr.DataArray([15, 10, 5], dims=["time"]),
            }
        )
        result = derived.cooling_degree_days(ds, base_temp=12)
        # tmean = [20, 15, 10], base = 12, so CDD = [8, 3, 0]
        expected = xr.DataArray([8, 3, 0], dims=["time"])
        xr.testing.assert_equal(result, expected)

    def test_cooling_degree_days_no_cooling_needed(self):
        """Test cooling degree days when no cooling is needed."""
        ds = xr.Dataset(
            {
                "tmax": xr.DataArray([15, 10, 5], dims=["time"]),
                "tmin": xr.DataArray([5, 0, -5], dims=["time"]),
            }
        )
        result = derived.cooling_degree_days(ds)
        # tmean = [10, 5, 0], base = 18, so CDD = [0, 0, 0]
        expected = xr.DataArray([0, 0, 0], dims=["time"])
        xr.testing.assert_equal(result, expected)


class TestWindBased:
    """Test wind_speed function."""

    def test_wind_speed_default_height(self):
        """Test wind speed at default height when wspd exists."""
        ds = xr.Dataset({"wspd": xr.DataArray([5, 10, 15], dims=["time"])})
        result = derived.wind_speed(ds)
        xr.testing.assert_equal(result, ds["wspd"])

    def test_wind_speed_specific_height_exists(self):
        """Test wind speed when specific height variable exists."""
        ds = xr.Dataset({"wspd50": xr.DataArray([8, 16, 24], dims=["time"])})
        result = derived.wind_speed(ds, h=50)
        xr.testing.assert_equal(result, ds["wspd50"])

    def test_wind_speed_power_law_calculation(self):
        """Test wind speed calculation using power law."""
        ds = xr.Dataset({"wspd100": xr.DataArray([10, 20, 30], dims=["time"])})
        result = derived.wind_speed(ds, h=10, h_ref=100)
        alpha = 1 / 7
        expected = ds["wspd100"] * (10 / 100) ** alpha
        xr.testing.assert_allclose(result, expected)

    def test_wind_speed_different_heights(self):
        """Test wind speed calculation for different heights."""
        ds = xr.Dataset({"wspd100": xr.DataArray([20], dims=["time"])})
        result_50 = derived.wind_speed(ds, h=50, h_ref=100)
        result_25 = derived.wind_speed(ds, h=25, h_ref=100)
        # Higher height should have higher wind speed
        assert result_50.values[0] > result_25.values[0]


class TestComputeQuantity:
    """Test compute_quantity function."""

    def test_compute_quantity_existing_variable(self):
        """Test compute_quantity when variable exists in dataset."""
        ds = xr.Dataset({"temp": xr.DataArray([22, 27, 32], dims=["time"])})
        result = derived.compute_quantity(ds, "temp")
        xr.testing.assert_equal(result, ds["temp"])

    def test_compute_quantity_derived_variable(self):
        """Test compute_quantity for derived variable."""
        ds = xr.Dataset(
            {
                "tmax": xr.DataArray([25, 30, 35], dims=["time"]),
                "tmin": xr.DataArray([15, 20, 25], dims=["time"]),
            }
        )
        result = derived.compute_quantity(ds, "tmean")
        expected = xr.DataArray([20, 25, 30], dims=["time"])
        xr.testing.assert_equal(result, expected)

    def test_compute_quantity_with_aliases(self):
        """Test compute_quantity with aliases."""
        ds = xr.Dataset(
            {
                "tmax": xr.DataArray([25, 30, 35], dims=["time"]),
                "tmin": xr.DataArray([15, 20, 25], dims=["time"]),
            }
        )
        result_hdd = derived.compute_quantity(ds, "hdd")
        result_cdd = derived.compute_quantity(ds, "cdd")

        # Should be equivalent to calling the functions directly
        expected_hdd = derived.heating_degree_days(ds)
        expected_cdd = derived.cooling_degree_days(ds)

        xr.testing.assert_equal(result_hdd, expected_hdd)
        xr.testing.assert_equal(result_cdd, expected_cdd)

    def test_compute_quantity_unknown_variable(self):
        """Test compute_quantity with unknown variable raises error."""
        ds = xr.Dataset({"temp": xr.DataArray([22, 27, 32], dims=["time"])})
        with pytest.raises(ValueError, match="Unknown quantity 'unknown_var'"):
            derived.compute_quantity(ds, "unknown_var")

    def test_compute_quantity_custom_database(self):
        """Test compute_quantity with custom database."""

        def custom_func(ds):
            return ds["temp"] * 2

        custom_db = {"double_temp": custom_func}
        ds = xr.Dataset({"temp": xr.DataArray([10, 20, 30], dims=["time"])})
        result = derived.compute_quantity(ds, "double_temp", database=custom_db)
        expected = xr.DataArray([20, 40, 60], dims=["time"])
        xr.testing.assert_equal(result, expected)


class TestCallableVariables:
    """Test CALLABLE_VARIABLES dictionary."""

    def test_callable_variables_completeness(self):
        """Test that all expected variables are in CALLABLE_VARIABLES."""
        expected_vars = ["tmean", "temp", "heat_index", "wind_chill", "hdd", "cdd"]
        for var in expected_vars:
            assert var in derived.CALLABLE_VARIABLES

    def test_callable_variables_functions(self):
        """Test that all functions in CALLABLE_VARIABLES are callable."""
        for name, func in derived.CALLABLE_VARIABLES.items():
            assert callable(func), f"{name} is not callable"

    def test_callable_variables_aliases(self):
        """Test that aliases point to correct functions."""
        assert derived.CALLABLE_VARIABLES["hdd"] == derived.heating_degree_days
        assert derived.CALLABLE_VARIABLES["cdd"] == derived.cooling_degree_days
