"""Tests for ForecastZarr functionality including credential handling and data access."""

import os
from unittest.mock import Mock, patch

import pytest
import xarray as xr

import salientsdk as sk
from salientsdk.forecast_zarr import ForecastZarr, ZarrBase, ZarrGEM, ZarrV9


class TestForecastZarrFactory:
    """Test the ForecastZarr factory class."""

    def test_factory_returns_zarr_gem_for_gem_model(self):
        """Test that ForecastZarr returns ZarrGEM for 'gem' model."""
        location = sk.Location(lat=40, lon=-90)
        zarr_obj = ForecastZarr(
            location=location,
            model="gem",
            key_id="test_id",
            key_secret="test_secret",
            direct_url="test_url",
        )
        assert isinstance(zarr_obj, ZarrGEM)

    def test_factory_returns_zarr_gem_for_baseline_model(self):
        """Test that ForecastZarr returns ZarrGEM for 'baseline' model."""
        location = sk.Location(lat=40, lon=-90)
        zarr_obj = ForecastZarr(
            location=location,
            model="baseline",
            key_id="test_id",
            key_secret="test_secret",
            direct_url="test_url",
        )
        assert isinstance(zarr_obj, ZarrGEM)

    def test_factory_returns_zarr_v9_for_other_models(self):
        """Test that ForecastZarr returns ZarrV9 for other models."""
        location = sk.Location(lat=40, lon=-90)
        zarr_obj = ForecastZarr(
            location=location,
            model="blend",
            timescale="sub-seasonal",
            key_id="test_id",
            key_secret="test_secret",
            direct_url="test_url",
        )
        assert isinstance(zarr_obj, ZarrV9)


class TestCredentialValidation:
    """Test credential validation with secrets fallback."""

    @patch.dict(
        os.environ,
        {
            "SALIENT_DIRECT_ID": "env_id",
            "SALIENT_DIRECT_SECRET": "env_secret",
            "SALIENT_DIRECT_URL": "env_url",
        },
    )
    @patch("salientsdk.forecast_zarr._get_secret")
    def test_credentials_from_env_vars_priority(self, mock_get_secret):
        """Test that environment variables take priority over secrets."""
        location = sk.Location(lat=40, lon=-90)
        zarr_obj = ForecastZarr(location=location, model="blend", timescale="sub-seasonal")

        assert zarr_obj._key_id == "env_id"
        assert zarr_obj._key_secret == "env_secret"
        assert zarr_obj._direct_url == "env_url"
        mock_get_secret.assert_not_called()

    def test_direct_credentials_override_all(self):
        """Test that directly provided credentials override everything."""
        location = sk.Location(lat=40, lon=-90)
        zarr_obj = ForecastZarr(
            location=location,
            model="blend",
            timescale="sub-seasonal",
            key_id="direct_id",
            key_secret="direct_secret",
            direct_url="direct_url",
        )

        assert zarr_obj._key_id == "direct_id"
        assert zarr_obj._key_secret == "direct_secret"
        assert zarr_obj._direct_url == "direct_url"


class TestZarrV9:
    """Test ZarrV9 specific functionality."""

    def test_zarr_v9_properties(self):
        """Test ZarrV9 property values."""
        location = sk.Location(lat=40, lon=-90)
        zarr_obj = ZarrV9(
            location=location, key_id="test_id", key_secret="test_secret", direct_url="test_url"
        )

        assert "temp" in zarr_obj.FORECAST_VARIABLES
        assert "precip" in zarr_obj.FORECAST_VARIABLES
        assert "blend" in zarr_obj.MODELS
        assert "north-america" in zarr_obj.REGIONS
        assert "anom" in zarr_obj.FIELDS
        assert "vals" in zarr_obj.FIELDS
        assert "sub-seasonal" in zarr_obj.TIMESCALES

    def test_zarr_v9_validation_invalid_field(self):
        """Test validation with invalid field."""
        location = sk.Location(lat=40, lon=-90)
        with pytest.raises(AssertionError, match="Invalid field"):
            ZarrV9(
                location=location,
                field="invalid_field",
                key_id="test_id",
                key_secret="test_secret",
                direct_url="test_url",
            )

    def test_zarr_v9_validation_invalid_model(self):
        """Test validation with invalid model."""
        location = sk.Location(lat=40, lon=-90)
        with pytest.raises(AssertionError, match="Invalid model"):
            ZarrV9(
                location=location,
                model="invalid_model",
                key_id="test_id",
                key_secret="test_secret",
                direct_url="test_url",
            )

    def test_zarr_v9_validation_invalid_variable(self):
        """Test validation with invalid variable."""
        location = sk.Location(lat=40, lon=-90)
        with pytest.raises(AssertionError, match="Invalid variable"):
            ZarrV9(
                location=location,
                variable="invalid_variable",
                key_id="test_id",
                key_secret="test_secret",
                direct_url="test_url",
            )

    def test_zarr_v9_validation_invalid_timescale(self):
        """Test validation with invalid timescale."""
        location = sk.Location(lat=40, lon=-90)
        with pytest.raises(AssertionError, match="Invalid timescale"):
            ZarrV9(
                location=location,
                timescale="invalid_timescale",
                key_id="test_id",
                key_secret="test_secret",
                direct_url="test_url",
            )

    def test_zarr_v9_date_validation(self):
        """Test date validation."""
        location = sk.Location(lat=40, lon=-90)
        with pytest.raises(
            AssertionError, match="The end date.*must be at or after the start date"
        ):
            ZarrV9(
                location=location,
                start="2023-12-01",
                end="2023-11-01",
                key_id="test_id",
                key_secret="test_secret",
                direct_url="test_url",
            )


class TestZarrGEM:
    """Test ZarrGEM specific functionality."""

    def test_zarr_gem_properties(self):
        """Test ZarrGEM property values."""
        location = sk.Location(lat=40, lon=-90)
        zarr_obj = ZarrGEM(
            location=location,
            model="gem",
            timescale="daily",
            key_id="test_id",
            key_secret="test_secret",
            direct_url="test_url",
        )

        assert "temp" in zarr_obj.FORECAST_VARIABLES
        assert "precip" in zarr_obj.FORECAST_VARIABLES
        assert "cdd" in zarr_obj.FORECAST_VARIABLES
        assert "hdd" in zarr_obj.FORECAST_VARIABLES
        assert "gem" in zarr_obj.MODELS
        assert "baseline" in zarr_obj.MODELS
        assert "north-america" in zarr_obj.REGIONS
        assert "anom" in zarr_obj.FIELDS
        assert "vals" in zarr_obj.FIELDS
        assert "anom_ens" in zarr_obj.FIELDS
        assert "vals_ens" in zarr_obj.FIELDS
        assert "daily" in zarr_obj.TIMESCALES
        assert len(zarr_obj.TIMESCALES) == 1  # Only daily for GEM

    def test_zarr_gem_multiple_fields(self):
        """Test ZarrGEM with multiple fields."""
        location = sk.Location(lat=40, lon=-90)
        zarr_obj = ZarrGEM(
            location=location,
            model="gem",
            timescale="daily",
            field=["anom", "vals"],
            key_id="test_id",
            key_secret="test_secret",
            direct_url="test_url",
        )

        assert zarr_obj.field == ["anom", "vals"]


class TestLocationSubsetting:
    """Test location-based subsetting functionality."""

    def test_region_location(self):
        """Test region-based location."""
        location = sk.Location(region="north-america")
        zarr_obj = ZarrV9(
            location=location, key_id="test_id", key_secret="test_secret", direct_url="test_url"
        )
        assert zarr_obj.region == "north-america"

    def test_lat_lon_location(self):
        """Test lat/lon location."""
        location = sk.Location(lat=40.0, lon=-90.0)
        zarr_obj = ZarrV9(
            location=location, key_id="test_id", key_secret="test_secret", direct_url="test_url"
        )
        assert zarr_obj.location.lat == 40.0
        assert zarr_obj.location.lon == -90.0

    def test_default_region_when_no_region_specified(self):
        """Test default region assignment."""
        location = sk.Location(lat=40.0, lon=-90.0)
        zarr_obj = ZarrV9(
            location=location, key_id="test_id", key_secret="test_secret", direct_url="test_url"
        )
        assert zarr_obj.region == "north-america"

    def test_invalid_region_raises_error(self):
        """Test that invalid region raises error."""
        location = sk.Location(region="invalid-region")
        with pytest.raises(AssertionError, match="Invalid region"):
            ZarrV9(
                location=location,
                key_id="test_id",
                key_secret="test_secret",
                direct_url="test_url",
            )


class TestMemoryChecking:
    """Test memory checking functionality."""

    def test_memory_check_large_dataset_raises_error(self):
        """Test that excessively large datasets raise error."""
        location = sk.Location(lat=40, lon=-90)
        zarr_obj = ZarrV9(
            location=location, key_id="test_id", key_secret="test_secret", direct_url="test_url"
        )

        # Create a mock dataset that would be too large
        mock_ds = Mock()
        mock_ds.nbytes = 65 * 1e9  # 65 GB

        with pytest.raises(sk.forecast_zarr.ExcessiveMemoryRequestError):
            zarr_obj._check_memory(mock_ds)

    def test_memory_check_warning_for_large_dataset(self):
        """Test that large datasets generate warnings."""
        location = sk.Location(lat=40, lon=-90)
        zarr_obj = ZarrV9(
            location=location, key_id="test_id", key_secret="test_secret", direct_url="test_url"
        )

        # Create a mock dataset that would generate a warning
        mock_ds = Mock()
        mock_ds.nbytes = 10 * 1e9  # 10 GB

        with pytest.warns(UserWarning, match="Loading.*GB of data into memory"):
            zarr_obj._check_memory(mock_ds)


class TestCoordinateUtilities:
    """Test coordinate manipulation utilities."""

    def test_make_coords_dataarrays_single_point(self):
        """Test making coordinate DataArrays for single point."""
        lat, lon = ZarrBase.make_coords_dataarrays(40.0, -90.0)

        assert isinstance(lat, xr.DataArray)
        assert isinstance(lon, xr.DataArray)
        assert lat.values[0] == 40.0
        assert lon.values[0] == -90.0
        assert lat.dims == ("location",)
        assert lon.dims == ("location",)

    def test_make_coords_dataarrays_multiple_points(self):
        """Test making coordinate DataArrays for multiple points."""
        lats = [40.0, 41.0, 42.0]
        lons = [-90.0, -91.0, -92.0]
        names = ["point1", "point2", "point3"]

        lat, lon = ZarrBase.make_coords_dataarrays(lats, lons, names)

        assert len(lat) == 3
        assert len(lon) == 3
        assert list(lat.values) == lats
        assert list(lon.values) == lons
        assert list(lat.location.values) == names
        assert list(lon.location.values) == names

    def test_make_coords_dataarrays_no_names(self):
        """Test making coordinate DataArrays without names."""
        lats = [40.0, 41.0]
        lons = [-90.0, -91.0]

        lat, lon = ZarrBase.make_coords_dataarrays(lats, lons)

        assert len(lat) == 2
        assert len(lon) == 2
        assert list(lat.values) == lats
        assert list(lon.values) == lons


# Integration tests that require actual credentials
class TestIntegration:
    """Integration tests requiring actual credentials."""

    @pytest.mark.parametrize(
        "variable,field",
        [("precip", "vals"), ("precip", "vals_ens"), ("precip", "anom"), ("precip", "anom_ens")],
    )
    def test_gem_base_nonneg(self, variable, field):
        """Test GEM data access and validate non-negative precipitation values."""
        location = sk.Location(lat=40, lon=-90)
        zarr_obj = sk.ForecastZarr(
            location=location,
            model="gem",
            variable=variable,
            field=field,
            timescale="daily",
            start="2025-01-01",
            end="2025-01-01",
            # REQUIRES credentials to be in the environment or stored in secret manager
        )
        # Reduce size of data accessed by selecting out two lead chunks
        ds = zarr_obj.open_dataset().isel(lead=list(range(7)) + list(range(98, 100))).compute()
        varname = f"{variable}_{field}"

        if "ens" in field:
            assert ds[varname].notnull().prod(["lead", "ensemble"]).item() == 1
        else:
            assert ds[varname].notnull().prod(["lead", "quantiles"]).item() == 1

        if variable == "precip" and "vals" in field:
            assert ds[varname].min().item() >= 0

    def test_temperature_bounds_relationship(self):
        """Test that tmax >= temp >= tmin for temperature data."""
        location = sk.Location(lat=40, lon=-90)

        # Get all temperature variables in a single request
        zarr_obj = sk.ForecastZarr(
            location=location,
            model="gem",
            variable=["temp", "tmax", "tmin"],
            field="vals",
            start="2023-01-01",
            end="2023-01-02",
        )

        ds = zarr_obj.open_dataset().isel(lead=list(range(7)) + list(range(98, 100))).compute()

        # Extract the temperature values
        temp_vals = ds["temp_vals"]
        tmax_vals = ds["tmax_vals"]
        tmin_vals = ds["tmin_vals"]

        # Verify tmax >= temp >= tmin relationship
        assert (tmax_vals >= temp_vals).all().item(), "tmax should be >= temp"
        assert (temp_vals >= tmin_vals).all().item(), "temp should be >= tmin"

    def test_zarr_v9_open_dataset_basic(self):
        """Test opening a basic ZarrV9 dataset."""
        location = sk.Location(lat=40, lon=-90)
        zarr_obj = ForecastZarr(
            location=location,
            model="blend",
            variable="temp",
            field="anom",
            timescale="sub-seasonal",
            start="2023-01-01",
            end="2023-01-02",
        )

        ds = zarr_obj.open_dataset()
        assert isinstance(ds, xr.Dataset)
        assert "anom" in ds.data_vars  # Variable name structure for ZarrV9
        assert "lat" in ds.coords
        assert "lon" in ds.coords
        assert "forecast_date" in ds.coords

    def test_zarr_gem_open_dataset_basic(self):
        """Test opening a basic ZarrGEM dataset."""
        location = sk.Location(lat=40, lon=-90)
        zarr_obj = ForecastZarr(
            location=location,
            model="gem",
            variable="temp",
            field="anom",
            start="2023-01-01",
            end="2023-01-02",
        )

        ds = zarr_obj.open_dataset()
        assert isinstance(ds, xr.Dataset)
        assert "temp_anom" in ds.data_vars
        assert "lat" in ds.coords
        assert "lon" in ds.coords
