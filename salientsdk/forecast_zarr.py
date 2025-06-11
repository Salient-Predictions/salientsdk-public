# Copyright Salient Predictions 2025

"""Interface to Salient forecast zarr stores.

Used in conjunction with the Salient SDK

```python
import salientsdk as sk
loc_region = sk.Location(region="north-america")
ds= sk.ForecastZarr(
    location=loc_region,
    variable="temp",
    field=["anom", "vals"],
    model="blend",
    timescale="sub-seasonal",
).open_dataset()
```
"""

import os
import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterable

import dask
import pandas as pd
import s3fs
import xarray as xr

from .gem_utils import get_gem_dataset
from .location import Location

MEMORY_WARN_GB = 8
MEMORY_MAX_GB = 64


class ExcessiveMemoryRequestError(Exception):
    """Exception raised when a memory request exceeds the allowed threshold for loading data."""

    pass


class ForecastZarr:
    """Factory class for accessing Salient forecast zarr stores."""

    def __new__(
        _cls,
        location: Location,
        field: str | list[str] = "anom",
        model: str | list[str] = "gem",
        timescale: str = "daily",
        variable: str = "temp",
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
        key_id: str | None = None,
        key_secret: str | None = None,
        direct_url: str | None = None,
    ):
        """Create and return an appropriate zarr accessor instance.

        Factory method that determines which zarr accessor class to instantiate
        based on the provided model parameter. Returns ZarrGEM for "gem" and
        "baseline" models, otherwise returns ZarrV9.

        Args:
            location: The Salient SDK location object.
            field: The forecast field or fields to retrieve.
            model: Which model to retrieve.
            timescale: The forecast timescale.
            variable: The forecast variable.
            start: The starting forecast date.
            end: The ending forecast date.
            key_id: The Key ID used to access the zarr stores.
            key_secret: The Key Secret used to access the zarr stores.
            direct_url: The URL for the Salient zarr store.

        Returns:
            An instance of ZarrGEM or ZarrV9, depending on the model.
        """
        if model in ["gem", "baseline"]:
            return ZarrGEM(
                location=location,
                field=field,
                model=model,
                timescale=timescale,
                variable=variable,
                start=start,
                end=end,
                key_id=key_id,
                key_secret=key_secret,
                direct_url=direct_url,
            )
        else:
            return ZarrV9(
                location=location,
                field=field,
                model=model,
                timescale=timescale,
                variable=variable,
                start=start,
                end=end,
                key_id=key_id,
                key_secret=key_secret,
                direct_url=direct_url,
            )


class ZarrBase(ABC):
    """Abstract base class for accessing Salient forecast zarr stores.

    Provides common functionality and defines the interface that concrete zarr
    accessor classes must implement. Handles parameter validation, credential
    management, location subsetting, and memory checking operations.
    """

    @property
    @abstractmethod
    def FORECAST_VARIABLES(self) -> list[str]:
        """Available forecast variables for this zarr store type."""
        pass

    @property
    @abstractmethod
    def MODELS(self) -> list[str]:
        """Available forecast models for this zarr store type."""
        pass

    @property
    @abstractmethod
    def REGIONS(self) -> list[str]:
        """Available geographic regions for this zarr store type."""
        pass

    @property
    @abstractmethod
    def FIELDS(self) -> list[str]:
        """Available forecast fields for this zarr store type."""
        pass

    @property
    @abstractmethod
    def TIMESCALES(self) -> list[str]:
        """Available forecast timescales for this zarr store type."""
        pass

    def __init__(
        self,
        location: Location,
        field: str | list[str] = "anom",
        model: str | list[str] = "blend",
        timescale: str = "sub-seasonal",
        variable: str = "temp",
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
        key_id: str | None = None,
        key_secret: str | None = None,
        direct_url: str | None = None,
    ):
        """Initialize the ForecastZarr object.

        Args:
            location: The Salient SDK location object
            field: The forecast field or fields to retrieve.
            model: Which model to retrieve.
            timescale: The forecast timescale, one of `sub-seasonal`, `seasonal`, or `long-range`.
            variable: The forecast variable.
            start: The starting forecast date.
            end: The ending forecast date.
            key_id: The Key ID used to access the zarr stores. If not provided will attempt to read
                from the environment variable `SALIENT_DIRECT_ID`.
            key_secret: The Key Secret used to access the zarr stores. If not provided will attempt
                to read from the environment variable `SALIENT_DIRECT_SECRET`.
            direct_url: The URL for the Salient zarr store. If not provided will attempt to read
                from the environment variable `SALIENT_DIRECT_URL`.
        """
        self._validate_credentials(key_id, key_secret, direct_url)
        self.location = location
        self.field = field
        self.model = model
        self.variable = variable
        self.timescale = timescale
        self.start = start
        self.end = end

        self._validate_forecast_params()

    def _validate_forecast_params(self):
        if isinstance(self.field, str):
            assert (
                self.field in self.FIELDS
            ), f"Invalid field: {self.field}. Available fields: {self.FIELDS}"
        else:
            assert all(
                [f in self.FIELDS for f in self.field]
            ), f"Invalid field: {self.field}. Available fields: {self.FIELDS}"

        assert (
            self.model in self.MODELS
        ), f"Invalid model: {self.model}. Available models: {self.MODELS}"

        variables = self.variable if isinstance(self.variable, list) else [self.variable]
        assert all(
            [v in self.FORECAST_VARIABLES for v in variables]
        ), f"Invalid variable: {self.variable}. Available variables: {self.FORECAST_VARIABLES}"

        assert (
            self.timescale in self.TIMESCALES
        ), f"Invalid timescale: {self.timescale}. Available timescales: {self.TIMESCALES}"

        if self.start:
            self.end = self.start if self.end is None else self.end
            assert pd.Timestamp(self.start) <= pd.Timestamp(
                self.end
            ), f"The end date ({self.end}) must be at or after the start date ({self.start})."

        if self.location.region:
            assert (
                self.location.region in self.REGIONS
            ), f"Invalid region: {self.location.region}. Available regions: {self.REGIONS}"
            self.region = self.location.region
        else:
            self.region = "north-america"

    def _validate_credentials(
        self, key_id: str | None, key_secret: str | None, direct_url: str | None
    ):
        """Validate the credentials for direct zarr store access.

        Will read from environment variables if no arguments are passed.
        """
        if not key_id:
            key_id = os.getenv("SALIENT_DIRECT_ID")
            assert (
                key_id is not None
            ), "Must provide `key_id` or set `SALIENT_DIRECT_ID` in your environment."
        if not key_secret:
            key_secret = os.getenv("SALIENT_DIRECT_SECRET")
            assert (
                key_id is not None
            ), "Must provide `key_secret` or set `SALIENT_DIRECT_SECRET` in your environment."
        if not direct_url:
            direct_url = os.getenv("SALIENT_DIRECT_URL")
            assert (
                direct_url is not None
            ), "Must provide `direct_url` or set `SALIENT_DIRECT_URL` in your environment."

        self._key_id = key_id
        self._key_secret = key_secret
        self._direct_url = direct_url

    @abstractmethod
    def open_dataset(self, in_memory: bool = False) -> xr.Dataset:
        """Open and return the forecast dataset.

        Args:
            in_memory: Whether to load the dataset into memory immediately.

        Returns:
            The xarray Dataset containing forecast data.
        """
        pass

    def subset_location(self, ds):
        """Subset the dataset to the passed `Location` object."""
        if self.location.location_file:
            df = self.location.load_location_file()
            lat, lon = ZarrBase.make_coords_dataarrays(df.lat, df.lon, df.name)
            return ZarrBase.interp_to_coords(ds, lat, lon)
        elif self.location.lat and self.location.lon:
            lat, lon = ZarrBase.make_coords_dataarrays(self.location.lat, self.location.lon)
            return ZarrBase.interp_to_coords(ds, lat, lon)
        elif self.location.region:
            return ds
        elif self.location.shapefile:
            raise NotImplementedError("Shapefile subsetting not yet implemented")
        else:
            raise ValueError("Location object must have either lat/lon, location_file, or region")

    @staticmethod
    def interp_to_coords(ds, lat=[], lon=[], method="linear"):
        """Interpolate the dataset to lat/lon locations."""
        # Interpolate to subgrid scale locations, either with sel or interp
        if method == "nearest":
            interpolated = ds.sel(lat=lat, lon=lon, method="nearest")
            interpolated = interpolated.assign_coords({"lon": lon, "lat": lat})
        else:
            # # xarray interp does sequential nd interpolation, so currently can't do this version without blowing up memory:
            # interpolated = ds.interp(lat=lat, lon=lon)
            # Instead, do a looping version through each location
            nonspatial_vars = None
            if isinstance(ds, xr.Dataset):
                # Concat tries to add location coord to nonspatial vars so separate those out
                nonspatial_vars = [
                    v for v in ds.data_vars if "lon" not in ds[v].dims and "lat" not in ds[v].dims
                ]
                ds_nonspatial = ds[nonspatial_vars]
                ds = ds.drop_vars(nonspatial_vars)
            interpolated = []
            for loc in lat.location.values:
                interpolated.append(
                    dask.delayed(ds.interp)(
                        lat=lat.sel(location=[loc]), lon=lon.sel(location=[loc]), method=method
                    )
                )
            interpolated = xr.concat(dask.compute(*interpolated), dim="location")
            if nonspatial_vars:
                interpolated = xr.merge([interpolated, ds_nonspatial])

        return interpolated

    @staticmethod
    def make_coords_dataarrays(lat, lon, names=None):
        """Make a coordinate data array from the provided location lat/lons."""
        if not isinstance(lat, Iterable):
            lat = [lat]
        if not isinstance(lon, Iterable):
            lon = [lon]
        lat = xr.DataArray(lat, dims="location", name="lat", attrs={"long_name": "Latitude"})
        lon = xr.DataArray(lon, dims="location", name="lon", attrs={"long_name": "Longitude"})
        if names is not None:
            lat = lat.assign_coords({"location": names})
            lon = lon.assign_coords({"location": names})
        return lat, lon

    def _check_memory(self, ds: xr.Dataset):
        num_gb = ds.nbytes / 1e9
        if num_gb > MEMORY_MAX_GB:
            raise ExcessiveMemoryRequestError(
                f"Loading {num_gb:,.1f} GB of data into memory exceeds the maximum allowed ({MEMORY_MAX_GB} GB)."
            )
        elif num_gb > MEMORY_WARN_GB:
            warnings.warn(
                f"Loading {num_gb:,.1f} GB of data into memory. Consider spatial selection to conserve memory and bandwidth."
            )


class ZarrV9(ZarrBase):
    """Access the native temporal resolution Salient forecast zarr stores.

    Used in order to get dask-backed Xarray datasets to the hindcast catalog of
    Salient forecasts in order to perform large-scale analyses without needing
    to make API calls.

    The available timescales are `sub-seasonal`, `seasonal`, and `long-range`.
    """

    _FORECAST_VARIABLES = ["temp", "precip", "wspd", "tsi", "cdd", "hdd"]
    _MODELS = ["blend", "noaa_gefs", "ecmwf_ens", "ecmwf_seas5", "truth"]
    _REGIONS = ["north-america"]
    _FIELDS = ["anom", "vals"]
    _TIMESCALES = ["sub-seasonal", "seasonal", "long-range"]

    @property
    def FORECAST_VARIABLES(self):
        """Available forecast variables for this zarr store type."""
        return self._FORECAST_VARIABLES

    @property
    def MODELS(self):
        """Available forecast models for this zarr store type."""
        return self._MODELS

    @property
    def REGIONS(self):
        """Available geographic regions for this zarr store type."""
        return self._REGIONS

    @property
    def FIELDS(self):
        """Available forecast fields for this zarr store type."""
        return self._FIELDS

    @property
    def TIMESCALES(self):
        """Available forecast timescales for this zarr store type."""
        return self._TIMESCALES

    def __init__(
        self,
        location: Location,
        field: str | list[str] = "anom",
        model: str | list[str] = "blend",
        timescale: str = "sub-seasonal",
        variable: str = "temp",
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
        key_id: str | None = None,
        key_secret: str | None = None,
        direct_url: str | None = None,
    ):
        """Initialize the ForecastZarr object.

        Args:
            location: The Salient SDK location object
            field: The forecast field or fields to retrieve.
            model: Which model to retrieve.
            timescale: The forecast timescale, one of `sub-seasonal`, `seasonal`, or `long-range`.
            variable: The forecast variable.
            start: The starting forecast date.
            end: The ending forecast date.
            key_id: The Key ID used to access the zarr stores. If not provided will attempt to read
                from the environment variable `SALIENT_DIRECT_ID`.
            key_secret: The Key Secret used to access the zarr stores. If not provided will attempt
                to read from the environment variable `SALIENT_DIRECT_SECRET`.
            direct_url: The URL for the Salient zarr store. If not provided will attempt to read
                from the environment variable `SALIENT_DIRECT_URL`.
        """
        super().__init__(
            location=location,
            field=field,
            model=model,
            timescale=timescale,
            variable=variable,
            start=start,
            end=end,
            key_id=key_id,
            key_secret=key_secret,
            direct_url=direct_url,
        )

    def open_dataset(self, in_memory: bool = False) -> xr.Dataset:
        """Open the dataset.

        Will be returned as dask-backed xr.Dataset, unless `in_memory=True`, and then the values
        will be loaded into memory.
        """
        fs = s3fs.S3FileSystem(
            key=self._key_id,
            secret=self._key_secret,
            client_kwargs=dict(endpoint_url=self._direct_url, region_name="enam"),
            s3_additional_kwargs=dict(ACL="private"),
        )

        fields = [self.field] if isinstance(self.field, str) else self.field
        if self.model == "truth":
            fields = [f"{field}_actual" for field in fields]

        store_path = f"{self.region}-{self.variable}/{self.timescale}/{self.model}"
        # Check is false since we're pointing to a directory, not an object
        store = s3fs.S3Map(root=store_path, s3=fs, check=False, create=False)
        ds = xr.open_zarr(store=store).sel(forecast_date=slice(self.start, self.end))[fields]
        ds = self.subset_location(ds)

        if in_memory:
            self._check_memory(ds)
            ds = ds.load()

        return ds


class ZarrGEM(ZarrBase):
    """Access the native Salient forecast zarr stores.

    Used in order to get dask-backed Xarray datasets to the hindcast catalog of
    Salient forecasts in order to perform large-scale analyses without needing
    to make API calls.
    """

    _BASE_VARS = [
        "cc",
        "hgt500",
        "mslp",
        "precip",
        "rh",
        "tmax",
        "tmin",
        "tsi",
        "wgst",
        "wspd",
        "wspd100",
    ]
    _DERIVED_VARS = ["cdd", "hdd", "heat_index", "wind_chill", "temp"]
    _FORECAST_VARIABLES = _BASE_VARS + _DERIVED_VARS

    _MODELS = ["gem", "baseline"]
    _REGIONS = ["north-america"]
    _FIELDS = ["anom", "vals", "anom_ens", "vals_ens"]
    _TIMESCALES = ["daily"]
    _AVAILABLE_DATES = xr.date_range("2020-10-16", "2025-04-30", freq="D")

    @property
    def FORECAST_VARIABLES(self):
        """Available forecast variables for this zarr store type."""
        return self._FORECAST_VARIABLES

    @property
    def MODELS(self):
        """Available forecast models for this zarr store type."""
        return self._MODELS

    @property
    def REGIONS(self):
        """Available geographic regions for this zarr store type."""
        return self._REGIONS

    @property
    def FIELDS(self):
        """Available forecast fields for this zarr store type."""
        return self._FIELDS

    @property
    def TIMESCALES(self):
        """Available forecast timescales for this zarr store type."""
        return self._TIMESCALES

    def __init__(
        self,
        location: Location,
        field: str | list[str] = "anom",
        model: str | list[str] = "blend",
        timescale: str = "sub-seasonal",
        variable: str = "temp",
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
        key_id: str | None = None,
        key_secret: str | None = None,
        direct_url: str | None = None,
    ):
        """Initialize the ForecastZarr object.

        Args:
            location: The Salient SDK location object
            field: The forecast field or fields to retrieve.
            model: Which model to retrieve.
            timescale: The forecast timescale, one of `sub-seasonal`, `seasonal`, or `long-range`.
            variable: The forecast variable.
            start: The starting forecast date.
            end: The ending forecast date.
            key_id: The Key ID used to access the zarr stores. If not provided will attempt to read
                from the environment variable `SALIENT_DIRECT_ID`.
            key_secret: The Key Secret used to access the zarr stores. If not provided will attempt
                to read from the environment variable `SALIENT_DIRECT_SECRET`.
            direct_url: The URL for the Salient zarr store. If not provided will attempt to read
                from the environment variable `SALIENT_DIRECT_URL`.
        """
        super().__init__(
            location=location,
            field=field,
            model=model,
            timescale=timescale,
            variable=variable,
            start=start,
            end=end,
            key_id=key_id,
            key_secret=key_secret,
            direct_url=direct_url,
        )

    def open_dataset(self, in_memory: bool = False) -> xr.Dataset:
        """Open the dataset.

        Will be returned as dask-backed xr.Dataset, unless `in_memory=True`, and then the values
        will be loaded into memory.
        """
        if not self.start:
            self.start = self._AVAILABLE_DATES.min()
        if not self.end:
            self.end = self._AVAILABLE_DATES.max()

        if isinstance(self.field, str):
            self.field = [self.field]
        ds = xr.merge(
            [
                get_gem_dataset(
                    variables=self.variable,
                    model=self.model,
                    field=field,
                    region=self.region,
                    start=self.start,
                    end=self.end,
                    key_id=self._key_id,
                    key_secret=self._key_secret,
                    direct_url=self._direct_url,
                )
                for field in self.field
            ]
        )
        ds = self.subset_location(ds)

        if in_memory:
            self._check_memory(ds)
            ds = ds.load()

        return ds
