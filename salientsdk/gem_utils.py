"""GEM forecast data utilities and adapters.

This module provides utilities for loading, processing, and adapting GEM (Global Environmental
Multiscale) forecast data from zarr stores. It includes functions for data clipping, quantile
computation, coordinate promotion, and format adaptation for API compatibility.

Key functionality:
- Loading GEM and baseline forecast data from S3-compatible zarr stores
- Variable clipping to enforce physical constraints (e.g., non-negative precipitation)
- Converting ensemble forecasts to quantile-based representations
- Adapting data formats for backwards compatibility with existing APIs
- Computing derived meteorological quantities
"""
from typing import Literal

import pandas as pd
import s3fs
import xarray as xr

from .derived import compute_quantity

CLIPPERS = dict(
    precip={"min": 0},
    rh={"min": 0, "max": 1},
    tsi={"min": 0},
    wspd={"min": 0},
    wspd100={"min": 0},
    wgst={"min": 0},
    cc={"min": 0, "max": 1},
)
QUANTILES = [
    0.01,
    0.025,
    0.05,
    0.1,
    0.15,
    0.2,
    0.25,
    0.3,
    0.35,
    0.4,
    0.45,
    0.5,
    0.55,
    0.6,
    0.65,
    0.7,
    0.75,
    0.8,
    0.85,
    0.9,
    0.95,
    0.975,
    0.99,
]


def _create_fs(key_id: str, key_secret: str, direct_url: str) -> s3fs.S3FileSystem:
    fs = s3fs.S3FileSystem(
        key=key_id,
        secret=key_secret,
        client_kwargs=dict(endpoint_url=direct_url, region_name="enam"),
        s3_additional_kwargs=dict(ACL="private"),
    )
    return fs


def _load_baseline(
    fs: s3fs.S3FileSystem,
    region: str,
    date: str | None = None,
    limit_samples: bool = True,
    chunks: dict | None = None,
) -> xr.Dataset:
    store_path = f"gem-{region}/baseline"
    # Check is false since we're pointing to a directory, not an object
    store = s3fs.S3Map(root=store_path, s3=fs, check=False, create=False)

    ds = xr.open_zarr(store, chunks=None, decode_timedelta=True).drop_vars("historical_date")
    chunks = chunks or dict(forecast_date=1, lead=7, sample=-1, lat=60, lon=60)

    # Handle date selection separately for the one year of baseline forecasts
    if date is not None:
        ds = ds.sel(forecast_date=[pd.Timestamp(date).replace(year=2024)])
        ds = ds.drop_vars("forecast_date").assign_coords(forecast_date=[pd.Timestamp(date)])

    ds = _subselect_and_chunk(ds, None, limit_samples, chunks=chunks)

    return ds


def _load_gem(
    fs: s3fs.S3FileSystem,
    region: str,
    date: str | None = None,
    limit_samples: bool = True,
    chunks: dict | None = None,
) -> xr.Dataset:
    store_path = f"gem-{region}/forecast"
    store = s3fs.S3Map(root=store_path, s3=fs, check=False, create=False)
    ds = xr.open_zarr(store, chunks=None, decode_timedelta=True)
    chunks = chunks or dict(forecast_date=1, lead=7, sample=-1, lat=60, lon=60)
    ds = _subselect_and_chunk(ds, date, limit_samples, chunks=chunks)
    return ds


def _align_baseline(ds: xr.Dataset, baseline: xr.Dataset) -> xr.Dataset:
    """Merge the baseline dataset into the main dataset for arbitrary range of forecast dates.

    The `baseline` dataset represents a climatology, but instead of having a `dayofyear` dimension,
    it has a `forecast_date` dimension so that it can easily be used seamlessly with the GEM
    dataset. However, its forecast dates are only for 2024, so to join to a range of non-2024
    `forecast_date`s in the GEM dataset, we loop through the years present in the GEM dataset
    to easily select the correct climatology for the baseline, then concatenate these datasets
    together so that the GEM and baseline datasets are aligned.

    Args:
        ds: The GEM dataset to align to. Can have any arbitrary range of `forecast_date`s
        baseline: The baseline dataset (i.e., climo) to join with the GEM dataset.

    Returns:
        A `baseline` dataset with the same `forecast_date`s as the GEM dataset to represent
            the climatology to be used for each `forecast_date`.
    """
    ds_years = set(ds.forecast_date.dt.year.values)
    datasets = []
    for year in sorted(ds_years):
        year_str = f"{year}"
        forecast_dates = ds.sel(forecast_date=slice(year_str, year_str)).forecast_date.values
        data = baseline.sel(
            forecast_date=[pd.Timestamp(date).replace(year=2024) for date in forecast_dates]
        )
        datasets.append(
            data.drop_vars("forecast_date").assign_coords(
                forecast_date=[pd.Timestamp(date) for date in forecast_dates]
            )
        )

    return xr.concat(datasets, dim="forecast_date").transpose(*list(ds.dims))


def _subselect_and_chunk(
    ds: xr.Dataset,
    date: str | None = None,
    limit_samples: bool = False,
    chunks: dict | None = None,
) -> xr.Dataset:
    if "ensemble" in ds.dims:
        ds = ds.rename(ensemble="sample")

    if date:
        ds = ds.sel(forecast_date=[date])

    if limit_samples:
        ds = ds.isel(sample=slice(100))

    if chunks is not None:
        ds = ds.chunk(chunks)

    return ds


def _clip_variables(ds: xr.Dataset) -> xr.Dataset:
    """Clip variable to defined min/max values.

    This is used because we use a heavy compression scheme on the store, so it's possible
    to have very small (<1e-3) negative values for say, precipitation or tsi, and we don't
    want to return these invalid values.
    """
    clipped_vars = set(ds.data_vars).intersection(CLIPPERS.keys())
    if clipped_vars:
        for var in clipped_vars:
            ds[var] = ds[var].clip(**CLIPPERS[var])
    return ds


def _promote_coords(
    ds: xr.Dataset, coords_to_promote: list[str] = ["forecast_date", "lead"]
) -> xr.Dataset:
    """Promote coordinates to dimensions if they are not already dimensions."""
    for coord in coords_to_promote:
        if coord in ds.coords and coord not in ds.dims:
            ds = ds.expand_dims(coord)
    return ds


def _get_quantiles(
    ds: xr.Dataset,
    quantiles: float | list[float] = QUANTILES,
    interp_method: str = "linear",
    quantile_dim_name: str = "quantiles",
    skipna: bool = True,
) -> xr.Dataset:
    """Convert v10 forecast from sample space to quantile space.

    Running `ds.quantile` will drop any coordinates that are single-valued that are not
    also dimensions so we promote the `forecast_date` and `lead` coordinates if they are
    not already dimensions.
    """
    ds = _promote_coords(ds, coords_to_promote=["forecast_date", "lead"])
    return ds.quantile(
        quantiles, dim="sample", method=interp_method, keep_attrs=True, skipna=skipna
    ).rename({"quantile": quantile_dim_name})


def v10_adapter(ds: xr.Dataset, decode_timedelta: bool = True) -> xr.Dataset:
    """Adapt the GEM dataset to the old daily GEFS format for use in API."""
    dims = ["forecast_date", "lead"]
    dims = dims + ["lat", "lon"] if "lat" in ds.dims and "lon" in ds.dims else dims
    dims = dims + ["ensemble"] if "sample" in ds.dims else dims
    renamer = {"sample": "ensemble"} if "sample" in ds.dims else {}

    if decode_timedelta:
        leads = ds.lead
    else:
        leads = pd.to_timedelta([l for l in ds.lead.values], unit="D")

    ds_new = ds.assign_coords({"lead": leads}).rename(**renamer).transpose(*dims, ...)

    for v in ds_new.data_vars:
        anomaly = " anomaly" if "anom" in str(v) else ""
        ensembles = " (ensembles)" if "ens" in str(v) else ""
        ds_new[v].attrs["long_name"] += anomaly + ensembles

    if not decode_timedelta:
        ds_new = ds_new.assign_coords(lead=ds_new.lead.dt.days)
        ds_new["lead"].attrs["units"] = "days"

    return ds_new


def get_gem_dataset(
    variables: str | list[str],
    model: Literal["gem", "baseline"] = "gem",
    field: str | list[str] | None = ["vals_ens"],
    date: str | pd.Timestamp | None = None,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    region: Literal["north-america"] = "north-america",
    decode_timedelta: bool = True,
    limit_samples: bool = False,
    chunks: dict | None = None,
    quantiles: float | list[float] = QUANTILES,
    adapt: bool = True,
    key_id: str | None = None,
    key_secret: str | None = None,
    direct_url: str | None = None,
    **kwargs,
) -> xr.Dataset:
    """Primary data adapter for GEM forecasts.

    The v10 dataset is stored in a "base" zarr store with all variables. Returns from this store
    will apend the field to the variable name, e.g., `temp_vals`, `precip_anom`, etc. to conform
    with other API integrations.

    Args:
        variables: Variable or list of variables to return.
        model: Model to return data for, either "gem" or "baseline".
        field: Field type to return, either "vals", "anom", or "vals_ens"/"anom_ens".
            If None, defaults to ["vals_ens"].
        date: Specific date to return data for. If None, returns all available dates.
        start: Start date for date range selection.
        end: End date for date range selection.
        region: The modeling region, currently only "north-america" supported.
        decode_timedelta: Whether to decode lead time as timedelta objects.
            If False, returns integer day values.
        limit_samples: Whether to limit samples to first 100 to reduce memory usage.
        chunks: Custom chunking configuration for the dataset.
        quantiles: Quantiles to compute for ensemble-to-quantile conversion.
        adapt: Whether to adapt dataset to legacy GEFS format for API usage.
        key_id: S3 access key ID for zarr store authentication.
        key_secret: S3 secret key for zarr store authentication.
        direct_url: Direct URL endpoint for S3-compatible zarr store.
        **kwargs: Additional arguments passed to loading functions.

    Returns:
        Processed xarray Dataset with requested variables and transformations applied.

    Raises:
        ValueError: If unsupported model is specified.
    """
    if isinstance(field, str):
        field = [field]

    if isinstance(variables, str):
        variables = [variables]

    fs = _create_fs(key_id, key_secret, direct_url)

    if model == "gem":
        ds = _load_gem(fs, region, date=date, limit_samples=limit_samples, chunks=chunks, **kwargs)
    elif model == "baseline":
        ds = _load_baseline(
            fs, region, date=date, limit_samples=limit_samples, chunks=chunks, **kwargs
        )
    else:
        raise ValueError(f"Model {model} not supported")

    ds = _clip_variables(ds)

    ds_processed = xr.Dataset()
    for v in variables:
        ds_processed[v] = compute_quantity(ds, v)
        ds_processed[v].attrs.pop("kind", None)

    ds = ds_processed
    # Drop timedelta encoding after so we can compute derived quantities properly
    if not decode_timedelta:
        ds = ds.assign_coords(lead=ds.lead.dt.days)
        ds["lead"].attrs["units"] = "days"

    ds = ds.rename({v: f"{v}_vals_ens" for v in variables})

    if "anom" in field or "anom_ens" in field:
        # Reduce the forecast date range as the baseline alignment is an expensive operation
        if start and end:
            ds = ds.sel(forecast_date=slice(start, end))
        # Call recursively to get baseline forecast as the climo reference
        baseline = get_gem_dataset(
            variables,
            model="baseline",
            field="vals_ens",
            date=date,
            region=region,
            decode_timedelta=decode_timedelta,
            adapt=False,  # Don't adapt until we compute anoms
            key_id=key_id,
            key_secret=key_secret,
            direct_url=direct_url,
            **kwargs,
        )
        baseline = _align_baseline(ds, baseline)
        baseline_mean = baseline.mean(dim="sample", keep_attrs=True)
        with xr.set_options(keep_attrs=True):
            ds = ds - baseline_mean
        ds = ds.rename({f"{v}_vals_ens": f"{v}_anom_ens" for v in variables})

    if "anom" in field or "vals" in field:
        v_a = "vals" if "vals" in field else "anom"
        ds = _get_quantiles(ds, quantiles=quantiles, skipna=False)
        ds = ds.rename({f"{v}_{v_a}_ens": f"{v}_{v_a}" for v in variables})

    if adapt:
        ds = v10_adapter(ds, decode_timedelta=decode_timedelta)

    if start and end:
        ds = ds.sel(forecast_date=slice(start, end))

    return ds
