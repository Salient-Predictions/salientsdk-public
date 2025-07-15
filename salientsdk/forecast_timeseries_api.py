#!/usr/bin/env python
# Copyright Salient Predictions 2025

"""Forecast data timeseries.

This module is an interface to the Salient `forecast_timeseries` API, which returns
probabilistic weather forecasts in subseasonal-to-seasonal timescales.

Command line usage example:

```
cd ~/salientsdk
# this will get a single variable in a single file:
python -m salientsdk forecast_timeseries -lat 42 -lon -73 --timescale seasonal -u username -p password
# this will get multiple variables in separate files:
python -m salientsdk forecast_timeseries -lat 42 -lon -73 -var temp,precip --timescale seasonal
```

"""

from datetime import datetime
from typing import Literal

import pandas as pd
import requests
import xarray as xr

from .constants import Format, Weights, _build_urls, _expand_comma, _validate_enum
from .location import Location
from .login_api import download_queries

Model = Literal[
    "gem",
    "clim",
    "gfs",
    "ecmwf",
    "blend",
    "salient_clim",
    "norm_5yr",
    "norm_10yr",
    "noaa_gfs",
    "noaa_gefs",
    "noaa_gefs_calib",
    "ecmwf_ens",
    "ecmwf_ens_calib",
    "ecmwf_seas5",
    "ecmwf_seas5_calib",
    "ai",
]

Variable = Literal[
    "cc",
    "cdd",
    "hdd",
    "heat_index",
    "hgt500",
    "mslp",
    "precip",
    "rh",
    "temp",
    "tmax",
    "tmin",
    "tsi",
    "wgst",
    "wind_chill",
    "wspd",
    "wspd100",
]

Field = Literal["anom", "vals", "anom_ens", "vals_ens"]

_EXCLUDE_ARGS = ["force", "session", "verify", "verbose", "destination", "strict", "loc", "kwargs"]


def forecast_timeseries(
    # API inputs -------
    loc: Location,
    date: str | list[str] = "-default",
    debias: bool = False,  # noqa: vulture
    field: Field | None = "anom",
    format: Format = "nc",
    model: Model | list[Model] = "blend",
    reference_clim: str = "salient",
    timescale="all",
    variable: Variable | list[Variable] | None = "temp",
    version: str | list[str] = "-default",
    weights: Weights = None,
    custom_quantity: str | list[str] | None = None,
    # non-API arguments ---
    destination: str = "-default",
    force: bool = False,
    session: requests.Session | None = None,
    apikey: str | None = None,
    verify: bool | None = None,
    verbose: bool = False,
    strict: bool = True,
    **kwargs,
) -> str | pd.DataFrame:
    """Get time series of S2S meteorological forecasts.

    This function is a convenience wrapper to the Salient
    [API](https://api.salientpredictions.com/v2/documentation/api/#/Forecasts/forecast_timeseries).

    Args:
        loc (Location): The location to query
        date (str | list[str]): The date the forecast was generated.
            Defaults to find the most recent forecast.
            Can be a specific date in the format `YYYY-MM-DD`.
            May be a vector of dates or a comma separated list of dates, in which case the function will return a `DataFrame`.
        debias (bool): If True, debias the data to local observations.
            Valid only for `model = blend`.
            [detail](https://salientpredictions.notion.site/Debiasing-2888d5759eef4fe89a5ba3e40cd72c8f)
        field (str): The field to query, defaults to `anom` which is an anomaly value from climatology.
            Also available: `vals`, which will return absolute values without regard to climatology.
        format (str): The file format of the response.
            Defaults to `nc` which returns a multivariate NetCDF file.
            Also available: `csv` which returns a CSV file.
        model (str | list[str]): The model to query.
            Defaults to `blend`, which is the Salient blended forecast.
            May be a list of strings or comma-separated string, which downloads multiple models.
        reference_clim (str):  Reference climatology for calculating anomalies.
            Ignored when `field=vals` since there are no anomalies to calculate.
            Defaults to `salient`, which is Salient's proprietary climatology.
        timescale (str): Forecast look-ahead.
            - 'daily' is valid for models 'gem' and 'noaa_gefs'
            - `sub-seasonal` is 1-5 weeks.  Will return a coordinate `forecast_date_weekly` and
                a data variable `anom_weekly` or `vals_weekly`.
            - `seasonal` is 1-3 months.  Will return a coordinate `forecast_date_monthly` and a
                data variable `anom_monthly` or `vals_monthly`.
            - `long-range` is 1-4 quarters.  Will return a coordinate `forecast_date_quarterly` and a
                data variable `anom_quarterly` or `vals_quarterly`.
            - `all` (default) will include `sub-seasonal`, `seasonal`, and `long-range` timescales
        variable (str | list[str] | None): The variable(s) to query, defaults to `temp`
            To request multiple variables, separate them with a comma `temp,precip` or use a `list`.
            This will download one file per variable
            See the
            [Data Fields](https://salientpredictions.notion.site/Variables-d88463032846402e80c9c0972412fe60)
            documentation for a full list of available historical and forecast variables.
        custom_quantity (str | list[str] | None): Name of a previously-uploaded custom quantity definition.
            Defaults to `None`.  If specified, ignores `variable`.
        version (str): The model version of the Salient `blend` forecast.
            To request multiple versions, provide a list or comma-separated string.
            `-default` calls `get_default_version()`.
        weights (Weights): Aggregation mechanism if using a `shapefile`.
            Default `None` performs no aggregation.
        destination (str): The destination directory for downloaded files.
            `-default` uses `get_file_destination()`
        force (bool): If False (default), don't download the data if it already exists
        session (requests.Session): The session object to use for the request
        apikey (str | None): The API key to use for the request.
            In most cases, this is not needed if a `session` is provided.
        verify (bool): If True (default), verify the SSL certificate.
            Defaults to use the value returned by `get_verify_ssl()`
        verbose (bool): If True (default False) print status messages
        strict (bool): If True (default) error if query is invalid.  If False,
            return `NA` for the file name and continue processing.
        **kwargs: Additional arguments to pass to the API

    Keyword Arguments:
        units (str): `SI` or `US`

    Returns:
        str | pd.DataFrame: the file name(s) of the downloaded data.
            File names are a hash of the query parameters.
            When `force=False` and the file already exists, the function will return the file name
            almost instantaneously without querying the API.
            If multiple variables, dates, or models are requested,
            returns a `DataFrame` with column `file_name` additional columns for vectorized queries.
    """
    format = _validate_enum(format, Format, name="format")
    weights = _validate_enum(weights, Weights, name="weights")

    date = _expand_comma(date, name="date", default=datetime.today().strftime("%Y-%m-%d"))
    custom_quantity = _expand_comma(custom_quantity, name="custom_quantity", default=None)
    if custom_quantity is None:
        variable = _expand_comma(variable, Variable, "variable", default="temp")
        assert field in ["anom", "vals", "anom_ens", "vals_ens"], f"Invalid field {field}"
    else:
        # Ignore these args if custom_quantity is specified
        variable = None
        field = None

    model = _expand_comma(model, Model, default="blend")

    # Validate parameters that are used dynamically via locals()
    assert isinstance(debias, bool), f"debias must be bool, got {type(debias)}"

    args = {k: v for k, v in {**locals(), **kwargs}.items() if k not in _EXCLUDE_ARGS}

    queries = _build_urls(
        endpoint="forecast_timeseries", args=loc.asdict(**args), destination=destination
    )

    # return file_name because strict=False will NA failures
    queries["file_name"] = download_queries(
        query=queries["query"].values,
        file_name=queries["file_name"].values,
        force=force,
        session=session,
        verify=verify,
        verbose=verbose,
        format=format,
        strict=strict,
    )

    if len(queries) == 1:
        return queries["file_name"].values[0]
    else:
        queries = queries.drop(columns="query")  # not needed anymore
        return queries


def stack_forecast(
    ds: xr.Dataset | str | list[str] | pd.DataFrame,
    compute: bool = True,
) -> xr.Dataset:
    """Align forecast_timeseries dataset(s) along `forecast_date` as a dimension.

    Args:
        ds: Input data, which can be:
            - `xr.Dataset`: A single forecast dataset with a `forecast_date` coordinate
            - `str`: Path to a single forecast dataset file
            - `list[str]`: List of paths to multiple forecast dataset files
            - `pd.DataFrame`: DataFrame with `file_name` column containing paths
        compute: If `False`, lazy-load datasets and delay computation (default `True`)

    Returns:
        xarray Dataset with `forecast_date` as a dimension and a `time` coordinate
        that depends on `forecast_date` and `lead`.

    See Also:
        `stack_history`: Similar functionality for historical data alignment.
    """
    FCST_DIM = "forecast_date"
    LEAD_DIM = "lead"
    if isinstance(ds, xr.Dataset) or isinstance(ds, xr.DataArray):
        assert FCST_DIM in ds.coords  # Ignore forecast_date_[weekly|monthly|quarterly]
        assert LEAD_DIM in ds.coords  # Ignore lead_[weekly|monthly|quarterly]
        forecast_date = pd.Timestamp(ds[FCST_DIM].values)
        leads = ds[LEAD_DIM].values
        offset = -pd.Timedelta("1D")  # In API v2, "lead" is 1-indexed. May change in the future.
        ds = ds.drop_dims("nbnds") if "nbnds" in ds.dims else ds
        time_values = forecast_date + leads + offset
        return (
            ds.drop_vars(FCST_DIM)
            .assign_coords({"time": ("lead", time_values)})
            .expand_dims(dim={FCST_DIM: [forecast_date]})
        )
    elif isinstance(ds, str):
        args = {"decode_timedelta": True}
        opened = xr.load_dataset(ds, **args) if compute else xr.open_dataset(ds, **args)
        return stack_forecast(opened, compute=compute)
    elif isinstance(ds, pd.DataFrame):
        return stack_forecast(ds.file_name)
    elif isinstance(ds, list) or isinstance(ds, pd.Series):
        ds = pd.Series(ds).dropna().tolist()
        stacked = xr.open_mfdataset(
            ds,
            preprocess=stack_forecast,
            concat_dim=FCST_DIM,
            combine="nested",
            decode_timedelta=True,
        )
        if compute:
            stacked = stacked.compute()
        return stacked
    else:
        raise TypeError(f"Unsupported input type: {type(ds)}")
