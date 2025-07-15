#!/usr/bin/env python
# Copyright Salient Predictions 2025

"""Historical data timeseries.

This module is an interface to the Salient `data_timeseries` API, which returns historical
observed data.  It also includes utility functions for operating on the returned data.

Command line usage example:

```
cd ~/salientsdk
# this will get a single variable in a single file:
python -m salientsdk data_timeseries -lat 42 -lon -73 -fld all --start 2020-01-01 --end 2020-12-31 --force  -u username -p password
# this will get multiple variables in separate files:
python -m salientsdk data_timeseries -lat 42 -lon -73 -fld all -var temp,precip -u username -p password
```

"""

from datetime import datetime
from typing import Literal

import dask
import numpy as np
import pandas as pd
import requests
import xarray as xr

from .constants import Format, Weights, _build_urls, _expand_comma, _validate_enum
from .location import Location
from .login_api import download_queries

Frequency = Literal["hourly", "daily", "weekly", "monthly", "3-monthly"]
Field = Literal[
    "anom", "anom_d", "anom_ds", "anom_qnt", "anom_s", "clim", "stdv", "trend", "vals", "all"
]
HourlyVariable = Literal[
    "cc",
    "precip",
    "sm",
    "snow",
    "st",
    "temp",
    "tsi",
    "dhi",
    "dni",
    "wdir",
    "wdir100",
    "wgst",
    "wspd",
    "wspd100",
]

_EXCLUDE_ARGS = ["force", "session", "verify", "verbose", "destination", "strict", "loc", "kwargs"]


def data_timeseries(
    # API inputs -------
    loc: Location,
    variable: str | list[str] | None = "temp",
    field: Field | list[Field] = "anom",
    debias: bool = False,  # noqa: vulture
    start: str = "1950-01-01",
    end: str = "-today",
    format: Format = "nc",
    frequency: Frequency = "daily",
    weights: Weights = None,
    custom_quantity: str | list[str] | None = None,
    # non-API arguments ---
    destination: str = "-default",
    force: bool = False,
    session: requests.Session | None = None,
    apikey: str | None = None,
    verify: bool | None = None,
    verbose: bool = False,
    **kwargs,
) -> str | pd.DataFrame:
    """Get a historical time series of ERA5 data.

    This function is a convenience wrapper to the Salient
    [API](https://api.salientpredictions.com/v2/documentation/api/#/Historical/get_data_timeseries).

    Args:
        loc (Location): The location to query.
            If using a `shapefile` or `location_file`, may input a vector of file names which
            will trigger multiple calls to `data_timeseries`.
        variable (str | list[str] | None): The variable to query, defaults to `temp`
            To request multiple variables, separate them with a comma `temp,precip`
            This will download one file per variable
            See the
            [Data Fields](https://salientpredictions.notion.site/Variables-d88463032846402e80c9c0972412fe60)
            documentation for a full list of available historical variables.
        custom_quantity (str | list[str] | None): Name of a previously-uploaded custom quantity definition.
            Defaults to `None`.  If specified, ignores `variable`.
        field (str): The field to query, defaults to "anom"
        debias (bool): If True, debias the data to local observations.
            Disabled for `shapefile` locations.
            [detail](https://salientpredictions.notion.site/Debiasing-2888d5759eef4fe89a5ba3e40cd72c8f)
        start (str): The start date of the time series
        end (str): The end date of the time series
        format (Format): The file format of the response
        frequency (Frequency): The temporal frequency of the time series
        weights (Weights): Aggregation mechanism if using a `shapefile`.
        destination (str): The directory to download the data to
        force (bool): If False (default), don't download the data if it already exists
        session (requests.Session): The session object to use for the request.
            If `None` (default) uses `get_current_session()`.
        apikey (str | None): The API key to use for the request.
            In most cases, this is not needed if a `session` is provided.
        verify (bool): If True (default), verify the SSL certificate
        verbose (bool): If True (default False) print status messages
        **kwargs: Additional arguments to pass to the API

    Keyword Arguments:
        units (str): `SI` or `US`

    Returns:
        str | pd.DataFrame:
            the file name of the downloaded data.  File names are a hash of the query parameters.
            When `force=False` and the file already exists, the function will return the file name
            almost instantaneously without querying the API.
            If multiple variables are requested, returns a `pd.DataFrame` with columns `file_name`
            and additional columns documenting the vectorized input arguments such as `location_file`
            or `variable`
    """
    format = _validate_enum(format, Format, name="format")
    frequency = _validate_enum(frequency, Frequency, name="frequency")
    weights = _validate_enum(weights, Weights, name="weights")

    custom_quantity = _expand_comma(custom_quantity, name="custom_quantity", default=None)
    if custom_quantity is None:
        variable = _expand_comma(
            variable, HourlyVariable if frequency == "hourly" else None, "variable"
        )
        field = _expand_comma(field, valid=Field, name="field")
        if field != "vals" and frequency == "hourly":
            raise ValueError("Only field `vals` is available for hourly frequency")
    else:
        # Ignore these if custom_quantity is specified
        variable = None
        field = None

    args = {k: v for k, v in {**locals(), **kwargs}.items() if k not in _EXCLUDE_ARGS}

    endpoint = "data_timeseries"
    queries = _build_urls(endpoint, loc.asdict(**args), destination)

    download_queries(
        query=queries["query"].values,
        file_name=queries["file_name"].values,
        force=force,
        session=session,
        verify=verify,
        verbose=verbose,
        format=format,
    )

    if len(queries) == 1:
        return queries["file_name"].values[0]
    else:
        # Now that we've executed the queries, we don't need it anymore:
        queries = queries.drop(columns="query")

        # load_multihistory needs either variable or custom_quantity to specify short_name
        if "variable" in queries:
            pass
        elif "custom_quantity" in queries:
            pass
        elif variable is not None:
            queries["variable"] = variable
        elif custom_quantity is not None:
            queries["custom_quantity"] = custom_quantity
        else:
            raise ValueError("Must specify variable or custom_quantity")

        return queries


def _load_history_row(row: pd.DataFrame, fields: list[str] = ["vals"]) -> xr.Dataset:
    """Load a single history file and prepare for merging with others."""
    variable = row["variable"] if "variable" in row else "variable"

    hst = xr.load_dataset(row["file_name"])
    hst = hst[fields]
    fields_new = [variable if field == "vals" else variable + "_" + field for field in fields]
    hst = hst.rename({field: field_new for field, field_new in zip(fields, fields_new)})
    for fld in fields_new:
        hst[fld].attrs = hst.attrs
    hst.attrs = {}

    if "location_file" in row:
        # Preserve the provenance of the source location_file
        location_files = np.repeat(row["location_file"], len(hst.location))
        hst = hst.assign_coords(location_file=("location", location_files))

    hst.close()

    return hst


def load_multihistory(files: pd.DataFrame, fields: list[str] = ["vals"]) -> xr.Dataset:
    """Load multiple .nc history files and merge them into a single dataset.

    Args:
        files (pd.DataFramme): Table of the type returned by
            `data_timeseries` when multiple `variable`s, `location_file`s
            or `shapefile`s are requested
            e.g. `data_timeseries(..., variable = "temp,precip")`

        fields (list[str]): List of fields to extract from the history files.
            Useful if when calling `data_timeseries(..., field = "all")`

    Returns:
        xr.Dataset: The merged dataset, where each field and variable is renamed
            to `<variable>_<field>` or simply `variable` if field = "vals".
            This will cause the format of a multi-variable file to match the data
            variable names of `downscale`, which natively supports multi-variable queries.
    """
    hst = [_load_history_row(row, fields) for _, row in files.iterrows()]
    hst = xr.merge(hst)
    return hst


def stack_history(
    hist: xr.Dataset | str,
    forecast_date: xr.DataArray | np.ndarray,
    lead: xr.DataArray | np.ndarray,
    compute: bool = True,
) -> xr.Dataset:
    """Restructure historical observations to match the structure of stacked forecast data.

    Args:
        hist: Historical observation dataset or path to dataset file
        forecast_date: Array of forecast dates
        lead: Array of lead times
        compute: If False, lazy-load datasets and delay computation (default True)

    Returns:
        xarray Dataset with historical observations structured like forecast data aligned
               along `forecast_date`.
    """
    if isinstance(hist, str):
        hist = xr.load_dataset(hist) if compute else xr.open_dataset(hist, chunks={})

    forecast_date = forecast_date.values if hasattr(forecast_date, "values") else forecast_date
    lead = lead.values if hasattr(lead, "values") else lead

    # We only want to stack variables dependent on "time"
    time_vars = [var_name for var_name, var in hist.data_vars.items() if "time" in var.dims]
    # Variables dependent on "day of year" should pass through unchanged:
    doy_vars = [var_name for var_name, var in hist.data_vars.items() if "time" not in var.dims]
    time_ds = hist[time_vars]
    # lead values returned by forecast_timeseries are 1-indexed.
    offset = -pd.Timedelta("1D")

    # Sort the time index to ensure proper initialization of pandas' index engine.
    # This prevents "Reindexing only valid with uniquely valued Index objects" errors
    # that can occur during dask's delayed computation setup. The sorting also
    # ensures a cleaner, ordered dataset for subsequent operations.
    time_ds = time_ds.assign_coords(time=time_ds.time.to_index().sort_values())

    def process_forecast_date(f_date):
        """Extract relevant dates and denominate by lead."""
        return (
            time_ds.sel(time=pd.Timestamp(f_date) + lead + offset)
            .assign_coords(lead=("time", lead))
            .swap_dims({"time": "lead"})
            .expand_dims(dim={"forecast_date": [pd.Timestamp(f_date)]})
        )

    if not compute:
        delayed_datasets = [
            dask.delayed(process_forecast_date)(f_date) for f_date in forecast_date
        ]
        computed_datasets = dask.compute(*delayed_datasets)
    else:
        computed_datasets = [process_forecast_date(f_date) for f_date in forecast_date]

    combined_time_ds = xr.concat(computed_datasets, dim="forecast_date")

    # Restore dayofyear non-time denominated variables if they exist
    if doy_vars:
        result_ds = xr.merge([combined_time_ds, hist[doy_vars]])
    else:
        result_ds = combined_time_ds

    # Determine chunking dynamically based on dataset dimensions
    if not compute:
        chunk_dict = {"forecast_date": 1}
        if "lead" in result_ds.dims:
            chunk_dict["lead"] = -1
        for dim in result_ds.dims:
            if dim not in chunk_dict and dim != "time":
                chunk_dict[dim] = -1
        result_ds = result_ds.chunk(chunk_dict)

    # Preserve attributes from the original dataset
    for attr_name, attr_value in hist.attrs.items():
        result_ds.attrs[attr_name] = attr_value

    return result_ds


def extrapolate_trend(
    # data_timeseries inputs -------
    loc: Location,
    variable: str | list[str] = "temp",
    # climo-specific inputs -------
    start: str | datetime | pd.Timestamp = "-today",
    end: str
    | datetime
    | pd.Timestamp
    | pd.tseries.offsets.BaseOffset
    | pd.DateOffset = pd.DateOffset(years=5),
    # Other args passed to data_timeseries ----
    stdv_mult: float = 0,
    verbose: bool = False,
    **kwargs,
) -> str | pd.DataFrame:
    """Use Salient's 30-year linear trend to generate a per-day timeseries on for any date range.

    Args:
        loc (Location): The location to query.
            If using a `shapefile` or `location_file`, may input a vector of file names which
            will trigger multiple calls to `data_timeseries`.
        variable (str | list[str]): The variable to query, defaults to `temp`
            To request multiple variables, separate them with a comma `temp,precip`
            This will download one file per variable
            See the
            [Data Fields](https://salientpredictions.notion.site/Variables-d88463032846402e80c9c0972412fe60)
            documentation for a full list of available historical variables.
        start (str | datetime | pd.Timestamp, optional):
            The start date for the projection.
            Default is "-today", which uses the current date.
        end (str | datetime | pd.Timestamp | pd.tseries.offsets.BaseOffset | pd.DateOffset, optional):
            The end date for the projection.
            Can be a specific date or an offset from the start date.
            Supports python offset aliases such as "5YE" for "5 year end" from now.
            Default is 5 years from the start date.
        stdv_mult (float): number of standard deviations above/below `clim` & `trend` to calculate (default 0)
        verbose (bool): If True (default False) print status messages
        **kwargs: Additional arguments passed to `data_timeseries`.

    Keyword Arguments:
        units (str): `SI` or `US`
        destination (str): The directory to download the data to
        force (bool): If False (default), don't download the data if it already exists
        session (requests.Session): The session object to use for the request.
            If `None` (default) uses `get_current_session()`.
        apikey (str | None): The API key to use for the request.
            In most cases, this is not needed if a `session` is provided.
        verify (bool): If True (default), verify the SSL certificate
    """
    field = ["clim", "trend", "stdv"]
    clim_files = data_timeseries(
        loc=loc,
        variable=variable,
        field=field,
        debias=False,  # climo doesn't support debias
        start="2020-01-01",  # unused - set for caching purposes
        end="2020-01-01",  # unused  - set for caching purposes
        format="nc",
        frequency="daily",  # future enhancement - hourly
        verbose=verbose,
        **kwargs,
    )

    # Reshape the list of downloaded files so that clim/trend/stdv get their own columns
    clim_files = clim_files.pivot(
        index=["variable"]
        + [col for col in clim_files.columns if col not in ["field", "file_name", "variable"]],
        columns="field",
        values="file_name",
    ).reset_index()
    if verbose:
        print(clim_files)

    date_range = _get_date_range(start, end)

    clim_timeseries = _compute_trend(
        clim=clim_files.clim,
        trend=clim_files.trend,
        stdv=clim_files.stdv,
        stdv_mult=stdv_mult,
        name=clim_files.variable,
        date_range=date_range,
    )

    return clim_timeseries


def _get_date_range(start="-today", end=pd.DateOffset(years=1)) -> pd.DatetimeIndex:
    """Turn a start and end date into a vector of dates."""
    start = (
        datetime.today() if isinstance(start, str) and start == "-today" else pd.to_datetime(start)
    )
    try:
        end = start + pd.tseries.frequencies.to_offset(end)
    except ValueError:
        end = pd.to_datetime(end)
    return pd.date_range(start.date(), end.date(), freq="D")


def _compute_trend(
    clim: xr.Dataset | str | list[str] | pd.Series,
    trend: xr.Dataset | str | list[str] | pd.Series,
    stdv: xr.Dataset | str | list[str] | pd.Series,
    stdv_mult: float = 0,
    name: str | list[str] | pd.Series = "value",
    date_range: pd.DatetimeIndex = _get_date_range(),
) -> xr.Dataset:
    """Extrapolates climatology and trend over a specified date range.

    This is the "worker" function for `extrapolate_trend` once all of the necessary
    inputs have been corralled.

    Args:
        clim (xr.Dataset | str | list[str] | pd.Series):
            The climatology dataset or path to the dataset, of the type returned by
            `data_timeseries(field="clim",...).
            Can also be a list or series of datasets/paths.
        trend (xr.Dataset | str | list[str] | pd.Series):
            The trend dataset or path to the dataset, of the type returned by
            `data_timeseries(field="trend",...).
            Can also be a list or series of datasets/paths.
        stdv (xr.Dataset | str | list[str] | pd.Series):
            The standard deviation dataset or path to the dataset, of the type returned by
            `data_timeseries(field="stdv",...).
            Can also be a list or series of datasets/paths.
        stdv_mult (float): standard deviation multiplier.  defaults to 0.
        name (str | list[str] | pd.Series, optional):
            The name for the output variable in the resulting dataset.
            Default is "value".
        date_range (pd.DatetimeIndex): the dates to project clim & trend onto.

    Returns:
        xr.Dataset: A dataset containing the projected climate data over the specified date range.
            Each named variable will be a separate `DataArray`.

    Raises:
        AssertionError: If `clim`, `trend`, `stdv`, and `name` are lists or series,
            they must all be of the same length.
    """
    vec_type = (list, pd.Series)
    if isinstance(clim, vec_type):
        assert isinstance(trend, vec_type) and isinstance(name, vec_type)
        assert len(clim) == len(trend) == len(stdv) == len(name)
        return xr.merge(
            [
                _compute_trend(c, t, s, stdv_mult, n, date_range)
                for c, t, s, n in zip(clim, trend, stdv, name)
            ]
        )

    DOY = "dayofyear"

    clim = xr.load_dataset(clim) if isinstance(clim, str) else clim
    trend = xr.load_dataset(trend) if isinstance(trend, str) else trend
    stdv = xr.load_dataset(stdv) if isinstance(stdv, str) else stdv

    year = date_range.year.values[:, None]
    dayofyear = date_range.dayofyear.values

    # select corresponding days, in order, from the dataset
    clim_days = clim["clim"].sel(dayofyear=dayofyear)
    trend_days = trend["trend"].sel(dayofyear=dayofyear)
    stdv_days = stdv["stdv"].sel(dayofyear=dayofyear)

    # Find the center of the climatology from the attributes:
    clim_start = int(clim.attrs["clim_start"][0:4])  # 2019
    clim_end = int(clim.attrs["clim_end"][0:4])  # 1990
    clim_length = clim_end - clim_start + 1  # 30
    clim_center = clim_start + clim_length / 2  # 2005

    climo_with_trend = (
        clim_days + trend_days * (year - clim_center) / clim_length + stdv_mult * stdv_days
    )
    climo_with_trend["time"] = ((DOY), date_range)

    ds_future = climo_with_trend.to_dataset(name=name).swap_dims({DOY: "time"}).drop(DOY)
    ds_future[name].attrs = clim.attrs
    return ds_future
