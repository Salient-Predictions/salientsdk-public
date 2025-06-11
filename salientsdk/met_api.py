#!/usr/bin/env python
# Copyright Salient Predictions 2025

"""Observed data timeseries.

This module acquires observed station meteorological data and converts it into a format
compatible with the `data_timeseries` function.
"""

from collections.abc import Iterable
from typing import Literal

import numpy as np
import pandas as pd
import requests
import xarray as xr

from .constants import _build_urls, _collapse_comma
from .location import Location
from .login_api import download_queries

_EXCLUDE_ARGS = ["force", "session", "verify", "verbose", "destination", "loc", "strict", "kwargs"]


MetVariable = Literal["precip", "temp", "tmax", "tmin", "snow", "wdir", "wspd"]


def met_stations(
    # API arguments -----
    loc: Location,
    variables: MetVariable | list[MetVariable] | None = None,
    start: str | None = "2020-01-01",
    end: str | None = None,
    max_distance: float = 10,
    # Non-API arguments --------
    destination: str = "-default",
    force: bool = False,
    session: requests.Session | None = None,
    apikey: str | None = None,
    verify: bool | None = None,
    verbose: bool = False,
    strict: bool = True,
    **kwargs,
) -> str | pd.DataFrame:
    """Search meteorological stations.

    For more detail, see the
    [api doc](https://api.salientpredictions.com/v2/documentation/api/#/Meteorological%20Stations/met_stations).

    Args:
        loc (Location): The location to query.
            If using a `shapefile` or `location_file`, may input a vector of file names which
            will trigger multiple calls to `downscale`.  This is useful because `downscale` requires
            that all points in a file be from the same continent.
        variables (MetVariable | list[MetVariable] | None): The variables that must be available,
            separated by commas or as a `list`.
            Must be one or more of precip, temp, tmax, tmin, snow, wdir, or wspd.
            Not all data is available at all stations.
            If omitted (default), does not filter stations by variable availability.
        start (str): Makes sure that data is available
            at least as long ago as this date.  If omitted, does not filter station list
            by availability date.
        end (str): Makes sure that data is available at least as recently as this date.
            If omitted (default) does not filter station list by availability date.
        max_distance (float): The maximum allowable distance in kilometers between a coordinate
            and its nearest station. If this distance is exceeded, no station will be returned
            for the coordinate. Only applies when `lat/lon` or `location_file` is specified.
        destination (str): The destination directory for downloaded files.
        force (bool): If False (default), don't download the data if it already exists
        session (requests.Session): The session object to use for the request
        apikey (str | None): The API key to use for the request.
            In most cases, this is not needed if a `session` is provided.
        verify (bool): If True (default), verify the SSL certificate
        verbose (bool): If True (default False) print status messages
        strict (bool): If True (default) error if query is invalid.  If False,
            return `NA` for the file name and continue processing.
        **kwargs: Additional arguments to pass to the API

    Returns:
        str | pd.DataFrame : If only one file was downloaded, return the name of the file.
            If multiple files were downloaded, return a table with column `file_name` and
            additional columns documenting the vectorized input arguments such as
            `location_file`.
    """
    format = "csv"
    variables = _collapse_comma(variables, valid=MetVariable)
    assert max_distance > 0
    args = {k: v for k, v in {**locals(), **kwargs}.items() if k not in _EXCLUDE_ARGS}

    queries = _build_urls(
        endpoint="met_stations", args=loc.asdict(**args), destination=destination
    )

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
        queries = queries.drop(columns="query")
        return queries


def met_observations(
    # API arguments -----
    loc: Location,
    variables: MetVariable | list[MetVariable] = ["tmax", "tmin", "precip"],
    start: str | None = "2020-01-01",
    end: str | None = None,
    max_distance: float = 10,
    format: str = "nc",
    # Non-API arguments --------
    destination: str = "-default",
    force: bool = False,
    session: requests.Session | None = None,
    apikey: str | None = None,
    verify: bool | None = None,
    verbose: bool = False,
    strict: bool = True,
    **kwargs,
) -> str | pd.DataFrame:
    """Fetch meteorological station observations.

    For more detail, see the
    [api doc](https://api.salientpredictions.com/v2/documentation/api/#/Meteorological%20Stations/met_observations).

    Args:
        loc (Location): The location to query.
            If using a `shapefile` or `location_file`, may input a vector of file names which
            will trigger multiple calls to `downscale`.  This is useful because `downscale` requires
            that all points in a file be from the same continent.
        variables (MetVariable | list[MetVariable] | None): The variables to download,
            separated by commas or as a `list`.
            Must be one or more of precip, temp, tmax, tmin, snow, wdir, or wspd.
            Not all data is available at all stations.
        start (str): Makes sure that data is available
            at least as long ago as this date.  If omitted, does not filter station list
            by availability date.
        end (str): Makes sure that data is available at least as recently as this date.
            If omitted (default) does not filter station list by availability date.
        max_distance (float): The maximum allowable distance in kilometers between a coordinate
            and its nearest station. If this distance is exceeded, no station will be returned
            for the coordinate. Only applies when `lat/lon` or `location_file` is specified.
        format (str): `nc` or `csv`
        destination (str): The destination directory for downloaded files.
        force (bool): If False (default), don't download the data if it already exists
        session (requests.Session): The session object to use for the request
        apikey (str | None): The API key to use for the request.
            In most cases, this is not needed if a `session` is provided.
        verify (bool): If True (default), verify the SSL certificate
        verbose (bool): If True (default False) print status messages
        strict (bool): If True (default) error if query is invalid.  If False,
            return `NA` for the file name and continue processing.
        **kwargs: Additional arguments to pass to the API

    Returns:
        str | pd.DataFrame : If only one file was downloaded, return the name of the file.
            If multiple files were downloaded, return a table with column `file_name` and
            additional columns documenting the vectorized input arguments such as
            `location_file`.
    """
    variables = _collapse_comma(variables, valid=MetVariable)
    assert max_distance > 0
    args = {k: v for k, v in {**locals(), **kwargs}.items() if k not in _EXCLUDE_ARGS}

    queries = _build_urls(
        endpoint="met_observations", args=loc.asdict(**args), destination=destination
    )

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
        queries = queries.drop(columns="query")
        return queries


def make_observed_ds(
    obs_df: pd.DataFrame | str | Iterable[pd.DataFrame | str],
    name: str | Iterable[str],
    variable: str | Iterable[str],
    time_col: str = "time",
) -> xr.Dataset:
    """Convert weather observation DataFrame(s) to xarray Dataset.

    This function converts tabular meteorological data into a format identical to
    `data_timeseries(..., frequency='daily') suitable for use with the `crps` function.

    Args:
        obs_df: Single DataFrame or a filename to a CSV that can be read as a dataframe.
            May also be an iterable vector of filenames or `DataFrame`s.
            Each DataFrame should have columns for `time` and the `variable` of interest.
            If the dataframe contains lat, lon, and elev metadata the function will
            preserve thse as coordinates. Function `get_ghcnd` will provide a compatible
            dataset, or you can provide your own.
        name: Station name(s) corresponding to the DataFrame(s). Must be a string if obs_df
            is a single DataFrame, or an iterable of strings matching the length of obs_df
            if multiple DataFrames are provided.
        variable: Name(s) of the column(s) in obs_df to extract the met data
            (e.g. 'temp', 'precip') or ['temp','precip']
        time_col: Name of the column in obs_df containing the time (default `time`)

    Returns:
        xarray Dataset containing the variable data and station metadata. Has dimensions
        'time' and 'location', with coordinates for station lat/lon/elevation.
    """

    def get_attrs(var_name):
        """Helper to collect attributes from a dataframe column."""
        attrs = {"short_name": var_name}
        if hasattr(obs_df[var_name], "attrs"):
            for attr in ["units", "long_name"]:
                if attr in obs_df[var_name].attrs:
                    attrs[attr] = obs_df[var_name].attrs[attr]
        return attrs

    if isinstance(obs_df, Iterable) and not isinstance(obs_df, pd.DataFrame):
        if name is None or isinstance(name, str):
            raise ValueError(
                "When obs_df is a list of DataFrames, name must be an iterable of strings"
            )

        assert len(obs_df) == len(
            name
        ), f"Length mismatch: got {len(obs_df)} DataFrames but {len(name)} names"

        ds = [
            make_observed_ds(obs_df=df, name=n, variable=variable, time_col=time_col)
            for df, n in zip(obs_df, name)
        ]
        return xr.concat(ds, dim="location")

    if isinstance(obs_df, str):
        obs_df = pd.read_csv(obs_df)

    name = str(name)
    attrs = {}

    if isinstance(variable, str):
        data_vars = {"vals": (("time", "location"), obs_df[variable].values[:, np.newaxis])}
        attrs = get_attrs(variable)
    else:
        data_vars = {}
        for var in variable:
            data_vars[var] = (
                ("time", "location"),
                obs_df[var].values[:, np.newaxis],
                get_attrs(var),
            )

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "time": pd.to_datetime(obs_df[time_col]),
            "location": [name],
        },
        attrs=attrs,
    )

    # Preserve station geo-coordinates if they exist as a column
    for coord in ["lat", "lon", "elev"]:
        if coord in obs_df.columns:
            ds = ds.assign_coords({f"{coord}_station": ("location", [obs_df[coord].mean()])})

    return ds
