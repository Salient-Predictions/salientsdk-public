#!/usr/bin/env python
# Copyright Salient Predictions 2025

"""Interface to the Salient data_timeseries API.

Command line usage example:
```
cd ~/salientsdk
python -m salientsdk downscale -lat 42 -lon -73 --date 2020-01-01 -u username -p password --force
```

"""

import os
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import xarray as xr

from .constants import _build_urls, _collapse_comma, _expand_comma
from .location import Location
from .login_api import download_queries

REFERENCE_CLIMS = ["30_yr", "10_yr", "5_yr", "salient"]
FREQUENCY = ["daily", "hourly"]

_EXCLUDE_ARGS = ["force", "session", "verify", "verbose", "destination", "loc", "strict", "kwargs"]


def downscale(
    # API arguments -----
    loc: Location,
    variables: str | list[str] = "temp,precip",
    date: str = "-today",
    members: int = 50,
    debias: bool = False,
    frequency="daily",
    reference_clim: str = "salient",
    version: str | list[str] = "-default",
    weights: str | None = None,  # noqa
    length: int = 366,
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
    """Temporal downscale of forecasts.

    Convert temporally coarse probabilistic forecasts into granular daily ensembles.
    For more detail, see the
    [api doc](https://api.salientpredictions.com/v2/documentation/api/#/Forecasts/downscale).

    Args:
        loc (Location): The location to query.
            If using a `shapefile` or `location_file`, may input a vector of file names which
            will trigger multiple calls to `downscale`.  This is useful because `downscale` requires
            that all points in a file be from the same continent.
        variables (str | list[str]): The variables to query, separated by commas or as a `list`
            See the
            [Data Fields](https://salientpredictions.notion.site/Variables-d88463032846402e80c9c0972412fe60)
            documentation for a full list of available variables.
            Note that `downscale` natively supports a list of variables, so passing a list of
            variables here will not necessarily trigger downloading multiple files.
        date (str): The start date of the time series.
            If `date` is `-today`, use the current date.
        members (int): The number of ensemble members to download
        frequency (str): The temporal resolution of the time series, `daily` (default) or `hourly`.
        reference_clim (str): Reference period to calculate anomalies
        debias (bool): If True, debias the data to observation stations
        version (str): The model version of the Salient `blend` forecast.
        weights (str): The variable that will be used to weight each grid point within a
            `shapefile`.  Defaults to unweighted, can also weight by `population`.
        length (int): The number of days to downscale for. Defaults to 366.
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

    Keyword Arguments:
        gdd_base (int): The base temperature for growing degree days
        units (str): US or SI

    Returns:
        str | pd.DataFrame : If only one file was downloaded, return the name of the file.
            If multiple files were downloaded, return a table with column `file_name` and
            additional columns documenting the vectorized input arguments such as
            `location_file`.
    """
    if os.getenv("USE_MOCK_DOWNSCALE", "False") == "True":
        # pytest --nbmake does not support mock fixtures, requiring a more manual
        # approach to mocking when running notebooks as tests:
        try:
            return _downscale_mock(
                loc=loc,
                variables=variables,
                date=date,
                members=members,
                frequency=frequency,
                destination=destination,
                verbose=verbose,
                **kwargs,
            )
        except NotImplementedError as err:
            # mocking not available for all cases
            pass

    assert members > 0, "members must be a positive integer"
    assert reference_clim in REFERENCE_CLIMS, f"reference_clim must be one of {REFERENCE_CLIMS}"

    if not isinstance(length, int) or length <= 0:
        raise TypeError("length must be a positive integer")
    elif length > 366:
        raise ValueError("length must be less than 366 days")

    format = "nc"
    model = "blend"
    date = datetime.today().strftime("%Y-%m-%d") if date == "-today" else date
    variables = _collapse_comma(variables)
    frequency = _expand_comma(frequency, valid=FREQUENCY, name="frequency", default="daily")
    args = {k: v for k, v in {**locals(), **kwargs}.items() if k not in _EXCLUDE_ARGS}

    queries = _build_urls(endpoint="downscale", args=loc.asdict(**args), destination=destination)

    # return file_name because strict=False will NA failures
    queries["file_name"] = download_queries(
        query=queries["query"].values,
        file_name=queries["file_name"].values,
        force=force,
        session=session,
        verify=verify,
        verbose=verbose,
        format=format,
        max_workers=5,  # downscale @limiter.limit("5 per second")
        strict=strict,
    )

    if len(queries) == 1:
        return queries["file_name"].values[0]
    else:
        # Now that we've executed the queries, we don't need it anymore:
        queries = queries.drop(columns="query")
        return queries


def _downscale_mock(
    loc: Location,
    variables: str | list[str] = "temp,precip",
    date: str = "2021-01-01",
    members: int = 21,
    # debias: bool = False,
    frequency="daily",
    # reference_clim: str = "salient",
    # version: str | list[str] = "-default",
    # weights: str | None = None,  # noqa
    # Non-API arguments --------
    destination: str = "-default",
    force: bool = False,
    # session: requests.Session | None = None,
    # apikey: str | None = None,
    # verify: bool | None = None,
    verbose: bool = False,
    **kwargs,
) -> str | pd.DataFrame:
    """Generate synthetic data quickly instead of calling the API."""
    format = "nc"
    model = "blend"
    date = datetime.today().strftime("%Y-%m-%d") if date == "-today" else date
    variables = _collapse_comma(variables)
    mock = True  # generate a different file_name than the real API call
    args = {k: v for k, v in {**locals(), **kwargs}.items() if k not in _EXCLUDE_ARGS}

    queries = _build_urls(endpoint="downscale", args=loc.asdict(**args), destination=destination)
    if len(queries) > 1:
        raise NotImplementedError("mock downscale expects scalar values")
    file_name = queries["file_name"].values[0]
    if os.path.exists(file_name) and not force:
        return file_name

    freq, time_cnt, tdim = (
        ("h", 8760, "time") if frequency == "hourly" else ("d", 365, "forecast_day")
    )
    variables = _expand_comma(variables)
    time = pd.date_range(start=date, periods=time_cnt, freq=freq)

    if loc.shapefile is not None:
        raise NotImplementedError("mock shapefile not available")
        # Shapefile downscales are shaped like:
        # (tdim,"ensemble","lat","lon")
    else:
        if loc.location_file is not None:
            geo = loc.load_location_file(destination=destination)
        elif loc.lat is not None:
            geo = pd.DataFrame({"name": ["0"], "lat": [loc.lat], "lon": [loc.lon]})
        loc_cnt = len(geo)

        data_vars = {
            var: ((tdim, "ensemble", "location"), np.random.rand(time_cnt, members, loc_cnt))
            for var in variables
        }
        coords = {
            tdim: (tdim, time),
            "ensemble": ("ensemble", np.array(range(members))),
            "location": ("location", geo["name"]),
            "lat": ("location", geo["lat"]),
            "lon": ("location", geo["lon"]),
            "forecast_date": np.datetime64(date, "ns"),
        }

    ds = xr.Dataset(data_vars=data_vars, coords=coords)
    if verbose:
        print(ds)
    ds.to_netcdf(file_name, encoding={"location": {"dtype": str}})
    return file_name
