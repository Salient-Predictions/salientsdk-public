#!/usr/bin/env python
# Copyright Salient Predictions 2025

"""Interface to the Salient geo API.

Command line usage example:
```
cd ~/salientsdk
python -m salientsdk geo -lat 42 -lon -73 -u username -p password --force
```

"""

import pandas as pd
import requests
import xarray as xr

from .constants import _build_urls, _collapse_comma
from .location import Location
from .login_api import download_queries

_EXCLUDE_ARGS = ["force", "session", "verify", "verbose", "destination", "loc", "kwargs"]

VARIABLES = [
    # Elevation
    "elevation",
    "slope",
    # Population
    "population",
    "pop_density",
    # Soil
    "bdod",
    "cec",
    "cfvo",
    "clay",
    "nitrogen",
    "phh2o",
    "sand",
    "silt",
    "soc",
    "ocd",
    "ocs",
    "wv0010",
    "wv0033",
    "wv1500",
    # Land Use / Land Cover
    "fal",
    "lai_hv",
    "lai_lv",
    "lulc_bgc",
    "lulc_bgc_per",
    "lulc_igbp",
    "lulc_igbp_per",
    "lulc_lai",
    "lulc_lai_per",
    "lulc_land_water",
    "lulc_land_water_per",
    "lulc_lccs_cover",
    "lulc_lccs_cover_per",
    "lulc_lccs_hydro",
    "lulc_lccs_hydro_per",
    "lulc_lccs_use",
    "lulc_lccs_use_per",
    "lulc_pft",
    "lulc_pft_per",
    "lulc_umd",
    "lulc_umd_per",
]

RESOLUTIONS = [1 / 4, 1 / 8, 1 / 16]


def geo(
    # API arguments -----
    loc: Location,
    variables: str | list[str] = "elevation",
    resolution: float = 0.25,
    start: str | None = None,
    end: str | None = None,
    format: str = "nc",
    # Non-API arguments --------
    destination: str = "-default",
    force: bool = False,
    session: requests.Session | None = None,
    apikey: str | None = None,
    verify: bool | None = None,
    verbose: bool = False,
    **kwargs,
) -> str | pd.DataFrame:
    """Get static geo-data.

    Args:
        loc (Location): The location to query.
            If using a `shapefile` or `location_file`, may input a vector of file names which
            will trigger multiple calls to `downscale`.  This is useful because `downscale` requires
            that all points in a file be from the same continent.
        variables (str): The variables to query, defaults to "elevation".
            Supports a comma separated list or list of variables.
        resolution (float): The spatial resolution of the data in degrees.
        start (str): The start date of the time series (optional).
        end (str): The end date of the time series (optional).
        format (str): The file format of the response.
            Defaults to `nc` which returns a multivariate NetCDF file.
            Also available: `csv` which returns a CSV file.
        destination (str): The destination directory for downloaded files.
        force (bool): If False (default), don't download the data if it already exists
        session (requests.Session): The session object to use for the request
        apikey (str | None): The API key to use for the request.
            In most cases, this is not needed if a `session` is provided.
        verify (bool): If True (default), verify the SSL certificate
        verbose (bool): If True (default False) print status messages
        **kwargs: Additional arguments to pass to the API

    Returns:
        str | pd.DataFrame : If only one file was downloaded, return the name of the file.
            If multiple files were downloaded, return a table with column `file_name` and
            additional columns documenting the vectorized input arguments such as
            `location_file`.
    """
    assert resolution in RESOLUTIONS, f"Resolution must be one of {RESOLUTIONS}"
    variables = _collapse_comma(variables, VARIABLES)

    args = {k: v for k, v in {**locals(), **kwargs}.items() if k not in _EXCLUDE_ARGS}
    queries = _build_urls(endpoint="geo", args=loc.asdict(**args), destination=destination)

    download_queries(
        query=queries["query"].values,
        file_name=queries["file_name"].values,
        force=force,
        session=session,
        verify=verify,
        verbose=verbose,
        format=format,
        max_workers=5,  # geo @limiter.limit("5 per second")
    )

    if len(queries) == 1:
        return queries["file_name"].values[0]
    else:
        queries = queries.drop(columns="query")
        return queries


def add_geo(
    ds: str | xr.Dataset,
    **kwargs,
) -> xr.Dataset:
    """Get static geo-data and add it to an existing dataset.

    Args:
        ds (str | xr.Dataset): The dataset (or filename to a dataset) to add geo parameters to
        **kwargs: passed to `geo()`

    Keyword Args:
        loc (Location): The location to query, by lat/lon, shapefile, or location_file.
        variables (str): The variables to query, defaults to "elevation".

    Returns:
        xr.Dataset: `ds` with `variables` added as a data variables
    """
    geo_data = xr.load_dataset(geo(**kwargs))

    if isinstance(ds, str):
        ds = xr.load_dataset(ds)

    ds = xr.merge([ds, geo_data], combine_attrs="override", compat="override")

    return ds
