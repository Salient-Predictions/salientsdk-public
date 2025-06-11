#!/usr/bin/env python
# Copyright Salient Predictions 2025

"""Hindcast Summary statistics.

This module is an interface to the Salient `hindcast_summmary` API, which returns
summary statistics for historical weather forecast quality.

Command line usage example:

```
cd ~/salientsdk
# this will get a single variable in a single file:
python -m salientsdk hindcast_summary -lat 42 -lon -73 -u username -p password --force
# to request multiple variables, separate them with a comma:
python -m salientsdk hindcast_summary -lat 42 -lon -73 --variable temp,precip
# to request variables AND multiple seasons:
python -m salientsdk hindcast_summary -lat 42 -lon -73 --variable temp,precip --season DJF,MAM

```

"""

import numpy as np
import pandas as pd
import requests

from .constants import _build_url, _build_urls, _expand_comma
from .location import Location
from .login_api import download_queries

REFERENCE_VALUES = [
    "-auto",
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
MODEL_VALUES = [m for m in REFERENCE_VALUES if m not in ("-auto", "ai")]


METRIC_VALUES = ["rps", "mae", "crps", "rps_skill_score", "mae_skill_score", "crps_skill_score"]
SEASON_VALUES = ["all", "DJF", "MAM", "JJA", "SON"]
SPLIT_SET_VALUES = ["all", "test", "validation"]
INTERP_METHOD_VALUES = ["nearest", "linear"]


COL_FILE = "file_name"
ENDPOINT = "hindcast_summary"


def hindcast_summary(
    # API inputs -------
    loc: Location,
    metric: str = "crps",
    model: str = "blend",
    reference: str = "-auto",
    season: str | list[str] = "all",
    split_set: str = "all",
    timescale="all",
    variable: str | list[str] = "temp",
    version: str | list[str] = "-default",
    interp_method: str = "nearest",
    # non-API arguments ---
    destination: str = "-default",
    force: bool = False,
    session: requests.Session | None = None,
    apikey: str | None = None,
    verify: bool | None = None,
    verbose: bool = False,
    **kwargs,
) -> str:
    """Get summary of accuracy metrics.

    This function is a convenience wrapper to the Salient
    [API](https://api.salientpredictions.com/v2/documentation/api/#/Validation/hindcast_summary).

    Args:
        loc (Location): The location to query
        metric (str): The accuracy metric to calculate. Defaults to "crps".
        interp_method (str): The interpolation method to use.
            - `nearest` (default) selects the nearest native gridpoint
            - `linear` interpolates from the native grid to the input point
        model (str): Forecast model. Defaults to "blend".
        reference (str): The reference dataset to compare against. Defaults to "-auto".
        season (str | list[str]): Meteorological season to consider.
            Defaults to "all".
            May also be a list of strings or a comma-separated string to vectorize the query.
            Also valid: DJF, MAM, JJA, SON.
        split_set (str): The date range over which to calculate scores.
            `all` (default) calculates scores back to 2000.
            `test` uses post-2015 data which was never used for model training.
        timescale (str): Forecast look-ahead.
            - `sub-seasonal` is 1-5 weeks.  Will return a coordinate `forecast_date_weekly` and
                a data variable `anom_weekly` or `vals_weekly`.
            - `seasonal` is 1-3 months.  Will return a coordinate `forecast_date_monthly` and a
                data variable `anom_monthly` or `vals_monthly`.
            - `long-range` is 1-4 quarters.  Will return a coordinate `forecast_date_quarterly` and a
                data variable `anom_quarterly` or `vals_quarterly`.
            - `all` (default) will include `sub-seasonal`, `seasonal`, and `long-range` timescales
        variable (str | list[str]): The variable to query, defaults to `temp`
            To request multiple variables, separate them with a comma `temp,precip` or use a list.
            This will download one file per variable
            See the
            [Data Fields](https://salientpredictions.notion.site/Variables-d88463032846402e80c9c0972412fe60)
            documentation for a full list of available historical variables.
        version (str | list[str]): The model version of the Salient `blend` forecast.
            To compare multiple versions, provide a list or comma-separated string.
            `-default` calls `get_default_version()`.
        destination (str): The destination directory for downloaded files.
            `-default` uses `get_file_destination()`
        force (bool): If False (default), don't download the data if it already exists
        session (requests.Session): The `Session` object to use for the request.
            Defaults to use get_current_session(), typically set during `login()`.
        apikey (str | None): The API key to use for the request.
            In most cases, this is not needed if a `session` is provided.
        verify (bool): Verify the SSL certificate.
            Defaults to use get_verify_ssl(), typically set during `login()`.
        verbose (bool): If True (default False) print status messages.
        **kwargs: Additional arguments to pass to the API

    Keyword Arguments:
        units (str): `SI` or `US`

    Returns:
        str: the file name of the downloaded data.
            File names are a hash of the query parameters.
            When `force=False` and the file already exists, the function will return the file name
            almost instantaneously without querying the API.
            If multiple files are requested, they will be concatenated into a single table.
    """
    format = "csv"  # hardcode, not supporting "table" in the sdk

    # These args aren't natively vectorized in the API, so we'll do it here
    variable = _expand_comma(variable)
    season = _expand_comma(season, SEASON_VALUES, "season")

    assert split_set in SPLIT_SET_VALUES, f"split_set must be one of {SPLIT_SET_VALUES}"
    assert model in MODEL_VALUES, f"model must be one of {MODEL_VALUES}"
    assert reference in REFERENCE_VALUES, f"reference must be one of {REFERENCE_VALUES}"
    # We can't expand_comma on "metric" since it changes the column headers
    # and messes up the multi-file concatenation.
    assert metric in METRIC_VALUES, f"metric must be one of {METRIC_VALUES}"
    assert (
        interp_method in INTERP_METHOD_VALUES
    ), f"interp_method must be one of {INTERP_METHOD_VALUES}"

    if reference == "-auto":
        reference = "gfs"

    args = loc.asdict(
        metric=metric,
        reference=reference,
        season=season,
        split_set=split_set,
        timescale=timescale,
        variable=variable,
        version=version,
        interp_method=interp_method,
        format=format,
        model=model,
        apikey=apikey,
        **kwargs,
    )

    queries = _build_urls(ENDPOINT, args, destination)

    download_queries(
        query=queries["query"].values,
        file_name=queries[COL_FILE].values,
        force=force,
        session=session,
        verify=verify,
        verbose=verbose,
        format=format,
    )

    file_name = _concatenate_hindcast_summary(queries, format, destination)

    if verbose:
        print(f"Saving combined table to {file_name}")

    return file_name


def _concatenate_hindcast_summary(
    queries: pd.DataFrame, format: str = "csv", destination: str = "-default"
) -> str:
    file_names = queries[COL_FILE].values
    if len(file_names) == 1:
        # Most of the time, we'll only have downloaded a single file.
        # No need to concatenate.
        return file_names[0]

    scores = [
        pd.read_csv(row[COL_FILE]).assign(
            **{col: row[col] for col in queries.columns if col not in ["query", COL_FILE]}
        )
        for index, row in queries.iterrows()
    ]

    # Check for unique column names for columns 2 and 3 (zero-based indexing)
    # The "Reference <metric>" and "Salient <metric>" column have units.
    # If they conflict, it will concatenate each into separate columns.
    # Rename columns to strip units for a clean concatenation.
    def standardize_column(dfs, col_idx):
        """Standardize column names across all DataFrames by removing units."""
        col_names = [df.columns[col_idx] for df in dfs]
        if len(set(col_names)) > 1:
            common_root = col_names[0].split(" (")[0]
            for df in dfs:
                df.rename(columns={df.columns[col_idx]: common_root}, inplace=True)

    standardize_column(scores, 2)  # Reference CRPS
    standardize_column(scores, 3)  # Salient CRPS

    scores = pd.concat(scores, ignore_index=True)

    # We don't care abut the url here - just calling it to get a consistent filename
    [url, file_name] = _build_url(
        endpoint=ENDPOINT,
        args={"concatenate_file_names": str(file_names), "format": format},
        destination=destination,
    )

    scores.to_csv(file_name, index=False)

    return file_name


def transpose_hindcast_summary(
    scores: str | pd.DataFrame,
    min_score: float = 0.0,
    weight_weeks=1.0,
    weight_months=0.5,
    weight_quarters=0.25,
) -> pd.DataFrame:
    """Transpose hindcast_summary long to wide, preserving groups.

    Transposes the hindcast summary data from a long format to a wide format
    where each 'Lead' row becomes a column.  Adds a column `mean` with a
    weighted average of the scores.

    Parameters:
        scores (str | pd.DataFrame): The hindcast scores data.
            This can be either a file path as a string or a pre-loaded DataFrame
            of the type returned by `hindcast_summary()`
        min_score (float): Render any scores below this threshold as NA
        weight_weeks (float): Weight for weeks 3-5, defaults to 1.0
        weight_months (float): Weight for months 2-3, defaults to 0.5
        weight_quarters (float): Weight for quarters 2-4, defaults to 0.25

    Returns:
        pd.DataFrame: The transposed DataFrame with 'Lead' categories as columns.
    """
    if isinstance(scores, str):
        scores = pd.read_csv(scores)

    # The first 5 columns are standard columns that are always present
    # Any additional columns represent the vectorized arguments
    # (and may not exist if arguments were not vectorized)

    # set the table index to be columns 6 to end:
    unstack_by = "Lead"
    vector_cols = scores.columns[5:].tolist()
    index_cols = vector_cols + [unstack_by]
    extract_col = 4  # This should be the relative skill score
    if isinstance(extract_col, int):
        extract_col = scores.columns[extract_col]

    scores.set_index(index_cols, inplace=True)
    scores = scores[[extract_col]]

    # Adds new rows with Lead="mean"
    scores = _add_mean(scores, weight_weeks, weight_months, weight_quarters)

    if len(index_cols) == 1:
        # If there is no vector expansion we don't need an unstack.
        # A simple transpose will do.
        scores = scores.T
    else:
        # Preserve the order of the original table rows
        unstack_row = scores.index.get_level_values(unstack_by).unique()
        scores = scores.unstack(unstack_by)
        # restore the original order of the rows, now in column form
        scores = scores[extract_col][unstack_row]

    # make any scores below min_score NA
    scores = scores.where(scores >= min_score)

    return scores


def _add_mean(scores: pd.DataFrame, weeks=1.0, months=0.5, quarters=0.25) -> pd.DataFrame:
    """Add a row with the mean of the scores.

    This adds a set of rows to the DataFrame with the mean of the scores
    weighted by the number of weeks, months, or quarters in the forecast lead.

    Args:
        scores (pd.DataFrame): The DataFrame of scores as returned by `hindcast_summary()`
        weeks (int): Weight for weeks 3-5
        months (float): Weight for months 2-3
        quarters (float): Weight for quarters 2-4

    Returns:
        pd.DataFrame: The DataFrame with the mean rows added.
    """
    # get the names of the index columns of scores
    unstack_by = "Lead"
    index_cols = scores.index.names
    vector_cols = index_cols.copy()
    vector_cols.remove(unstack_by)

    WGT = "Weight"
    weights = {
        "Week 1": 0,
        "Week 2": 0,
        "Week 3": weeks,
        "Week 4": weeks,
        "Week 5": weeks,
        "Month 1": 0,
        "Month 2": months,
        "Month 3": months,
        "Months 1-3": 0,
        "Months 4-6": quarters,
        "Months 7-9": quarters,
        "Months 10-12": quarters,
    }
    extract_col = scores.columns[0]

    weights = pd.DataFrame.from_dict(weights, orient="index", columns=[WGT])
    weights.index.name = unstack_by
    scores = scores.merge(weights, how="left", left_index=True, right_index=True)

    def _weighted_mean(group):
        if WGT in group.columns and extract_col in group.columns:
            # Ensure there are no NaN values in weights or the extract_col
            group = group.dropna(subset=[WGT, extract_col])
            if not group.empty:
                return np.average(group[extract_col], weights=group[WGT]).round(2)

    if len(vector_cols) == 0:
        avg = pd.DataFrame({extract_col: [_weighted_mean(scores)], unstack_by: ["mean"]})
    else:
        avg = (
            scores.groupby(level=vector_cols)  # don't include "Lead" in the groupby
            .apply(_weighted_mean)
            .reset_index(name=extract_col)
        )
        avg[unstack_by] = "mean"
    avg.set_index(index_cols, inplace=True)

    scores.drop(columns=WGT, inplace=True)  # don't need this anymore
    scores = pd.concat([scores, avg], ignore_index=False, axis=0)
    return scores
