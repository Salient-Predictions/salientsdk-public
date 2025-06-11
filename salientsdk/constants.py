#!/usr/bin/env python
# Copyright Salient Predictions 2025

"""Constants for the Salient SDK.

This module contains constants used throughout the Salient SDK.

"""

import datetime
import hashlib
import itertools
import logging
import os
import shutil
import urllib
from importlib import resources
from typing import Any, Literal, get_args, get_origin

import pandas as pd

# This is the base URL for the Salient API.
# The env-var override is for testing vs development builds.
URL = os.getenv(key="SALIENTSDK_URL", default="https://api.salientpredictions.com/")

API_VERSION = "v2"

MODEL_VERSION = "v9"
MODEL_VERSIONS = ["v8", "v9"]


DESTINATION = "salient_data"


def get_file_destination(destination: str = "-default", make: bool = True) -> str:
    """Get the default destination for downloaded files.

    Args:
        destination (str): If `-default`, return the destination
            set by `set_file_destination`.  Otherwise, return the
            value of `destination`.
        make (bool): If True, create the directory if it doesn't exist.

    Returns:
        str: The destination for downloaded files.
             Note: future versions may return pathlib.Path.  Take care
             using string methods like `.replace()` on the returned value.
    """
    if destination is None:
        return destination

    destination = str(destination)
    if destination == "-default":
        destination = DESTINATION

    if destination is not None and make:
        os.makedirs(destination, exist_ok=True)

    return destination


def set_file_destination(destination: str = "salient_data", force: bool = False) -> str:
    """Set the default destination for downloaded files.

    Args:
        destination (str): The default destination for downloaded files.
        force (bool): If True (default False), clear the cached contents of the
            destination directory.  This will force all subsequent downloads
            to re-download fresh versions of the files.

    Returns:
        str: The destination that was set by the function.

    """
    global DESTINATION
    if destination is not None:
        destination = str(destination)

        if force and os.path.exists(destination):
            shutil.rmtree(destination)

    DESTINATION = destination
    return DESTINATION


def _build_url(
    endpoint: str, args: None | dict = None, destination: str = "-default"
) -> tuple[str, str]:
    """Build an API query and associated file name for a given endpoint and arguments.

    Args:
        endpoint (str): The API endpoint to query.
        args (dict): The arguments to pass to the endpoint.
        destination (str): The destination directory for downloaded files.
            If `-default`, will call `get_file_destination()`

    Returns:
        tuple[str, str]: A tuple containing the query URL and the file name to
            download to.
            File name will be `/<destination>/<endpoint>_<argshash>.<format>`.
    """
    url = URL + API_VERSION + "/" + endpoint
    file_name = endpoint

    destination = get_file_destination(destination)
    if destination is not None:
        file_name = os.path.join(destination, file_name)

    if args:
        # apikey will often be None when we're using a persistent session.
        # Eliminate it (and anything else that's None too.)
        args = {k: v for k, v in args.items() if v is not None}

        url += "?"
        url += urllib.parse.urlencode(args, safe=",")

        # apikey doesn't influence the file contents, so shouldn't be in the hash:
        if "apikey" in args:
            del args["apikey"]

        file_name += "_"
        file_name += hashlib.md5(str(args).encode()).hexdigest()

        if "format" in args:
            file_name += "." + args["format"]

    return (url, file_name)


def _build_urls(
    endpoint: str, args: None | dict = None, destination: str = "-default"
) -> pd.DataFrame:
    """Build URLs for a given endpoint and arguments.

    This is the vectorized version of `_build_url`.

    Args:
        endpoint (str): The API endpoint to query.
        args (dict): The arguments to pass to the endpoint.  If any values
            in `args` are lists or tuples, the function will perform a
            combinatoric expansion on all vectorized arguments.
        destination (str): The destination directory for downloaded files.

    Returns:
        pd.DataFrame: A DataFrame containing the queries and file names.
            Will have a single row if all `args` are scalar, or multiple
            rows if any `args` are vectorized.
    """
    if args:
        vector_args = {k: v for k, v in args.items() if isinstance(v, (list, tuple))}

        if vector_args:
            scalar_args = {k: v for k, v in args.items() if not isinstance(v, (list, tuple))}

            expanded_args = list(itertools.product(*vector_args.values()))
            expanded_args = [dict(zip(vector_args.keys(), values)) for values in expanded_args]
            queries = [
                _build_urls(
                    endpoint=endpoint, args={**arg, **scalar_args}, destination=destination
                ).assign(**arg)
                for arg in expanded_args
            ]
            return pd.concat(queries, ignore_index=True)

    (url, file_name) = _build_url(endpoint, args, destination)
    return pd.DataFrame([{"query": url, "file_name": file_name}])


def _validate_date(date: str | datetime.datetime) -> str:
    if isinstance(date, str) and date == "-today":
        date = datetime.datetime.today()

    if isinstance(date, datetime.datetime):
        date = date.strftime("%Y-%m-%d")

    # ENHANCEMENT: accept other date formats like numpy datetime64, pandas Timestamp, etc
    # ENHANCEMENT: make sure date is properly formatted

    return date


def get_model_version(version: str = "-default") -> str:
    """Get the current default model version.

    Args:
        version (str): If `-default`, return the current model version.
            Otherwise, return the value of `version`.

    Returns:
        str: The current model version

    """
    if version is None or version == "-default":
        version = MODEL_VERSION

    return version


def set_model_version(version: str) -> None:
    """Set the default model version.

    Args:
        version (str): The model version to set

    """
    version = str(version)
    assert version in MODEL_VERSIONS
    global MODEL_VERSION
    MODEL_VERSION = version


def _expand_comma(
    val: str | list[str] | None,
    valid: list[str] | None = None,
    name="value",
    default: str | None = None,
) -> list[str] | str | None:
    """Expand a comma-separated string into a list of strings.

    See also `_collapse_comma()`, which does the opposite.

    Args:
        val (str | list[str] | None): A single string that may contain commas.
            If a list of strings, convert to a single string if length == 1.
            If None, return None.
        valid (list[str] | None): A list of valid values for the string.
            If None (default) no validation is performed.
            If provided, asserts if any `val` is not in `valid`.
        name (str): The name of the value to use in error messages.
            Not used if `valid` is None.
        default (str): If not `None`, will replace a string value of `-default` with this

    Returns:
        list[str] | str | None: A list of strings if commas are present,
            otherwise the original string or list of strings.
    """
    if val is None:
        return val

    if isinstance(val, str) and "," in val:
        val = val.split(",")

    # Check to see if val is a list of strings
    if isinstance(val, list):
        if len(val) == 1:
            val = val[0]

    DEFAULT = "-default"
    if default is not None:
        if isinstance(val, list):
            if DEFAULT in val:
                val[val.index(DEFAULT)] = default
        elif isinstance(val, str):
            val = default if val == DEFAULT else val

    if valid:
        if isinstance(val, list):
            for v in val:
                assert v in valid, f"{name} {v} not in {valid}"
        else:
            assert val in valid, f"{name} {val} not in {valid}"

    return val


def _collapse_comma(val: str | list[str] | None, valid: list[str] | Any | None = None) -> str:
    """Validate and regularize arguments that are natively comma-separated.

    This is used with API endpoints that are natively vectorized and accept
    a comma-separated string of arguments.  This function helps the user to pass
    a either a comma-separated string or a list of strings, just like the other
    api functions that vectorize API calls with only one argument.

    This function is a companion to `_expand_comma()`.

    Args:
        val (str | list[str]): The value to validate, either a comma-separated
            list of strings, or a list of strings
        valid (list[str] | Any | None): The valid values for the argument, if any.
            Can be a list of strings or a typing.Literal type.
            If a Literal type, will extract the valid values automatically.
            Ignored if None.

    Returns:
        str: A comma-separated string of values.

    """
    if val is None:
        return None
    elif isinstance(val, list):
        val_str = ",".join(val)
        val_vec = val
    else:
        val_str = val
        val_vec = val.split(",")

    if valid is not None:
        # If valid is a Literal type, extract the values
        if get_origin(valid) is Literal:
            valid = list(get_args(valid))

        for v in val_vec:
            assert v in valid, f"Invalid argument: {v}"

    # The downstream function is expecting a single comma-separated list
    return val_str


def get_hindcast_dates(
    start_date: str = "2015-01-01",
    end_date: str = "2022-12-31",
    timescale: str = "sub-seasonal",
    extend: bool = False,
) -> list[str]:
    """Get a list of dates matching the Salient reforecast schedule within a given date range.

    This function reads dates from a CSV file and returns a list of
    dates that fall within the specified start and end dates for the given timescale.

    Args:
        start_date (str): The start date of the range in 'YYYY-MM-DD' format.
        end_date (str): The end date of the range in 'YYYY-MM-DD' format.
        timescale (str): The timescale for which to retrieve dates.
           - `sub-seasonal` - blend weeks 1-5
           - `seasonal` - blend months 1-3
           - `long-range` - blend quarters 1-4
           - `ecmwf_ens` - ECMWF ENS daily
        extend (bool): If `True` (default `False`) fill in any dates after the end of
            the hindcast history and `end_date`.

    Returns:
        list[str]: A list of subseasonal dates within the specified range,
            formatted as strings in 'YYYY-MM-DD' format.
    """
    try:
        start_date = pd.Timestamp(start_date)
        end_date = pd.Timestamp(end_date)
    except ValueError as e:
        raise ValueError(f"Invalid date format: {e}. Please use 'YYYY-MM-DD' format.")

    try:
        with resources.files("salientsdk.data").joinpath(f"{timescale}_dates.csv").open("r") as f:
            hindcast_dates = pd.read_csv(f, header=None, parse_dates=[0], index_col=0).index
    except (FileNotFoundError, ValueError):
        raise ValueError(
            f"Unknown timescale {timescale} - use sub-seasonal, seasonal, long-range, or ecmwf_ens"
        )

    date_range = hindcast_dates[
        (hindcast_dates >= start_date) & (hindcast_dates <= end_date)
    ].dropna()

    if extend and len(date_range) > 0:
        last_date = date_range[-1]
        if last_date < end_date:
            last_date = last_date + pd.Timedelta(days=1)
            date_range = date_range.append(pd.date_range(start=last_date, end=end_date, freq="D"))

    return date_range.strftime("%Y-%m-%d").tolist()


def get_logger(name: str) -> logging.Logger:
    """Get logger for logging.

    Args:
        name (str): Name of the logger.

    Returns:
        logging.Logger instance for logging.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(name)
