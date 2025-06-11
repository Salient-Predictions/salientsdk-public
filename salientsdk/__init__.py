#!/usr/bin/env python
# Copyright Salient Predictions 2025

"""Salient Predictions Software Development Kit."""

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import toml

from . import event, hydro, skill, solar, wind  # noqa: F401 access via submodule
from .constants import (
    get_file_destination,
    get_hindcast_dates,
    get_model_version,
    set_file_destination,
    set_model_version,
)
from .data_timeseries_api import data_timeseries, load_multihistory
from .downscale_api import downscale
from .forecast_timeseries_api import forecast_timeseries
from .forecast_zarr import ForecastZarr
from .geo_api import geo
from .hindcast_summary_api import hindcast_summary, transpose_hindcast_summary
from .location import Location, merge_location_data
from .login_api import (
    get_current_session,
    get_verify_ssl,
    login,
    set_current_session,
    set_verify_ssl,
)
from .met_api import met_observations, met_stations
from .upload_file_api import (  # upload_file_example, candidate
    upload_bounding_box,
    upload_file,
    upload_location_file,
    upload_shapefile,
    user_files,
)


def _get_version(pkgname):
    pyproject_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
    if pyproject_path.exists():
        pyproject_content = toml.load(pyproject_path)
        return pyproject_content["tool"]["poetry"]["version"]
    else:
        # Try to import the version from installed package metadata
        # Only works for python v3.9 or later.
        try:
            return version(pkgname)
        except PackageNotFoundError:
            return "unknown"  # Development fallback


__version__ = _get_version("salientsdk")
__author__ = "Salient Predictions"
__all__ = [
    "login",
    "data_timeseries",
    "downscale",
    "ForecastZarr",
    "forecast_timeseries",
    "geo",
    "get_current_session",
    "set_file_destination",
    "get_model_version",
    "get_verify_ssl",
    "hindcast_summary",
    "transpose_hindcast_summary",
    "load_multihistory",
    "Location",
    "merge_location_data",
    "met_stations",
    "met_observations",
    "set_current_session",
    "get_file_destination",
    "get_hindcast_dates",
    "set_model_version",
    "set_verify_ssl",
    "upload_file",
    # "upload_file_example", candidate
    "upload_bounding_box",
    "upload_location_file",
    "upload_shapefile",
    "user_files",
]

if __name__ == "__main__":
    print(f"ver: {__version__} by: {__author__}")
