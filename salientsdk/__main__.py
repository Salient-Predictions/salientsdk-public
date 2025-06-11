#!/usr/bin/env python
# Copyright Salient Predictions 2025

"""Command line interface for the Salient SDK."""

import argparse
import os
from importlib import resources
from pathlib import Path

import pandas as pd
import xarray as xr

from .__init__ import __version__
from .data_timeseries_api import data_timeseries
from .downscale_api import downscale
from .forecast_timeseries_api import forecast_timeseries
from .geo_api import geo
from .hindcast_summary_api import hindcast_summary
from .location import Location
from .login_api import get_verify_ssl, login


def main() -> None:
    """Command line interface for the Salient SDK."""
    parser = _get_parser()

    # Convert arguments into action -------
    args = parser.parse_args()
    args_dict = vars(args)
    cmd = args_dict.pop("command")

    args_dict = _login_from_arg(args_dict)
    args_dict = _location_from_arg(args_dict)

    # Dispatch to the appropriate function based on the command
    if cmd == "forecast_timeseries":
        file_name = forecast_timeseries(**args_dict)
    elif cmd == "downscale":
        file_name = downscale(**args_dict)
    elif cmd == "data_timeseries":
        file_name = data_timeseries(**args_dict)
    elif cmd == "geo":
        file_name = geo(**args_dict)
    elif cmd == "hindcast_summary":
        file_name = hindcast_summary(**args_dict)
    elif cmd == "version":
        file_name = __version__
        args.verbose = False
    elif cmd == "login":
        file_name = args_dict["session"]
    elif cmd == "examples":
        file_name = "\n".join(_list_examples())
    else:
        # print(f"Command '{cmd}' not recognized")
        parser.print_help()
        file_name = None

    if file_name is None:
        pass
    elif (
        "verbose" in args
        and args.verbose
        and isinstance(file_name, str)
        and os.path.exists(file_name)
    ):
        ext = os.path.splitext(file_name)[1]
        if ext == ".nc":
            print(xr.open_dataset(file_name, decode_timedelta=True))
        elif ext == ".csv":
            print(pd.read_csv(file_name).head())
        else:
            print(file_name)
    else:
        print(file_name)


def _location_from_arg(arg: dict) -> dict:
    if all(key in arg for key in ["latitude", "longitude", "location_file", "shapefile"]):
        if not "region" in arg:
            # "region" is only used by hindcast_summary
            arg["region"] = None

        arg["loc"] = Location(
            lat=arg.pop("latitude"),
            lon=arg.pop("longitude"),
            location_file=arg.pop("location_file"),
            shapefile=arg.pop("shapefile"),
            region=arg.pop("region"),
        )

    return arg


def _login_from_arg(arg: dict) -> dict:
    if all(key in arg for key in ["username", "password", "apikey"]):
        arg["session"] = login(
            username=arg.pop("username"),
            password=arg.pop("password"),
            apikey=arg["apikey"],  # don't pop, we need this later
            verify=arg.pop("verify"),  # will reset momentarily
            verbose=arg["verbose"],  # don't pop, used elsewhere
        )
        # login() may set this as a side effect:
        arg["verify"] = get_verify_ssl()
    return arg


def _get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="salientsdk command line interface")
    subparsers = parser.add_subparsers(dest="command")

    # version command ---------
    subparsers.add_parser("version", help="Print the Salient SDK version")

    # examples command ---------
    subparsers.add_parser("examples", help="List example notebooks")

    # login command ----------
    _add_common_args(
        subparsers.add_parser("login", help="Login to the Salient API"),
        [],
        location_args=False,
        login_args=True,
        force=False,
    )

    # forecast_timeseries command ------------
    forecast_parser = _add_common_args(
        subparsers.add_parser("forecast_timeseries", help="Run the forecast_timeseries function"),
        ["date", "debias", "version", "variable"],
    )
    forecast_parser.add_argument("-fld", "--field", type=str, default="anom")
    forecast_parser.add_argument("-fmt", "--format", type=str, default="nc")
    forecast_parser.add_argument("-mdl", "--model", type=str, default="blend")
    forecast_parser.add_argument("-ref", "--reference_clim", type=str, default="30_yr")
    forecast_parser.add_argument("--timescale", type=str, default="all")

    # downscale command -----------
    downscale_parser = _add_common_args(
        subparsers.add_parser("downscale", help="Run the downscale function"),
        ["date", "debias", "version"],
    )
    # Note plural "variables", not "variable"
    downscale_parser.add_argument("-var", "--variables", type=str, default="temp,precip")
    downscale_parser.add_argument("--members", type=int, default=50)
    # These are currently the only available values.  Not necessary to expose
    # these as options from the command line.
    # argparser.add_argument("--format", type=str, default="nc")
    # argparser.add_argument("--frequency", type=str, default="daily")

    # data_timeseries command --------------
    data_parser = _add_common_args(
        subparsers.add_parser(
            "data_timeseries", help="Get historical ERA5 from the data_timeseries API"
        ),
        ["debias", "variable"],
    )
    data_parser.add_argument("-fld", "--field", type=str, default="anom")
    data_parser.add_argument("--start", type=str, default="1950-01-01")
    data_parser.add_argument("--end", type=str, default="-today")
    data_parser.add_argument("--format", type=str, default="nc")
    data_parser.add_argument("--frequency", type=str, default="daily")

    # hindcast_summary command ------------
    hnd = _add_common_args(
        subparsers.add_parser(
            "hindcast_summary",
            help="Get summary statistics for historical weather forecast quality",
        ),
        ["variable", "version", "force", "region"],
    )
    hnd.add_argument(
        "-met", "--metric", type=str, default="crps", help="The accuracy metric to calculate"
    )
    hnd.add_argument(
        "-ref",
        "--reference",
        type=str,
        default="-auto",
        help="The reference dataset to compare against",
    )
    hnd.add_argument("--season", type=str, default="all", help="The season to consider")
    hnd.add_argument("--split_set", type=str, default="all", help="The split set to use")
    hnd.add_argument("--timescale", type=str, default="all", help="Forecast look-ahead timescale")

    # geo command --------------
    geo_parser = _add_common_args(
        subparsers.add_parser("geo", help="Get static geo data from the geo API.")
    )
    geo_parser.add_argument(
        "-var",
        "--variables",
        type=str,
        default="temp,precip",
        help="Comma-separated list of variables to return.",
    )
    geo_parser.add_argument(
        "-fmt", "--format", type=str, default="nc", help="Format of the returned data."
    )
    geo_parser.add_argument(
        "-res",
        "--resolution",
        type=float,
        default=0.25,
        help="The spatial resolution of returned data in degrees.",
    )

    return parser


def _add_common_args(
    argparser: argparse.ArgumentParser,
    args: list[str] = [],
    location_args: bool = True,
    login_args: bool = True,
    force: bool = True,
) -> argparse.ArgumentParser:
    """Add standard arguments to a subparser.

    Args:
        argparser (argparse.ArgumentParser): The subparser to add arguments to.
        args (list[str]): Additional standard/shared arguments to add to the parser.
        location_args (bool): If True (default), add standard location arguments.
        login_args (bool): If True (default), add username/password arguments
        force (bool): If True (default), add a `--force` argument to overwrite existing files.
            Most API functions have this argument.

    Generate an argument parser for the Location class with consistent arguments
    that can be used across multiple `main()` functions and the command line.
    """
    if login_args:
        # Users will login with either a username/password or an API key:
        argparser.add_argument(
            "-u",
            "--username",
            type=str,
            default="SALIENT_USERNAME",  # pull from env var
            help="Salient-issued user name",
        )
        argparser.add_argument(
            "-p",
            "--password",
            type=str,
            default="SALIENT_PASSWORD",  # pull from env var
            help="Salient-issued password",
        )
        argparser.add_argument(
            "--apikey",
            type=str,  # Use "SALIENT_APIKEY" to pull from env var
            default=None,
            help="Salient-issued API key (ignores username/password)",
        )
        verify_group = argparser.add_mutually_exclusive_group(required=False)
        verify_group.add_argument(
            "--verify",
            dest="verify",
            action="store_true",
            help="Force verification of SSL certificates.",
        )
        verify_group.add_argument(
            "--noverify",
            dest="verify",
            action="store_false",
            help="Disable verification of SSL certificates.",
        )
        argparser.set_defaults(verify=None)

        verbosity_group = argparser.add_mutually_exclusive_group(required=False)
        verbosity_group.add_argument(
            "--verbose",
            dest="verbose",
            action="store_true",
            help="Print status messages (default behavior)",
        )
        verbosity_group.add_argument(
            "--quiet", dest="verbose", action="store_false", help="Suppress status messages"
        )
        argparser.set_defaults(verbose=True)

    if location_args:
        argparser.add_argument(
            "-lat",
            "--latitude",
            type=float,
            default=None,
            help="Decimal latitude -90 to 90 (also requires longitude)",
        )
        argparser.add_argument(
            "-lon",
            "--longitude",
            type=float,
            default=None,
            help="Decimal longitude -180 to 180 (also requires latitude)",
        )
        argparser.add_argument("-loc", "--location_file", type=str, default=None)
        argparser.add_argument("-shp", "--shapefile", type=str, default=None)

    if force:
        argparser.add_argument(
            "--force", action="store_true", help="Overwrite existing files (default is to cache)"
        )

    if "debias" in args:
        argparser.add_argument(
            "--debias",
            action="store_true",
            help="Debias to observation stations (default is no debiasing)",
        )

    if "version" in args:
        argparser.add_argument(
            "-ver", "--version", type=str, default="-default", help="Model version to use"
        )

    if "date" in args:
        argparser.add_argument("--date", type=str, default="-today")

    if "region" in args:
        argparser.add_argument("--region", type=str, default=None)

    if "variable" in args:
        argparser.add_argument("-var", "--variable", type=str, default="temp")

    return argparser


def _list_examples():
    try:
        examples_path = resources.files("salientsdk") / "examples"
        return [
            str(Path(f).resolve())
            for f in examples_path.iterdir()
            if f.name.endswith(".ipynb") and not f.name.startswith(".")
        ]
    except Exception as e:
        raise FileNotFoundError(f"Could not find examples directory. Error: {e}")


if __name__ == "__main__":
    main()
