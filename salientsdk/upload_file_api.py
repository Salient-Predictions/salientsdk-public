#!/usr/bin/env python
# Copyright Salient Predictions 2025

"""Interface to the Salient `upload_file` API.

There is no command line interface for this module.
"""

import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from importlib import resources
from pathlib import Path
from typing import Literal

import pandas as pd
import requests
from backoff import expo, on_exception
from ratelimit import RateLimitException, limits

from .constants import _build_url, get_file_destination
from .login_api import (
    CALLS_PER_DAY,
    CALLS_PER_MINUTE,
    DetailedHTTPError,
    download_query,
    get_current_session,
    get_verify_ssl,
    rate_limit_handler,
)


@on_exception(expo, RateLimitException, on_backoff=rate_limit_handler)
@limits(calls=CALLS_PER_MINUTE, period=60)
@limits(calls=CALLS_PER_DAY, period=86400)
def _upload_single_file(
    file: str,
    verbose: bool = True,
    session: requests.Session | None = None,
    apikey: str | None = None,
    verify: bool = True,
    update: bool = False,
) -> None:
    """Internal function to upload a single file with rate limiting."""
    if not os.path.exists(file):
        raise FileNotFoundError(f"File not found: {file}")
    if session is None:
        session = get_current_session()

    args = {"apikey": apikey, "update": update}
    (url, loc_file) = _build_url("upload_file", args=args)

    with open(file, "rb") as f:
        response = session.post(url, files={"file": f}, verify=verify)
        try:
            response.raise_for_status()
        except requests.HTTPError as err:
            raise DetailedHTTPError(err) from None

        if verbose:
            print(response.text)
    return None


def upload_file(
    file: str | list[str],
    verbose: bool = False,
    session: requests.Session | None = None,
    apikey: str | None = None,
    verify: bool | None = None,
    update: bool = False,
) -> None:
    """Upload location_file or shapefile.

    An interface to to the Salient
    [upload_file](https://api.salientpredictions.com/v2/documentation/api/#/General/upload_file)
    API endpoint.

    Args:
        file: Single file path or list of file paths to upload
        verbose: Whether to print progress information
        session: Optional session object, will create new session if None
        apikey (str | None): The API key to use for the request.
            In most cases, this is not needed if a `session` is provided.
        verify (bool): If True (default), verify the SSL certificate
        update: Boolean whether to overwrite any potential custom quantities
            with the same name.

    Returns:
        File ID(s) returned by the API

    Raises:
        FileNotFoundError: If any file doesn't exist
        HTTPError: If upload fails
        RateLimitException: If API rate limits are exceeded
    """
    if session is None:
        session = get_current_session()
    verify = get_verify_ssl(verify)

    args = {
        "verbose": verbose,
        "session": session,
        "apikey": apikey,
        "verify": verify,
        "update": update,
    }
    if isinstance(file, str):
        return _upload_single_file(file=file, **args)
    if not file:
        if verbose:
            print("Empty file list.  No files uploaded.")
        return None
    with ThreadPoolExecutor(max_workers=os.cpu_count() * 2) as executor:
        futures = [executor.submit(_upload_single_file, file=f, **args) for f in file]
        [future.result() for future in futures]

    return None


def _upload_file_example(
    geoname: str,
    destination: str = "-default",
    force: bool = False,
    verbose: bool = False,
    session: requests.Session | None = None,
    apikey: str | None = None,
) -> str:
    """Upload an example location_file or shapefile from the SDK's data directory.

    salientsdk contains example `location_file`s that reflect common queries.

    Status: Not currently used.  Under consideration for export.

    Args:
        geoname (str): Name of the location_file or shapefile to use.
           - `cmeus`: Chicago Mercantile Exchange USA HDD/CDD airport locations
        destination (str): Copy the file from the sdk to this local directory.
        force (bool): When False, if the file already exists don't upload it
        verbose (bool): If True, print status messages
        session (requests.Session): The session object to use for the upload request
        apikey (str | None): The API key to use for the request.
            In most cases, this is not needed if a `session` is provided.

    Returns:
        str: File name of the location_file or shapefile
    """
    # Get all matching files using importlib.resources
    data_files = [
        f for f in resources.files("salientsdk.data").iterdir() if f.name.startswith(f"{geoname}.")
    ]

    if not data_files:
        raise FileNotFoundError(f"No file found with name '{geoname}' in package data")
    elif len(data_files) > 1:
        raise ValueError(f"Multiple files found with name '{geoname}' in package data")

    src_file = data_files[0]
    src_name = src_file.name

    dst_path = get_file_destination(destination)
    if dst_path is not None:
        dst_file = Path(dst_path) / src_name
    else:
        dst_file = Path(src_name)

    if not force and dst_file.exists():
        if verbose:
            print(f"File {src_name} already exists")
        return src_name

    with src_file.open("rb") as src, open(dst_file, "wb") as dst:
        shutil.copyfileobj(src, dst)

    upload_file(file=str(dst_file), verbose=verbose, session=session, apikey=apikey)

    return src_name


def upload_bounding_box(
    # API arguments ----------
    north: float,
    south: float,
    east: float,
    west: float,
    geoname: str,
    # Non-API arguments --------
    destination: str | None = "-default",
    force: bool = False,
    verbose: bool = False,
    session: requests.Session | None = None,
    apikey: str | None = None,
) -> str:
    """Upload a bounding box.

    Create and upload a GeoJSON shapefile with a rectangular bounding box
    for later use with the `shapefile` location argument.

    Args:
        north (float): Northern extent decimal latitude
        south (float): Southern extent decimal latitude
        east (float): Eastern extent decimal longitude
        west (float): Western extent decimal longitude
        geoname (str): Name of the GeoJSON file and object to create
        destination (str): The destination directory for the generated file
        force (bool): If the file already exists, don't upload it
        verbose (bool): Whether to print status messages
        session (requests.Session): The session object to use for the request
        apikey (str | None): The API key to use for the request.
            In most cases, this is not needed if a `session` is provided.

    Returns:
        str: File name of the GeoJSON file
    """
    assert west < east, "West must be less than East"
    assert south < north, "South must be less than North"
    coords = [
        (west, north),
        (east, north),
        (east, south),
        (west, south),
    ]  # upload_shapefile will close the polygon for us
    return upload_shapefile(
        coords=coords,
        geoname=geoname,
        destination=destination,
        force=force,
        verbose=verbose,
        session=session,
        apikey=apikey,
    )


def upload_shapefile(
    coords: list[tuple[float, float]],
    geoname: str,
    # Non-API arguments --------
    destination: str | None = "-default",
    force: bool = False,
    verbose: bool = False,
    session: requests.Session | None = None,
    apikey: str | None = None,
):
    """Upload a custom shapefile defined by a a list of lat/lon pairs.

    This will often be used with `Location(shapefile...)`

    Args:
        coords (list[tuple]): List of (longitude, latitude) pairs defining the polygon.
        geoname (str): Name of the GeoJSON file and object to create.
        destination (str): The destination directory for the generated file.
        force (bool): If True, overwrite the existing file if it exists.
        verbose (bool): Whether to print status messages.
        session (requests.Session): The session object to use for the request.
        apikey (str | None): The API key to use for the request.
            In most cases, this is not needed if a `session` is provided.

    Returns:
        str: File name of the GeoJSON file.
    """
    geofile = geoname + ".geojson"
    destination = get_file_destination(destination)
    if destination is not None:
        geofile = os.path.join(destination, geofile)
    session = get_current_session() if session is None else session

    if not force and os.path.exists(geofile):
        if verbose:
            print(f"File {geofile} already exists")
        return geofile

    # Check to see if the polygon is closed, and close it if not:
    if coords[0] != coords[-1]:
        coords.append(coords[0])

    # Create the GeoJSON structure
    geoshape = {
        "type": "Feature",
        "properties": {"name": geoname},
        "geometry": {
            "type": "Polygon",
            "coordinates": [coords],
        },
    }

    # Write the GeoJSON to a file
    with open(geofile, "w") as f:
        json.dump(geoshape, f)

    upload_file(file=geofile, verbose=verbose, session=session, apikey=apikey)

    return geofile


def upload_location_file(
    lats: list[float] | pd.Series,
    lons: list[float] | pd.Series,
    names: list[str] | pd.Series,
    geoname: str,
    destination: str = "-default",
    force: bool = False,
    verbose: bool = False,
    session: requests.Session | None = None,
    apikey: str | None = None,
    **kwargs,
) -> str:
    """Upload a vector of locations.

    Create and upload a CSV file with a list of locations for
    later use with the `location_file` location argument.

    Args:
        lats (list[float] | pd.Series): List of decimal latitudes
        lons (list[float] | pd.Series): List of decimal longitudes
        names (list[str] | pd.Series): List of names for the locations
        geoname (str): Name of the CSV file and object to create
        destination (str): The destination directory for the generated file
        force (bool): When False, if the file already exists don't upload it
        verbose (bool): If True, print status messages
        session (requests.Session): The session object to use for the request
        apikey (str | None): The API key to use for the request.
            In most cases, this is not needed if a `session` is provided.
        **kwargs: Additional columns to include in the CSV file

    Returns:
        str: File name of the CSV file
    """
    geofile = geoname + ".csv"
    destination = get_file_destination(destination)
    if destination is not None:
        geofile = os.path.join(destination, geofile)

    if not force and os.path.exists(geofile):
        if verbose:
            print(f"File {geofile} already exists")
        return geofile

    loc_table = pd.DataFrame({"lat": lats, "lon": lons, "name": names, **kwargs})
    loc_table.to_csv(geofile, index=False)

    upload_file(file=geofile, verbose=verbose, session=session)

    return geofile


def user_files(
    type: Literal["location", "derived"] = "location",
    # Non-API arguments --------
    destination: str = "-default",
    session: requests.Session | None = None,
    apikey: str | None = None,
    verify: bool | None = None,
    verbose: bool = False,
) -> str:
    """List the location and shape files uploaded by the user.

    This will call the [get_files](https://api.salientpredictions.com/v2/documentation/api/#/General/get_files)
    API endpoint and return a json file.

    Args:
        type (str): Either `location` (default) for shapefiles and location_files or `derived`
            for custom quantities.
        destination (str): The destination directory for the resulting JSON file
        session (requests.Session): The session object to use for the request
        apikey (str | None): The API key to use for the request.
            In most cases, this is not needed if a `session` is provided.
        verify (bool): Whether to verify the SSL certificate.
            Defaults to use the value returned by `get_verify_ssl()`
        verbose (bool): If True, print the full contents of the file.

    Returns:
        str: the location of the JSON file containing top-level entries
             `coordinates` (for `location_file` inputs) and `shapefiles`.
    """
    format = "json"
    endpoint = "user_files"
    (url, loc_file) = _build_url(endpoint, args={"type": type}, destination=destination)
    loc_file = f"{loc_file}.{format}"

    download_query(
        query=url,
        file_name=loc_file,
        format=format,
        session=session,
        verify=verify,
        verbose=verbose,
        force=True,
    )

    if verbose:
        # parse the json file and print the results:
        with open(loc_file, "r") as f:
            data = json.load(f)
        for key, value in data.items():
            if isinstance(value, list):
                items = ", ".join(str(item) for item in value)
                print(f"{key}: {items}")
            else:
                print(f"{key}: {value}")

    return loc_file


def _mock_upload_location_file(
    destination: str = "-default",
    **kwargs,
) -> str:
    """Creates a location_file without uploading it."""
    geofile = os.path.join(get_file_destination(destination), "CA_Airports.csv")
    lats = [37.7749, 33.9416, 32.7336]
    lons = [-122.4194, -118.4085, -117.1897]
    names = ["SFO", "LAX", "SAN"]
    pd.DataFrame({"lat": lats, "lon": lons, "name": names}).to_csv(geofile, index=False)
    return geofile
