#!/usr/bin/env python
# Copyright Salient Predictions 2025

"""Define the Location class."""

# Usage example:
# python location.py --lat 42 --lon -73

import math
import os
from collections.abc import Iterable
from functools import lru_cache
from importlib import resources
from typing import Literal

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import xarray as xr
from sklearn.cluster import KMeans

from . import constants, login_api, upload_file_api

# location_file supports a variety of potential column names:
# fmt: off
LON_NAMES = ["lon", "Lon", "LON", "longitude", "Longitude", "LONGITUDE", "lng", "x", "X"]
LAT_NAMES = ["lat", "Lat", "LAT", "latitude", "Latitude", "LATITUDE", "y", "Y"]
NAM_NAMES = ["Location", "location", "loc", "id", "ID", "Place", "place", "Coordinates", "coordinates", "Name", "name"]
# fmt:on


class Location:
    """Define geographical parameters for an API query.

    The Salient API defines location by a single latitude/longitude,
    multiple latitude/longitude pairs in a single location_file, or a polygon
    defined in a shapefile.  This class manages these options.
    """

    def __init__(
        self,
        lat: float | None = None,
        lon: float | None = None,
        location_file: str | list[str] | None = None,
        shapefile: str | list[str] | None = None,
        region: str | list[str] | None = None,
    ):
        """Initialize a Location object.

        Only one of the following 4 options should be used at a time: lat/lon,
        location_file, shapefile, or region.

        Args:
            lat (float): Latitude in degrees, -90 to 90.
            lon (float): Longitude in degrees, -180 to 180.
            location_file (str | list[str]): Path(s) to CSV file(s) with latitude and longitude columns.
            shapefile (str | list[str]): Path(s) to a shapefile(s) with a polygon defining the location.
            region (str | list[str]): Accepts continents, countries, or U.S. states (e.g. "usa")
                Only available for `hindcast_summary()`
        """
        self.lat = lat
        self.lon = lon
        self.location_file = self._expand_user_files(location_file)
        self.shapefile = self._expand_user_files(shapefile)
        self.region = region

        self._validate()

    @staticmethod
    def _expand_user_files(files: str | list[str] | None) -> str | list[str] | None:  # noqua: D103
        # Strip directory names from location_file and shapefile.
        #
        # When referencing user files, the user may specify them with
        # a directory name because that's how functions like upload_*
        # create them.  But the API doesn't have a directory structure and
        # only uses the file name.  So we need to strip off the directory.
        #
        # Also, handles vectorization of file names in case user passes in
        # a list or a comma-separated string.

        files = constants._expand_comma(files)

        if files is None:
            pass
        elif isinstance(files, str):
            files = os.path.basename(files)
        elif isinstance(files, list):
            files = [os.path.basename(f) for f in files]

        return files

    def asdict(self, **kwargs) -> dict:
        """Render as a dictionary.

        Generates a dictionary representation that can be encoded into a URL.
        Will contain one and only one location_file, shapefile, or lat/lon pair.

        Args:
            **kwargs: Additional key-value pairs to include in the dictionary.
                Will validate some common arguments that are shared across API calls.
        """
        if self.location_file:
            dct = {"location_file": self.location_file, **kwargs}
        elif self.shapefile:
            dct = {"shapefile": self.shapefile, **kwargs}
        elif self.region:
            dct = {"region": self.region, **kwargs}
        else:
            dct = {"lat": self.lat, "lon": self.lon, **kwargs}

        if "apikey" in dct and dct["apikey"] is not None:
            dct["apikey"] = login_api._get_api_key(dct["apikey"])

        if "start" in dct:
            dct["start"] = constants._validate_date(dct["start"])
        if "end" in dct:
            dct["end"] = constants._validate_date(dct["end"])
        if "forecast_date" in dct:
            dct["forecast_date"] = constants._validate_date(dct["forecast_date"])

        if "version" in dct:
            dct["version"] = constants._expand_comma(
                val=dct["version"],
                valid=constants.MODEL_VERSIONS,
                name="version",
                default=constants.get_model_version(),
            )

        if "shapefile" in dct and "debias" in dct and dct["debias"]:
            raise ValueError("Cannot debias with shapefile locations")

        return dct

    def load_location_file(self, destination: str = "-default") -> gpd.GeoDataFrame:
        """Load the location file(s) into a DataFrame.

        Args:
            destination (str): The directory where the file is located.
                Defaults to the default directory via `get_file_destination`.

        Returns:
            gpd.GeoDataFrame: The location data in a DataFrame.  If multiple files are loaded,
                the DataFrames will be concatenated into a single DataFrame with an
                additional column `file_name` that documents the source file.

                location_file suports a variety of column names for latitude and longitude.
                This method standardizes them to `lat` and `lon`.
        """
        assert self.location_file is not None

        destination = constants.get_file_destination(destination)

        if isinstance(self.location_file, str):
            geo = pd.read_csv(os.path.join(destination, self.location_file))
        else:
            # location files may be a list of files.  Load all of them.
            geo = pd.concat(
                [
                    pd.read_csv(os.path.join(destination, f)).assign(file_name=f)
                    for f in self.location_file
                ]
            )

        return self._as_geoframe(geo)

    def cluster(
        self,
        cluster_size=256,
        upload: Literal["changed", "all", "none"] = "changed",
        destination: str = "-default",
        verbose: bool = False,
        session: requests.Session | None = None,
        verify: bool | None = None,
    ) -> "Location":
        """Cluster a vector of lat/lon points into separate `location_file`s.

        Groups locations into clusters of specified size and saves each cluster
        to a separate CSV file. Can optionally upload the resulting files.

        Args:
            cluster_size: Target number of locations per cluster (default: 256)
            upload: Upload strategy for cluster files:
                * `changed` - only upload modified files (default)
                * `all` - upload all cluster files, even if unchanged
                * `none` - skip uploading
            destination (str): Generate location files to this local directory.
            verbose: Whether to print progress information
            session: Optional session object to use, if uploding files
            verify (bool): Verify the SSL certificate, if uploading files

        Returns:
            Location: A new Location object with the `location_file` field
                as a vector of all generated files.

                Adds instance variable `any_changed` if any written
                files did not match existing ones. Useful for setting
                `force` on downstream processes.
        """
        assert upload in [
            "changed",
            "all",
            "none",
        ], f"upload must be one of: changed, all, none. Got: {upload}"
        destination = constants.get_file_destination(destination)

        base_file = os.path.join(destination, _get_basename(self.location_file))
        geo = self.load_location_file().drop("geometry", axis=1)

        geo["cluster"] = _cluster_region(lat=geo.lat, lon=geo.lon, cluster_size=cluster_size)

        upload = False
        clusters = geo["cluster"].unique()
        cluster_files = [f"{base_file}_{cluster}.csv" for cluster in clusters]
        changed_files = []
        for cluster, filename in zip(clusters, cluster_files):
            cluster_df = geo[geo["cluster"] == cluster].drop("cluster", axis=1)
            if _has_geo_changed(cluster_df, filename):
                cluster_df.to_csv(filename, index=False)
                changed_files.append(filename)

        any_changed = len(changed_files) > 0
        changed_files = (
            cluster_files if upload == "all" else [] if upload == "none" else changed_files
        )
        upload_file_api.upload_file(
            file=changed_files, verbose=verbose, session=session, verify=verify
        )

        cluster_loc = Location(location_file=cluster_files)
        cluster_loc.any_changed = any_changed

        return cluster_loc

    @staticmethod
    def _as_geoframe(geo: pd.DataFrame) -> gpd.GeoDataFrame:
        """Normalize column names on a DataFrame and convert to a GeoDataFrame.

        Regardless of the column names in the input DataFrame, this method
        will return a GeoDataFrame with columns `lat`,`lon`, and `name`
        """
        col_names = set(geo.columns)
        lon_name = next((name for name in LON_NAMES if name in col_names), None)
        lat_name = next((name for name in LAT_NAMES if name in col_names), None)
        nam_name = next((name for name in NAM_NAMES if name in col_names), None)

        assert lon_name is not None, f"Missing longitude column in {col_names}"
        assert lat_name is not None, f"Missing latitude column in {col_names}"

        geo.rename(columns={lon_name: "lon", lat_name: "lat"}, inplace=True)

        if nam_name is not None:
            # "name" column is optional and not guaranteed to be present.
            geo.rename(columns={nam_name: "name"}, inplace=True)
            # Sometimes location_files will contain all-numeric "names" like
            # HUC watersheds.  The API will coerce string datatypes for all
            # NetCDFs, so the SDK must do the same thing for clean merges.
            geo["name"] = geo["name"].astype(str)

        geo = gpd.GeoDataFrame(geo, geometry=gpd.points_from_xy(geo.lon, geo.lat))

        return geo

    def _validate(self):
        if self.location_file:
            assert not self.lat, "Cannot specify both lat and location_file"
            assert not self.lon, "Cannot specify both lon and location_file"
            assert not self.region, "Cannot specify both region and location_file"
            assert not self.shapefile, "Cannot specify both shape_file and location_file"
        elif self.shapefile:
            assert not self.region, "Cannot specify both region and shapefile"
            assert not self.lat, "Cannot specify both lat and shape_file"
            assert not self.lon, "Cannot specify both lon and shape_file"
        elif self.region:
            assert not self.lat, "Cannot specify both lat and region"
            assert not self.lon, "Cannot specify both lon and region"
            assert not self.location_file, "Cannot specify both location_file and region"
            assert not self.shapefile, "Cannot specify both shape_file and region"
        else:
            assert self.lat, "Must specify lat & lon, location_file, shapefile, or region"
            assert self.lon, "Must specify lat & lon, location_file, shapefile, or region"
            assert -90 <= self.lat <= 90, "Latitude must be between -90 and 90 degrees"
            assert -180 <= self.lon <= 180, "Longitude must be between -180 and 180 degrees"

    def plot_locations(
        self,
        title: str = None,
        weight: str = None,
        pad: float = 1.0,
        names=True,
    ):
        """Show location points on a map.

        Args:
            title (str): The title of the plot.
            weight (str): The column name in the location file to use to weight the points.
            pad (float): Adds extent to the lat/lon bounding boxon the map.
            names (bool): Plot location string names
        """
        geo = self.load_location_file()

        if title is None:
            title = str(self.location_file)

        if weight is not None:
            weight = geo[weight] * 10

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": ccrs.Mercator()})
        min_lon, max_lon = geo["lon"].min() - pad, geo["lon"].max() + pad
        min_lat, max_lat = geo["lat"].min() - pad, geo["lat"].max() + pad

        ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.BORDERS)
        # ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.LAND, edgecolor="black")
        ax.add_feature(cfeature.OCEAN, facecolor="aqua")
        ax.add_feature(cfeature.LAKES, facecolor="aqua")

        ax.scatter(
            geo["lon"],
            geo["lat"],
            s=weight,
            color="dodgerblue",
            alpha=0.5,
            transform=ccrs.PlateCarree(),
        )

        if names:
            for lon, lat, name in zip(geo["lon"], geo["lat"], geo["name"]):
                ax.text(
                    lon,
                    lat,
                    name,
                    fontsize=8,
                    ha="center",
                    va="center",
                    transform=ccrs.PlateCarree(),
                )

        plt.title(title)
        plt.show()

        return fig, ax

    def __str__(self):  # noqa: D105
        if self.location_file:
            return f"location file: {self.location_file}"
        elif self.shapefile:
            return f"shape file: {self.shapefile}"
        elif self.region:
            return f"region: {self.region}"
        else:
            return f"({self.lat}, {self.lon})"

    def __eq__(self, other):  # noqa: D105
        if self.location_file:
            return self.location_file == other.location_file
        elif self.shapefile:
            return self.shapefile == other.shape_file
        elif self.region:
            return self.region == other.region
        else:
            return self.lat == other.lat and self.lon == other.lon

    def __ne__(self, other):  # noqa: D105
        return not self.__eq__(other)


@lru_cache(maxsize=1)
def _load_regions_mask():
    """Lazy loader for mask dataset."""
    f = resources.files("salientsdk.data").joinpath("region.nc")
    return xr.load_dataset(str(f), engine="netcdf4")

    # this isn't compatible with netcdf4.
    # with resources.files("salientsdk.data").joinpath("region.nc").open("rb") as f:
    #    return xr.load_dataset(f, engine="netcdf4")


def _find_region(lat: Iterable[float], lon: Iterable[float]) -> np.ndarray:
    """Find region codes for given latitude and longitude coordinates.

    Args:
        lat: Iterable of latitude values
        lon: Iterable of longitude values

    Returns:
        numpy array of integer region codes.  Region zero is unrecognized.
    """
    mask = _load_regions_mask()
    return (
        mask.mask.sel(
            lat=xr.DataArray(np.array(lat), dims="points"),
            lon=xr.DataArray(np.array(lon), dims="points"),
            method="nearest",
        )
        .fillna(0)
        .values.astype(int)
    )


def _cluster_region(
    lat: Iterable[float],
    lon: Iterable[float],
    region: Iterable | None = None,
    cluster_size: int = 256,
) -> np.ndarray:
    """Cluster points within regions based on geographic proximity.

    Args:
        lat: Iterable of latitude values
        lon: Iterable of longitude values
        region: Region codes. If None, will be determined using find_region(lat,lon)
        cluster_size: Minimum size for a region to be clustered. Defaults to 256,
            which is appropriate for `downscale` with 50 ensembles and 5 variables.
            Note that this is a target value, and not a hard limit.  Clusters may
            contain up to 25% more points than the target size.

    Returns:
        numpy array of strings in format "<region>_<cluster>"
    """
    df = pd.DataFrame(
        {
            "lat": lat,
            "lon": lon,
            "region": _find_region(lat, lon) if region is None else region,
        }
    )
    max_region_len = max(len(str(r)) for r in np.unique(df["region"]))
    result = np.empty(len(df), dtype=f"<U{max_region_len + 3}")

    # We'll allow clusters to be slightly larger than the target cluster size
    cluster_max = 1.25 * cluster_size
    for r in np.unique(df["region"]):
        mask = df["region"] == r
        region_df = df[mask]

        if len(region_df) > cluster_max:
            n_clusters = math.ceil(len(region_df) / cluster_size)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(region_df[["lat", "lon"]])

            # Keep splitting until no cluster exceeds threshold
            next_cluster = clusters.max() + 1
            while True:
                sizes = pd.Series(clusters).value_counts()
                largest_cluster = sizes.index[sizes > cluster_max]

                if len(largest_cluster) == 0:
                    break

                # Split the largest remaining oversized cluster
                c = largest_cluster[0]
                cluster_mask = clusters == c
                subset = region_df[cluster_mask]

                # Split into two new clusters
                sub_kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
                sub_clusters = sub_kmeans.fit_predict(subset[["lat", "lon"]])

                # Replace old cluster with two new ones
                clusters[cluster_mask] = np.where(sub_clusters == 0, c, next_cluster)
                next_cluster += 1

        else:
            clusters = np.zeros(len(region_df))

        result[mask] = [f"{r}_{int(c):02d}" for c in clusters]

    return result


def _has_geo_changed(
    geo1: pd.DataFrame | str,
    geo2: pd.DataFrame | str,
) -> bool:
    """Check if two geographic datasets differ.

    Args:
        geo1: First dataset (DataFrame or path to CSV)
        geo2: Second dataset (DataFrame or path to CSV)

    Returns:
        True if datasets differ, False if they match
    """

    def _to_df(geo):
        if not isinstance(geo, pd.DataFrame):
            if not os.path.exists(geo):
                return None
            geo = pd.read_csv(geo)
        return geo.reset_index(drop=True)

    # Convert inputs and check for existence
    df1 = _to_df(geo1)
    if df1 is None:
        return True

    df2 = _to_df(geo2)
    if df2 is None:
        return True

    if len(df1) != len(df2):
        return True

    # Compare numeric columns with tolerance
    numeric_cols = df1.select_dtypes(include=["float64", "float32"]).columns
    for col in numeric_cols:
        if not np.allclose(df1[col], df2[col], rtol=1e-5, atol=1e-8):
            return True

    # Compare non-numeric columns exactly
    other_cols = [col for col in df1.columns if col not in numeric_cols]
    for col in other_cols:
        if not df1[col].equals(df2[col]):
            return True

    return False


def _get_basename(geofile: str | Iterable[str] | None) -> str:
    """Get basename from either a single file or multiple files.

    Args:
        geofile: Either a single filepath string or iterable of filepaths

    Returns:
        Basename string with path and extensions removed
    """
    if isinstance(geofile, str):
        return os.path.splitext(os.path.basename(geofile))[0]
    elif geofile is None:
        raise ValueError("shapefile or location_file must be non-empty")
    else:
        return "_".join(_get_basename(f) for f in geofile)


def merge_location_data(
    ds: xr.Dataset | str,
    loc_file: "str | Location",
    as_data_vars: bool = True,
) -> xr.Dataset:
    """Merge additional data columns from a location_file into a dataset.

    Will add any additional columns from `loc_file` to `ds`.

    Args:
        ds (xr.Dataset | str): A `Dataset` with a vector `location` coordinate,
            typically resulting from requesting a `location_file`
        loc_file (str | Location): Path to a CSV file containing location data
            or a `Location` object with a `location_file` attribute.
        as_data_vars (bool): If True (default), adds columns as data variables.
            If False, adds them as coordinates.

    Returns:
        xr.Dataset: A new `Dataset` with the additional columns from `loc_file`
            along the `location` coordinate.
    """
    if isinstance(ds, str):
        ds = xr.load_dataset(str)

    geo = (
        loc_file.load_location_file().drop(columns=["geometry"])  # geopandas -> pandas
        if isinstance(loc_file, Location)
        else pd.read_csv(loc_file)
    )

    geo = geo.drop(columns=["lat", "lon"])  # redundant
    geo = geo.rename(columns={"name": "location"}).set_index("location")
    geo = xr.Dataset.from_dataframe(geo)

    if not as_data_vars:
        geo = geo.set_coords(geo.data_vars)

    return xr.merge([geo, ds], combine_attrs="override")
