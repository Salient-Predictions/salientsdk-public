#!/usr/bin/env python
# Copyright Salient Predictions 2025

"""Tools for solar energy analysis.

Connects Salient timeseries data to pvlib.
https://salientpredictions.atlassian.net/browse/RD-1259
"""

# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pvlib as pv
import requests
import xarray as xr

from .data_timeseries_api import data_timeseries, load_multihistory
from .downscale_api import downscale
from .geo_api import add_geo
from .location import Location

SOLAR_VARIABLES = ["temp", "wspd", "tsi", "dhi", "dni"]


def data_timeseries_solar(
    # API inputs -------
    loc: Location,
    start: str = "1950-01-01",
    end: str = "-today",
    variable: list[str] = SOLAR_VARIABLES,
    debias: bool = False,
    # non-API arguments ---
    destination: str = "-default",
    force: bool = False,
    session: requests.Session | None = None,
    apikey: str | None = None,
    verify: bool | None = None,
    verbose: bool = False,
) -> xr.Dataset:
    """Get a historical time series of pvlib-compatible solar weather inputs.

    A convenience wrapper around `data_timeseries()` to get solar weather data.
    Generates a past weather dataset suitable to pass to `run_pvlib_dataset()`.

    Args:
        loc (Location): The location to query.
        start (str): The start date of the time series
        end (str): The end date of the time series
        variable (list[str]): The variables to download.
            Defaults to `["temp", "wspd", "tsi", "dhi", "dni"]` which are the inputs
            needed for `run_pvlib_dataset()`.
        debias (bool): If `True`, debias the data to local observations.
            Defaults to `False` since debiasing does not currently support
            solar components like `tsi`, `dhi`, and `dni`.

        destination (str): The directory to download the data to
        force (bool): If False (default), don't download the data if it already exists
        session (requests.Session): The session object to use for the request.
            If `None` (default) uses `get_current_session()`.
        apikey (str | None): The API key to use for the request.
            In most cases, this is not needed if a `session` is provided.
        verify (bool): If True (default), verify the SSL certificate
        verbose (bool): If True (default False) print status messages and check for NaN.

    Returns:
        xr.Dataset: An hourly dataset with the following variables:
            - temp: temperature (C)
            - wspd: wind speed (m/s)
            - tsi: total solar irradiance, aka Global Horizontal Irradiance (W/m^2)
            - dhi: diffuse horizontal irradiance (W/m^2)
            - dni: direct normal irradiance (W/m^2)
            - elevation: elevation (m)
    """
    field = "vals"
    format = "nc"
    frequency = "hourly"
    # units = "SI" - unnecessary, since it's the default

    # Validate parameters that are used dynamically via locals()
    assert isinstance(debias, bool), f"debias must be bool, got {type(debias)}"

    file = data_timeseries(**locals())
    vals = load_multihistory(file)
    vals = add_geo(
        loc=loc,
        ds=vals,
        variables="elevation",
        destination=destination,
        force=force,
        session=session,
        apikey=apikey,
        verify=verify,
        verbose=verbose,
    )

    if verbose:
        _check_nans(vals)

    return vals


def downscale_solar(
    # API inputs -------
    loc: Location,
    date: str = "-today",
    members: int = 50,
    variables: list[str] = SOLAR_VARIABLES,
    # non-API arguments ---
    destination: str = "-default",
    force: bool = False,
    session: requests.Session | None = None,
    apikey: str | None = None,
    verify: bool | None = None,
    verbose: bool = False,
    **kwargs,
) -> xr.Dataset:
    """Get a forecast time series of pvlib-compatible solar weather inputs.

    A convenience wrapper around `downscale()` to get solar weather data.
    Generates a future weather dataset suitable to pass to `run_pvlib_dataset()`.

    Args:
        loc (Location): The location to query.
        date (str): The start date of the time series.
            If `date` is `-today`, use the current date.
        members (int): The number of ensemble members to download
        variables (list[str]): The variables to download.
            Defaults to `["temp", "wspd", "tsi", "dhi", "dni"]` which are the inputs
            needed for `run_pvlib_dataset()`.

        destination (str): The directory to download the data to
        force (bool): If False (default), don't download the data if it already exists
        session (requests.Session): The session object to use for the request.
            If `None` (default) uses `get_current_session()`.
        apikey (str | None): The API key to use for the request.
            In most cases, this is not needed if a `session` is provided.
        verify (bool): If True (default), verify the SSL certificate
        verbose (bool): If True (default False) print status messages and check for NaN
        **kwargs: Additional arguments to pass to `downscale()`

    Keyword Arguments:
        reference_clim (str): Reference period to calculate anomalies.
        version (str): The model version of the Salient `blend` forecast.
        debias (bool): Defaults to `False`, note that `tsi`, `dhi`, and `dni` are not debiased.


    Returns:
        xr.Dataset: An hourly dataset with the following variables:
            - temp: temperature (C)
            - wspd: wind speed (m/s)
            - tsi: total solar irradiance, aka Global Horizontal Irradiance (W/m^2)
            - dhi: diffuse horizontal irradiance (W/m^2)
            - dni: direct normal irradiance (W/m^2)
            - elevation: elevation (m)
    """
    frequency = "hourly"
    # units = "SI" unnecessary since it's the default
    file = downscale(**{**{k: v for k, v in locals().items() if k != "kwargs"}, **kwargs})
    # enhancement: if "location" is vectorized, need to load multiple datasets:
    vals = xr.load_dataset(file)

    vals = add_geo(
        loc=loc,
        ds=vals,
        variables="elevation",
        destination=destination,
        force=force,
        session=session,
        apikey=apikey,
        verify=verify,
        verbose=verbose,
    )

    if verbose:
        _check_nans(vals)

    return vals


def run_pvlib_dataset(
    weather: xr.Dataset | str,
    timedim: str = "time",
    model_chain: pv.modelchain.ModelChain | list[pv.modelchain.ModelChain] | None = None,
    nan_check: float = 0.01,
) -> xr.Dataset:
    """Execute pvlib multiple times on a weather-inputs dataset.

    Args:
        weather (xr.Dataset | str): Solar meteorological inputs,
            of the form returned by `data_timeseries_solar()`
            May also be the filename to a valid dataset.
        timedim (str): The name of the time dimension
        model_chain (pv.modelchain.ModelChain): The model chain(s) to use.
            If `None`, calls `dataset_to_modelchain(weather)`.
        nan_check (float): issue a warning if any month in the dataset contains
            more than this ratio of NaNs. Suppress with `None`.



    Returns:
        xr.Dataset: A dataset with matching coordinates to `ds` containing data
            * `ac` = Alternating Current power
            * `dc` = Direct Current power
            * `effective_irradiance` = Effective Irradiance
            * `poa_direct` = Plane of Array Direct
            * `poa_sky_diffuse` = Plane of Array Sky Diffuse
            * `poa_ground_diffuse` = Plane of Array Ground Diffuse

            `poa_global` = `poa_direct` + `poa+diffuse`
            `poa_diffuse` = `poa_sky_diffuse` + `poa_ground_diffuse`


    """
    if isinstance(weather, str):
        weather = xr.open_dataset(weather)

    if model_chain is None:
        model_chain = dataset_to_modelchain(weather)

    if isinstance(model_chain, list) and len(model_chain) > 1:
        if "location" not in weather.coords:
            raise ValueError(
                "Weather dataset must have a 'location' coordinate when using multiple ModelChains"
            )
        if len(model_chain) != len(weather.location):
            raise ValueError(
                f"Number of ModelChains ({len(model_chain)}) must match number of locations ({len(weather.location)})"
            )
        mc_index = xr.DataArray(
            np.arange(len(model_chain)), dims=["location"], coords={"location": weather.location}
        )
    else:
        mc_index = xr.DataArray(0)

    (ac, dc, ei, poa_direct, poa_sky_diffuse, poa_ground_diffuse) = xr.apply_ufunc(
        lambda *args: _run_pvlib_slice(*args, model_chain=model_chain),
        weather[timedim],
        weather["tsi"],
        weather["dhi"],
        weather["dni"],
        weather["temp"],
        weather["wspd"],
        mc_index,
        input_core_dims=[
            [timedim],  # time
            [timedim],  # tsi
            [timedim],  # dhi
            [timedim],  # dni
            [timedim],  # temp
            [timedim],  # wspd
            [],  # mc_index
        ],
        output_core_dims=[
            [timedim],  # ac
            [timedim],  # dc
            [timedim],  # effective_irradiance
            [timedim],  # poa_direct
            [timedim],  # poa_sky_diffuse
            [timedim],  # poa_ground_diffuse
        ],
        vectorize=True,
    )

    pwr = xr.Dataset(
        {
            "ac": ac,
            "dc": dc,
            "effective_irradiance": ei,
            "poa_direct": poa_direct,
            "poa_sky_diffuse": poa_sky_diffuse,
            "poa_ground_diffuse": poa_ground_diffuse,
        },
        coords=weather.coords,
    )
    pwr["ac"].attrs = {"units": "W", "long_name": "AC Power", "standard_name": "ac"}

    pwr["dc"].attrs = {"units": "W", "long_name": "DC Power", "standard_name": "dc"}

    pwr["effective_irradiance"].attrs = {
        "units": "W m**-2",
        "long_name": "Effective Irradiance",
        "standard_name": "effective_irradiance",
    }

    pwr["poa_direct"].attrs = {
        "units": "W m**-2",
        "long_name": "Plane of Array Direct",
        "standard_name": "poa_direct",
    }

    pwr["poa_sky_diffuse"].attrs = {
        "units": "W m**-2",
        "long_name": "Plane of Array Sky Diffuse",
        "standard_name": "poa_sky_diffuse",
    }

    pwr["poa_ground_diffuse"].attrs = {
        "units": "W m**-2",
        "long_name": "Plane of Array Ground Diffuse",
        "standard_name": "poa_ground_diffuse",
    }

    _check_nans(pwr, nan_check)
    return pwr


def _run_pvlib_slice(
    time: list[float],  # "time",
    tsi: list[float],  # "ghi",
    dhi: list[float],  # "dhi",
    dni: list[float],  # "dni",
    temp: list[float],  # "temp_air",
    wspd: list[float],  # "wind_speed",
    mc_index: int = 0,
    model_chain: pv.modelchain.ModelChain | None = None,
) -> tuple[
    list[float],  # ac
    list[float],  # dc
    list[float],  # effective_irradiance
    list[float],  # poa_direct
    list[float],  # poa_sky_diffuse
    list[float],  # poa_ground_diffuse
]:
    """Run a single instance of a pvlib model chain.

    Intended to be called inside a ufunc to iterate over multiple
    lat/lon locations and also downscale ensembles.
    """
    if isinstance(model_chain, list):
        model_chain = model_chain[int(mc_index)]

    # Assemble all of the columns into a single pandas dataframe
    weather = pd.DataFrame(
        {
            "ghi": tsi,
            "dhi": dhi,
            "dni": dni,
            "temp_air": temp,
            "wind_speed": wspd,
        },
        index=pd.to_datetime(time, utc=True),
    )

    model_chain.run_model(weather)

    # Known issue: SingleAxisTracker mounts produce NaNs when the sun is near the horizon
    # https://github.com/pvlib/pvlib-python/issues/656
    # Workaround: fill NaNs with zeroes
    def _has_tracker(mc: pv.modelchain.ModelChain) -> bool:
        """Return True if any array mounts have trackers."""
        for arr in mc.system.arrays:
            if isinstance(arr.mount, pv.pvsystem.SingleAxisTrackerMount):
                return True
        return False

    if _has_tracker(model_chain):
        attributes_to_scrub = ["dc", "ac", "effective_irradiance", "total_irrad"]

        for attr in attributes_to_scrub:
            if hasattr(model_chain.results, attr):
                setattr(model_chain.results, attr, getattr(model_chain.results, attr).fillna(0))

    return (
        model_chain.results.ac,
        model_chain.results.dc,
        model_chain.results.effective_irradiance,
        model_chain.results.total_irrad["poa_direct"],
        model_chain.results.total_irrad["poa_sky_diffuse"],
        model_chain.results.total_irrad["poa_ground_diffuse"],
    )


def _dataset_to_modelchain_single(ds: xr.Dataset, idx: int = 0) -> pv.modelchain.ModelChain:
    """Generate a single pvlib modelchain from dataset."""

    def xtract(var, default=None):
        if var in ds:
            values = ds[var].values
            return values if values.shape == () else values[idx]
        elif default is None:
            raise ValueError(f"Variable '{var}' not available in the dataset")
        else:
            return default

    lat = xtract("lat")
    azi = xtract("azimuth", 180 if lat > 0 else 0)
    mnt = xtract("mount", "Fixed")

    if mnt == "Fixed":
        mount = pv.pvsystem.FixedMount(
            surface_tilt=xtract("tilt", lat),
            surface_azimuth=azi,
            # racking_model =
            # module_height =
        )
    elif mnt == "SingleAxisTracker":
        mount = pv.pvsystem.SingleAxisTrackerMount(
            axis_tilt=xtract("tilt", 0),
            axis_azimuth=azi,
            max_angle=80,  # default 90
            # backtrack, # default True
            # gcr, # default 2/7
            # cross_axistilt,
            # rackingmodel,
            # module_height)
        )
    elif mnt == "TwoAxisTracker":
        mount = pv.pvsystem.TwoAxisTrackerMount()
    else:
        raise ValueError(f"Unrecognized mount type {mnt} [Fixed|SingleAxis|TwoAxis]")

    mc = pv.modelchain.ModelChain(
        system=pv.pvsystem.PVSystem(
            arrays=pv.pvsystem.Array(
                mount=mount,
                module_parameters=dict(pdc0=1, gamma_pdc=-0.004),
                temperature_model_parameters=dict(a=-3.56, b=-0.075, deltaT=3),
                # module,
                # module_type,
                # modules_per_string,
                # strings,
                # array_losses_parameters,
                # albedo = None,
                # surface_type=,
            ),
            inverter_parameters=dict(pdc0=3),
        ),
        location=pv.location.Location(
            latitude=lat,
            longitude=xtract("lon"),
            tz="UTC",
            altitude=xtract("elevation", 0),
        ),
        aoi_model="physical",
        spectral_model="no_loss",
        name=xtract("location", "model"),
    )

    return mc


def dataset_to_modelchain(
    ds: xr.Dataset,
) -> list[pv.modelchain.ModelChain]:
    """Generate pvlib ModelChains from site characteristics in an xarray Dataset.

    Recognized values:
        * lat - latitude, decimal degrees.
        * lon - longitude, decimal degrees.
        * location - name of the site.  typically supplied by location_file.
        * elevation - elevation in meters.
        * azimuth - north/south orientation in degrees.  defaults to 180 in the northern hemisphere.
        * tilt - angle in degrees
        * mount - Fixed|SingleAxisTracker|TwoAxisTracker

    Args:
        ds (xr.Dataset): xarray dataset containing site characteristics.

    Returns:
        list[pvlib.modelchain.ModelChain]: a vector of `ModelChain` objects
            corresponding to the site characteristics in the input dataset.
            If `ds` has a one lat/lon (and no `location`) or a 2-d array of
            lat/lon from a `shapefile` polygon, will return a single `ModelChain`.
    """
    if "location" in ds.dims and "lat" in ds.coords and "lon" in ds.coords:
        # Location(location_file) - we may have different system properties at each location
        model_chain = [_dataset_to_modelchain_single(ds, i) for i in range(len(ds.location))]
    else:
        if len(ds.coords["lat"]) > 1 or len(ds.coords["lon"]) > 1:
            # Location(shapefile) gridded data with lat/lon.
            # Collapse to a single location.
            ds = ds.coarsen(lat=len(ds.lat), lon=len(ds.lon), boundary="trim").mean()
        model_chain = [_dataset_to_modelchain_single(ds)]

    return model_chain


def _check_nans(ds: xr.Dataset, nan_check=0.01, timedim="time") -> xr.Dataset:
    """Inspect a timeseries for NaN values and take action.

    Args:
        ds (xr.Dataset): The input dataset to check for NaN values.
        nan_check (float, optional): The threshold percentage of NaN values above which
        to issue a warning. Defaults to 0.05 (5%).
        timedim (str, optional): The name of the time dimension in the dataset.
            Defaults to "time".

    Returns:
        xr.Dataset: A dataset containing the percentage of NaN values for each variable,
            grouped by `month` and `location` (if available).
    """
    sum_dims = [timedim, "location"] if "location" in ds.dims else [timedim]
    nan_vars = [var for var in ds.data_vars if all(dim in ds[var].dims for dim in sum_dims)]
    nan_mask = ds[nan_vars].isnull()

    nan_mask["month"] = nan_mask[timedim].dt.strftime("%Y-%m")
    nan_mask = nan_mask.swap_dims({timedim: "month"})
    avg_dims = [dim for dim in nan_mask.dims if dim not in ["month", "location"]]
    nan_pct = nan_mask.groupby("month").mean().mean(avg_dims)

    if nan_check is not None:
        nan_bad = nan_pct.where(nan_pct > nan_check).dropna(dim="month", how="all")
        bad_tabl = nan_bad.to_dataframe()[nan_vars].dropna(how="all").dropna(axis=1, how="all")

        if not bad_tabl.empty:
            print(f"Warning: {len(bad_tabl.index)} months have NaNs > {nan_check:.1%}:")
            print(bad_tabl)

    return nan_pct
