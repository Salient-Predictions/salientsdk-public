#!/usr/bin/env python
# Copyright Salient Predictions 2025

"""Hydrology functions."""

import datetime as dt
import hashlib
import os
import subprocess
import tempfile
from importlib import resources

import numpy as np
import pandas as pd
import pyproj
import requests
import xarray as xr
from shapely.geometry import box

from .constants import get_file_destination, get_logger
from .data_timeseries_api import data_timeseries, load_multihistory
from .geo_api import geo
from .location import Location

_LOGGER = get_logger("salientsdk.hydro")

# Dictionary mapping Salient variables to VIC parameters
_SALIENT_VIC_PARAMS = {
    "elevation": {"name": "elev", "dtype": "float64"},
    "bdod": {"name": "bulk_density", "dtype": "float64"},
    "fal": {"name": "albedo", "dtype": "float64"},
}


_VIC_SOIL_TEXTURE = "vic_soil_texture_lookup.csv"
_VIC_VEG = "vic_veg_lookup.csv"
_VIC_VALID_SOIL_DEPTHS = [0.05, 2.0]

VIC_VALID_DATE_RANGE = [dt.date(2016, 1, 1), dt.date(2024, 12, 31)]


def calc_swe(met: xr.Dataset, timedim: str = "forecast_day") -> xr.DataArray:
    """Call the `snow17` model for each location and ensemble member.

    Acknowledgements: Based on Anderson (2006) and Mark Raleigh's matlab code.

    Primary Citations:
    1.  Anderson, E. A. (1973), National Weather Service River Forecast System
    Snow   Accumulation   and   Ablation   Model,   NOAA   Tech.   Memo.   NWS
    HYDro-17, 217 pp., U.S. Dep. of Commer., Silver Spring, Md.
    2.  Anderson, E. A. (1976), A point energy and mass balance model of a snow
    cover, NOAA Tech. Rep. 19, 150 pp., U.S. Dep. of Commer., Silver Spring, Md.

    Written by Joe Hamman April, 2013

    Args:
        met (xr.Dataset): a dataset containing `timedim`, `precip`, `temp`, & `lat`.
            If `met` contains a field `elevation` it will be used.
        timedim (str): the name of the time dimension in `met`
            The time step in `timedim` can be `hourly` or `daily`.

    Returns:
        xr.DataArray: a dataset containing snow water equivalent (mm)
    """
    elev = met["elevation"] if "elevation" in met else 0
    dt = _get_timestep_hours(met[timedim])  # 1 = hourly, 24 = daily frequency
    # snow17 needs a vector of datetime objects to call timetuple()
    time = pd.DatetimeIndex(met[timedim].values).to_pydatetime()

    # Apply the snow model over all locations and all ensembles
    (swe, outflow) = xr.apply_ufunc(
        snow17,
        time,  # time,
        met.precip,  # prec,
        met.temp,  # tair,
        met.lat,  # lat
        elev,  # elevation
        dt,  # dt
        input_core_dims=[[timedim], [timedim], [timedim], [], [], []],
        output_core_dims=[[timedim], [timedim]],
        vectorize=True,
    )

    # After calling ufunc, swe will have timedim last.  Force it to match others:
    swe = swe.transpose(*met.temp.dims)

    swe.attrs = {"short_name": "swe", "long_name": "Snow Water Equivalent", "units": "mm"}

    return swe


def _get_timestep_hours(time_array: xr.DataArray) -> float:
    """Return the timestep of a datetime array in hours."""
    t0, t1 = time_array[:2]
    return ((t1 - t0) / np.timedelta64(1, "h")).item()


def snow17(
    time,
    prec,
    tair,
    lat=50,
    elevation=0,
    dt=24,
    scf=1.0,
    rvs=1,
    uadj=0.04,
    mbase=1.0,
    mfmax=1.05,
    mfmin=0.6,
    tipm=0.1,
    nmf=0.15,
    plwhc=0.04,
    pxtemp=1.0,
    pxtemp1=-1.0,
    pxtemp2=3.0,
):
    """Snow-17 accumulation and ablation model.

    This version of Snow-17 is intended for use at a point location.
    The time steps for precipitation and temperature must be equal for this
    code.

    Args:
        time (1d numpy.ndarray or scalar): Array of datetime objects.
        prec (1d numpy.ndarray or scalar): Array of precipitation forcings, size of `time`.
        tair (1d numpy.ndarray or scalar): Array of air temperature forcings, size of `time`.
        lat (float, optional): Latitude of simulation point or grid cell. Defaults to None.
        elevation (float, optional): Elevation of simulation point or grid cell. Defaults to 0.
        dt (float, optional): Timestep in hours. Defaults to 24 hours but should always match the timestep in `time`.
        scf (float, optional): Gauge under-catch snow correction factor. Defaults to 1.0.
        rvs ({0, 1, 2}, optional): Rain vs. Snow option. Default value of 1 is a linear transition between 2 temperatures (pxtemp1 and pxtemp2).
        uadj (float, optional): Average wind function during rain on snow (mm/mb). Defaults to 0.04, based on data from the American River Basin (Shamir & Georgakakos 2007).
        mbase (float, optional): Base temperature above which melt typically occurs (deg C). Defaults to 1.0, based on data from the American River Basin (Shamir & Georgakakos 2007). Must be greater than 0 deg C.
        mfmax (float, optional): Maximum melt factor during non-rain periods (mm/deg C 6 hr) - in western facing slope assumed to occur on June 21. Defaults to 1.05, based on data from the American River Basin (Shamir & Georgakakos 2007).
        mfmin (float, optional): Minimum melt factor during non-rain periods (mm/deg C 6 hr) - in western facing slope assumed to occur on December 21. Defaults to 0.60, based on data from the American River Basin (Shamir & Georgakakos 2007).
        tipm (float, optional): Model parameter (>0.0 and <1.0) - Anderson Manual recommends 0.1 to 0.2 for deep snowpack areas. Defaults to 0.1.
        nmf (float, optional): Percent liquid water holding capacity of the snow pack - max is 0.4. Defaults to 0.04, based on data from the American River Basin (Shamir & Georgakakos 2007).
        plwhc (float, optional): Percent liquid water holding capacity of the snow pack - max is 0.4. Defaults to 0.04, based on data from the American River Basin (Shamir & Georgakakos 2007).
        pxtemp (float, optional): Temperature dividing rain from snow, deg C - if temp is less than or equal to pxtemp, all precip is snow. Otherwise, it is rain. Defaults to 1.0.
        pxtemp1 (float, optional): Lower Limit Temperature dividing transition from snow, deg C - if temp is less than or equal to pxtemp1, all precip is snow. Otherwise, it is mixed linearly. Defaults to -1.0.
        pxtemp2 (float, optional): Upper Limit Temperature dividing rain from transition, deg C - if temp is greater than or equal to pxtemp2, all precip is rain. Otherwise, it is mixed linearly. Defaults to 3.0.

    Returns:
        model_swe (numpy.ndarray): Simulated snow water equivalent.
        outflow (numpy.ndarray): Simulated runoff outflow.
    """
    # Convert to numpy array if scalars
    time = np.asarray(time)
    prec = np.asarray(prec)
    tair = np.asarray(tair)

    assert time.shape == prec.shape == tair.shape

    # Initialization
    # Antecedent Temperature Index, deg C
    ait = 0.0
    # Liquid water capacity
    w_qx = 0.0
    # Liquid water held by the snow (mm)
    w_q = 0.0
    # accumulated water equivalent of the iceportion of the snow cover (mm)
    w_i = 0.0
    # Heat deficit, also known as NEGHS, Negative Heat Storage
    deficit = 0.0

    # number of time steps
    nsteps = len(time)
    model_swe = np.zeros(nsteps)
    outflow = np.zeros(nsteps)

    # Stefan-Boltzman constant (mm/K/hr)
    stefan = 6.12 * (10 ** (-10))
    # atmospheric pressure (mb) where elevation is in HUNDREDS of meters
    # (this is incorrectly stated in the manual)
    p_atm = 33.86 * (29.9 - (0.335 * elevation / 100) + (0.00022 * ((elevation / 100) ** 2.4)))

    transitionx = [pxtemp1, pxtemp2]
    transitiony = [1.0, 0.0]

    tipm_dt = 1.0 - ((1.0 - tipm) ** (dt / 6))

    # Model Execution
    for i, t in enumerate(time):
        mf = melt_function(t, dt, lat, mfmax, mfmin)

        # air temperature at this time step (deg C)
        t_air_mean = tair[i]
        # precipitation at this time step (mm)
        precip = prec[i]

        # Divide rain and snow
        if rvs == 0:
            if t_air_mean <= pxtemp:
                # then the air temperature is cold enough for snow to occur
                fracsnow = 1.0
            else:
                # then the air temperature is warm enough for rain
                fracsnow = 0.0
        elif rvs == 1:
            if t_air_mean <= pxtemp1:
                fracsnow = 1.0
            elif t_air_mean >= pxtemp2:
                fracsnow = 0.0
            else:
                fracsnow = np.interp(t_air_mean, transitionx, transitiony)
        elif rvs == 2:
            fracsnow = 1.0
        else:
            raise ValueError("Invalid rain vs snow option")

        fracrain = 1.0 - fracsnow

        # Snow Accumulation
        # water equivalent of new snowfall (mm)
        pn = precip * fracsnow * scf
        # w_i = accumulated water equivalent of the ice portion of the snow
        # cover (mm)
        w_i += pn
        e = 0.0
        # amount of precip (mm) that is rain during this time step
        rain = fracrain * precip

        # Temperature and Heat deficit from new Snow
        if t_air_mean < 0.0:
            t_snow_new = t_air_mean
            # delta_hd_snow = change in the heat deficit due to snowfall (mm)
            delta_hd_snow = -(t_snow_new * pn) / (80 / 0.5)
            t_rain = pxtemp
        else:
            t_snow_new = 0.0
            delta_hd_snow = 0.0
            t_rain = t_air_mean

        # Antecedent temperature Index
        if pn > (1.5 * dt):
            ait = t_snow_new
        else:
            # Antecedent temperature index
            ait = ait + tipm_dt * (t_air_mean - ait)
        if ait > 0:
            ait = 0

        # Heat Exchange when no Surface melt
        # delta_hd_t = change in heat deficit due to a temperature gradient(mm)
        delta_hd_t = nmf * (dt / 6.0) * ((mf) / mfmax) * (ait - t_snow_new)

        # Rain-on-snow melt
        # saturated vapor pressure at t_air_mean (mb)
        e_sat = 2.7489 * (10**8) * np.exp((-4278.63 / (t_air_mean + 242.792)))
        # 1.5 mm/ 6 hrs
        if rain > (0.25 * dt):
            # melt (mm) during rain-on-snow periods is:
            m_ros1 = np.maximum(stefan * dt * (((t_air_mean + 273) ** 4) - (273**4)), 0.0)
            m_ros2 = np.maximum((0.0125 * rain * t_rain), 0.0)
            m_ros3 = np.maximum(
                (
                    8.5
                    * uadj
                    * (dt / 6.0)
                    * (((0.9 * e_sat) - 6.11) + (0.00057 * p_atm * t_air_mean))
                ),
                0.0,
            )
            m_ros = m_ros1 + m_ros2 + m_ros3
        else:
            m_ros = 0.0

        # Non-Rain melt
        if rain <= (0.25 * dt) and (t_air_mean > mbase):
            # melt during non-rain periods is:
            m_nr = (mf * (t_air_mean - mbase)) + (0.0125 * rain * t_rain)
        else:
            m_nr = 0.0

        # Ripeness of the snow cover
        melt = m_ros + m_nr
        if melt <= 0:
            melt = 0.0

        if melt < w_i:
            w_i = w_i - melt
        else:
            melt = w_i + w_q
            w_i = 0.0

        # qw = liquid water available melted/rained at the snow surface (mm)
        qw = melt + rain
        # w_qx = liquid water capacity (mm)
        w_qx = plwhc * w_i
        # deficit = heat deficit (mm)
        deficit += delta_hd_snow + delta_hd_t

        # limits of heat deficit
        if deficit < 0:
            deficit = 0.0
        elif deficit > 0.33 * w_i:
            deficit = 0.33 * w_i

        # Snow cover is ripe when both (deficit=0) & (w_q = w_qx)
        if w_i > 0.0:
            if (qw + w_q) > ((deficit * (1 + plwhc)) + w_qx):
                # THEN the snow is RIPE
                # Excess liquid water (mm)
                e = qw + w_q - w_qx - (deficit * (1 + plwhc))
                # fills liquid water capacity
                w_q = w_qx
                # w_i increases because water refreezes as heat deficit is
                # decreased
                w_i = w_i + deficit
                deficit = 0.0
            elif (qw >= deficit) and ((qw + w_q) <= ((deficit * (1 + plwhc)) + w_qx)):
                # ait((qw + w_q) <= ((deficit * (1 + plwhc)) + w_qx))):  BUG???
                # https://github.com/UW-Hydro/tonic/issues/78
                # THEN the snow is NOT yet ripe, but ice is being melted
                e = 0.0
                w_q = w_q + qw - deficit
                # w_i increases because water refreezes as heat deficit is
                # decreased
                w_i = w_i + deficit
                deficit = 0.0
            else:
                # (qw < deficit) %elseif ((qw + w_q) < deficit):
                # THEN the snow is NOT yet ripe
                e = 0.0
                # w_i increases because water refreezes as heat deficit is
                # decreased
                w_i = w_i + qw
                deficit = deficit - qw
            swe = w_i + w_q
        else:
            e = qw
            swe = 0

        if deficit == 0:
            ait = 0

        # End of model execution
        model_swe[i] = swe  # total swe (mm) at this time step
        outflow[i] = e

    return model_swe, outflow


def melt_function(t, dt, lat, mfmax, mfmin):
    """Seasonal variation calcs - indexed for Non-Rain melt.

    Args:
        t (datetime): Datetime object for the current timestep.
        dt (float): Timestep duration in hours.
        lat (float): Latitude of the simulation point or grid cell.
        mfmax (float): Maximum melt factor during non-rain periods (mm/deg C per 6 hours),
            typically occurring on June 21. The default value of 1.05 is based on data from
            the American River Basin (Shamir & Georgakakos, 2007).
        mfmin (float): Minimum melt factor during non-rain periods (mm/deg C per 6 hours),
            typically occurring on December 21. The default value of 0.60 is based on data from
            the American River Basin (Shamir & Georgakakos, 2007).

    Returns:
        float: Melt function value for the current timestep.
    """
    tt = t.timetuple()
    jday = tt[-2]
    n_mar21 = jday - 80
    days = 365

    # seasonal variation
    sv = (0.5 * np.sin((n_mar21 * 2 * np.pi) / days)) + 0.5
    if lat < 54:
        # latitude parameter, av=1.0 when lat < 54 deg N
        av = 1.0
    else:
        if jday <= 77 or jday >= 267:
            # av = 0.0 from September 24 to March 18,
            av = 0.0
        elif jday >= 117 and jday <= 227:
            # av = 1.0 from April 27 to August 15
            av = 1.0
        elif jday >= 78 and jday <= 116:
            # av varies linearly between 0.0 and 1.0 from 3/19-4/26 and
            # between 1.0 and 0.0 from 8/16-9/23.
            av = np.interp(jday, [78, 116], [0, 1])
        elif jday >= 228 and jday <= 266:
            av = np.interp(jday, [228, 266], [1, 0])
    meltf = (dt / 6) * ((sv * av * (mfmax - mfmin)) + mfmin)

    return meltf


def exec_vic(
    # API arguments -----
    loc: Location,
    start: str,
    end: str,
    soil_depths: list[float] = [0.1, 0.3, 1],
    reference_clim: int = 10,
    # Non-API arguments --------
    vic_image_path: str = "vic_image.exe",
    destination: str = "-default",
    force: bool = False,
    session: requests.Session | None = None,
    apikey: str | None = None,
    verify: bool | None = None,
    verbose: bool = False,
    **kwargs,
):
    """Execute the Variable Infiltration Capacity model for a location over a time range.

    Args:
        loc (Location): The location to query. This location must be a shapefile with one or more polygons defining the location.
        start (str): The start date of the time series.
        end (str): The end date of the time series.
        soil_depths (list): Bottom of each soil layer in meters.
        reference_clim (int): Reference period in years to calculate average annual precipitation.
        vic_image_path (str): Path to vic_image.exe.  Default assumes vic_image.exe is in PATH or current directory.
        destination (str): The destination directory for downloaded files.
        force (bool): If False (default), don't download the data if it already exists
        session (requests.Session): The session object to use for the request
        apikey (str | None): The API key to use for the request.
            In most cases, this is not needed if a `session` is provided.
        verify (bool): If True (default), verify the SSL certificate
        verbose (bool): If True (default False) print status messages
        **kwargs: Additional arguments to pass to the API

    Returns:
        Dictionary of paths to VIC inputs/outputs.
    """
    _LOGGER.warning("salientsdk.hydro.exec_vic is experimental and may change in future releases.")

    vic_paths = _build_vic_inputs(
        loc=loc,
        start=start,
        end=end,
        soil_depths=soil_depths,
        reference_clim=reference_clim,
        destination=destination,
        force=force,
        session=session,
        apikey=apikey,
        verify=verify,
        verbose=verbose,
        **kwargs,
    )

    try:
        with open(vic_paths["vic_log_path"], "w") as log_file:
            _LOGGER.info("Begin VIC execution...")
            sp_out = subprocess.run(
                [vic_image_path, "-g", vic_paths["global_params_path"]],
                stdout=log_file,
                stderr=log_file,
            )

        assert (
            sp_out.returncode == 0
        ), f"""
        An error occurred while executing VIC. Check the log file for more details:
            {vic_paths["vic_log_path"]}
        """
        _LOGGER.info("VIC execution complete!")
        return vic_paths
    except FileNotFoundError as e:
        print(
            f"""
        The below path to vic_image.exe was not found:
              {vic_image_path}
        Please reference the documentation for compiling the VIC Image Driver:
        https://vic.readthedocs.io/en/master/Documentation/Drivers/Image/RunVIC/.
        Once compiled, input the path to the VIC image driver via the vic_image_path
        argument or add the directory that contains the VIC image driver to PATH.
        """
        )
        raise e


def _build_vic_inputs(
    # API arguments -----
    loc: Location,
    start: str,
    end: str,
    soil_depths: list[float] = [0.1, 0.3, 1],
    reference_clim: int = 10,
    # Non-API arguments --------
    destination: str = "-default",
    force: bool = False,
    session: requests.Session | None = None,
    apikey: str | None = None,
    verify: bool | None = None,
    verbose: bool = False,
    **kwargs,
) -> dict:
    """Generate VIC domain, forcings, parameters, and global parameters files.

    Args:
        loc (Location): The location to query. This location must be a shapefile with one or more polygons defining the location.
        start (str): The start date of the time series.
        end (str): The end date of the time series.
        soil_depths (list): Bottom of each soil layer in meters.
        reference_clim (int): Reference period in years to calculate average annual precipitation.
        destination (str): The destination directory for downloaded files.
        force (bool): If False (default), don't download the data if it already exists
        session (requests.Session): The session object to use for the request
        apikey (str | None): The API key to use for the request.
            In most cases, this is not needed if a `session` is provided.
        verify (bool): If True (default), verify the SSL certificate
        verbose (bool): If True (default False) print status messages
        **kwargs: Additional arguments to pass to the API

    Returns:
        dict: Path to the global parameter file used as input to the VIC Image Driver.
    """
    _LOGGER.info("Generating VIC inputs...")
    assert loc.shapefile is not None, "The input location must be a shapefile!"

    if isinstance(loc.shapefile, list):
        assert (
            len(loc.shapefile) == 1
        ), "The input location mast be a single shapefile with a single polygon defining the location!"

    # Ensure start and end times are within valid date range for VIC
    start_date = dt.datetime.fromisoformat(start).date()
    end_date = dt.datetime.fromisoformat(end).date()
    start_is_valid = (
        start_date >= VIC_VALID_DATE_RANGE[0] and start_date <= VIC_VALID_DATE_RANGE[1]
    )
    end_is_valid = end_date >= VIC_VALID_DATE_RANGE[0] and end_date <= VIC_VALID_DATE_RANGE[1]
    assert (
        start_is_valid and end_is_valid
    ), f"Start and end times must be within the date range {VIC_VALID_DATE_RANGE[0]}-{VIC_VALID_DATE_RANGE[1]}!"

    # Ensure start is less than end time
    assert start_date < end_date, "Start must be less than end time!"

    out_paths = _init_vic_destinations(
        destination=destination,
        loc=loc.shapefile,
        start=start,
        end=end,
        reference_clim=reference_clim,
    )

    _LOGGER.info(f"VIC root directory: {out_paths['root_dir']}")

    if os.path.exists(out_paths["global_params_path"]) and force is False:
        _LOGGER.info("It appears that VIC inputs have already been generated. Skipping...")
        return out_paths

    salient_ds = _get_salient_params(
        loc=loc,
        start=start,
        end=end,
        reference_clim=reference_clim,
        destination=destination,
        force=force,
        session=session,
        apikey=apikey,
        verify=verify,
        verbose=verbose,
        **kwargs,
    )

    domain_ds = _build_vic_domain(salient_ds=salient_ds, out_path=out_paths["domain_path"])

    _build_vic_forcings(
        start=start,
        end=end,
        salient_ds=salient_ds,
        domain_path=out_paths["domain_path"],
        out_path=out_paths["forcings_path"],
    )

    _build_vic_params(
        salient_ds=salient_ds,
        domain_ds=domain_ds,
        start=start,
        end=end,
        soil_depths=soil_depths,
        out_path=out_paths["params_path"],
    )

    _build_vic_global_params(
        start=start, end=end, n_soil_layers=len(soil_depths), out_paths=out_paths
    )

    _LOGGER.info("VIC input generation complete!")
    return out_paths


def _init_vic_destinations(destination: str, **kwargs) -> dict:
    """Make directories and generate paths for VIC inputs and outputs.

    Args:
        destination (str): The destination directory for downloaded files.
        **kwargs: Additional arguments used to construct the file name MD5 hash.

    Returns:
        Dictionary of paths to VIC inputs and outputs.
    """
    dest_dir = get_file_destination(destination)
    md5 = hashlib.md5(str(kwargs).encode()).hexdigest()
    root_dir = f"{dest_dir}/vic_{md5}"
    inputs_dir = f"{root_dir}/inputs"
    os.makedirs(inputs_dir, exist_ok=True)
    outputs_dir = f"{root_dir}/outputs"
    os.makedirs(outputs_dir, exist_ok=True)
    return {
        "root_dir": root_dir,
        "inputs_dir": inputs_dir,
        "outputs_dir": outputs_dir,
        "domain_path": f"{inputs_dir}/domain.nc",
        "forcings_path": f"{inputs_dir}/forcings.",
        "params_path": f"{inputs_dir}/params.nc",
        "global_params_path": f"{inputs_dir}/global_params.txt",
        "vic_log_path": f"{root_dir}/vic.log",
    }


def _get_salient_params(
    # API arguments -----
    loc: Location,
    start: str,
    end: str,
    reference_clim: int = 10,
    # Non-API arguments --------
    destination: str = "-default",
    force: bool = False,
    session: requests.Session | None = None,
    apikey: str | None = None,
    verify: bool | None = None,
    verbose: bool = False,
    **kwargs,
) -> xr.Dataset:
    """Get Salient variables used to generate VIC inputs.

    Args:
        loc (Location): The location to query. This location must be a shapefile with one or more polygons defining the location.
        start (str): The start date.
        end (str): The end date.
        reference_clim (int): Reference period in years to calculate average annual precipitation.
        destination (str): The destination directory for downloaded files.
        force (bool): If False (default), don't download the data if it already exists
        session (requests.Session): The session object to use for the request
        apikey (str | None): The API key to use for the request.
            In most cases, this is not needed if a `session` is provided.
        verify (bool): If True (default), verify the SSL certificate
        verbose (bool): If True (default False) print status messages
        **kwargs: Additional arguments to pass to the API

    Returns:
        xr.Dataset: xarray.Dataset of weather and geo data.
    """
    _LOGGER.info(f"Fetching data from Salient API...")
    # Get geo data used for VIC domain and parameters.
    geo_path = geo(
        loc=loc,
        variables=[
            "elevation",
            "slope",
            "sand",
            "clay",
            "bdod",
            "lulc_igbp",
            "lulc_igbp_per",
            "fal",
            "lai_hv",
            "lai_lv",
        ],
        resolution=0.25,
        format="nc",
        destination=destination,
        force=force,
        session=session,
        apikey=apikey,
        verify=verify,
        verbose=verbose,
        **kwargs,
    )
    geo_ds = xr.open_dataset(geo_path)

    # Get weather data for average annual precipitation
    end_year = pd.Timestamp(start).year - 1
    start_year = end_year - reference_clim
    clim_precip_paths = data_timeseries(
        loc=loc,
        variable=["precip", "st"],
        field="vals",
        start=f"{start_year}-01-01",
        end=f"{end_year}-12-31",
        format="nc",
        destination=destination,
        force=force,
        session=session,
        apikey=apikey,
        verify=verify,
        verbose=verbose,
        **kwargs,
    )

    clim_ds = (
        load_multihistory(clim_precip_paths).squeeze("region").drop_vars(["region", "spatial_ref"])
    )

    # Compute average annual precipitation
    annual_prec_da = clim_ds["precip"].resample(time="YS").sum("time")
    geo_ds["annual_prec"] = annual_prec_da.mean("time")

    # Compute average soil temperature
    geo_ds["avg_T"] = clim_ds["st"].mean("time")

    # Get weather data used for VIC forcings.
    offset_start = str(dt.datetime.fromisoformat(start) - dt.timedelta(days=90))
    weather_paths = data_timeseries(
        loc=loc,
        variable=["tmax", "tmin", "precip", "wspd"],
        field="vals",
        start=offset_start,
        end=end,
        format="nc",
        destination=destination,
        force=force,
        session=session,
        apikey=apikey,
        verify=verify,
        verbose=verbose,
        **kwargs,
    )

    # Reformat weather dataset
    weather_ds = load_multihistory(weather_paths)
    assert weather_ds["region"].size == 1, "Location files can only include a single polygon!"
    weather_ds = weather_ds.squeeze("region").drop_vars(["region", "spatial_ref"], errors="ignore")

    # Merge datasets into a single "Salient" dataset
    salient_ds = xr.merge([geo_ds, weather_ds])

    # Prepare Salient dataset for VIC
    vic_das = {}
    for salient_var in salient_ds.data_vars:
        vic_param = _SALIENT_VIC_PARAMS.get(salient_var)
        if vic_param is None:
            vic_das[salient_var] = salient_ds[salient_var]
        else:
            vic_das[vic_param["name"]] = salient_ds[salient_var].astype(vic_param["dtype"])

    salient_ds = xr.Dataset(vic_das)
    salient_ds["lat"] = salient_ds["lat"].astype("float64")
    salient_ds["lon"] = salient_ds["lon"].astype("float64")
    _LOGGER.info(f"Salient data fetch complete!")
    return salient_ds


def _build_vic_domain(salient_ds: xr.Dataset, out_path: str) -> xr.Dataset:
    """Generate the VIC domain dataset and write to output path.

    Args:
        salient_ds (xr.Dataset): Salient variables generated by _get_salient_params()).
        out_path (str): Path to output VIC domain NetCDF dataset generated by _init_vic_destinations().

    Returns:
        xr.Dataset of domain variables.
    """
    _LOGGER.info(f"Generating VIC domain dataset!")
    # mask
    not_null = salient_ds["elev"].notnull()
    mask_da = salient_ds["elev"].where(not_null, 0)
    mask_da = mask_da.where(~not_null.values, 1).astype("int32")
    mask_da.name = "mask"

    # area
    # TODO: Optimize area computations.
    geod = pyproj.Geod(ellps="WGS84")
    areas = []
    for lat in mask_da.lat:
        poly = box(0, lat - 0.25 / 2, 0.25, lat + 0.25 / 2)
        area, _ = geod.geometry_area_perimeter(poly)
        areas.append(area)
    areas = np.array(areas).reshape(-1, 1)
    area_da = mask_da.where(mask_da == 1, 1) * areas

    # lats, lons, gridcell
    df = mask_da.to_dataframe().reset_index().drop("mask", axis=1)
    df = df.set_index(["lat", "lon"], drop=False)
    df = df.rename({"lat": "lats", "lon": "lons"}, axis=1)
    df["gridcell"] = np.arange(1, df.shape[0] + 1, dtype="float64")

    domain_ds = xr.Dataset(
        {
            "elev": salient_ds["elev"],
            "mask": mask_da,
            "area": area_da,
            "frac": mask_da.astype("float64"),
            "run_cell": mask_da,
        }
    )
    domain_ds = xr.merge([domain_ds, df.to_xarray()])
    domain_ds.to_netcdf(out_path)
    _LOGGER.info(f"VIC domain dataset complete!")
    return domain_ds


def _build_vic_forcings(
    start: str, end: str, domain_path: str, salient_ds: xr.Dataset, out_path: str
) -> xr.Dataset:
    """Generate the VIC forcings dataset and write each year to a NetCDF file.

    Args:
        start (str): The start date of the time series.
        end (str): The end date of the time series.
        domain_path (str): Path to the VIC domain dataset generated by _build_vic_domain().
        salient_ds (xr.Dataset): Salient variables generated by _get_salient_params().
        out_path (str): Path to output VIC forcings NetCDF datasets generated by _init_vic_destinations().

    Returns:
        xr.Dataset of VIC forcings.
    """
    _LOGGER.info(f"Generating VIC forcings datasets...")
    try:
        from metsim import MetSim
    except ImportError:
        raise ImportError("The metsim package is required. Install with 'poetry add metsim'")

    weather_ds = salient_ds[["tmax", "tmin", "precip", "wspd"]]

    # Simulate air_pressure, shortware, longwave, and vapor_pressure
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write weather dataset
        weather_path = f"{tmpdir}/weather.nc"
        weather_ds.to_netcdf(weather_path)

        # Simulate w/ Metsim
        metsim_params = _get_metsim_params(
            start=start,
            end=end,
            domain_path=domain_path,
            weather_path=weather_path,
            out_dir=tmpdir,
        )
        ms = MetSim(metsim_params)
        ms.run()

        # Convert MetSim output to VIC forcings
        metsim_ds = xr.open_mfdataset(f"{tmpdir}/metsim_output*.nc")
        for year, forcing_ds in metsim_ds.groupby("time.year"):
            forcing_path = f"{out_path}{year}.nc"
            forcing_ds.to_netcdf(forcing_path)
    _LOGGER.info(f"VIC forcings datasets complete!")
    return xr.open_mfdataset(f"{out_path}*.nc")


def _get_metsim_params(
    start: str, end: str, domain_path: str, weather_path: str, out_dir: str
) -> dict:
    """Get the parameters used as input for MetSim to simulate weather variables necessary for VIC.

    Args:
        start (str): The start date of the time series.
        end (str): The end date of the time series.
        domain_path (str): Path to the VIC domain dataset generated by _build_vic_domain().
        weather_path (str): Path to the weather NetCDF dataset with variables tmin, tmax, precip, and wspd.
        out_dir (str): Path to output directory where results will be written.

    Returns:
        Dictionary of parameters used to execute MetSim.
    """
    params = {
        # Time step in minutes. Must be divisible by 1440.
        # 360 minutes is the coarsest resolution for most variables.
        # See https://metsim.readthedocs.io/en/1.1.0/data.html#output-specifications
        "time_step": 360,
        "start": start,
        "stop": end,
        "forcing": weather_path,
        "domain": domain_path,
        "state": weather_path,
        "forcing_fmt": "netcdf",
        "out_dir": out_dir,
        "out_prefix": "metsim_output",
        "chunks": {},
        "forcing_vars": {"precip": "prec", "tmax": "t_max", "tmin": "t_min", "wspd": "wind"},
        "out_vars": {
            "temp": {"out_name": "temp"},
            "prec": {"out_name": "prec"},
            "air_pressure": {"out_name": "air_pressure"},
            "shortwave": {"out_name": "shortwave"},
            "longwave": {"out_name": "longwave"},
            "vapor_pressure": {"out_name": "vapor_pressure"},
            "wind": {"out_name": "wind"},
        },
        "state_vars": {"precip": "prec", "tmax": "t_max", "tmin": "t_min"},
        "domain_vars": {"elev": "elev", "lat": "lat", "lon": "lon", "mask": "mask"},
    }
    return params


def _build_vic_params(
    salient_ds: xr.Dataset,
    domain_ds: xr.Dataset,
    start: str,
    end: str,
    out_path: str,
    soil_depths: list[float] = [0.1, 0.3, 1],
) -> xr.Dataset:
    """Generate the VIC parameters dataset.

    Args:
        salient_ds (xr.Dataset): Salient variables generated by _get_salient_params().
        domain_ds (xr.Dataset): VIC domain dataset generated by _build_vic_domain().
        start (str): The start date of the time series.
        end (str): The end date of the time series.
        out_path (str): Path to output VIC parameters NetCDF dataset generated by _init_vic_destinations().
        soil_depths (list): Bottom of each soil layer in meters. Default values are 0.1, 0.3, and 1 meter(s).

    Returns:
        xr.Dataset of VIC parameters.
    """
    _LOGGER.info(f"Generating VIC parameters dataset...")
    soil_ds = _build_vic_soil(salient_ds, soil_depths)
    veg_ds = _build_vic_veg(salient_ds, start, end)
    param_domain_ds = domain_ds[["run_cell", "gridcell", "mask", "lats", "lons"]]
    params_ds = xr.merge([param_domain_ds, soil_ds, veg_ds])
    params_ds.to_netcdf(out_path)
    _LOGGER.info(f"VIC parameters dataset complete!")
    return params_ds


def _build_vic_soil(
    salient_ds: xr.Dataset, soil_depths: list[float] = [0.1, 0.3, 1]
) -> xr.Dataset:
    """Generate VIC soil parameters.

    Args:
        salient_ds (xr.Dataset): Salient variables generated by _get_salient_params().
        soil_depths (list): Bottom of each soil layer in meters. Default values are 0.1, 0.3, and 1 meter(s).

    Returns:
        xr.Dataset of VIC soil parameters.
    """
    try:
        from soiltexture import getTexture
    except ImportError:
        raise ImportError(
            "The soiltexture package is required. Install with 'poetry add soiltexture'"
        )

    # Ensure soil depths are within the domain of soil variable depths.
    for soil_depth in soil_depths:
        assert (
            soil_depth >= _VIC_VALID_SOIL_DEPTHS[0] and soil_depth <= _VIC_VALID_SOIL_DEPTHS[1]
        ), f"{soil_depth} meter(s) is an invalid soil depth! Soil depths must be between {_VIC_VALID_SOIL_DEPTHS[0]} and {_VIC_VALID_SOIL_DEPTHS[1]} meters."

    # Interpolate soil data to input soil_depths
    f_ds = salient_ds[["bulk_density", "sand", "clay"]]
    f_ds = f_ds.interp(depth=soil_depths, method="linear", kwargs={"fill_value": "extrapolate"})

    # Convert bulk density from g/cm3 to kg/m3
    f_ds["bulk_density"] = f_ds["bulk_density"] * 1000

    #
    # Add all variables with dimensions (depth, lat, lon)
    #

    depth_df = f_ds.to_dataframe().reset_index()

    # Add thickness and layer number
    thicks = {}
    top = 0
    for depth in soil_depths:
        thicks[depth] = depth - top
        top = depth
    depth_df["thickness"] = depth_df["depth"].replace(thicks)
    depth_df["nlayer"] = (
        depth_df["depth"]
        .replace({depth: idx + 1 for idx, depth in enumerate(sorted(soil_depths))})
        .astype("int32")
    )

    # Add soil texture lookup
    with resources.files("salientsdk.data").joinpath(_VIC_SOIL_TEXTURE).open("r") as f:
        text_lu = pd.read_csv(f)
    depth_df["texture"] = depth_df.apply(lambda x: getTexture(x["sand"], x["clay"]), axis=1)
    depth_df = depth_df.merge(text_lu, on="texture", how="left")

    # Add constants
    depth_df["phi_s"] = -999.0
    depth_df["init_moist"] = (
        depth_df["Wcr_FRACT"] * depth_df["porosity"] * depth_df["thickness"] * 1000
    )
    depth_df["bubble"] = 0.32 * depth_df["expt"] + 4.3
    depth_df["soil_density"] = 2685.0
    depth_df["resid_moist"] = 0.0

    depth_df = depth_df.drop(["sand", "clay", "texture", "porosity", "depth"], axis=1)
    depth_df = depth_df.rename({"thickness": "depth"}, axis=1)
    depth_ds = depth_df.set_index(["nlayer", "lat", "lon"]).to_xarray()
    depth_ds["layer"] = depth_ds["nlayer"]

    #
    # Add all variables with dimensions (lat, lon)
    #

    lat_lon = depth_df.reset_index().loc[:, ["lat", "lon"]].drop_duplicates()
    non_depth_df = lat_lon.set_index(["lat", "lon"])
    non_depth_df["infilt"] = 0.2
    non_depth_df["Ds"] = 0.001
    non_depth_df["Ws"] = 0.9
    non_depth_df["off_gmt"] = 0.0
    non_depth_df["rough"] = 0.001
    non_depth_df["snow_rough"] = 0.0005
    non_depth_df["fs_active"] = 0.0
    non_depth_df["c"] = 2.0
    non_depth_df["dp"] = 4.0
    non_depth_df = non_depth_df.rename({"lat": "lats", "lon": "lons"}, axis=1)

    non_depth_ds = non_depth_df.to_xarray()
    non_depth_ds["Dsmax"] = depth_ds["Ksat"].mean(dim="nlayer") * salient_ds["slope"] / 100
    return xr.merge(
        [salient_ds[["elev", "annual_prec", "avg_T"]].astype("float64"), non_depth_ds, depth_ds]
    )


def _build_vic_veg(salient_ds: xr.Dataset, start: str, end: str) -> xr.Dataset:
    """Generate VIC vegetation parameters.

    Args:
        salient_ds (xr.Dataset): Salient variables generated by _get_salient_params().
        start (str): The start date of the time series.
        end (str): The end date of the time series.

    Returns:
        xr.Dataset of VIC vegetation parameters.
    """
    # Use the most recent IGBP land cover data in the time range.
    years = np.arange(np.datetime64(start, "Y"), np.datetime64(end, "Y") + np.timedelta64(1, "Y"))
    for year in reversed(years):
        lc_ds = salient_ds[["lulc_igbp", "lulc_igbp_per"]].sel(
            time=f"{year}-01-01", method="nearest"
        )
        ds_is_empty = lc_ds["lulc_igbp_per"].isnull().all()
        if not ds_is_empty:
            break

    # Vegetation lookup table
    with resources.files("salientsdk.data").joinpath(_VIC_VEG).open("r") as f:
        veg_lu = pd.read_csv(f)
    veg_lu = veg_lu.astype({"overstory": "float64", "veg_class": "int32"})

    #
    # Vegetation params w/ dimensions (lat, lon)
    #

    # Number of vegetation classes per grid cell (Nveg)
    f_nveg_da = xr.where(lc_ds["lulc_igbp_per"] > 0, 1, 0).sum(dim="igbp_class")
    f_nveg_da = f_nveg_da.where(f_nveg_da != 0)
    f_nveg_da.name = "Nveg"

    #
    # Vegetation params w/ dimensions (veg_class, lat, lon)
    #

    igbp_df = lc_ds["lulc_igbp_per"].to_dataframe().reset_index()
    igbp_df = igbp_df.merge(veg_lu, on="igbp_class", how="left")
    veg_df = igbp_df.loc[
        :, ["veg_class", "lat", "lon", "lulc_igbp_per", "overstory", "RGL", "veg_height"]
    ]

    # Percentage of vegetation class per grid cell (Cv)
    veg_df["Cv"] = (veg_df["lulc_igbp_per"] / 100).astype("float64")

    # Constants
    veg_df["rarc"] = 2.0
    veg_df["rmin"] = 100.0
    veg_df["rad_atten"] = 0.5
    veg_df["wind_atten"] = 0.5
    veg_df["trunk_ratio"] = 0.2

    # Wind height measurement is 2 meters higher than vegetation height
    veg_df["wind_h"] = veg_df["veg_height"] + 2.0

    veg_df = veg_df.set_index(["veg_class", "lat", "lon"]).drop(
        ["lulc_igbp_per", "veg_height"], axis=1
    )
    veg_ds = veg_df.to_xarray()
    veg_ds["veg_descr"] = xr.DataArray(
        data=lc_ds["igbp_class"].values.astype("<U36"), coords={"veg_class": veg_ds["veg_class"]}
    )

    #
    # Vegetation params w/ dimensions (veg_class, root_zone, lat, lon)
    #

    dfs = []
    for root_zone in [1, 2, 3]:
        depth_col = f"root_depth_{root_zone}"
        fract_col = f"root_fract_{root_zone}"
        root_zone_cols = ["veg_class", "lat", "lon", depth_col, fract_col]
        root_zone_df = igbp_df.loc[:, root_zone_cols]
        root_zone_df = root_zone_df.rename(
            {depth_col: "root_depth", fract_col: "root_fract"}, axis=1
        )
        root_zone_df["root_zone"] = np.int32(root_zone)
        root_zone_df = root_zone_df.set_index(["veg_class", "root_zone", "lat", "lon"])
        dfs.append(root_zone_df)
    veg_root_df = pd.concat(dfs)
    veg_root_ds = veg_root_df.to_xarray()

    #
    # Vegetation params w/ dimensions (veg_class, month, lat, lon)
    #

    # Compute monthly averages
    veg_month_ds = salient_ds.sel(time=slice(start, end))[["albedo", "lai_lv", "lai_hv"]]
    if veg_month_ds.time.size < 12:
        end_dt = dt.datetime.fromisoformat(end)
        start_dt = dt.datetime(end_dt.year - 1, end_dt.month + 1, 1)
        veg_month_ds = salient_ds.sel(time=slice(start_dt, end_dt))[["albedo", "lai_lv", "lai_hv"]]
    veg_month_ds = veg_month_ds.groupby("time.month").mean()
    veg_month_ds["month"] = veg_month_ds["month"].astype("int32")
    veg_month_ds = xr.merge(
        [
            lc_ds["lulc_igbp"].drop_vars("time").expand_dims(month=veg_month_ds["month"]),
            veg_month_ds,
        ]
    )

    # Convert to dataframe and compute variables
    veg_month_df = veg_month_ds.to_dataframe().reset_index()
    veg_month_df = veg_month_df.merge(
        veg_lu.loc[:, ["veg_class", "igbp_value", "overstory"]],
        left_on="lulc_igbp",
        right_on="igbp_value",
        how="left",
    )
    veg_month_df["LAI"] = veg_month_df.apply(
        lambda x: x["lai_hv"] if x["overstory"] == 1 else x["lai_lv"], axis=1
    )
    veg_month_df["mask"] = ~veg_month_df["albedo"].isnull()

    # Average albedo and LAI by month for each vegetation class
    veg_month_agg_df = (
        veg_month_df.loc[:, ["veg_class", "month", "albedo", "LAI"]]
        .groupby(["veg_class", "month"])
        .mean()
        .reset_index()
    )

    # Manually set LAI for Permanent Snow and Ice
    veg_month_agg_df.loc[veg_month_agg_df["veg_class"] == 15, "LAI"] = 0

    # Manually set albedo and LAI for Open Water
    veg_month_agg_df.loc[veg_month_agg_df["veg_class"] == 17, "albedo"] = 0.5
    veg_month_agg_df.loc[veg_month_agg_df["veg_class"] == 17, "LAI"] = 0

    # Add veg_class level to index and merge averages by month with original grid
    veg_month_df = (
        pd.concat(
            {
                veg_class: veg_month_df.loc[:, ["month", "lat", "lon", "mask"]]
                for veg_class in veg_ds["veg_class"].values
            },
            names=["veg_class"],
        )
        .reset_index()
        .drop("level_1", axis=1)
    )
    veg_month_df = veg_month_df.merge(veg_month_agg_df, on=["veg_class", "month"], how="left")

    # Necessary if all vegetation classes are not present in dataset
    veg_month_df = veg_month_df.fillna(0)

    # Add veg_rough and displacement
    veg_month_df = veg_month_df.merge(
        veg_lu.loc[:, ["veg_class", "veg_height"]], on="veg_class", how="left"
    )
    veg_month_df["veg_rough"] = veg_month_df["veg_height"] * 0.123
    veg_month_df["displacement"] = veg_month_df["veg_height"] * 0.67
    veg_month_df = veg_month_df.drop("veg_height", axis=1)

    # Apply mask to variables
    for data_var in ["LAI", "albedo", "veg_rough", "displacement"]:
        veg_month_df[data_var] = veg_month_df[data_var].where(veg_month_df["mask"], np.nan)

    # Finalize
    veg_month_df = veg_month_df.drop("mask", axis=1)
    veg_month_ds = veg_month_df.set_index(["veg_class", "month", "lat", "lon"]).to_xarray()

    #
    # Build final dataset
    #

    return xr.merge([f_nveg_da, veg_ds, veg_root_ds, veg_month_ds]).drop_vars("time")


def _build_vic_global_params(start: str, end: str, n_soil_layers: int, out_paths: dict):
    """Write VIC global parameters to text file.

    Args:
        start (str): The start date of the time series.
        end (str): The end date of the time series.
        n_soil_layers (int): The number of soil layers.
        out_paths (dict): Dictionary of paths to VIC inputs and outputs generated by _init_vic_destinations().

    Returns:
        None
    """
    _LOGGER.info(f"Generating VIC global parameters text file...")
    start_date = dt.datetime.fromisoformat(start).date()
    end_date = dt.datetime.fromisoformat(end).date()

    params = f"""
        MODEL_STEPS_PER_DAY     4
        SNOW_STEPS_PER_DAY      4
        RUNOFF_STEPS_PER_DAY    4

        STARTYEAR   {start_date.year}
        STARTMONTH  {start_date.month}
        STARTDAY    {start_date.day}
        ENDYEAR     {end_date.year}
        ENDMONTH    {end_date.month}
        ENDDAY      {end_date.day}
        CALENDAR    STANDARD

        FULL_ENERGY FALSE
        QUICK_FLUX  TRUE
        FROZEN_SOIL FALSE

        AERO_RESIST_CANSNOW AR_406

        DOMAIN      {out_paths["domain_path"]}
        DOMAIN_TYPE LAT     lat
        DOMAIN_TYPE LON     lon
        DOMAIN_TYPE MASK    mask
        DOMAIN_TYPE AREA    area
        DOMAIN_TYPE FRAC    frac
        DOMAIN_TYPE YDIM    lat
        DOMAIN_TYPE XDIM    lon

        FORCING1    {out_paths["forcings_path"]}
        FORCE_TYPE  AIR_TEMP    temp
        FORCE_TYPE  PREC        prec
        FORCE_TYPE  PRESSURE    air_pressure
        FORCE_TYPE  SWDOWN      shortwave
        FORCE_TYPE  LWDOWN      longwave
        FORCE_TYPE  VP          vapor_pressure
        FORCE_TYPE  WIND        wind
        WIND_H      10.0

        PARAMETERS  {out_paths["params_path"]}
        LAI_SRC     FROM_VEGPARAM
        FCAN_SRC    FROM_DEFAULT
        ALB_SRC     FROM_VEGPARAM
        NODES       {n_soil_layers}
        SNOW_BAND   FALSE

        RESULT_DIR  {out_paths["outputs_dir"]}
        OUTFILE     fluxes
        COMPRESS    FALSE
        OUT_FORMAT  NETCDF4
        AGGFREQ     NDAYS 1
        HISTFREQ    NMONTHS 1
        OUTVAR      OUT_SOIL_MOIST
    """
    f_lines = [line.strip() for line in params.split(os.linesep)]
    with open(out_paths["global_params_path"], "w") as f:
        f.write(os.linesep.join(f_lines))
    _LOGGER.info(f"VIC global parameters complete!")
