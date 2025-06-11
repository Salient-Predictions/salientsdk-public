#!/usr/bin/env python
# Copyright Salient Predictions 2025

"""Tools for wind energy analysis.

These functions do not call the Salient API.  They are built to use the
outputs in the form returned by the API.  The functions are `xarray`-native
to enable geospatial analysis at scale.

In most cases, you will only need to use `calc_wind_power_all`, which performs
all necessary shearing, density correction, and power curve interpolation.  The
other functions are provided for applications that require greater control
or customization.
"""

from importlib import resources

import numpy as np
import pandas as pd
import xarray as xr


def calc_wind_power_all(
    met: xr.Dataset,
    wspd: str | xr.DataArray = "wspd",
    wspd100: str | xr.DataArray = "wspd100",
    temp: str | xr.DataArray = "temp",
    turbine_model: str | xr.DataArray = "turbine_model",
    hub_height: str | xr.DataArray = "hub_height",
    elevation: str | xr.DataArray = "elevation",
) -> xr.Dataset:
    """Calculate wind power from meteorological data.

    This is a convenience function that shears wind to hub height, density-corrects it,
    then calculates wind power.  It operates on a single dataset as a source of
    the necessary data.

    The arguments `wspd`, `wspd100`, `temp`, `turbine_model`, `hub_height`, and `elevation`
    can be either strings or `xr.DataArray` objects. If they are strings, they are assumed
    to be keys in the `met` dataset.

    Args:
        met: Meteorological data.
        wspd: Wind speed at hub height.
        wspd100: Wind speed at 100m.
        temp: Temperature
        turbine_model: Turbine model.
        hub_height: Hub height (m).
        elevation: Elevation above sea level (m).

    Returns:
        xr.Dataset: with fields
            `wspdhh` for density-corrected wind speed at hub height
            `power` for generated wind power in MW.
    """

    def _extract_field(field) -> xr.DataArray:
        return met[field] if isinstance(field, str) and field in met else field

    wspd = _extract_field(wspd)
    wspd100 = _extract_field(wspd100)
    temp = _extract_field(temp)
    turbine_model = _extract_field(turbine_model)
    hub_height = _extract_field(hub_height)
    elevation = _extract_field(elevation)

    wspdhh = shear_wind(wspd100, wspd, hub_height)
    wspdhh = correct_wind_density(wspdhh, temp=temp, elev=elevation + hub_height)
    power = calc_wind_power(wspdhh, turbine_model)

    pwr = xr.Dataset({"wspdhh": wspdhh, "power": power})

    # preserve provenance?
    # def _add_data(ds, dat):
    #    ds = ds.assign({dat.name: dat})
    #    return ds
    # pwr = _add_data(pwr, turbine_model)
    # pwr = _add_data(pwr, hub_height)
    # pwr = _add_data(pwr, elevation)

    return pwr


def calc_wind_power(
    wind_speed: xr.DataArray,
    turbine_model: str | xr.DataArray = "default",
    keep_dims: set | None = None,
) -> xr.DataArray:
    """Calculate power from power curve & density-corrected hub-height wind.

    Args:
        wind_speed: Wind speed at hub height.
        turbine_model: Turbine model to use.
            If a string, call `get_power_curve(turbine_model)`
            If a `DataArray`, iterate over locations to apply a different power curve at each location.
        keep_dims: Set of coordinates to preserve in the output.
            If `None` (default) uses `wind_speed.dims`.

    Returns:
        xr.DataArray: Wind power in MW.
    """
    if keep_dims is None:
        keep_dims = set(wind_speed.dims)

    LOC = "location"
    if (
        isinstance(turbine_model, xr.DataArray)
        and LOC in turbine_model.coords
        and turbine_model[LOC].size > 1
    ):
        # Vectorized form, iterate over location:
        power = [
            calc_wind_power(
                wind_speed=wind_speed.sel(location=loc, drop=False),
                turbine_model=turbine_model.sel(location=loc),
                keep_dims=keep_dims,
            )
            for loc in turbine_model[LOC].values
        ]
        power = xr.concat(power, dim=LOC, combine_attrs="override")
        power = power.transpose(*wind_speed.dims)

        return power

    power_curve = get_power_curve(turbine_model=turbine_model)
    power = power_curve.interp(wind_speed=wind_speed, method="linear")
    power.name = "power"
    power.attrs["long_name"] = "Wind Power"
    power.attrs["standard_name"] = "power"
    power.attrs["units"] = "MW"

    # Make sure that interp didn't add unwanted extra coordinates and data:
    if "turbine_model" in power.dims:
        power = power.squeeze("turbine_model", drop=True)
    power = power.drop_vars(set(power.coords) - keep_dims)
    power = power.transpose(*wind_speed.dims)

    return power


def celsius_to_kelvin(temp: float | xr.DataArray) -> float | xr.DataArray:
    """Convert temperature from Celsius to Kelvin.

    Args:
        temp: Temperature in Celsius.
            If a `DataArray` with `units` of `degK`, will skip celsius-to-kelvin conversion

    Returns:
        float | xr.DataArray: Temperature in Kelvin, same data type as input.
    """
    C2K = 273.15
    MIN_TEMP = -100 + C2K
    UNITS_K = "degK"

    if isinstance(temp, xr.DataArray):
        if not ("units" in temp.attrs and temp.attrs["units"] == UNITS_K):
            temp += C2K
            temp.attrs["units"] = UNITS_K
        mintemp = temp.min(skipna=True).item()
    else:
        temp = temp + C2K
        mintemp = temp

    assert mintemp >= MIN_TEMP, f"Unreasonable temperature: {mintemp} < {MIN_TEMP} K"

    return temp


def calc_air_density(
    temp: xr.DataArray | float = 13.0,
    elev: xr.DataArray | float = 0.0,
) -> xr.DataArray:
    """Calculate air density.

    Args:
        temp: Temperature (degrees C)
        elev: Elevation above sea level (m).
            Corrects to sea level by default.

    Returns:
        xr.DataArray: air density (kg/m^3)
    """
    AIR_DENS_SEA = 1.225  # Air density at sea level, typically "g", kg/m^3
    GRAVITY = 9.807  # Acceleration of gravity, m/s^2
    MOLAR_AIR = 0.029  # Molar mass of air, typically "M", kg/mol
    GAS_CONSTANT = 8.314  # Typically "R", J/(mol·K)

    temp = celsius_to_kelvin(temp)

    # rho = rho_0 * exp(-g * M * h / (R * T))
    air_density = AIR_DENS_SEA * np.exp(-GRAVITY * MOLAR_AIR * elev / (GAS_CONSTANT * temp))
    if isinstance(temp, xr.DataArray):
        air_density = air_density.transpose(*temp.dims)
    if isinstance(air_density, xr.DataArray):
        air_density.name = "air_density"
        air_density.attrs["long_name"] = "Air Density"
        air_density.attrs["short_name"] = "air_density"
        air_density.attrs["units"] = "kg mm**-3"

    return air_density


def correct_wind_density(
    wspd: xr.DataArray,
    dens: xr.DataArray | float = 1.225,
    temp: xr.DataArray | float = 286.0,
    elev: xr.DataArray | float = 0.0,
) -> xr.DataArray:
    """Correct wind speed for air density.

    Values `dens`, `temp`, and `elev` can be scalar numbers or `DataArray`s with
    coordinates compatible with `wspd`.

    Args:
        wspd: Wind speed at hub height.
        dens: Target air density to correct to (kg/m^3).
            Corrects to sea level air density by default.
        temp: Temperature (degrees C)
        elev: Elevation above sea level (m).
            Corrects to sea level by default.

    Returns:
        xr.DataArray: Wind speed corrected for air density.
    """
    dens_src = calc_air_density(temp, elev)
    wspd_dc = wspd * (dens / dens_src) ** (1 / 3)
    wspd_dc.attrs = wspd.attrs
    wspd_dc.attrs["density_corrected"] = True
    return wspd_dc


def shear_wind(
    wnd_hi: xr.DataArray,
    wnd_lo: xr.DataArray,
    hgt_hh: xr.DataArray,
    hgt_hi: float = 100.0,
    hgt_lo: float = 10.0,
) -> xr.DataArray:
    """Shear wind speed to hub height.

    Uses the Power Law to shear wind speed from two heights to a hub height.
    Will interpolate or extrapolate, as needed.

    Args:
        wnd_hi: Wind speed at height `hgt_hi`.
        wnd_lo: Wind speed at height `hgt_lo`.
        hgt_hh: Hub height above ground level
        hgt_hi: Height above ground level of `wnd_hi`.
        hgt_lo: Height above ground level of `wnd_lo`.

    Returns:
        xr.DataArray: Wind speed at hub height, same dimensions as `wnd_hi`.
    """
    SHEAR_MIN = 0.05  # very unstable atmospheric conditions
    SHEAR_MAX = 0.60  # unusual but plausibly stable conditions
    EPS = 1e-10  # protect vs divide-by-zero and log(0)

    assert hgt_lo > 0, f"hgt_lo {hgt_lo} must be positive"
    assert hgt_hi > hgt_lo, f"hgt_hi {hgt_hi} must be greater than hgt_lo {hgt_lo}"

    wnd_hi = wnd_hi.clip(min=EPS)
    wnd_lo = wnd_lo.clip(min=EPS)
    shear = np.log(wnd_hi / wnd_lo) / np.log(hgt_hi / hgt_lo)
    shear = shear.clip(min=SHEAR_MIN, max=SHEAR_MAX)

    wnd_hh = wnd_hi * (hgt_hh / hgt_hi) ** shear
    wnd_hh = wnd_hh.clip(min=0.0)

    wnd_hh.name = "wspdhh"
    wnd_hh.attrs["long_name"] = "Wind Speed at hub height"
    wnd_hh.attrs["short_name"] = "wspdhh"
    wnd_hh.attrs["units"] = "m s**-1"

    return wnd_hh


def get_power_curve(
    turbine_model: None | str | list[str] | xr.DataArray = "default",
) -> xr.DataArray:
    """Get a power curve for a specific turbine model.

    Args:
        turbine_model: Turbine model to select.
            If None, return all available models.
            If a string, return the model with that name.
                If string is `default`, chooses a popular turbine model.
            If a list of strings or a string `DataArray`, will load all turbines in the list.

    Return:
        xr.DataArray: Power curve for the selected turbine model(s).
            Coordinate `wind_speed` is the wind speed at hub height.
    """
    if isinstance(turbine_model, xr.DataArray):
        if turbine_model.ndim == 0:
            turbine_model = str(turbine_model.item())
        else:
            turbine_model = list(turbine_model.astype(str).values)

    WSPD = "wind_speed"
    if turbine_model is None:
        usecols = None  # just read everything
    elif isinstance(turbine_model, str):
        usecols = [WSPD, turbine_model]
    elif isinstance(turbine_model, (list, tuple)):
        usecols = [WSPD] + turbine_model
    else:
        raise ValueError("turbine_model must be a str, [str], DataArray, or None")

    if turbine_model is not None:
        DEFAULT_TURBINE = "SWT-2.3-93"
        usecols = [DEFAULT_TURBINE if model == "default" else model for model in usecols]

    with resources.files("salientsdk.data").joinpath("power_curves.csv").open("r") as f:
        power_table = pd.read_csv(f, index_col=WSPD, usecols=usecols)

    power_curve = xr.DataArray(
        power_table,
        dims=["wind_speed", "turbine_model"],
        name="power_curve",
        attrs={"units": "MW", "long_name": "Power", "standard_name": "power_curve"},
    )
    power_curve["wind_speed"].attrs["units"] = "m s**-1"
    power_curve["wind_speed"].attrs["long_name"] = "Wind Speed at hub height"
    power_curve["wind_speed"].attrs["short_name"] = "wspdhh"

    # power_curve["turbine_model"].attrs["units"] = ""
    power_curve["turbine_model"].attrs["long_name"] = "Turbine Model"
    power_curve["turbine_model"].attrs["short_name"] = "turbine_model"

    return power_curve
