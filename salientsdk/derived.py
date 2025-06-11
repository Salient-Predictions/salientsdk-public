"""Derived meteorological variable calculations.

This module provides functions for computing derived meteorological variables from
primary forecast variables. It includes temperature-based indices, comfort metrics, and
degree day calculations.

All functions operate on xarray Datasets and return DataArrays with appropriate
metadata and units. The module supports both direct variable access and computed
derivations based on available input data.
"""
import numpy as np
import pint  # noqa: F401
import pint_xarray  # noqa: F401
import xarray as xr


def tmean(ds: xr.Dataset) -> xr.DataArray:
    """Conventional definition of tmean as the mean of tmin and tmax (°C)."""
    return ((ds["tmax"] + ds["tmin"]) / 2).assign_attrs(
        long_name="Average temperature at 2m", units="degC"
    )


def temp(ds: xr.Dataset) -> xr.DataArray:
    """Take direct daily average temp variable if it exists, otherwise compute tmean."""
    if "temp" in ds.data_vars:
        return ds["temp"]
    else:
        return tmean(ds)


def heat_index(ds: xr.Dataset) -> xr.DataArray:
    """Maximum daily heat index computed from tmax and relative humidity.

    This uses the definition used by the National Weather Service and ECMWF, and requires the
    conversion to `degF` in order to perform the calculation. The units are then converted back
    to the source units using the `pint` library.

    Ref:
        Rothfusz L (1990) The heat index “equation” (or, more than you ever wanted to know about
            heat index). National Weather Service Technical Attachment SR 90-23.
            National Weather Service, USA

        https://www.weather.gov/media/epz/wxcalc/heatIndex.pdf
    """

    def _lt80(ds):
        """Calculation when temp is less than 80°F."""
        ds["heat_index"] = 0.5 * (
            ds["tmax"] + 61.0 + ((ds["tmax"] - 68.0) * 1.2) + ds["rh"] * 0.094
        )
        return ds

    def _ge80(ds):
        """Calculation when temp is greater than or equal to 80°F."""
        ds["heat_index"] = (
            -42.379
            + 2.04901523 * ds["tmax"]
            + 10.14333127 * ds["rh"]
            - 0.22475541 * ds["tmax"] * ds["rh"]
            - 0.00683783 * ds["tmax"] ** 2
            - 0.05481717 * ds["rh"] ** 2
            + 0.00122874 * ds["tmax"] ** 2 * ds["rh"]
            + 0.00085282 * ds["tmax"] * ds["rh"] ** 2
            - 0.00000199 * ds["tmax"] ** 2 * ds["rh"] ** 2
        )
        return ds

    ds = ds.copy()
    orig_units = ds["tmax"].attrs.get("units", "degC")
    if orig_units != "degF":
        ds["tmax"] = ds["tmax"].pint.quantify(orig_units).pint.to("degF").pint.dequantify()

    ds["heat_index"] = ds["tmax"] * np.nan
    ds["rh"] = ds["rh"] * 100
    hi_ge80 = _ge80(ds)  # Calculate heat index for tmax >= 80°F
    hi_lt80 = _lt80(ds)  # Calculate heat index for tmax < 80°F

    ds["heat_index"] = ds["heat_index"].where(ds["tmax"] < 80, hi_ge80["heat_index"])
    ds["heat_index"] = ds["heat_index"].where(ds["tmax"] >= 80, hi_lt80["heat_index"])

    out = ds["heat_index"].assign_attrs(units="degF", long_name="Maximum daily heat index")
    out = (
        out.pint.quantify("degF").pint.to(orig_units).pint.dequantify()
    )  # Convert to original units
    out.attrs["units"] = orig_units
    return out


def wind_chill(ds: xr.Dataset) -> xr.DataArray:
    """Minimum daily wind chill (°C) computed from tmin and wind speed."""
    t = ds["tmin"]
    w = wind_speed(ds, h=10)
    wp = w**0.16
    wc = 13.12 + 0.6215 * t - 11.37 * wp + 0.3965 * t * wp
    mask = (t < 10) & (w > 4.8)
    return wc.where(mask, t).assign_attrs(units="degC", long_name="Minimum daily wind chill")


def heating_degree_days(ds: xr.Dataset, base_temp: float = 18) -> xr.DataArray:
    """Daily heating degree days (base 18°C)."""
    return (
        (base_temp - tmean(ds))
        .clip(min=0)
        .assign_attrs(long_name="Heating degree days", units="HDD day**-1 degC")
    )


def cooling_degree_days(ds: xr.Dataset, base_temp: float = 18) -> xr.DataArray:
    """Daily cooling degree days (base 18°C)."""
    return (
        (tmean(ds) - base_temp)
        .clip(min=0)
        .assign_attrs(long_name="Cooling degree days", units="CDD day**-1 degC")
    )


# Include some aliases for convenience
CALLABLE_VARIABLES = dict(
    tmean=tmean,
    temp=temp,
    heat_index=heat_index,
    wind_chill=wind_chill,
    hdd=heating_degree_days,
    cdd=cooling_degree_days,
)


def compute_quantity(
    data: xr.Dataset, name: str, database: dict = CALLABLE_VARIABLES
) -> xr.DataArray:
    """Helper function to compute derived quantities from input dataset.

    Args:
        data: The input data to compute the quantity from. Can either be a Dataset
            with each variable as an array, or a DataArray with a variable dimension.
        name: The name of the quantity to compute.
        database: A dictionary of derived quantity definitions.
    """
    if name in data.data_vars:
        # If the variable already exists in the data, return it directly
        return data[name]
    elif name in database:
        da = database[name](data)
        return da
    else:
        raise ValueError(
            f"Unknown quantity '{name}' to compute. Available quantities: {list(database.keys())}."
        )
