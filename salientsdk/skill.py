#!/usr/bin/env python
# Copyright Salient Predictions 2025

"""Skill and validation."""

from collections.abc import Iterable
from functools import partial

import numpy as np
import pandas as pd
import xarray as xr

from .data_timeseries_api import load_multihistory

# ===================== Skill Functions =============================


def crpss(
    forecast: xr.Dataset,
    reference: xr.Dataset,
) -> xr.Dataset:
    """Continuous Ranked Probability Skill Score.

    CRPSS measures the relative skill improvement of one CRPS forecast over another.
    Positive values can roughly interpreted as percentages, so a CRPSS of 0.10 means a 10% improvement.
    Zero values mean no improvement, and negative values mean the forecast is worse than the reference.

    Args:
        forecast: The forecast data
        reference: The reference baseline data

    Returns:
        xr.DataArray: The CRPSS result
    """
    skill_score = 1 - (forecast / reference)

    skill_score = _set_array_name(skill_score, "crpss")

    return skill_score


def crps(
    observations: xr.Dataset | xr.DataArray | str,
    forecasts: xr.Dataset | xr.DataArray | str | list[str] | pd.DataFrame,
    qnt_dim: str = "quantiles",
) -> xr.Dataset:
    """Calculate Continuous Ranked Probability Score.

    CRPS is used to calculate the skill of a probabilistic forecast.
    CRPS is defined as 2 times the integral of the quantile loss over the distribution.
    Zero CRPS indicates a perfect forecast.

    Args:
        observations: `Dataset` of observed values, aka "truth", or a `nc` filename.
            `observations` should usually have the same coordinates as `forecasts`.
            If `observations` has a `daily` timescale to match
            a `weekly`, `monthly`, or `quarterly` timescale for `forecasts.`
        forecasts: `Dataset` of forecasted values.  May also be:
            - a file reference to a `nc` file
            - a vector of file references to `nc` files
            - a `DataFrame` with a `file_name` column
        qnt_dim: Name of the quantile dimension in the forecast array.
            Defaults to `quantiles`, which is the dimension name returned by
            `forecast_timeseries`.

    Returns:
        xr.Dataset: The CRPS of the `forecasts` quantiles vs. the `observations`.
            If `forecasts` is a vector of multiple `forecast_date_<timescale>`s,
            will also return a data variable `crps_<timescale>_all` denominated by
            `forecast_date` to enable subsetting or trend analysis.
    """
    return _calc_skill(observations, forecasts, skill_func=_crps_core, qnt_dim=qnt_dim)


def _crps_core(
    observations: xr.Dataset | xr.DataArray,
    forecasts: xr.Dataset | xr.DataArray,
    qnt_dim="quantiles",
) -> xr.Dataset | xr.DataArray:
    """The fundamental Continuous Ranked Probability Score calculation.

    This function differs from the user-visible `crps` function in that it does not
    perform any vectorization, coordinate alignment, or validation.  It assumes that
    observations and forecasts have been properly aligned and are dimension-compatible.
    """
    diff = observations - forecasts
    qnt_val = diff[qnt_dim]

    qnt_score = 2 * np.maximum(qnt_val * diff, (qnt_val - 1) * diff)
    skill = qnt_score.integrate(coord=qnt_dim)
    skill.attrs.update(forecasts.attrs)
    if hasattr(forecasts, "data_vars"):  # xr.Dataset only...
        for var_name in forecasts.data_vars:
            skill[var_name].attrs.update(forecasts[var_name].attrs)

    skill = _set_array_name(skill, "crps")

    return skill


def crps_ensemble(
    observations: xr.Dataset | xr.DataArray | str,
    forecasts: xr.Dataset | xr.DataArray | str | list[str] | pd.DataFrame,
    ens_dim: str = "ensemble",
) -> xr.Dataset:
    """Calculate Continuous Ranked Probability Score for ensemble forecasts.

    Args:
        observations: Observed values or path to NetCDF file
        forecasts: Ensemble forecast values or path(s) to NetCDF file(s)
        ens_dim: Name of the ensemble dimension (default: "ensemble")

    Returns:
        xr.Dataset: The ensemble CRPS score
    """
    return _calc_skill(
        observations,
        forecasts,
        skill_func=_crps_ensemble_core,
        ens_dim=ens_dim,
    )


def _crps_ensemble_core(
    observations: xr.Dataset | xr.DataArray,
    forecasts: xr.Dataset | xr.DataArray,
    ens_dim: str = "ensemble",
) -> xr.Dataset | xr.DataArray:
    """Ensemble CRPS. Faster than the properscoring version and doesn't require numba.

    Args:
        observations: the observed data
        forecasts: ensemble forecast data
        ens_dim: name of the ensemble dimension

    Returns:
        crps: ensemble CRPS

    References:
    https://doi.org/10.1198/016214506000001437
    https://doi.org/10.1007/s11004-017-9709-7
    """

    def _crps(observations, forecasts):
        skill = np.abs(forecasts - observations[..., None]).mean(axis=-1)
        forecasts_sorted = np.sort(forecasts, axis=-1)
        n = forecasts.shape[-1]
        r = np.arange(1, n + 1, dtype=forecasts.dtype)
        weights = (2 * r - n - 1) / n**2
        integral_term = np.einsum("...j,j->...", forecasts_sorted, weights)
        crps = skill - integral_term
        return crps

    crps = xr.apply_ufunc(
        _crps,
        observations,
        forecasts,
        input_core_dims=[[], [ens_dim]],
        output_core_dims=[[]],
        dask="parallelized",
        #        output_dtypes=[forecasts.dtype],
    )
    crps.attrs.update(forecasts.attrs)
    if hasattr(forecasts, "data_vars"):  # xr.Dataset only...
        for var_name in forecasts.data_vars:
            crps[var_name].attrs.update(forecasts[var_name].attrs)

    crps = _set_array_name(crps, "crps")

    return crps


def _calc_skill(
    observations: xr.Dataset | xr.DataArray | str | pd.DataFrame,
    forecasts: xr.Dataset | xr.DataArray | str | list[str] | pd.DataFrame,
    skill_func: callable,
    **kwargs,
) -> xr.DataArray:
    """Calculate a skill score.

    Handles vectorization and coordinate alignment for skill functions.  Skill itself
    is calculated by the `skill_func` function.

    Args:
        observations: `DataArray` or `Dataset` of observed ("truth") values , or a `nc` filename.
            `observations` should usually have the same coordinates as `forecasts`.
            If a `pd.DataFrame` will call `load_multihistory` to aggregate multiple variables.
            If `observations` has a `daily` timescale, the system will aggregate match
            a `weekly`, `monthly`, or `quarterly` timescale for `forecasts.`
        forecasts: `Dataset` or `DataArray` of forecasted values.  May also be:
          * a file reference to a `nc` file
          * a vector of file references to `nc` files, which will calculate skill for
            each file and return the average across all forecasts.
          * a DataFrame with a `file_name` column
        skill_func: The skill function to use.
        **kwargs: Additional arguments to pass to `skill_func`.

    Returns:
        xr.DataArray: The skill of the `forecast` values vs. the `observation`s.

    """

    def extract_df_files(obj, col_name="file_name"):
        """Extract file names from a DataFrame, if passed in."""
        if isinstance(obj, pd.DataFrame):
            assert col_name in forecasts.columns, f"DataFrame must have a '{col_name}' column."
            obj = obj[col_name].tolist()
        return obj

    if observations is None or observations is pd.NA:
        return None
    elif isinstance(observations, pd.DataFrame):
        observations = load_multihistory(observations)
    elif isinstance(observations, str):
        observations = xr.load_dataset(observations)
    elif not isinstance(observations, xr.DataArray) and not isinstance(observations, xr.Dataset):
        raise ValueError(
            f"observations {type(observations)} must be Dataset, DataArray, or filename"
        )

    forecasts = extract_df_files(forecasts)
    if forecasts is None or forecasts is pd.NA:
        return None
    elif isinstance(forecasts, str):
        # We want to load_datasaet instead of load_dataarray in order to preserve forecast_period
        forecasts = xr.load_dataset(forecasts, decode_timedelta=True)
    elif isinstance(forecasts, list):
        # This is the most common entry point - iterate over several
        skill = [_calc_skill(observations, fcst, skill_func, **kwargs) for fcst in forecasts]
        skill = [s for s in skill if s is not None]
        if len(skill) == 0:
            return None
        elif len(skill) == 1:
            return skill[0]

        skill_all = _concat_by_forecast_date(skill)
        return _mean_forecast(skill_all)
    elif not isinstance(forecasts, xr.DataArray) and not isinstance(forecasts, xr.Dataset):
        raise ValueError(
            f"forecast {type(forecasts)} must be a Dataset, DataArray, DataFrame[file_name], or filename."
        )

    # At this point, all vectorization and file loading should be done.  Now we need to make sure
    # that the coordinates match.
    if "time" in observations.coords:
        if any(coord.startswith("lead_") for coord in forecasts.coords):
            # floating timescale sub-seasonal/seasonal/long-range/all
            observations = align_daily_obs_to_lead(observations, forecasts)
        elif _is_downscale(forecasts):
            (observations, forecasts) = _align_downscale(observations, forecasts)
        elif "time" in forecasts.coords:
            # calendar-locked weekly/monthly/quarterly
            (observations, forecasts) = _align_time_to_lead(observations, forecasts)

    skill = skill_func(observations, forecasts, **kwargs)

    if "forecast_date" in skill.dims and skill.sizes["forecast_date"] > 1:
        # It is possible that the user has aligned multiple forecast dates into
        # a single dataset, for example by use of bulk downscale.  If so, make the
        # resulting dataset look like that produced when "forecasts" is a list of
        # downscales.  Preserve the original per-date skill with "_all" and calculate
        # a single mean skill score.
        skill = _mean_forecast(skill)

    return skill


# ============== Utility functions ==================================


def _set_array_name(ds: xr.Dataset | xr.DataArray, name: str) -> xr.Dataset | xr.DataArray:
    """Set the name of the data array or dataset variable(s), respecting timescale suffixes.

    For example, transform "vals_weekly" to "crps_weekly" or "anom_monthly" to "mae_monthly"
    """

    def _change_name(
        original_str: str | None,
        name: str,
        other_names: list[str] | None = None,
        delim: str = "_",
    ) -> str:
        """Change the string format based on the presence of a delimiter.

        Args:
            original_str: The original string to be altered.
            name: The new name to replace the value part of the string.
            other_names: The other available names in the dataset.
            delim: the intermediate string to search for

        Returns:
            str: The altered string.
        """
        if original_str is None:
            # DataArray case
            return name
        elif delim in original_str:
            # If the dataset contains multiple timescales like "vals_weekly", replace "vals"
            return f"{name}{delim}{original_str.split(delim, 1)[1]}"
        elif other_names is not None and len(other_names) > 1:
            # If a dataset contains multiple named variables like "temp" and "precip",
            # preserve each of them separately.
            return f"{name}{delim}{original_str}"
        else:
            # If we just have a single variable like "temp" or don't have time delimiters
            # so the name is just "vals":
            return name

    if isinstance(ds, xr.DataArray):
        ds.name = _change_name(ds.name, name)
    elif isinstance(ds, xr.Dataset):
        ds = ds.rename_vars(
            {var_name: _change_name(var_name, name, ds.data_vars) for var_name in ds.data_vars}
        )

    ds.attrs["short_name"] = name
    ds.attrs["long_name"] = (
        ds.attrs["long_name"] + " " if "long_name" in ds.attrs else ""
    ) + name.upper()

    return ds


def _extract_array_name(ds: xr.Dataset | xr.DataArray, search: str | None = None) -> str:
    """Extract the name of the first matching data variable in a dataset."""
    if isinstance(ds, xr.Dataset):
        # Iterate over data variables to find the first match
        for var_name in ds.data_vars:
            if search is None or (search in var_name):
                return var_name
        raise ValueError(f"No data variable found matching search string '{search}'.")
    elif isinstance(ds, xr.DataArray):
        if search is None or (search in ds.name):
            return ds.name
        else:
            raise ValueError(
                f"DataArray name '{ds.name}' does not match search string '{search}'."
            )
    else:
        raise ValueError(f"Must be a Dataset or DataArray, not {type(ds)}")


def align_daily_obs_to_lead(
    observations: str | xr.Dataset | xr.DataArray,
    forecasts: str | xr.Dataset | xr.DataArray,
    timescale: str | Iterable[str] = "all",
) -> xr.Dataset | xr.DataArray:
    """Convert daily observations to match forecasts denominated by a coarse lead time.

    Args:
        observations: Daily observed values with coordinate `time`.  If `str` load NetCDF.
        forecasts: The forecasted values with coordinate `lead_<timescale>` and
            `forecast_period_<timescale>`.  If `str`, load NetCDF.
        timescale: The forecast period of the forecast, corresponding to data variables
            `lead_<timescale>` and `forecast_period_<timescale>`.
            Will typically be `weekly`, `monthly`, or `quarterly`.
            If `all` (default), will work across each of the possible timescales.

    Returns:
        xr.Dataset | xr.DataArray: An aggregated version of `observations` with coordinate(s)
            `lead_<timescale>` instead of `time`.
    """
    if isinstance(observations, str):
        observations = xr.load_dataset(observations)
    if isinstance(forecasts, str):
        forecasts = xr.load_dataset(forecasts)

    if isinstance(timescale, str) and timescale == "all":
        # This will usually be ["weekly","monthly","quarterly"], but let's dynamically
        # locate the leads within the forecast dataset in case it has been manipulated:
        timescale = [
            coord.split("_", 1)[1] for coord in forecasts.coords if coord.startswith("lead_")
        ]
        timescale = timescale[0] if len(timescale) == 1 else timescale

    if isinstance(timescale, Iterable) and not isinstance(timescale, str):
        # If we have a vector of leads, operate over each of them individually and reassemble:
        aligned = xr.merge(
            [align_daily_obs_to_lead(observations, forecasts, ts) for ts in timescale]
        )
        aligned, _ = xr.align(aligned, forecasts, join="inner", copy=False)
        return aligned

    lead_name = f"lead_{timescale}"
    data_name = _extract_array_name(forecasts, timescale)
    period_name = f"forecast_period_{timescale}"
    forecast_name = f"forecast_date_{timescale}"

    lead_vals = forecasts[lead_name].values

    # groupby_bins creates bins that are right-inclusive, so we start the
    # binning one day early:
    first_day = forecasts[period_name].isel(nbnds=0)[0] - np.timedelta64(1, "D")
    bins = np.append(first_day, forecasts[period_name].isel(nbnds=1))

    observations = (
        observations.groupby_bins("time", bins)
        .mean()
        .assign_coords(lead=("time_bins", lead_vals))
        .rename({"lead": lead_name})
        .swap_dims({"time_bins": lead_name})
        .drop_vars("time_bins")
    )

    if isinstance(observations, xr.Dataset):
        obs_name = _extract_array_name(observations)
        observations = observations.rename({obs_name: data_name})
    elif isinstance(observations, xr.DataArray):
        observations.name = data_name

    if forecast_name in forecasts.coords:
        observations = observations.assign_coords({forecast_name: forecasts[forecast_name]})

    return observations


def _find_coord(
    ds: xr.DataArray | xr.Dataset, starts_with: str, strict: bool = True
) -> str | None:
    """Find a coordinate that starts with a given string.

    Args:
        ds: The dataset or dataarray to search
        starts_with: Search for coordinates that begin with this string
        strict: If True, raise an error if no match is found.

    Returns:
        str: The name of the coordinate that starts with `starts_with`
            or `None` if no coordinate was found
    """
    found = next((coord for coord in ds.coords if coord.startswith(starts_with)), None)
    if strict:
        assert found is not None, f"No {starts_with} coordinate found."
    return found


def _is_downscale(forecasts: xr.Dataset) -> bool:
    """Check to see if a dataset came from sk.downscale."""
    # forecast_date = daily downscale
    # time = hourly downscale
    return "ensemble" in forecasts.dims and any(
        dim in forecasts.dims for dim in ["forecast_day", "time"]
    )


def _align_downscale(observations: xr.Dataset, forecasts: xr.Dataset):
    """Prepare downscale and obs so that they have the same coordinates."""
    if "forecast_day" in forecasts:
        # Daily downscale uses forecast_day as its time coordinate.
        # Hourly uses "time".  Let's convert to "time"
        forecasts = forecasts.rename({"forecast_day": "time"})

    # We don't care about any of the climatology or anomaly variables.
    # Only validate in vals space for now.
    drop_suffix = ("_clim", "_anom", "_bias_correction")
    forecasts = forecasts.drop_vars(
        [var for var in forecasts.data_vars if var.endswith(drop_suffix)]
    )
    # We don't need this and it interferes with vectorization:
    if "dayofyear" in forecasts.dims:
        forecasts = forecasts.drop_dims("dayofyear")

    # data_timeseries returns "vals" instead of variable name.
    # Force consistency.
    if "vals" in observations.data_vars:
        fcst_vars = list(forecasts.data_vars)
        if len(fcst_vars) != 1:
            raise ValueError(f"Expected single data variable in forecasts, found: {fcst_vars}")
        observations = observations.rename({"vals": fcst_vars[0]})

    # So we can vectorize multiple forecast_dates, denominate by lead
    # instead of absolute datetime.
    observations, forecasts = xr.align(observations, forecasts, join="inner")
    lead = (forecasts.time - forecasts.forecast_date).data
    forecasts = (
        forecasts.assign_coords(lead=("time", lead)).swap_dims({"time": "lead"}).drop_vars("time")
    )
    observations = (
        observations.assign_coords(lead=("time", lead))
        .swap_dims({"time": "lead"})
        .drop_vars("time")
    )

    return (observations, forecasts)


def _align_time_to_lead(
    observations: xr.DataArray, forecasts: xr.DataArray
) -> tuple[xr.DataArray, xr.DataArray]:
    """Align or aggregate observations to match the granularity of forecasts.

    Args:
        observations (xr.DataArray): The observed values with coordinate `time`.
        forecasts (xr.DataArray): The forecasted values with coordinate `time`.

    Returns:
        tuple: Aligned or aggregated observations and forecasts with updated 'lead' coordinates.
    """
    # Convert time coordinates to pd.DatetimeIndex and infer frequencies

    # obs_time = _ensure_freq(observations.time.values)
    fcst_time = _ensure_freq(forecasts.time.values)

    lead_times = range(1, len(forecasts.time) + 1)
    forecasts = forecasts.assign_coords(lead=("time", lead_times))

    observations = observations.resample(time=fcst_time.freq).mean()

    observations, forecasts = xr.align(observations, forecasts, join="inner")
    observations = observations.assign_coords(lead=("time", forecasts.lead.values))

    forecasts = forecasts.swap_dims({"time": "lead"})
    observations = observations.swap_dims({"time": "lead"})

    return observations, forecasts


def _ensure_freq(time) -> pd.DatetimeIndex:
    """Ensure that a time vector has a 'freq' attribute for resampling."""
    if not isinstance(time, pd.DatetimeIndex):
        time = pd.to_datetime(time)

    if time.freq is not None:
        # Frequency is already set
        return time
    elif len(time) < 2:
        # Not enough data to infer or calculate frequency
        raise ValueError("Not enough data points to determine frequency.")
    elif len(time) == 2:
        # Manually calculate frequency from two dates.
        step = time[1] - time[0]
        days = step.days
        if days == 1:
            freq = "D"
        elif days == 7:
            day_of_week = time[0].day_name()
            freq = f"W-{day_of_week[:3].upper()}"
        elif 28 <= days <= 31:
            freq = "MS"  # ms= momth start
        elif 89 <= days <= 92:
            # Get the month for the first date
            month_of_year = time[0].month
            # Map month to the corresponding quarter start
            if month_of_year in [1, 2, 3]:
                freq = "Q-JAN"
            elif month_of_year in [4, 5, 6]:
                freq = "Q-APR"
            elif month_of_year in [7, 8, 9]:
                freq = "Q-JUL"
            elif month_of_year in [10, 11, 12]:
                freq = "Q-OCT"
        else:
            raise ValueError(f"Unknown time step: {days} days.")
    else:
        # Infer frequency from three or more dates
        freq = pd.infer_freq(time)
        if freq is None:
            raise ValueError("Unable to infer frequency from data.")

    return pd.DatetimeIndex(time, freq=freq)


def _find_timescales(ds: xr.Dataset) -> set[str]:
    """Find and return the set of timescales present in the dataset.

    Args:
        ds: The xarray Dataset to inspect.

    Returns:
        A set of timescales found in the dataset's coordinates.
        Will typically be some subset of `weekly`, `monthly`, and `quarterly`,
        or an empty string if a generic 'lead' coordinate is present.
    """
    # First search for any timescale named "lead_<timescale>", which we use for
    # floating forecasts.
    timescales = {coord.split("_")[1] for coord in ds.coords if coord.startswith("lead_")}

    # calendar-locked forecasts just say "lead" with no timescale, which we indicate
    # with an empty timescale:
    if "lead" in ds.coords:
        timescales.add("")

    return timescales


def _extract_timescale(ds: xr.Dataset, timescale: str) -> xr.Dataset:
    """Extract components of the dataset relevant to the specified timescale.

    In forecast dataset resulting from timescale=all, the dataset will contain coords:
        - lead_[weekly|monthly|quarterly]
        - forecast_period_[weekly|monthly|quarterly]
        - [anom|vals]_[weekly|monthly|quarterly]
        - quantiles, lat, lon, and sometimes location

    Args:
        ds: The xarray Dataset to be filtered.
        timescale: The timescale to retain (e.g., 'weekly', 'monthly', 'quarterly').

    Returns:
        A new xarray Dataset containing only the components relevant to the specified timescale,
        or an empty Dataset if the timescale is not present.
    """
    all_timescales = _find_timescales(ds)

    if timescale not in all_timescales:
        return xr.Dataset()
    elif timescale == "":
        # calendar-locked forecasts don't name themselves by timescale:
        return ds

    bad_timescales = {ts for ts in all_timescales if ts != timescale}
    bad_crd = [crd for crd in ds.coords if any(crd.endswith(f"_{ts}") for ts in bad_timescales)]
    bad_var = [var for var in ds.data_vars if any(var.endswith(f"_{ts}") for ts in bad_timescales)]
    filtered_ds = ds.drop_vars(bad_var, errors="ignore").drop_vars(bad_crd, errors="ignore")

    return filtered_ds


def _concat_by_forecast_date(src: Iterable[str] | Iterable[xr.Dataset]) -> xr.Dataset:
    """Process an iterable of datasets or filenames and merge them by timescale.

    Expects forecast-style Datasets with contain coords:
        - lead_[weekly|monthly|quarterly]
        - forecast_period_[weekly|monthly|quarterly]
        - [anom|vals]_[weekly|monthly|quarterly]
        - quantiles, lat, lon, and sometimes location
    Not all timescales must be present.

    Args:
        src: An iterable of xarray Datasets or filenames to datasets.

    Returns:
        A merged xarray Dataset containing data from all specified timescales.
    """
    timescales = ["", "weekly", "monthly", "quarterly"]
    first_element = next(iter(src), None)
    if isinstance(first_element, str):
        # Assume src is an iterable of filenames
        timescales_ds = [
            xr.open_mfdataset(
                src,
                concat_dim=f"forecast_date_{timescale}" if timescale else "forecast_date",
                preprocess=partial(_extract_timescale, timescale=timescale),
                combine="nested",
            ).load()
            for timescale in timescales
        ]
    elif isinstance(first_element, xr.Dataset):
        # Assume src is an iterable of Datasets
        timescales_ds = [
            xr.concat(
                [_extract_timescale(ds, timescale) for ds in src],
                dim=f"forecast_date_{timescale}" if timescale else "forecast_date",
                combine_attrs="override",
            )
            for timescale in timescales
        ]
    else:
        raise ValueError("The source iterable must contain either filenames or xarray Datasets.")

    return xr.merge(timescales_ds, compat="override")


import xarray as xr


def _mean_forecast(ds: xr.Dataset, dim_prefix: str = "forecast_date") -> xr.Dataset:
    """Compute the mean of forecast data along the specified dimension for each timescale.

    Args:
        ds: The xarray Dataset containing forecast data.
        dim_prefix: The prefix of the dimension along which to compute the mean (default is "forecast_date").

    Returns:
        A new xarray Dataset with the mean values computed for each timescale, preserving
        the original timescale data with an `_all` suffix.
    """
    timescales = _find_timescales(ds)

    ds_avg = xr.merge(
        [
            extracted_ds.mean(
                dim=dim_prefix if timescale == "" else f"{dim_prefix}_{timescale}",
                keep_attrs=True,
            )
            for timescale in timescales
            for extracted_ds in [_extract_timescale(ds, timescale)]
            if extracted_ds.data_vars
        ],
        compat="override",
    )

    ds = ds.rename_vars({var: f"{var}_all" for var in ds.data_vars})

    return xr.merge([ds, ds_avg])
