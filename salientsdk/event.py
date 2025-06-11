#!/usr/bin/env python
# Copyright Salient Predictions 2025

"""Detect and analyze events."""

from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr
from pandas.io.formats.style import Styler
from sklearn.linear_model import LogisticRegression

# Default payoff matrix
PAYOFF = {
    # 2x2 simple matrix payoff coefficients:
    "pp": 100,  #    Benefit of correctly predicting an event
    "fp": -10,  #    Cost of falsely predicting an event
    "tn": 5,  #      Benefit of correctly predicting no event
    "fn": -50,  #    Cost of missing an event
    # 4x2 joint matrix payoff coefficients:
    "npp": -10,
    "ppp": 100,  #   Both agree there will be an extreme.
    "npn": -20,
    "ppn": 300,  #   Forecast extreme, ref normal.  Differentiated alpha
    "nnp": -20,
    "pnp": 300,  #   Forecast normal, ref extreme.  Differentiated alpha.
    "nnn": 0,
    "pnn": 0,  #     Neither extreme, take no action.
    # For setting axis labels
    "units": "$",
}


def _safe_divide(
    numerator: xr.Dataset | xr.DataArray,
    denominator: xr.Dataset | xr.DataArray,
    long_name: str | None = None,
    units: str | None = None,
    **attrs,
) -> xr.Dataset | xr.DataArray:
    """Avoid divide-by-zero errors in division.

    Args:
        numerator: The dividend in the division operation
        denominator: The divisor in the division operation
        long_name: Optional name to add to the result's attributes
        units: Optional units to add to the result's attributes
        **attrs: Additional attributes to add to the result.
            Overridden by long_name and units, if present.

    Returns:
        The result of numerator/denominator, with 0.0 returned for any cases where
        denominator is zero to avoid division errors. Return type matches input type.
    """
    with xr.set_options(keep_attrs=True):
        # the double-where idiom here is necessary to be dask-friendly
        masked_denominator = xr.where(denominator == 0, 1.0, denominator)
        result = numerator / masked_denominator
        result = xr.where(denominator != 0, result, 0.0)

    attrs = attrs.copy()  # prevent changing by reference
    if long_name is not None:
        attrs["long_name"] = long_name
    if units is not None:
        attrs["units"] = units
    if attrs:
        result.attrs.update(attrs)

    return result


def classify_event(
    x: xr.DataArray,
    target: float | xr.DataArray,
    width: float = 0.5,
    dim: str | list[str] | None = None,
    greater_than: bool = True,
) -> xr.DataArray:
    """Calculate an extreme event probability.

    Args:
        x: Input DataArray of classifier values
        target: Target threshold to define an extreme.
        width: Transition width for a sigmoid function.
            If 0, use binary classification.
        dim: Optional dimension to reduce over. If provided, will calculate
            the mean probability across that dimension.
        greater_than: If `True`, classify as x >= target. If False, classify as x <= target.

    Returns:
        DataArray with values between 0 and 1 indicating probability of extreme event
    """
    attrs = x.attrs.copy()
    attrs["greater_than"] = greater_than
    if isinstance(target, float):
        attrs["target"] = target

    if width == 0:
        # xarray comparisons result in False where "x" is nan.  Preserve nans:
        result = xr.where(np.isnan(x), np.nan, (x >= target) if greater_than else (x <= target))
        result.name = "extreme"
        attrs.update({"long_name": "Event", "units": "T/F"})
    else:
        sign = 1 if greater_than else -1
        result = 1 / (1 + np.exp(-sign * (x - target) / width))
        result.name = "extreme_pct"
        attrs.update({"long_name": "Event probability", "units": "%", "width": width})

    if dim is not None:
        result = result.mean(dim=dim)

    result.attrs = attrs
    return result


def calibrate_event(
    observed: xr.DataArray,
    forecast: xr.DataArray,
    greater_than: bool = True,
    groupby: str | list[str] | None = None,
) -> xr.DataArray:
    """Calculate calibrated extreme event probabilities using a calibrated logistic regression.

    Args:
        observed: Boolean DataArray indicating observed extreme events
        forecast: Continuous DataArray with forecast values
        groupby: Optional dimension(s) to group by when fitting logistic models.
            If None, fit a single model to all data.
        greater_than: If `True`, calibrate from upper quantiles.

    Returns:
        DataArray with calibrated probabilities between 0 and 1 indicating
        extreme event likelihood
    """
    result = (
        xr.zeros_like(observed, dtype=float)
        .rename("extreme_pct")
        .assign_attrs(forecast.attrs)
        .assign_attrs({"long_name": "Event Probability", "units": "%"})
    )
    qnt = [0.90, 0.75] if greater_than else [0.10, 0.25]

    if groupby is None:
        dim = [d for d in forecast.dims if d not in observed.dims]
        x = [forecast.quantile(q, dim=dim) for q in qnt] if dim else [forecast]

        # Sklearn expects non-NaN numpy (not xarray)
        x = np.column_stack([xcol.values.ravel() for xcol in x])
        y = observed.values.ravel()
        ok = ~np.isnan(y) & ~np.any(np.isnan(x), axis=1)
        # astype(int) fails on nan values, so we convert to int AFTER filtering:
        x_ok, y_ok = x[ok], y[ok].astype(int)

        y_sum = np.sum(y_ok)
        if y_sum == 0:  # Classifier fails if all observations are F
            probs = np.zeros_like(y)
        elif y_sum == len(y_ok):  # Classifier fails if all observations are T
            probs = np.ones_like(y)
        else:
            mdl = LogisticRegression(fit_intercept=True, solver="lbfgs")
            mdl.fit(x_ok, y_ok)
            probs = mdl.predict_proba(x)[:, 1]

        result.values = probs.reshape(observed.shape)
    else:
        # Calibrate each group individually and reassemble:
        groupby = [groupby] if isinstance(groupby, str) else groupby
        for group_key, group_fcst in forecast.groupby(groupby):
            sel_dict = dict(
                zip(groupby, group_key if isinstance(group_key, tuple) else [group_key])
            )
            group_obs = observed.sel(sel_dict)
            group_result = calibrate_event(group_obs, group_fcst, groupby=None)
            result.loc[sel_dict] = group_result

    return result


def build_confusion_matrix(
    observed: xr.DataArray,
    forecast: xr.DataArray,
    reference: xr.DataArray | None = None,
    groupby: str | list[str] | None = None,
) -> xr.Dataset:
    """Calculate confusion matrix categories.

    Args:
        observed: Boolean `DataArray` indicating observed events
        forecast: Boolean `DataArray` indicating forecasted events
        reference: Optional boolean `DataArray` for paired confusion matrix
        groupby: Dimension(s) to group by/preserve (if `None`, calculate single score)

    Returns:
        Dataset containing observation percentages (0-1)
        If reference is None, simple confusion matrix:
            -`pp`: true positive (positive truth, positive forecast)
            -`nn`: true negative (negative truth, negative forecast)
            -`np`: false positive (negative truth, positive forecast)
            -`pn`: false negative (positive truth, negative forecast)
        If reference is provided, paired confusion matrix:
            -`nnn`: both true negative (negative truth, both negative)
            -`nnp`: reference false positive (negative truth, forecast negative, reference positive)
            -`npn`: forecast false positive (negative truth, forecast positive, reference negative)
            -`npp`: both false positive (negative truth, both positive)
            -`pnn`: both miss (positive truth, both negative)
            -`pnp`: reference true positive, forecast miss (positive truth, forecast negative, reference positive)
            -`ppn`: forecast true positive, reference miss (positive truth, forecast positive, reference negative)
            -`ppp`: both true positive (positive truth, both positive)
    """
    observed = observed.astype(bool)
    forecast = forecast.astype(bool)

    # Check if groupby is in coordinates but not dimensions, and swap if needed
    if groupby is not None:
        groupby_list = [groupby] if isinstance(groupby, str) else groupby
        for g in groupby_list:
            if g in observed.coords and g not in observed.dims:
                # Find the parent dimension of this coordinate
                parent_dim = [d for d in observed.dims if g in observed[d].coords][0]
                observed = observed.swap_dims({parent_dim: g})
                forecast = forecast.swap_dims({parent_dim: g})

    # Get model names from attributes, with fallbacks
    obs_name = getattr(observed, "model_name", "Observed")
    fcst_name = getattr(forecast, "model_name", "Forecast")

    if groupby is None:
        dims = observed.dims
    else:
        groupby = [groupby] if isinstance(groupby, str) else groupby
        dims = [dim for dim in observed.dims if dim not in groupby]

    total = observed.count(dim=dims)

    # Create filtered attributes without conflicting keys
    attrs = forecast.attrs.copy()
    attrs.pop("long_name", None)
    attrs.pop("units", None)
    attrs.pop("type", None)

    if reference is None:
        return xr.Dataset(
            {
                "nn": _safe_divide(
                    (~observed & ~forecast).sum(dim=dims),
                    total,
                    long_name="True Negative",
                    units="%",
                    type="tn",
                    **attrs,
                ),
                "np": _safe_divide(
                    (~observed & forecast).sum(dim=dims),
                    total,
                    long_name="False Positive",
                    units="%",
                    type="fp",
                    **attrs,
                ),
                "pn": _safe_divide(
                    (observed & ~forecast).sum(dim=dims),
                    total,
                    long_name="False Negative",
                    units="%",
                    type="fn",
                    **attrs,
                ),
                "pp": _safe_divide(
                    (observed & forecast).sum(dim=dims),
                    total,
                    long_name="True Positive",
                    units="%",
                    type="tp",
                    **attrs,
                ),
            }
        ).assign_attrs(
            observed_model_name=obs_name,
            forecast_model_name=fcst_name,
        )
    else:
        reference = reference.astype(bool)
        ref_name = getattr(reference, "model_name", "Reference")

        return xr.Dataset(
            {
                # Hit - True Negative
                "nnn": _safe_divide(
                    (~observed & ~forecast & ~reference).sum(dim=dims),
                    total,
                    long_name="Both True Negative",
                    units="%",
                    type="tn",
                    **attrs,
                ),
                # Three types of false positives:
                "nnp": _safe_divide(
                    (~observed & ~forecast & reference).sum(dim=dims),
                    total,
                    long_name=f"{ref_name} False Positive, {fcst_name} True Negative",
                    units="%",
                    type="fptn",  # false positive / true negative combo
                    **attrs,
                ),
                "npn": _safe_divide(
                    (~observed & forecast & ~reference).sum(dim=dims),
                    total,
                    long_name=f"{fcst_name} False Positive, {ref_name} True Negative",
                    units="%",
                    type="fptn",  # false positive / true negative combo
                    **attrs,
                ),
                "npp": _safe_divide(
                    (~observed & forecast & reference).sum(dim=dims),
                    total,
                    long_name="Both False Positive",
                    units="%",
                    type="fp",
                    **attrs,
                ),
                # Three types off False negatives (failure to detect):
                "pnn": _safe_divide(
                    (observed & ~forecast & ~reference).sum(dim=dims),
                    total,
                    long_name="Both False Negative",
                    units="%",
                    type="fn",
                    **attrs,
                ),
                "pnp": _safe_divide(
                    (observed & ~forecast & reference).sum(dim=dims),
                    total,
                    long_name=f"{ref_name} True Positive, {fcst_name} False Negative",
                    units="%",
                    type="fntp",  # false negative / true positive combo
                    **attrs,
                ),
                "ppn": _safe_divide(
                    (observed & forecast & ~reference).sum(dim=dims),
                    total,
                    long_name=f"{fcst_name} True Positive, {ref_name} False Negative",
                    units="%",
                    type="fntp",  # false negative / true positive combo
                    **attrs,
                ),
                # Hit - True Positive
                "ppp": _safe_divide(
                    (observed & forecast & reference).sum(dim=dims),
                    total,
                    long_name="Both True Positive",
                    units="%",
                    type="tp",
                    **attrs,
                ),
            }
        ).assign_attrs(
            observed_model_name=obs_name,
            forecast_model_name=fcst_name,
            reference_model_name=ref_name,
        )


def _calc_payoff(cm: xr.Dataset, payoff: dict | None = PAYOFF, **attrs) -> xr.Dataset | None:
    """Calculate payoff from a confusion matrix using provided coefficients.

    Args:
        cm: Confusion matrix Dataset containing data variables to score
        payoff: Dictionary mapping confusion matrix categories to their values
        **attrs: Additional attributes to add to the result

    Returns:
        Dataset containing:
            - Individual payoff components for each outcome
            - 'total': Sum of all outcome payoffs
        Returns None if `payoff` is None
    """
    if payoff is None:
        return None

    attrs["units"] = payoff.get("units", "$")
    coeffs = xr.Dataset({k: xr.DataArray(payoff.get(k, 0)) for k in cm.data_vars})
    with xr.set_options(keep_attrs=True):
        outcome = cm * coeffs
        for var in outcome:
            outcome[var].attrs["payoff"] = payoff.get(var, 0)
        attrs.update(long_name="Expected Payoff")
        attrs.update({k: payoff.get(k, 0) for k in cm.data_vars})
        outcome["total"] = outcome.to_array(dim="variable").sum("variable").assign_attrs(attrs)

    return outcome


def calc_f_score(
    observed: xr.DataArray,
    forecast: xr.DataArray,
    reference: xr.DataArray | None = None,
    groupby: str | list[str] | None = None,
    beta: float = 1.0,
    payoff: dict = PAYOFF,
) -> xr.Dataset:
    """Calculate F-score components and optional payoff for event analysis.

    Args:
        observed: Boolean DataArray indicating observed extreme events
        forecast: Boolean DataArray indicating forecasted extreme events
        reference: Optional Boolean DataArray for calculating 4x2 paired payouts
        groupby: Dimension(s) to group by/preserve (if None, calculate single score)
        beta: Weight of recall in F-score calculation (default: 1.0 for F1 score)
        payoff: Optional dictionary with keys 'pp', 'nn', 'np', 'pn' specifying
            the value for each confusion matrix element. If provided, adds a
            'payoff' array to the output Dataset.

    Returns:
        Dataset containing:
            - `payoff`: Value calculated using payoff matrix
            - `f_score`: (1 + beta^2) * (precision * recall) / (beta^2 * precision + recall)
            - `precision`: True positives / (True positives + False positives, aka tpr)
            - `recall`: True positives / (True positives + False negatives)
            - `fpr`: False positive rate (False positives / Total negatives)
    """
    if reference is not None:
        cm_paired = build_confusion_matrix(observed, forecast, reference, groupby=groupby)
        po_paired = xr.Dataset({"payoff": _calc_payoff(cm_paired, payoff)["total"]})

        scores = [
            calc_f_score(observed, forecast, None, groupby, beta, payoff),
            calc_f_score(observed, reference, None, groupby, beta, payoff),
            po_paired,
        ]
        return xr.concat(
            scores, dim=pd.Index(["forecast", "reference", "paired"], name="forecast")
        ).assign_coords(
            model_name=(
                "forecast",
                [
                    getattr(forecast, "model_name", "Forecast"),
                    getattr(reference, "model_name", "Reference"),
                    "Paired",
                ],
            ),
            color=(
                "forecast",
                [
                    getattr(forecast, "color", "#1f77b4"),
                    getattr(reference, "color", "#ff7f0e"),
                    "#9467bd",  # "#2ca02c",
                ],
            ),
        )

    beta2 = beta**2
    cm = build_confusion_matrix(observed=observed, forecast=forecast, groupby=groupby)

    attrs = forecast.attrs.copy()
    attrs.pop("long_name", None)
    attrs.pop("units", None)

    payoff_value = _calc_payoff(cm, payoff, **attrs)
    if payoff_value is not None:
        payoff_value = payoff_value["total"]

    attrs["units"] = "%"
    precision = _safe_divide(cm.pp, cm.pp + cm.np, long_name="Precision", **attrs)
    recall = _safe_divide(cm.pp, cm.pp + cm.pn, long_name="Recall", alias="tpr", **attrs)
    fpr = _safe_divide(cm.np, cm.np + cm.nn, long_name="False Positive Rate", **attrs)
    f_score = _safe_divide(
        (1 + beta2) * (precision * recall),
        (beta2 * precision + recall),
        long_name=f"F{beta}-score",
        beta=beta,
        **attrs,
    )

    return xr.Dataset(
        {
            "payoff": payoff_value,
            "f_score": f_score,
            "precision": precision,
            "recall": recall,
            "fpr": fpr,
        }
    )


def style_confusion_matrix(
    observed: xr.DataArray,
    forecast: xr.DataArray,
    reference: xr.DataArray | None = None,
    beta: float = 1,
    payoff: dict | None = None,
    ndigits: int = 2,
) -> Styler:
    """Create a styled DataFrame showing confusion matrix results.

    Args:
        observed: Boolean DataArray indicating observed events
        forecast: Boolean DataArray indicating forecasted events
        reference: Optional boolean DataArray for paired confusion matrix
        beta: Weight of recall in F-score calculation (default: 1.0 for F1 score)
        payoff: Payoff matrix.  If supplied (default none) also displays payoff values.
        ndigits: Number of digits of precision to display in the table

    Returns:
        Styled pandas DataFrame.
    """
    cm = build_confusion_matrix(observed, forecast, reference, groupby=None)
    fcst_name = getattr(forecast, "model_name", "Forecast")
    f_score = calc_f_score(observed, forecast, beta=beta).f_score.item()

    def mk_cell(type: str | list[str]):
        """Build the value to show in a cell of the table."""
        types = [type] if isinstance(type, str) else type
        pct = round(sum((100 * cm[t]).round(ndigits).values for t in types), ndigits)

        # Force grand total to exactly 100 to prevent near-100 rounding errors
        if set(types) == set(cm.data_vars):
            pct = 100.0

        if payoff is None:
            return f"<b>{pct}%</b>"
        else:
            value = round(sum(round(payoff[t] * cm[t].values, ndigits) for t in types), ndigits)
            pyoff = (
                payoff[types[0]]
                if len(types) == 1
                else round(value / (pct / 100) if pct != 0 else 0, ndigits)
            )
            return f"<b>{pct}%</b><br><small><i>*{pyoff} = {value}</i></small>"

    # Create data based on matrix type
    if reference is None:
        data = [
            [mk_cell("np"), mk_cell("pp"), mk_cell(["np", "pp"])],
            [mk_cell("nn"), mk_cell("pn"), mk_cell(["nn", "pn"])],
            [mk_cell(["np", "nn"]), mk_cell(["pp", "pn"]), mk_cell(list(cm.data_vars))],
        ]
        index = ["Extreme", "Normal", "Total"]
        cell_types = [
            [cm.np.attrs["type"], cm.pp.attrs["type"], "total"],
            [cm.nn.attrs["type"], cm.pn.attrs["type"], "total"],
            ["total", "total", "total"],
        ]
        caption = f"{fcst_name} F-score: {f_score:.3f}"
    else:
        ref_name = getattr(reference, "model_name", "Reference")
        ref_f_score = calc_f_score(observed, reference, beta=beta).f_score.item()

        data = [
            [mk_cell("npp"), mk_cell("ppp"), mk_cell(["npp", "ppp"])],
            [mk_cell("npn"), mk_cell("ppn"), mk_cell(["npn", "ppn"])],
            [mk_cell("nnp"), mk_cell("pnp"), mk_cell(["nnp", "pnp"])],
            [mk_cell("nnn"), mk_cell("pnn"), mk_cell(["nnn", "pnn"])],
            [
                mk_cell(["npp", "npn", "nnp", "nnn"]),
                mk_cell(["ppp", "ppn", "pnp", "pnn"]),
                mk_cell(list(cm.data_vars)),
            ],
        ]
        index = ["Both Extreme", f"{fcst_name} Only", f"{ref_name} Only", "Neither", "Total"]
        cell_types = [
            [cm.npp.attrs["type"], cm.ppp.attrs["type"], "total"],
            [cm.npn.attrs["type"], cm.ppn.attrs["type"], "total"],
            [cm.nnp.attrs["type"], cm.pnp.attrs["type"], "total"],
            [cm.nnn.attrs["type"], cm.pnn.attrs["type"], "total"],
            ["total", "total", "total"],
        ]
        caption = f"{fcst_name} F-score: {f_score:.3f}\n{ref_name} F-score: {ref_f_score:.3f}"

    df = pd.DataFrame(data, columns=["Normal", "Extreme", "Total"], index=index)
    df.columns.name = "Observed"
    df.index.name = "Forecast"

    color_map = {
        "fp": "background-color: rgba(255, 0, 0, 0.2)",  # red
        "fn": "background-color: rgba(255, 255, 0, 0.2)",  # yellow
        "tp": "background-color: rgba(0, 255, 0, 0.2)",  # green
        "tn": "background-color: rgba(0, 255, 0, 0.2)",  # green
        "total": "background-color: rgba(128, 128, 128, 0.2)",  # gray
        "fptn": "background: repeating-linear-gradient(45deg, rgba(255, 0, 0, 0.2) 0px, rgba(255, 0, 0, 0.2) 5px, rgba(0, 255, 0, 0.2) 5px, rgba(0, 255, 0, 0.2) 10px)",
        "fntp": "background: repeating-linear-gradient(45deg, rgba(255, 255, 0, 0.2) 0px, rgba(255, 255, 0, 0.2) 5px, rgba(0, 255, 0, 0.2) 5px, rgba(0, 255, 0, 0.2) 10px)",
    }

    styler = df.style.set_properties(**{"white-space": "pre-wrap"})

    def style_cells(x, cell_type: str):
        """Apply styling based on cell type."""
        mask = [[cell == cell_type for cell in row] for row in cell_types]
        return np.where(mask, color_map[cell_type], "")

    for cell_type in color_map:
        styler = styler.apply(style_cells, cell_type=cell_type, axis=None)

    return styler.set_caption(caption)


def search_threshold(
    observed: xr.DataArray,
    forecast: xr.DataArray,
    objective: Literal["f_score", "payoff", "precision", "recall"] = "f_score",
    payoff: dict = PAYOFF,
    beta: float = 1,
    n_grid: int = 201,
) -> xr.Dataset:
    """Search for optimal thresholds by evaluating metrics across threshold values.

    Args:
        observed: Binary array of observed events (0 or 1)
        forecast: Numeric array of forecast probabilities
        objective: Metric to optimize ("f_score", "payoff", "precision", "recall")
        payoff: Dictionary specifying the value for each confusion matrix element
        beta: Weight parameter for F-score calculation
        n_grid: Number of threshold values to test

    Returns:
        Dataset containing all metrics for each threshold and the optimal threshold
    """
    # Create threshold grid adapted to forecast range
    TRIALS = "thresholds"
    thresholds = np.linspace(0, float(forecast.max().item()), n_grid)

    # Cartesian expansion of threshold and forecast's native
    threshold_dim = xr.DataArray(thresholds, dims=TRIALS, coords={TRIALS: thresholds})
    extreme = forecast >= threshold_dim

    # Calculate metrics across thresholds
    scores = calc_f_score(observed, extreme, groupby=TRIALS, payoff=payoff, beta=beta)
    scores = scores.assign_coords({TRIALS: thresholds})

    # Find optimal threshold
    objective_vals = scores[objective]
    optimal_idx = objective_vals.argmax(TRIALS).item()
    optimal_threshold = thresholds[optimal_idx]
    scores["threshold"] = xr.DataArray(optimal_threshold)
    scores["index"] = xr.DataArray(optimal_idx)

    # Handle metadata
    scores.attrs.update(forecast.attrs)
    scores.attrs.update({"objective": objective})
    scores.attrs.pop("long_name", None)
    scores.attrs.pop("units", None)

    return scores


def optimize_threshold(
    observed: xr.DataArray,
    forecast: xr.DataArray,
    objective: Literal["f_score", "payoff", "precision", "recall"] = "f_score",
    groupby: str | list[str] | None = None,
    payoff: dict = PAYOFF,
    beta: float = 1,
    n_grid: int = 201,
) -> xr.DataArray:
    """Find optimal thresholds for forecast probabilities that maximize an objective.

    Args:
        observed: Binary array of observed events (0 or 1)
        forecast: Numeric array of forecast probabilities
        objective: Metric to optimize ("f_score", "payoff", "precision", "recall")
        groupby: Dimension(s) to group by when computing thresholds
        payoff: Dictionary specifying the value for each confusion matrix element
        beta: Weight parameter for F-score calculation
        n_grid: Number of threshold values to test

    Returns:
        DataArray containing optimal thresholds for each group
    """
    if groupby is None:
        # If no groupby, just get the optimal threshold from search_threshold
        result = search_threshold(
            observed, forecast, objective=objective, payoff=payoff, beta=beta, n_grid=n_grid
        )["threshold"]
    else:
        # Convert groupby to list if it's a string
        groupby = [groupby] if isinstance(groupby, str) else groupby

        # Create output DataArray with dimensions from groupby
        coords = {dim: forecast[dim] for dim in groupby}
        result = xr.DataArray(np.nan, coords=coords, dims=groupby, name="threshold")

        # Process each group
        for group_key, group_data in forecast.groupby(groupby):
            sel_dict = dict(
                zip(groupby, group_key if isinstance(group_key, tuple) else [group_key])
            )

            # Get optimal threshold for this group using search_threshold
            result.loc[sel_dict] = search_threshold(
                observed.sel(sel_dict),
                group_data,
                objective=objective,
                beta=beta,
                payoff=payoff,
                n_grid=n_grid,
            )["threshold"].item()

    # Add metadata
    result.attrs.update(forecast.attrs)
    result.attrs.update(
        {
            "long_name": f"Optimal {forecast.attrs.get('long_name', '')} threshold",
            "objective": objective,
        }
    )

    if objective == "f_score":
        result.attrs["beta"] = beta
    elif objective == "payoff":
        result.attrs.update(
            {
                "payoff_pp": payoff.get("pp", 0),
                "payoff_nn": payoff.get("nn", 0),
                "payoff_np": payoff.get("np", 0),
                "payoff_pn": payoff.get("pn", 0),
            }
        )

    return result
