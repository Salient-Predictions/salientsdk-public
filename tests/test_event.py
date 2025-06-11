#!/usr/bin/env python
# Copyright Salient Predictions 2025

"""Tests for event.py.

Usage:
```
python -s -m pytest tests/test_event.py
```

"""

import numpy as np
import pandas as pd
import xarray as xr
from pandas.io.formats.style import Styler

import salientsdk as sk


def test_classify_event():
    """Test sigmoid/binary classifier."""
    obs, gem, ref = mock_event_data()
    target = 0.15

    dims = list(gem.extreme_pct.dims)
    cls_bin = sk.event.classify_event(gem.extreme_pct, target, width=0.0)
    cls_sig = sk.event.classify_event(gem.extreme_pct, target, width=0.5)
    cls_avg = sk.event.classify_event(gem.extreme_pct, target, width=0.5, dim=dims)

    assert len(cls_bin.lead) == len(gem.lead)
    assert len(cls_sig.lead) == len(gem.lead)
    assert cls_avg.size == 1


def test_calibrate_event():
    """Test calibration of extreme event probabilities using logistic regression."""
    # Get test data without pre-computed extremes
    obs, gem, ref = mock_event_data(add_extreme=True, seed=42)

    # Since mock_event_data doesn't have an ensemble dimension, use dim=None
    cal_probs = sk.event.calibrate_event(
        observed=obs.extreme,
        forecast=gem.extreme_pct,
        groupby="lead",
    )

    # Basic checks
    assert isinstance(cal_probs, xr.DataArray)
    assert cal_probs.dims == gem.extreme_pct.dims
    assert cal_probs.shape == gem.extreme_pct.shape
    assert np.all((cal_probs >= 0) & (cal_probs <= 1))

    # Test model performance
    # Create binary forecasts at probability threshold 0.5
    cal_extreme = cal_probs >= 0.5

    # Calculate F-scores for calibrated vs raw probabilities at 0.5 threshold
    raw_f = sk.event.calc_f_score(obs.extreme, gem.extreme_pct >= 0.5).f_score.item()
    cal_f = sk.event.calc_f_score(obs.extreme, cal_extreme).f_score.item()

    # Calibration should generally improve the F-score
    assert cal_f >= raw_f * 0.8  # Allow some flexibility as improvement depends on data


def test_optimize_threshold():
    """Test optimization of threshold for extreme event detection."""
    obs, gem, ref = mock_event_data(add_extreme=False)

    objective = "payoff"

    payoff1 = {"pp": 5.0, "np": -1.0}
    gem["threshold1"] = sk.event.optimize_threshold(
        observed=obs.extreme,
        forecast=gem.extreme_pct,
        groupby="lead",
        payoff=payoff1,
        objective=objective,
    )

    assert isinstance(gem.threshold1, xr.DataArray)
    assert "lead" in gem.threshold1.dims
    assert gem.threshold1.shape == (100,)  # One threshold per lead
    assert np.all(
        (gem.threshold1 >= 0) & (gem.threshold1 <= 1)
    )  # Thresholds should be between 0 and 1

    # Test optimization with custom payoff that punishes false negatives more
    payoff2 = {"pp": 5.0, "np": -3.0}
    gem["threshold2"] = sk.event.optimize_threshold(
        observed=obs.extreme,
        forecast=gem.extreme_pct,
        groupby="lead",
        payoff=payoff2,
        objective=objective,
    )

    # The higher cost for false negatives should generally lead to lower thresholds
    # (since we want to catch more events even at the expense of false positives)
    assert np.all(gem.threshold2 >= gem.threshold1)

    # Test applying thresholds and compare performance
    # Extract threshold for each lead time
    with xr.set_options(keep_attrs=True):
        gem["extreme1"] = gem.extreme_pct >= gem.threshold1
        gem["extreme2"] = gem.extreme_pct >= gem.threshold2

    # The optimization with a harsher pentalty for false positives should be more precise:
    gem_score1 = sk.event.calc_f_score(obs.extreme, gem.extreme1, payoff=payoff1)
    gem_score2 = sk.event.calc_f_score(obs.extreme, gem.extreme2, payoff=payoff2)
    assert gem_score1.precision < gem_score2.precision
    assert gem_score1.fpr > gem_score2.fpr


def test_build_confusion_matrix():
    """Test event detection and analysis functions."""
    obs, gem, ref = mock_event_data()

    # Get model names from the data
    obs_name = obs.extreme.attrs["model_name"]
    fcst_name = gem.extreme.attrs["model_name"]
    ref_name = ref.extreme.attrs["model_name"]

    # 1. Test 4x2 confusion matrix (with reference)
    cm_4x2 = sk.event.build_confusion_matrix(
        observed=obs.extreme,
        forecast=gem.extreme,
        reference=ref.extreme,
    )

    # Basic sanity checks for 4x2
    assert len(cm_4x2.dims) == 0  # Should have no dimensions
    assert np.allclose(
        (
            cm_4x2.nnn
            + cm_4x2.nnp
            + cm_4x2.npn
            + cm_4x2.npp
            + cm_4x2.pnn
            + cm_4x2.pnp
            + cm_4x2.ppn
            + cm_4x2.ppp
        ).values,
        1.0,
    )

    # Check model names are preserved in attributes
    assert cm_4x2.attrs["observed_model_name"] == obs_name
    assert cm_4x2.attrs["forecast_model_name"] == fcst_name
    assert cm_4x2.attrs["reference_model_name"] == ref_name

    # Check asymmetric cases include model names
    assert fcst_name in cm_4x2.npn.attrs["long_name"]
    assert ref_name in cm_4x2.npn.attrs["long_name"]
    assert fcst_name in cm_4x2.pnp.attrs["long_name"]
    assert ref_name in cm_4x2.pnp.attrs["long_name"]

    # 2. Test 2x2 confusion matrix (no reference)
    cm_2x2 = sk.event.build_confusion_matrix(
        observed=obs.extreme,
        forecast=gem.extreme,
    )

    # Basic sanity checks for 2x2
    assert len(cm_2x2.dims) == 0
    assert np.allclose((cm_2x2.nn + cm_2x2.np + cm_2x2.pn + cm_2x2.pp).values, 1.0)

    # Check 2x2 matrix also preserves names
    assert cm_2x2.attrs["observed_model_name"] == obs_name
    assert cm_2x2.attrs["forecast_model_name"] == fcst_name

    # 3. Test 4x2 matrix, groupby lead time
    cm_by_lead = sk.event.build_confusion_matrix(
        observed=obs.extreme,
        forecast=gem.extreme,
        groupby="lead",
    )

    # Checks for lead-grouped results
    assert len(cm_by_lead.dims) == 1
    assert "lead" in cm_by_lead.dims
    assert cm_by_lead.lead.size == 100
    # Each lead time should sum to 1.0
    assert np.allclose((cm_by_lead.nn + cm_by_lead.np + cm_by_lead.pn + cm_by_lead.pp), 1.0)


def test_calc_f_score():
    """Test F-score calculation with different grouping options."""
    obs, gem, ref = mock_event_data()

    # test paired f-score
    xxx = sk.event.calc_f_score(obs.extreme, gem.extreme, ref.extreme)
    print(xxx.payoff)
    print(xxx.f_score)

    # Test aggregated F-scores (no groupby)
    f_scores = sk.event.calc_f_score(
        observed=obs.extreme,
        forecast=gem.extreme,
        beta=1.0,  # F1 score
    )

    # Basic sanity checks for aggregated scores
    assert len(f_scores.dims) == 0
    for metric in ["f_score", "precision", "recall", "fpr"]:
        assert metric in f_scores
        assert np.all((f_scores[metric] >= 0) & (f_scores[metric] <= 1))

    # Test F-scores grouped by lead time
    f_scores_by_lead = sk.event.calc_f_score(
        observed=obs.extreme,
        forecast=gem.extreme,
        groupby="lead",
        beta=1.0,
    )

    # Checks for lead-grouped results
    assert len(f_scores_by_lead.dims) == 1
    assert "lead" in f_scores_by_lead.dims
    assert f_scores_by_lead.lead.size == 100

    # Check all metrics are present and in valid range
    for metric in ["f_score", "precision", "recall", "fpr"]:
        assert metric in f_scores_by_lead
        assert np.all((f_scores_by_lead[metric] >= 0) & (f_scores_by_lead[metric] <= 1))

    # Test with different beta value
    f2_scores = sk.event.calc_f_score(
        observed=obs.extreme,
        forecast=gem.extreme,
        beta=2.0,  # F2 score weights recall higher
    )

    # F2 score should be lower than recall but higher than precision
    # when recall > precision (typical for our test data setup)
    assert np.all(
        (f2_scores.f_score >= f2_scores.precision) & (f2_scores.f_score <= f2_scores.recall)
    )


def test_style_confusion_matrix(verbose: bool = False):
    """Test the styling of confusion matrices.

    Args:
        verbose: If True, print styled matrices to console
    """
    # Get test data using existing mock function
    obs, gem, ref = mock_event_data()

    gem_name = gem.extreme.attrs["model_name"]
    ref_name = ref.extreme.attrs["model_name"]

    # Test 2x2 matrix
    styled_2x2 = sk.event.style_confusion_matrix(obs.extreme, gem.extreme, payoff=None)
    df_2x2 = styled_2x2.data

    # Basic structure tests for 2x2
    assert isinstance(styled_2x2, Styler)
    assert df_2x2.shape == (3, 3)  # 3 rows (Extreme, Normal, Total) x 3 cols
    assert f"{gem_name} F-score:" in styled_2x2.caption

    # Test 4x2 matrix
    styled_4x2 = sk.event.style_confusion_matrix(
        obs.extreme, gem.extreme, ref.extreme, payoff=None
    )
    df_4x2 = styled_4x2.data

    # Basic structure tests for 4x2
    assert isinstance(styled_4x2, Styler)
    assert df_4x2.shape == (5, 3)  # 5 rows x 3 cols
    assert f"{gem_name} F-score:" in styled_4x2.caption
    assert f"{ref_name} F-score:" in styled_4x2.caption

    if verbose:
        # Print styled matrices with ANSI color codes
        print("\n2x2 Confusion Matrix:")
        print("--------------------")
        print(styled_2x2.data)

        print("\n4x2 Confusion Matrix:")
        print("--------------------")
        print(styled_4x2.data)


def mock_event_data(
    add_extreme: bool = True, seed: int = 42
) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    """Create synthetic datasets for testing event detection and analysis.

    Args:
        add_extreme: If True, add boolean extreme events using default thresholds
            gem: extreme_pct >= 0.15
            ref: extreme_pct >= 0.20
        seed: Random seed for reproducibility

    Returns:
        tuple of (obs, gem, ref) datasets where:
        - obs: observed data with 10% true event rate
        - gem: "good" forecast with well-calibrated ~10% event rate
        - ref: "bad" forecast with overconfident ~20% event rate
    """
    # Set random seed for reproducibility
    rng = np.random.default_rng(seed)

    n_lead = 100  # Smaller number of leads for faster testing
    n_dates = 100  # Multiple forecast dates
    lead = np.arange(1, n_lead + 1)
    forecast_dates = pd.date_range("2021-01-01", periods=n_dates)

    # Create observed events with 10% event rate
    obs_extreme = rng.random(size=(n_dates, n_lead)) < 0.1
    # We want one of our leads to have no hits, to make sure that downstream
    # processes are robust to sparse environments:
    obs_extreme[:, 42] = False

    # Create "good" forecast probabilities with skill
    gem_extreme_pct = np.zeros((n_dates, n_lead))
    gem_extreme_pct[obs_extreme] = rng.uniform(0.0, 0.5, size=np.sum(obs_extreme))
    gem_extreme_pct[~obs_extreme] = rng.uniform(0.0, 0.2, size=np.sum(~obs_extreme))

    # Create "bad" forecast probabilities with less skill
    ref_extreme_pct = np.zeros((n_dates, n_lead))
    ref_extreme_pct[obs_extreme] = rng.uniform(0, 0.4, size=np.sum(obs_extreme))
    ref_extreme_pct[~obs_extreme] = rng.uniform(0, 0.3, size=np.sum(~obs_extreme))

    # Create datasets
    obs = xr.Dataset(
        {
            "extreme": xr.DataArray(
                obs_extreme,
                dims=["forecast_date", "lead"],
                coords={"forecast_date": forecast_dates, "lead": lead},
                attrs={"long_name": "Observed Extreme Event", "model_name": "era5", "target": 0.1},
            )
        }
    )

    gem = xr.Dataset(
        {
            "extreme_pct": xr.DataArray(
                gem_extreme_pct,
                dims=["forecast_date", "lead"],
                coords={"forecast_date": forecast_dates, "lead": lead},
                attrs={"long_name": "Probability of Extreme Event", "model_name": "GEM"},
            ),
        }
    )

    ref = xr.Dataset(
        {
            "extreme_pct": xr.DataArray(
                ref_extreme_pct,
                dims=["forecast_date", "lead"],
                coords={"forecast_date": forecast_dates, "lead": lead},
                attrs={"long_name": "Probability of Extreme Event", "model_name": "BAD"},
            ),
        }
    )

    if add_extreme:
        with xr.set_options(keep_attrs=True):
            gem["extreme"] = gem.extreme_pct >= 0.15
            ref["extreme"] = ref.extreme_pct >= 0.20
            gem.extreme.attrs["long_name"] = "Extreme Event"
            ref.extreme.attrs["long_name"] = "Extreme Event"

    return obs, gem, ref


def main():
    """Run the tests, without the overhead of the testing infrastructure.

    python tests/test_event.py
    """
    # test_build_confusion_matrix()
    test_style_confusion_matrix(verbose=True)
    # test_calc_f_score()
    # test_classify_event()
    # test_calibrate_event()
    # test_optimize_threshold()


if __name__ == "__main__":
    main()
