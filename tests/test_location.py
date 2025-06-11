#!/usr/bin/env python
# Copyright Salient Predictions 2025


"""Tests for solar.py.

Usage:
```
python -s -m pytest tests/test_solar.py
```

"""

# import pytest

import numpy as np
import pandas as pd
import xarray as xr

import salientsdk as sk


def sample_regions(
    rgn: xr.DataArray = sk.location._load_regions_mask().mask, n_samples: int = 30
) -> pd.DataFrame:
    """Sample N points from each unique region in the mask.

    Args:
        rgn: xarray DataArray containing region mask
        n_samples: Number of points to sample per region

    Returns:
        DataFrame with lat, lon, and region_ref columns
    """
    samples = []
    regions = np.unique(rgn.values)

    for region in regions:
        if np.isnan(region):
            points = np.where(np.isnan(rgn.values))
            region = int(0)
        else:
            points = np.where(rgn.values == region)
            region = int(region)

        n = min(n_samples, len(points[0]))
        idx = np.random.choice(len(points[0]), size=n, replace=False)
        lats = rgn.lat.values[points[0][idx]]
        lons = rgn.lon.values[points[1][idx]]
        samples.append(
            pd.DataFrame(
                {
                    "lat": lats,
                    "lon": lons,
                    "name": [f"{region}_{i:02d}" for i in range(len(lats))],
                    "region_ref": region,
                }
            )
        )

    return pd.concat(samples, ignore_index=True)


def test_region():
    """Test find_regions and _cluster_region."""
    siz = 10
    rgn = sample_regions(n_samples=3 * siz)

    rgn["region"] = sk.location._find_region(lat=rgn["lat"], lon=rgn["lon"])
    pd.testing.assert_series_equal(rgn["region"], rgn["region_ref"], check_names=False)

    rgn["cluster"] = sk.location._cluster_region(
        lat=rgn["lat"], lon=rgn["lon"], region=rgn["region"], cluster_size=siz
    )
    assert len(np.unique(rgn["cluster"])) > len(
        np.unique(rgn["region"], equal_nan=True)
    ), "Clustering should create more groups than original regions"

    with pd.option_context("display.max_rows", None):
        print(rgn)


def main():
    """Run the tests, without the overhead of the testing infrastructure.

    python tests/test_location.py
    """
    # sk.constants.URL = "https://api-beta.salientpredictions.com/"
    # session = sk.login()

    test_region()


if __name__ == "__main__":
    main()
