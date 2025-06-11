#!/usr/bin/env python
# Copyright Salient Predictions 2025

"""Test for the downscale module.

Usage example:
```
cd ~/salientsdk
python -m pytest -s tests/test_downscale_api.py
```

"""

import os
from unittest.mock import patch

import pytest
import xarray as xr

import salientsdk as sk
from salientsdk.__main__ import main

verbose = False


TEST_DATE = "2020-01-01"
TEST_VARS = "temp,precip"
TEST_DEST = "test_downscale"


def test_downscale_success(session):  # noqa: D103
    loc = sk.Location(lat=42, lon=-73)
    fil = sk.downscale(
        loc=loc,
        variables=TEST_VARS,
        date=TEST_DATE,
        members=11,
        force=True,
        session=session,
        verbose=verbose,
        destination=TEST_DEST,
    )

    assert os.path.exists(fil)

    ts = xr.open_dataset(fil)
    assert "forecast_day" in ts
    assert "analog" in ts
    assert "temp" in ts
    assert "precip" in ts
    assert "temp_clim" in ts


def test_downscale_failure(session):  # noqa: D103
    loc = sk.Location(lat=42, lon=-73)
    with pytest.raises(AssertionError) as ae:
        fil = sk.downscale(
            loc=loc,
            variables=TEST_VARS,
            start=TEST_DATE,
            members=-20,  # negative member count will trigger failure
            force=True,
            session=session,
            verbose=verbose,
            destination=TEST_DEST,
        )


@patch(
    "sys.argv",
    [
        "salientsdk",
        "downscale",
        "-lat",
        "42",
        "-lon",
        "-73",
        "--variables",
        TEST_VARS,
        "--date",
        TEST_DATE,
        "--verbose",  # needed to test output
    ],
)
def test_command_line(capsys):  # noqa: D103
    main()

    captured = capsys.readouterr()
    assert "Data variables:" in captured.out
    assert "ensemble" in captured.out


def local_test(session=None):
    """Execute functions without the overhead of pytest.

    python tests/test_downscale_api.py
    """
    sk.set_file_destination(TEST_DEST)
    loc = sk.Location(location_file=sk.upload_file_api._mock_upload_location_file())
    dsc = sk.downscale_api._downscale_mock(loc, verbose=True, frequency="hourly")
    assert dsc is not None


if __name__ == "__main__":
    # sk.constants.URL = "https://api-beta.salientpredictions.com/"
    # session = sk.login()
    # local_test(session)
    local_test()
