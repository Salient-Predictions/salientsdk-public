#!/usr/bin/env python
# Copyright Salient Predictions 2025

"""Test for the hindcast_summary_api module.

Usage example:
```
python -m pytest -s tests/test_hindcast_summary_api.py
```

"""

from unittest.mock import patch

import pandas as pd
import pytest

import salientsdk as sk
from salientsdk.__main__ import main

TEST_LAT = 42.0
TEST_LON = -73.0

TEST_DIR = "test_hindcast_summary"


def test_hindcast_summary_success(session, verbose=False):  # noqa: D103
    loc = sk.Location(lat=TEST_LAT, lon=TEST_LON)
    fil = sk.hindcast_summary(
        loc=loc,
        force=True,
        session=session,
        verbose=verbose,
        destination=TEST_DIR,
    )

    df = pd.read_csv(fil)
    assert "Lead" in df.columns
    assert "Reference Model" in df.columns
    assert any(col.startswith("Reference CRPS") for col in df.columns)
    assert any(col.startswith("Salient CRPS") for col in df.columns)
    assert any(col.startswith("Salient CRPS Skill Score") for col in df.columns)

    xscores = sk.transpose_hindcast_summary(fil)
    assert "mean" in xscores.columns


def test_hindcast_summary_multi_success(session, verbose=False):  # noqa: D103
    # Note that temp & precip have different units, so this is a good test of
    # _concatenate_hindcast_summary, which has to do some fancy footwork to concatenate
    # Reference CRPS (unit) when the units conflict.

    loc = sk.Location(lat=TEST_LAT, lon=TEST_LON)
    fil = sk.hindcast_summary(
        loc=loc,
        variable="temp,precip",
        force=False,
        session=session,
        verbose=verbose,
        destination=TEST_DIR,
    )

    # because we requested multiple variables, the single file returned by
    # hindcast_summary will be the concatenation of the individual files,
    # with an additional column "variable" to indicate which variable each row corresponds to
    scores = pd.read_csv(fil)
    assert "variable" in scores.columns
    assert "Lead" in scores.columns
    assert "Reference Model" in scores.columns
    assert any(col.startswith("Reference CRPS") for col in scores.columns)
    assert any(col.startswith("Salient CRPS") for col in scores.columns)

    if verbose:
        print(scores)

    xscores = sk.transpose_hindcast_summary(fil)
    assert "variable" in xscores.index.names
    assert "mean" in xscores.columns

    if verbose:
        print(xscores)


def test_hindcast_summary_failure(session, verbose=False):  # noqa: D103
    loc = sk.Location(lat=TEST_LAT, lon=TEST_LON)
    with pytest.raises(AssertionError) as ae:
        fil = sk.hindcast_summary(
            loc=loc,
            force=True,
            reference="bad_reference",
            session=session,
            verbose=verbose,
            destination=TEST_DIR,
        )


@patch(
    "sys.argv",
    [
        "salientsdk",
        "hindcast_summary",
        "--lat",
        "42",
        "--lon",
        "-73",
        "--verbose",  # needed to test the output
    ],
)
def test_command_line(capsys):  # noqa: D103
    main()  # from sk.__main__
    captured = capsys.readouterr()
    assert "Lead" in captured.out


def run_local():
    """Run the tests, without the overhead of the testing infrastructure.

    python tests/test_hindcast_summary_api.py
    """
    sk.constants.URL = "https://api-beta.salientpredictions.com/"
    # sk.constants.set_model_version("v9")
    session = sk.login()

    # test_hindcast_summary_success(session)
    test_hindcast_summary_multi_success(session, verbose=True)


if __name__ == "__main__":
    run_local()
