#!/usr/bin/env python
# Copyright Salient Predictions 2025


"""Tests for constants.py.

Usage:
```
python -s -m pytest tests/test_constants.py
python tests/test_constants.py
```

"""

import pytest

import salientsdk as sk


def test_get_hindcast_dates():
    """Test get_hindcast_dates."""
    valid_timescales = ["sub-seasonal", "seasonal", "long-range"]

    for timescale in valid_timescales:
        dat = sk.get_hindcast_dates(timescale=timescale)
        assert isinstance(dat, list), f"Expected dat to be a list for timescale {timescale}"
        assert len(dat) > 0, f"Expected dat to be a non-empty list for timescale {timescale}"

    with pytest.raises(ValueError) as baddate:
        sk.get_hindcast_dates(start_date="not_a_date")

    assert "Invalid date format" in str(baddate.value)

    with pytest.raises(ValueError) as excinfo:
        sk.get_hindcast_dates(timescale="not-a-timescale")
    assert "not-a-timescale" in str(excinfo.value)


def main():
    """Run the tests, without the overhead of the testing infrastructure.

    python tests/test_constants.py
    """
    # the one function we test here is standalone and doesn't need a session
    # sk.constants.URL = "https://api-beta.salientpredictions.com/"
    # session = sk.login()

    test_get_hindcast_dates()


if __name__ == "__main__":
    main()
