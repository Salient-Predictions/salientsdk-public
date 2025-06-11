#!/usr/bin/env python
# Copyright Salient Predictions 2025

"""Test for the command line interface.

Usage example:
```
python -m pytest -s tests/test___main__.py
```

"""

from unittest.mock import patch

import pytest

from salientsdk.__main__ import main


def test_help(capsys):  # noqa: D103
    with patch("sys.argv", ["salientsdk", "--help"]):
        with pytest.raises(SystemExit) as e:
            main()
        assert e.value.code == 0

    captured = capsys.readouterr()
    assert "usage:" in captured.out


def test_examples(capsys):  # noqa: D103
    with patch("sys.argv", ["salientsdk", "examples"]):
        main()

    captured = capsys.readouterr()

    assert "examples/downscale.ipynb" in captured.out
