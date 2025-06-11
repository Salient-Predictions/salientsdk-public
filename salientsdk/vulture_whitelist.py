#!/usr/bin/env python
# Copyright Salient Predictions 2025

"""Whitelist file for vulture.

We use the vulture tool to make sure that we don't have dead code
hanging around. Sometimes vulture falsely flags a function as unused.
In that case, we add the function to this whitelist file so they are
explicitly registered as used.
"""

# from .upload_file_api import user_files
# assert user_files  # unused function (salientsdk/upload_file_api.py:220)

from .forecast_zarr import ForecastZarr
from .hydro import exec_vic
from .solar import _downscale_solar

assert _downscale_solar  # currently under development and skipped in testing
assert exec_vic  # Ignore b/c VIC Image Driver is provided by user, not the SDK.
assert ForecastZarr.__new__  # Vulture doesn't like the factory class method
