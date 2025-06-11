#!/usr/bin/env python
# Copyright Salient Predictions 2025

"""test files whitelist file for vulture.

We use the vulture tool to make sure that we don't have dead code
hanging around. Sometimes vulture falsely flags a function as unused.
In that case, we add the function to this whitelist file so they are
explicitly registered as used.
"""

from .conftest import set_api_url

# Explicitly reference the fixture to prevent Vulture from considering it unused
assert set_api_url
