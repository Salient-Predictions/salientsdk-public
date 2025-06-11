"""Pytest configuration file for the examples directory.

Sets the "USE_MOCK_DOWNSCALE" environment variable to "True" so that the
downscale function will use the mock implementation, which accelerates tests.

```
# To test all notebooks:
pytest --nbmake -v "./salientsdk/examples/"
# To test a single notebook:
pytest --nbmake -v "./salientsdk/examples/enwex.ipynb"
```

This conftest creates the equivalent of:
```
USE_MOCK_DOWNSCALE=True pytest --nbmake -v "./examples/"
```

"""

import os

import pytest


@pytest.fixture(scope="session", autouse=True)
def use_mock_downscale():  # noqa: Vulture
    """Set the USE_MOCK_DOWNSCALE environment variable to "True" for the duration of the test session."""
    # Save the original value so it can be restored later
    key = "USE_MOCK_DOWNSCALE"
    original_value = os.environ.get(key)

    os.environ[key] = "True"

    try:
        yield
    finally:
        if original_value is None:
            del os.environ[key]
        else:
            os.environ[key] = original_value


if __name__ == "__main__":
    """Quick command line test to make sure the fixture is accessible."""

    print(use_mock_downscale)
