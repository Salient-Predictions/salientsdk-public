#!/usr/bin/env python
# Copyright Salient Predictions 2025

"""Tests for login_api.

Usage:
```
python -s -m pytest tests/test_login_api.py
```

"""

import os

import salientsdk as sk

DST_DIR = "test_login"


def test_login_apikey():
    """Test logging in with apikey to make sure it works.

    Note that most Salient tests take a "session" argument generated in conftest.
    Because the whole point of the login function is to create a session, we
    shouldn't be passing a session here.
    """
    sk.set_file_destination(DST_DIR)

    session = sk.login(apikey="SALIENT_APIKEY")
    assert session.adapters and session.adapters["https://"]

    files_response = sk.user_files(session=session, destination=DST_DIR)
    assert os.path.exists(files_response)


def local_test():
    """Execute functions without the overhead of pytest.

    python tests/test_login_api.py
    """
    # sk.constants.URL = "https://api-beta.salientpredictions.com/"
    test_login_apikey()

    # print(sk.login())


if __name__ == "__main__":
    local_test()
