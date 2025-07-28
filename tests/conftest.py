"""Set up and tear down for the test suite."""

import os
import shutil
import tempfile
from pathlib import Path

import pytest

import salientsdk as sk


def get_test_dir(path_or_name: str | Path | None = None) -> Path:
    """Get a user-specific temporary directory for test files.

    Arguments:
        path_or_name: If specified, will create a subdirectory underneath the
            primary testing directory. Can be either:
            - A subdirectory name
            - A path to a file (e.g. __file__) - the filename will be extracted

    Returns:
        Path: Base directory for test files, unique to current user
    """
    base_dir = Path(tempfile.gettempdir()) / f"salientsdk_test_{os.getuid()}"
    base_dir.mkdir(mode=0o700, exist_ok=True)

    return base_dir if path_or_name is None else base_dir / Path(path_or_name).stem


@pytest.fixture(scope="session", autouse=True)
def set_api_url():
    """Use api-beta for testing."""
    original_url = sk.constants.URL
    sk.constants.URL = "https://api-beta.salientpredictions.com/"
    yield
    sk.constants.URL = original_url


@pytest.fixture(scope="session")
def session():
    """Shared session for all tests in the test suite.

    Raises:
        RuntimeError: If the session setup fails.

    Create a session with login and close it after the test.
    This session will be used by all the tests in the test suite.
    """
    try:
        # Setup: attempt to create a session with login
        test_session = sk.login()
        if not test_session:  # Replace this check with your actual condition for a bad session
            raise RuntimeError("Failed to set up a test session.")
    except Exception as e:
        # You can log the exception here if needed
        pytest.exit(f"Test session setup failed: {e}", returncode=1)

    yield test_session

    # Teardown: close the session
    test_session.close()


@pytest.fixture(scope="session")
def destination():
    """Destination directory for test output available for whole test session."""
    try:
        tmdir = tempfile.TemporaryDirectory()
        os.chmod(tmdir.name, 0o755)
        yield tmdir.name
    finally:
        try:
            # Force remove read-only files
            def remove_readonly(func, path, _):
                os.chmod(path, 0o755)
                func(path)

            shutil.rmtree(tmdir.name, onerror=remove_readonly)
        except Exception as e:
            print(f"Warning: Failed to cleanup {tmdir.name}: {e}")
