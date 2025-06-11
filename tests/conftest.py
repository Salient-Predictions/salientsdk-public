"""Set up and tear down for the test suite."""

import shutil
import tempfile

import pytest

import salientsdk as sk


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
        yield tmdir.name
    finally:
        shutil.rmtree(tmdir.name)
