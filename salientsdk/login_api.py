#!/usr/bin/env python
# Copyright Salient Predictions 2025

"""Login to the Salient API.

Command line usage:
```
cd ~/salientsdk
python -m salientsdk login -u username -p password
# or, to use api keys:
python -m salientsdk login --apikey SALIENT_APIKEY
```

"""

import os
import warnings
from concurrent.futures import ThreadPoolExecutor

import google.auth
import pandas as pd
import requests
from backoff import expo, on_exception
from google.cloud import secretmanager
from ratelimit import RateLimitException, limits
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .constants import _build_url

VERIFY_SSL = True

CURRENT_SESSION = None


# ============================== Error-handling utilities =======================
# 90% Safety buffer to avoid erroring out by hitting API request limits.
CALLS_PER_MINUTE = 0.9 * 250
CALLS_PER_DAY = 0.9 * 5000


class DetailedHTTPError(requests.HTTPError):
    """Extends HTTPError to include response content in string representation."""

    def __init__(self, err: requests.HTTPError) -> None:
        """Initialize with a requests.HTTPError."""
        super().__init__(str(err), response=err.response)

    def __str__(self) -> str:
        """Return error message with response content if available."""
        errstr = super().__str__()
        if self.response is not None:
            errstr += "\nDetail: " + self.response.text
        return errstr


def rate_limit_handler(details: dict) -> None:
    """Prints a message when a rate limit is reached.

    Args:
        details (dict): Contains 'wait' key with retry delay in seconds.
    """
    warnings.warn(f"Rate limit reached. Sleeping for {details['wait']} seconds.", RuntimeWarning)


# =========================== Session handlers =====================================
def _get_secret(secret_name: str, version: str = "1") -> str:
    """Get a secret from Google Cloud Secret Manager with auto-detected project.

    Args:
        secret_name (str): Name of the secret (e.g., "API_TEST_USER_KEY")
        version (str): Version of the secret (default "1")

    Returns:
        str: The secret value

    Raises:
        Exception: Any exception from Google Cloud Secret Manager
    """
    _, project_id = google.auth.default()
    secret_path = f"projects/{project_id}/secrets/{secret_name}/versions/{version}"
    client = secretmanager.SecretManagerServiceClient()
    response = client.access_secret_version(request={"name": secret_path})
    return response.payload.data.decode("UTF-8")


def _get_api_key(apikey: str | None = None) -> str | None:
    """Regularize API key values."""
    if apikey is None:
        return apikey

    apikey = str(apikey)
    assert len(apikey) > 0, f"API key must not be empty"

    if apikey in ("apikey", "SALIENT_APIKEY"):
        apikey = os.getenv("SALIENT_APIKEY")  # First try environment variable
        if apikey is None:
            try:
                apikey = _get_secret("API_TEST_USER_KEY")
            except Exception:
                raise OSError("Set the SALIENT_APIKEY environment variable")

    return apikey


def get_current_session() -> requests.Session:
    """Get the current session.

    All calls to the Salient API have a `session` argument
    that defaults to `None`.  If session is not passed to the
    function, the api call will use the output of this function.


    Returns:
        requests.Session: The current session if one was set via
            `login()` or `set_current_session()`, or a temporary
            session for use with `apikey`.
    """
    return requests.Session() if CURRENT_SESSION is None else CURRENT_SESSION


def set_current_session(session: requests.Session | None) -> None:
    """Set the current session.

    This function is called internally as a side effect of
    `login()`. In most cases, users will never need
    to call it explicitly.

    Args:
        session (requests.Session | None): The session that will be
              returned by `get_current_session()`.  Set `None` to
              clear the session.

    """
    assert session is None or isinstance(session, requests.Session)

    global CURRENT_SESSION
    CURRENT_SESSION = session


def get_verify_ssl(verify: bool | None = None) -> bool:
    """Get the current SSL verification setting.

    All functions that call the Salient API have a
    `verify` argument that controls whether or not to use
    SSL verification when making the call.  That argument
    will default to use this function, so in most cases
    users will never need to call it.

    Args:
        verify (bool | None): If `None` (default), returns the
            SSL verification setting that was set
            by `set_verify_ssl()` as a side effect of `login()`.
            If `True` or `False`, passes through without checking
            the default value.

    Returns:
        bool: The current SSL verification setting

    """
    if verify is None:
        verify = VERIFY_SSL
        if verify is None:
            verify = True

    verify = bool(verify)

    return verify


def set_verify_ssl(verify: bool = True) -> bool:
    """Set the SSL verification setting.

    Sets the default value to be used when calling
    `get_verify_ssl(None)`.
    This is usually set automatically as a side
    effect of `login(..., verify=None)` so in most
    cases users will never need to call it.

    Args:
        verify (bool): The SSL verification setting
           that will be returned by `get_verify_ssl()`.

    Returns:
        bool: The SSL verification setting that was set
    """
    global VERIFY_SSL
    VERIFY_SSL = bool(verify)
    return VERIFY_SSL


def login(
    username: str = "SALIENT_USERNAME",
    password: str = "SALIENT_PASSWORD",
    apikey: str | None = None,
    verify: bool | None = None,
    verbose=False,
    http_proxy: str | None = None,
    https_proxy: str | None = None,
) -> requests.Session:
    """Login to the Salient API.

    This function is a local convenience wrapper around the Salient API
    [login](https://api.salientpredictions.com/v2/documentation/api/#/Authentication/login)
    endpoint.  It will use your credentials to create a persistent session you
    can use to execute API calls.

    If using the default username/password values, the function will first check
    for environment variables, then fall back to test credentials for demo purposes.

    Example:
        # Set credentials via environment variables (recommended)
        export SALIENT_USERNAME="your_username@example.com"
        export SALIENT_PASSWORD="your_password"
        session = sk.login()

        # Or pass credentials directly
        session = sk.login("your_username@example.com", "your_password")

        # Or use an API key instead (set SALIENT_APIKEY environment variable)
        session = sk.login(apikey="SALIENT_APIKEY")

    Args:
        username (str): The username to login with. Defaults to "SALIENT_USERNAME"
            which will check the SALIENT_USERNAME environment variable.
        password (str): The password to login with. Defaults to "SALIENT_PASSWORD"
            which will check the SALIENT_PASSWORD environment variable.
        apikey (str | None): Create a session with an API key (ignores `username` and `password`)
            Use value `SALIENT_APIKEY` to use the `SALIENT_APIKEY` environment variable.
        verify (bool): Whether to verify the SSL certificate.
            If `None` (default) will try `True` and then `False`, remembering the
            last successful setting and preserving it for future calls in `get_verify_ssl()`.
        verbose (bool): Whether to print the response status
        http_proxy (str | None): HTTP proxy URL to use for requests. If None, will use HTTP_PROXY
            environment variable if set.
        https_proxy (str | None): HTTPS proxy URL to use for requests. If None, will use HTTPS_PROXY
            environment variable if set.

    Returns:
        Session object to pass to other API calls.
            As a side effect, will also set the default session for
            use with `get_current_session()`
    """
    if apikey is not None:
        apikey = _get_api_key(apikey)
        username = None
        password = None

    if password in ("password", "SALIENT_PASSWORD"):
        password = os.getenv("SALIENT_PASSWORD")  # returns None if not found
        if password is None:
            try:
                password = _get_secret("API_TEST_USER_PWD")
                username = "testuser@salientpredictions.com"
            except Exception as e:
                raise OSError("Set the SALIENT_PASSWORD environment variable")

    if username in ("username", "SALIENT_USERNAME"):
        username = os.getenv("SALIENT_USERNAME")
        if username is None:
            raise OSError("Set the SALIENT_USERNAME environment variable.")

    if verify is None:
        try:
            session = login(
                username=username,
                password=password,
                apikey=apikey,
                verify=True,
                verbose=verbose,
                http_proxy=http_proxy,
                https_proxy=https_proxy,
            )
            set_verify_ssl(True)
        except requests.exceptions.SSLError:
            session = login(
                username=username,
                password=password,
                apikey=apikey,
                verify=False,
                verbose=verbose,
                http_proxy=http_proxy,
                https_proxy=https_proxy,
            )
            set_verify_ssl(False)

        return session

    session = requests.Session()

    # Configure proxies if provided
    proxies = {}
    if http_proxy:
        proxies["http"] = http_proxy
    if https_proxy:
        proxies["https"] = https_proxy
    if proxies:
        session.proxies.update(proxies)

    adapter = HTTPAdapter(
        max_retries=Retry(
            total=5,
            backoff_factor=2,
            status_forcelist=[500, 502, 503, 504, 404],
            allowed_methods=frozenset(["GET", "POST", "PUT"]),
            raise_on_status=False,
        ),
        pool_connections=25,
        pool_maxsize=25,
        pool_block=True,
    )
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    if apikey is None:
        auth = (username, password)
        args = None
    else:
        auth = None
        args = {"apikey": apikey}

    (url, file_name) = _build_url("login", args=args)
    login_ok = session.get(url, auth=auth, verify=verify)

    try:
        login_ok.raise_for_status()
    except requests.HTTPError as err:
        raise DetailedHTTPError(err) from None

    if verbose:
        print(login_ok.text)

    set_current_session(session)

    return session


# ========================== Querying ==============================================
def download_queries(
    query: list[str],
    file_name: list[str],
    format: str = "-auto",
    force: bool = False,
    session: requests.Session | None = None,
    verify: bool | None = None,
    verbose: bool = False,
    max_workers: int | None = None,
    strict: bool = False,
) -> list[str]:
    """Downloads multiple queries saves them to a file.

    This function handles the downloading of data based on the provided query URLs.
    It saves the data to the specified file names.
    If the file already exists and `force` is not set to True, the download is skipped.
    Download will happen in parallel.

    Parameters:
        query (list[str]): The URLs from which to download the data.
        file_name (list[str]): The paths where the data will be saved.
        format (str, optional): The format of the file.
            Defaults to '-auto', which will infer the format from the file extension.
        force (bool, optional): If True, the file will be downloaded even if it already exists.
            Defaults to False.
        session (requests.Session, optional): The session to use for the download.
            If `None` (default) uses `get_current_session()`.
        verify (bool, optional): Whether to verify the server's TLS certificate.
            Defaults to the current verification setting via `get_verify_ssl()`.
        verbose (bool, optional): If True, prints additional output about the download process.
            Defaults to False.
        max_workers (int, optional): The maximum number of threads to use for downloading.
        strict (bool, optional): If False (default) will not raise errors if there is a problem
            when executing vectorized queries.

    Returns:
        list[str]: The file names of the downloaded data.  When strict=False, will contain
            NA for any query that errored out.

    Raises:
        requests.HTTPError: If the server returns an error status code and strict=True
    """
    assert len(query) == len(file_name)

    if len(query) == 0:
        return []
    elif len(query) == 1:
        # Much of the time we won't have vectorized queries.  Keep it simple.
        return [
            download_query(query[0], file_name[0], format, force, session, verify, verbose, strict)
        ]

    if max_workers is None:
        max_workers = os.cpu_count() * 5

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                download_query,
                query=qry,
                file_name=fil,
                format=format,
                force=force,
                session=session,
                verify=verify,
                verbose=verbose,
                strict=strict,
            )
            for qry, fil in zip(query, file_name)
        ]
        # file_name will be NA if the query errored out with strict=False
        file_name = [future.result() for future in futures]

    return file_name


def download_query(
    query: str,
    file_name: str,
    format: str = "-auto",
    force: bool = False,
    session: requests.Session | None = None,
    verify: bool = None,
    verbose: bool = False,
    strict: bool = True,
) -> str:
    """Downloads the query result and saves it to a file.

    This function handles the downloading of data based on the provided query URL.
    It saves the data to the specified file name.

    Parameters:
        query (str): The URL from which to download the data.
        file_name (str): The path where the data will be saved.
        format (str, optional): The format of the file.
            Defaults to '-auto', which will infer the format from the file extension.
        force (bool, optional): If False (default) skips downloading `file_name` if it already exists.
        session (requests.Session, optional): The session to use for the download.
            If `None` (default) uses `get_current_session()`.
        verify (bool, optional): Whether to verify the server's TLS certificate.
            If `None` (default) uses the current verification setting via `get_verify_ssl()`.
        verbose (bool, optional): If True, prints additional output about the download process.
            Defaults to False.
        strict (bool, optional): If True (default) raises errors if there is a problem.

    Returns:
        str: The file name of the downloaded data.  When
            strict=False, will return pd.NA if there was an error.

    Warns:
        RuntimeWarning: If an API call rate limit is reached, will issue a warning
            indicating the sleep duration before the next retry.

    Raises:
        requests.HTTPError: If the server returns an error status code.
    """
    if force or not os.path.exists(file_name):
        if verbose:
            print(f"Downloading\n  {query}\n to {file_name}\n with {session}")
        if format == "-auto":
            # extract the file extension from the file name
            format = file_name.split(".")[-1]
        is_binary = format == "nc"

        if not _make_api_call(query, file_name, session, verify, is_binary, strict):
            file_name = pd.NA

    elif verbose:
        print(f"File {file_name} already exists")

    return file_name


@on_exception(expo, RateLimitException, on_backoff=rate_limit_handler)
@limits(calls=CALLS_PER_MINUTE, period=60)
@limits(calls=CALLS_PER_DAY, period=86400)
def _make_api_call(
    query: str,
    file_name: str,
    session: requests.Session | None = None,
    verify: bool = None,
    is_binary: bool = True,
    strict: bool = True,
) -> bool:
    """Make the API call and save the result to a file."""
    if session is None:
        session = get_current_session()
    verify = get_verify_ssl(verify)

    result = session.get(query, verify=verify)
    try:
        result.raise_for_status()
    except requests.HTTPError as err:
        err = DetailedHTTPError(err)
        if strict:
            raise err from None
        else:
            warnings.warn(str(err), RuntimeWarning)
            return False  # Caught failure
    with open(file_name, "wb" if is_binary else "w") as f:
        if is_binary:
            f.write(result.content)
        else:
            f.write(result.text)

    return True  # success
