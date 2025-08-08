# tests/conftest.py
"""
Pytest-konfiguration der får alle tests grønne på Windows / Python 3.13.

* Ét globalt event-loop (vi lukker det ikke manuelt).
* Playwright stubbes, så   `await p.chromium.launch()` virker.
* AsyncHtmlClient bruger aldrig Playwright i tests.
* AsyncHtmlClient.close() sluger RuntimeError når loop allerede er lukket.
"""

import asyncio
import types
import logging
from contextlib import asynccontextmanager
import sys
import os # Import os to handle paths
from typing import Optional, Dict

# --- BEGIN ADDED FOR MODULE DISCOVERY (MUST BE AT TOP) ---
# Ensure the 'src' directory is in the Python path for module discovery
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)
# --- END ADDED FOR MODULE DISCOVERY ---

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock
from vexto.scoring.http_client import AsyncHtmlClient, html_cache
import httpx # Required for mocking httpx exceptions in tests

# ---------------------------------------------------------------------------
# 1)   Samlet event-loop for hele test-sessionen
# ---------------------------------------------------------------------------
@pytest_asyncio.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    #   Ingen loop.close()   →   httpx kan stadig afslutte sine sockets bagefter.


# ---------------------------------------------------------------------------
# 2)   Fuldt “await-bart” Playwright-stub (for tests that might use playwright.async_api directly)
# ---------------------------------------------------------------------------
try:
    import playwright.async_api as _pw_async_api
except ModuleNotFoundError:
    _pw_async_api = None
else:

    class _DummyBrowser:
        async def close(self):   # noqa: D401
            pass

        def is_connected(self) -> bool:      # noqa: D401
            return True

        async def new_page(self, *args, **kwargs):
            # Return a mock page that can be used for basic operations
            mock_page = MagicMock()
            mock_page.goto = AsyncMock()
            mock_page.content = AsyncMock(return_value="<html><body>Mock HTML Content</body></html>")
            mock_page.title = AsyncMock(return_value="Mock Page Title")
            mock_page.close = AsyncMock()
            return mock_page


    async def _launch(*_a, **_kw):
        return _DummyBrowser()

    _dummy_browser_type = types.SimpleNamespace(launch=_launch)

    @asynccontextmanager
    async def _dummy_pw_cm():
        yield types.SimpleNamespace(
            chromium=_dummy_browser_type,
            firefox=_dummy_browser_type,
            webkit=_dummy_browser_type,
        )

    # Monkey-patch fabrikken
    _pw_async_api.async_playwright = lambda *_a, **_kw: _dummy_pw_cm()


# ---------------------------------------------------------------------------
# 3)   Mocking the AsyncHtmlClient for tests (NEW/RE-ADDED APPROACH)
# ---------------------------------------------------------------------------
@pytest_asyncio.fixture(scope="function") # Changed scope to function for isolation per test
async def mock_async_html_client():
    """
    Returns a MagicMock that acts like an AsyncHtmlClient.
    It supports async with, has httpx_get, _httpx_client (for close), and _pw_thread (for close).
    """
    mock_client = MagicMock(spec=AsyncHtmlClient)
    
    # Mock __aenter__ and __aexit__ for async with support
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    # Mock _httpx_client and its aclose() for cleanup
    # Tests that use mock_async_html_client should expect mock_client.httpx_get
    # not direct _httpx_client usage.
    mock_client._httpx_client = MagicMock(spec=httpx.AsyncClient)
    mock_client._httpx_client.aclose = AsyncMock()

    # Mock _pw_thread and its close()
    mock_client._pw_thread = MagicMock()
    mock_client._pw_thread.close = AsyncMock()
    mock_client._pw_thread._is_ready = False # Default to Playwright not ready in mocks, for most unit tests

    # Mock critical public methods that tests might call
    mock_client.httpx_get = AsyncMock()
    mock_client.get_raw_html = AsyncMock()
    mock_client.get_soup = AsyncMock()
    mock_client.head = AsyncMock()
    mock_client.close = AsyncMock() # Mock the close method directly

    # Clear html_cache before each test using this mock, if tests interact with it
    html_cache.clear()

    yield mock_client

    # The mock_client.close() is now an AsyncMock, so awaiting it is fine.
    # It will implicitly be called as part of the fixture teardown by pytest-asyncio.
    pass # No explicit call here needed, as yield handles teardown


# ---------------------------------------------------------------------------
# 4)   Gør lukningen mere tolerant (loop kan være lukket af pytest)
# This section is now mainly for the real AsyncHtmlClient instance,
# not for the mock_async_html_client.
# ---------------------------------------------------------------------------
# The original _safe_close was monkey-patching AsyncHtmlClient.close
# With the new _PlaywrightThreadFetcher.close, the main client.close()
# should ideally handle its own cleanup correctly.
# If pytest_asyncio handles fixture teardown (which it does), the default
# client.close() should be called.
# Let's revert this monkey-patching for AsyncHtmlClient.close as the class's
# own close method should be robust. If there's still a RuntimeError about loop
# being closed, we can revisit.
# For now, remove this monkey-patching, as it might conflict with the mock.

# async def _safe_close(self):
#   try:
#       await self._httpx_client.aclose()
#   except RuntimeError:
#       pass
#   try:
#       await self._pw_thread.close()
#   except Exception as e:
#       logging.getLogger(__name__).warning(f"Error closing Playwright thread in tests: {e}")
# AsyncHtmlClient.close = _safe_close # REMOVE THIS LINE


# ---------------------------------------------------------------------------
# 5)   Fælles klient-fixture (for tests that need a real AsyncHtmlClient instance)
# ---------------------------------------------------------------------------
@pytest_asyncio.fixture(scope="session")
async def async_html_client():
    # In tests, we want Playwright to be explicitly NOT available unless
    # Playwright binaries are installed and a test specifically allows it.
    client = AsyncHtmlClient()
    # By default, Playwright is shut down for tests that don't specifically need it
    # to avoid real browser launches during unit tests.
    client._pw_thread._is_ready = False 
    yield client
    # The client.close() method of the *actual* AsyncHtmlClient is called here
    # by pytest-asyncio after the session ends.
    # Its internal logic handles both httpx and Playwright thread cleanup.
    await client.close()


@pytest_asyncio.fixture(scope="session")
async def client(async_html_client):
    """Alias så alle eksisterende tests kan bruge 'client'."""
    return async_html_client

# ---------------------------------------------------------------------------
# 6) setup_authority_tests fixture (add mocker to its signature)
# --- START PÅ KIRURGISK INDGREB ---
# Rettet fra autouse=True for at undgå at forstyrre andre tests.
# Rettet lambda til at acceptere et 'default'-argument for at undgå TypeError.
# ---------------------------------------------------------------------------
@pytest.fixture
def setup_authority_tests(mocker):
    """Ensures OPR_API_KEY is set and cache is cleared for authority tests."""
    mocker.patch('os.getenv', side_effect=lambda key, default=None: 'DUMMY_OPR_KEY' if key == 'OPENPAGERANK_API_KEY' else default)
    html_cache.clear()
    yield
    html_cache.clear()
# --- SLUT PÅ KIRURGISK INDGREB ---

# Mock httpx.Response for tests (if it's used by multiple fetchers tests)
class MockHttpxResponse:
    def __init__(self, status_code: int, text: str = "", json_data: Optional[Dict] = None, request: Optional[httpx.Request] = None):
        self.status_code = status_code
        self._text = text
        self._json_data = json_data
        self.request = request if request else httpx.Request("GET", "http://mock-url.com")
        self.url = self.request.url # Mock the .url attribute

    def json(self) -> Dict:
        return self._json_data if self._json_data is not None else {}

    @property
    def text(self) -> str:
        return self._text

    def raise_for_status(self) -> None:
        if 400 <= self.status_code < 600:
            raise httpx.HTTPStatusError(
                f"Client error: {self.status_code}", request=self.request, response=self
            )

# Add mock data if test_authority_fetcher.py relies on them globally
# (Otherwise, move these into test_authority_fetcher.py if they're only used there)
# MOCK_OPR_SUCCESS_RESPONSE = {"data": {"pageRank": [{"domain": "example.com", "pageRank": 50, "rank": 7, "globalRank": 12345}]}}
# MOCK_OPR_NO_DATA_RESPONSE = {"data": {"pageRank": []}}