import pytest
import os
import json
import httpx
from unittest.mock import AsyncMock # Use AsyncMock for async methods
from urllib.parse import urlparse # Import urlparse for cache key check

# Import the function to be tested and the AsyncHtmlClient
from vexto.scoring.authority_fetcher import get_authority
from vexto.scoring.http_client import AsyncHtmlClient, html_cache # Import html_cache for potential cache clearing
from vexto.scoring.schemas import AuthorityMetrics # Import schema for type checking assertions

# --- Mock Data for API Responses ---
MOCK_OPR_SUCCESS_RESPONSE = {
    "status": "success",
    "response": [
        {
            "domain": "example.com",
            "page_rank_integer": 7,
            "rank": 12345,
            "domain_authority": 50.0
        }
    ]
}

MOCK_OPR_NO_DATA_RESPONSE = {
    "status": "success",
    "response": [
        # OPR returns nulls, not an empty list for "no data" for a specific domain query
        {"domain": "nonexistent.com", "page_rank_integer": None, "rank": None, "domain_authority": None}
    ]
}


# Helper to create a mock httpx.Response
class MockHttpxResponse:
    def __init__(self, status_code, json_data=None):
        self.status_code = status_code
        self._json_data = json_data
        self.url = "https://openpagerank.com/api/v1.0/getPageRank?domains[]=example.com" # Dummy URL

    def json(self):
        if self._json_data is not None:
            return self._json_data
        raise json.JSONDecodeError("No JSON data", "", 0)

    def raise_for_status(self):
        if 400 <= self.status_code < 600:
            raise httpx.HTTPStatusError(
                f"Client error: {self.status_code}",
                request=httpx.Request("GET", "http://mock-url.com"),
                response=self
            )

# --- Pytest Fixtures ---
@pytest.fixture
def mock_async_html_client():
    """Provides a mock AsyncHtmlClient instance."""
    client = AsyncHtmlClient()
    client.httpx_get = AsyncMock()
    return client

@pytest.fixture
def setup_authority_tests(mocker):
    """Ensures OPR_API_KEY is set and cache is cleared for authority tests."""
    # --- START PÅ KIRURGISK INDGREB ---
    # Lambda-funktionen er opdateret til at acceptere et 'default' argument.
    # Dette forhindrer TypeError, når f.eks. Playwright kalder os.getenv med to argumenter.
    mocker.patch('os.getenv', side_effect=lambda key, default=None: 'DUMMY_OPR_KEY' if key == 'OPENPAGERANK_API_KEY' else default)
    # --- SLUT PÅ KIRURGISK INDGREB ---
    html_cache.clear()
    yield
    html_cache.clear()


# --- Test Cases ---

@pytest.mark.asyncio
async def test_get_authority_success(mock_async_html_client, setup_authority_tests):
    """Tests successful retrieval of authority data from OPR."""
    mock_response = MockHttpxResponse(200, json_data=MOCK_OPR_SUCCESS_RESPONSE)
    mock_async_html_client.httpx_get.return_value = mock_response

    url_to_test = "https://example.com"
    result = await get_authority(mock_async_html_client, url_to_test)

    expected: AuthorityMetrics = {
        "domain_authority": 50.0,
        "page_authority": 7.0,
        "global_rank": 12345,
        "authority_status": "ok_openpagerank",
    }
    assert result == expected
    mock_async_html_client.httpx_get.assert_called_once_with(
        "https://openpagerank.com/api/v1.0/getPageRank",
        params={"domains[]": "example.com"},
        headers={"API-OPR": "DUMMY_OPR_KEY"},
        timeout=15
    )
    assert html_cache.get(f"opr_{urlparse(url_to_test).hostname}") == expected
    await mock_async_html_client.close()

@pytest.mark.asyncio
async def test_get_authority_api_key_missing(mock_async_html_client, mocker):
    """Tests when OPR_API_KEY is not set."""
    mocker.patch('os.getenv', side_effect=lambda key, default=None: None if key == 'OPENPAGERANK_API_KEY' else default)

    result = await get_authority(mock_async_html_client, "https://any-domain.com")

    assert result is None
    mock_async_html_client.httpx_get.assert_not_called()
    await mock_async_html_client.close()


@pytest.mark.asyncio
async def test_get_authority_domain_not_found(mock_async_html_client, setup_authority_tests):
    """Tests authority fetch when OPR API returns data with nulls for a domain not found."""
    mock_response = MockHttpxResponse(200, json_data=MOCK_OPR_NO_DATA_RESPONSE)
    mock_async_html_client.httpx_get.return_value = mock_response

    url_to_test = "https://nonexistent.com"
    result = await get_authority(mock_async_html_client, url_to_test)

    expected: AuthorityMetrics = {
        "domain_authority": None,
        "page_authority": None,
        "global_rank": None,
        "authority_status": "no_data_openpagerank",
    }
    assert result == expected
    mock_async_html_client.httpx_get.assert_called_once()
    assert html_cache.get(f"opr_{urlparse(url_to_test).hostname}") == expected
    await mock_async_html_client.close()

@pytest.mark.asyncio
async def test_get_authority_api_http_error(mock_async_html_client, setup_authority_tests):
    """Tests when OPR API returns a non-2xx status code (e.g., 400, 500)."""
    mock_response = MockHttpxResponse(400, json_data={"error": "bad request"})
    mock_async_html_client.httpx_get.return_value = mock_response

    url_to_test = "https://error-domain.com"
    result = await get_authority(mock_async_html_client, url_to_test)

    expected: AuthorityMetrics = {
        "domain_authority": None, "page_authority": None, "global_rank": None, "authority_status": "http_error_400"
    }
    assert result == expected
    mock_async_html_client.httpx_get.assert_called_once()
    await mock_async_html_client.close()

@pytest.mark.asyncio
async def test_get_authority_network_error(mock_async_html_client, setup_authority_tests):
    """Tests when a network error occurs during OPR API call."""
    mock_async_html_client.httpx_get.side_effect = httpx.RequestError("Mock Network Error", request=httpx.Request("GET", "http://mock-url.com"))

    url_to_test = "https://example.com"
    result = await get_authority(mock_async_html_client, url_to_test)

    expected: AuthorityMetrics = {
        "domain_authority": None, "page_authority": None, "global_rank": None, "authority_status": "network_error"
    }
    assert result == expected
    mock_async_html_client.httpx_get.assert_called_once()
    await mock_async_html_client.close()