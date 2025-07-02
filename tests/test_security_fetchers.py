# tests/test_security_fetchers.py

import pytest
import respx
from httpx import Response

# Bemærk at vi ikke længere importerer AsyncHtmlClient her, da fixturen klarer det.
from vexto.scoring.security_fetchers import fetch_security_headers

@pytest.mark.asyncio
@respx.mock
async def test_fetch_security_headers_all_present(client): # <-- Testen beder om 'client' fixturen
    """
    Tests that the function correctly identifies all security headers when they are present.
    """
    url = "https://secure-site.com"
    headers = {
        'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
        'Content-Security-Policy': "default-src 'self'",
        'X-Content-Type-Options': 'nosniff',
        'X-Frame-Options': 'DENY'
    }
    respx.head(url).mock(return_value=Response(200, headers=headers))

    # Ingen try/finally er nødvendig. Fixturen håndterer oprydning.
    result = await fetch_security_headers(client, url)

    assert result['hsts_enabled'] is True
    assert result['csp_enabled'] is True
    assert result['x_content_type_options_enabled'] is True
    assert result['x_frame_options_enabled'] is True

@pytest.mark.asyncio
@respx.mock
async def test_fetch_security_headers_none_present(client): # <-- Testen beder om 'client' fixturen
    """
    Tests that the function returns False for all flags when no security headers are present.
    """
    url = "https://insecure-site.com"
    respx.head(url).mock(return_value=Response(200, headers={'content-type': 'text/html'}))

    result = await fetch_security_headers(client, url)

    assert result['hsts_enabled'] is False
    assert result['csp_enabled'] is False
    assert result['x_content_type_options_enabled'] is False
    assert result['x_frame_options_enabled'] is False

@pytest.mark.asyncio
@respx.mock
async def test_fetch_security_headers_network_error(client): # <-- Testen beder om 'client' fixturen
    """
    Tests that the function handles network errors gracefully and returns default values.
    """
    url = "https://error-site.com"
    respx.head(url).mock(side_effect=Exception("Network error"))

    result = await fetch_security_headers(client, url)

    assert result['hsts_enabled'] is False
    assert result['csp_enabled'] is False