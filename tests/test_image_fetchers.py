# tests/test_image_fetchers.py

import pytest
import respx
from httpx import Response
from bs4 import BeautifulSoup
from vexto.scoring.image_fetchers import fetch_image_stats

HTML_WITH_IMAGES = """
<html><body>
    <img src="/logo.png" alt="logo">
    <img src="https://example.com/image.jpg" alt="a descriptive alt">
    <img src="/no-alt.gif">
    <img src="/broken.jpg">
</body></html>
"""

@pytest.mark.asyncio
@respx.mock
async def test_fetch_image_stats_works(client):
    base_url = "https://test.com"
    
    # Mock netværkskaldene til billederne
    respx.head(f"{base_url}/logo.png").mock(return_value=Response(200, headers={'Content-Length': '10240'})) # 10 KB
    respx.head("https://example.com/image.jpg").mock(return_value=Response(200, headers={'Content-Length': '51200'})) # 50 KB
    respx.head(f"{base_url}/broken.jpg").mock(side_effect=Exception("network error")) # Simuler at ét billede fejler
    
    soup = BeautifulSoup(HTML_WITH_IMAGES, 'lxml')
    result = await fetch_image_stats(client, soup, base_url)
    
    assert result['image_count'] == 4
    assert result['image_alt_count'] == 2
    assert result['image_alt_pct'] == 50
    # Forventet gennemsnit: (10 KB + 50 KB) / 4 billeder = 15 KB
    assert result['avg_image_size_kb'] == 15