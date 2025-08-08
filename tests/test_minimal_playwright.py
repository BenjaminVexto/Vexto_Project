# tests/test_minimal_playwright.py
import pytest
from playwright.async_api import async_playwright

@pytest.mark.asyncio
async def test_playwright_starts():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        assert browser.is_connected()
        await browser.close()