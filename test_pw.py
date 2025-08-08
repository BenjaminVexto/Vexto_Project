# -*- coding: utf-8 -*-
import asyncio
from playwright.async_api import async_playwright

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page    = await browser.new_page()
        await page.goto("https://example.org")
        print("Page title:", await page.title())       # Expect: Example Domain
        await browser.close()

asyncio.run(main())
