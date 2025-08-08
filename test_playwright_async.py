# test_playwright_async.py
import asyncio, sitecustomize        # sitecustomize aktiverer Selector-policy
from playwright.async_api import async_playwright

async def run():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto("https://example.com", timeout=60_000)
        print("SUCCESS – Title is:", await page.title())
        await browser.close()

if __name__ == "__main__":
    asyncio.run(run())
