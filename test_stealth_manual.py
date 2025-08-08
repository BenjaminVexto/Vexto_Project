# test_stealth_manual.py
import asyncio
# --- MODIFIED: Use async_playwright instead of sync_playwright ---
from playwright.async_api import async_playwright
# --- END MODIFIED ---
from undetected_playwright import stealth_sync

async def run(): # This function is already async
    print("Starting Playwright with undetected_playwright stealth...")
    # --- MODIFIED: Use async with async_playwright() ---
    async with async_playwright() as p:
    # --- END MODIFIED ---
        browser = await p.chromium.launch(headless=False) # Keep headless=False for visual check
        context = await browser.new_context() # Await context creation
        page = await context.new_page() # Await page creation
        
        stealth_sync(page) # Apply stealth (stealth_sync should work on async page object)
        
        print(f"Navigating to https://www.jyskebank.dk with User-Agent: {await page.evaluate('navigator.userAgent')}") # Await evaluate
        print(f"Checking navigator.webdriver: {await page.evaluate('navigator.webdriver')}") # Await evaluate

        await page.goto("https://www.jyskebank.dk", timeout=60_000, wait_until="domcontentloaded") # Await goto
        
        print("Page title:", await page.title()) # Await title
        
        # Save a screenshot to visually inspect the loaded page content
        await page.screenshot(path="jyskebank_stealth.png") # Await screenshot
        print("Screenshot saved to jyskebank_stealth.png")

        await browser.close() # Await close
    print("Playwright session closed.")

if __name__ == "__main__":
    asyncio.run(run())