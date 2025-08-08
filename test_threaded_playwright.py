# test_threaded_playwright.py
import asyncio
import logging
import sys
# --- MODIFIED: Import get_random_user_agent directly ---
from src.vexto.scoring.http_client import AsyncHtmlClient, get_random_user_agent
# --- END MODIFIED ---

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

async def run_threaded_playwright_test():
    log.info("Starting threaded Playwright test for Jyske Bank...")
    client = None
    try:
        client = AsyncHtmlClient(max_connections=1) # Use a single connection for this isolated test
        
        url_to_test = "https://www.jyskebank.dk"
        log.info(f"Attempting to fetch {url_to_test} using Playwright thread...")
        
        # --- MODIFIED: Call get_random_user_agent directly ---
        html_content = await client._pw_thread.fetch(url_to_test, get_random_user_agent())
        # --- END MODIFIED ---

        if html_content:
            html_snippet_display = html_content[:500].replace('\n', ' ').strip()
            log.info(f"Successfully fetched HTML from {url_to_test}. First 500 chars: {html_snippet_display}")
            
            if "cdn-cgi/challenge" in html_content.lower() or "just a moment..." in html_content.lower() or "enable javascript and cookies" in html_content.lower():
                log.warning("Received Cloudflare challenge page content. Stealth not fully bypassed.")
            else:
                log.info("Looks like real content received!")
            
            with open("jyskebank_threaded_test.html", "w", encoding="utf-8") as f:
                f.write(html_content)
            log.info("HTML content saved to jyskebank_threaded_test.html")
        else:
            log.error(f"Failed to fetch HTML from {url_to_test} via Playwright thread.")

    except Exception as e:
        log.error(f"An unexpected error occurred: {e}", exc_info=True)
    finally:
        if client:
            log.info("Closing AsyncHtmlClient resources.")
            await client.close()
        log.info("Threaded Playwright test finished.")

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        log.info("Set WindowsProactorEventLoopPolicy for main thread.")
    asyncio.run(run_threaded_playwright_test())