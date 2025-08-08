# test_bank_access.py
import asyncio
import logging
import sys

from playwright.async_api import async_playwright

# Configure logging to see Playwright's internal logs if any
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

async def test_jyske_bank_direct():
    log.info("Starting direct Playwright test for Jyske Bank...")
    browser = None
    try:
        async with async_playwright() as p:
            log.info("Launching Chromium browser...")
            browser = await p.chromium.launch(headless=False) # Keep headless=False for visual inspection
            
            log.info("Creating new browser context...")
            context = await browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
                locale="da-DK",
                viewport={"width": 1920, "height": 1080}
            )

            log.info("Creating new page...")
            page = await context.new_page()

            # Apply the comprehensive stealth patches
            log.info("Applying stealth patches...")
            await page.add_init_script("""
                // Stealth: Hide webdriver property
                Object.defineProperty(navigator, 'webdriver', { get: () => undefined });

                // Stealth: Spoof plugins
                Object.defineProperty(navigator, 'plugins', {
                    get: () => {
                        const plugins = [
                            {
                                description: 'Portable Document Format',
                                filename: 'internal-pdf-viewer',
                                length: 1,
                                name: 'Chrome PDF Plugin',
                            },
                            {
                                description: 'Portable Document Format',
                                filename: 'mhjfbmdgcfjbbgmofphofjgnbnphflgn',
                                length: 1,
                                name: 'Chrome PDF Viewer',
                            },
                            {
                                description: 'Shockwave Flash 10.2 r154',
                                filename: 'pepflashplayer.dll',
                                length: 1,
                                name: 'Shockwave Flash',
                            }
                        ];
                        plugins.item = (index) => plugins[index];
                        plugins.namedItem = (name) => plugins.find(p => p.name === name);
                        return plugins;
                    },
                });

                // Stealth: Spoof languages
                Object.defineProperty(navigator, 'languages', {
                    get: () => ['da-DK', 'da', 'en-US', 'en'], // Prioritize Danish, then English
                });

                // Stealth: Spoof chrome object
                window.chrome = {
                    runtime: {},
                    app: {},
                    csi: function() {},
                    loadTimes: function() {}
                };

                // Stealth: Hide WebGL vendor and renderer (more robust)
                try {
                    const getParameter = WebGLRenderingContext.prototype.getParameter;
                    WebGLRenderingContext.prototype.getParameter = function(parameter) {
                        // UNMASKED_VENDOR_WEBGL
                        if (parameter === 37445) {
                            return 'Google Inc.';
                        }
                        // UNMASKED_RENDERER_WEBGL
                        if (parameter === 37446) {
                            return 'ANGLE (Google, Inc. OpenGL ES 2.0 Renderer)';
                        }
                        return getParameter.call(this, parameter);
                    };
                } catch (e) {
                    // console.log("WebGL spoofing failed:", e);
                }

                // Stealth: Spoof permissions (e.g., notifications)
                const originalQuery = window.navigator.permissions.query;
                window.navigator.permissions.query = (parameters) =>
                    parameters.name === 'notifications'
                        ? Promise.resolve({ state: Notification.permission })
                        : originalQuery(parameters);

                // Stealth: Hide specific Playwright properties (if they exist)
                try {
                    delete navigator.__proto__.webdriver;
                } catch (e) {}
                try {
                    delete navigator.__proto__.chrome;
                } catch (e) {}
                try {
                    delete navigator.__proto__.plugins;
                } catch (e) {}
                try {
                    delete navigator.__proto__.languages;
                } catch (e) {}
            """)

            target_url = "https://www.jyskebank.dk" # Test with Jyske Bank first
            log.info(f"Navigating to {target_url}...")
            
            # --- Capture console and network events ---
            page.on("console", lambda msg: log.info(f"Browser Console ({msg.type}): {msg.text}"))
            page.on("requestfailed", lambda request: log.warning(f"Request failed: {request.url} - {request.failure().error_text}"))
            page.on("response", lambda response: log.info(f"Response: {response.status} {response.url}"))
            # --- End capturing ---

            await page.goto(target_url, timeout=60_000, wait_until="domcontentloaded") # Increased timeout
            
            # Check navigator.webdriver directly in the browser
            webdriver_status = await page.evaluate("navigator.webdriver")
            log.info(f"Checked navigator.webdriver: {webdriver_status}")

            page_title = await page.title()
            log.info(f"Page title: {page_title}")
            
            # Save a screenshot to confirm what was rendered
            screenshot_path = "jyskebank_direct_test.png"
            await page.screenshot(path=screenshot_path)
            log.info(f"Screenshot saved to {screenshot_path}")

            # Optionally, save the HTML content
            html_content = await page.content()
            html_path = "jyskebank_direct_test.html"
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            log.info(f"HTML content saved to {html_path}")

    except Exception as e:
        log.error(f"An error occurred during direct Playwright test: {e}", exc_info=True)
    finally:
        if browser:
            log.info("Closing browser...")
            await browser.close()
        log.info("Direct Playwright test finished.")

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        log.info("Set WindowsProactorEventLoopPolicy for main thread.")
    asyncio.run(test_jyske_bank_direct())